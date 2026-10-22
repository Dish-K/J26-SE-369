import asyncio
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from pydantic import ValidationError
from datetime import UTC, datetime, timedelta

from backend.app.db.pool import close_db_pool, create_db_pool
from backend.app.main import app
from backend.app.telemetry.clock import HANDSHAKE_SAMPLES
from backend.app.telemetry.ingest import EVENT_COLUMNS, Batch, event_rows
from backend.app.telemetry.router import acked_up_to, store_batch

SESSION = uuid4()
TASK = uuid4()
OTHER_TASK = uuid4()


def _batch(*events, batch_seq=0, session=SESSION, load_seq=0):
    return Batch.model_validate(
        {"session_id": str(session), "load_seq": load_seq, "batch_seq": batch_seq, "events": list(events)}
    )


def _key(seq, perf_now, kind=0, task=TASK):
    return {"seq_in_batch": seq, "kind": kind, "perf_now": perf_now, "task_id": str(task), "key_code": "KeyA"}


def _column(rows, name):
    return [row[EVENT_COLUMNS.index(name)] for row in rows]


def test_the_browser_cannot_send_gap_ms():
    with pytest.raises(ValidationError):
        _batch(_key(0, 100.0) | {"gap_ms": 5.0})


def test_a_keystroke_without_a_task_is_refused():
    with pytest.raises(ValidationError):
        _batch({"seq_in_batch": 0, "kind": 0, "perf_now": 1.0})


def test_an_empty_batch_is_refused():
    with pytest.raises(ValidationError):
        _batch()


def test_gap_ms_is_measured_from_the_previous_typing_event():
    rows = event_rows(_batch(_key(0, 100.0), _key(1, 250.0, kind=1), _key(2, 400.0)), {}, 0.0)
    assert _column(rows, "gap_ms") == [None, 150.0, 150.0]


def test_gap_ms_continues_from_the_previous_batch():
    # {TASK: 400.0} stands in for the last typing event already in the database.
    rows = event_rows(_batch(_key(0, 1_000.0), batch_seq=1), {TASK: 400.0}, 0.0)
    assert _column(rows, "gap_ms") == [600.0]


def test_a_non_typing_event_gets_no_gap_and_does_not_split_a_pause():
    # The 18 s pause before the last keystroke must survive the blur in between.
    blur = {"seq_in_batch": 1, "kind": 3, "perf_now": 5_000.0}
    rows = event_rows(_batch(_key(0, 1_000.0), blur, _key(2, 19_000.0)), {}, 0.0)
    assert _column(rows, "gap_ms") == [None, None, 18_000.0]


def test_gap_ms_restarts_for_each_task():
    rows = event_rows(_batch(_key(0, 100.0), _key(1, 200.0, task=OTHER_TASK)), {}, 0.0)
    assert _column(rows, "gap_ms") == [None, None]


def test_time_is_perf_now_plus_the_offset():
    rows = event_rows(_batch(_key(0, 1_500.0)), {}, 1_000.0)
    assert _column(rows, "time") == [datetime(1970, 1, 1, 0, 0, 2, 500_000, tzinfo=UTC)]


def test_rows_carry_the_batch_fields_in_column_order():
    row = event_rows(_batch(_key(0, 1.0), batch_seq=7), {}, 0.0)[0]
    assert len(row) == len(EVENT_COLUMNS) == 38
    assert row[EVENT_COLUMNS.index("session_id")] == SESSION
    assert row[EVENT_COLUMNS.index("batch_seq")] == 7
    assert row[EVENT_COLUMNS.index("seq_in_batch")] == 0


# Skips rather than fails without a database, so C2 can still run tests/telemetry alone.
@pytest.fixture(scope="module")
def database():
    async def connect():
        await close_db_pool(await create_db_pool())

    try:
        asyncio.run(connect())
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"database not available: {exc}")


_NEW_SESSION = (
    "INSERT INTO sessions (session_id, candidate_id, language, started_at) "
    "VALUES ($1, $2, 'python', now())"
)
# sessions last: the other tables point to it.
_SESSION_TABLES = ("events", "ingested_batches", "clock_sync_samples", "feature_vectors", "task_submissions", "tasks", "sessions" )


# Each test gets its own session row and removes its rows afterwards, even on failure.
async def _in_new_session(scenario):
    pool = await create_db_pool()
    session = uuid4()
    try:
        async with pool.acquire() as connection:
            await connection.execute(_NEW_SESSION, session, uuid4())
            try:
                await scenario(connection, session)
            finally:
                for table in _SESSION_TABLES:
                    await connection.execute(f"DELETE FROM {table} WHERE session_id = $1", session)
    finally:
        await close_db_pool(pool)


async def _stored(connection, session):
    return await connection.fetch(
        "SELECT batch_seq, seq_in_batch, gap_ms, time FROM events "
        "WHERE session_id = $1 ORDER BY load_seq, batch_seq, seq_in_batch",
        session,
    )


# The repo has no async test plugin, so each test runs its scenario with asyncio.run.
def test_a_batch_sent_twice_is_stored_once(database):
    async def scenario(connection, session):
        batch = _batch(_key(0, 100.0), _key(1, 200.0), session=session)
        assert await store_batch(connection, batch, 0.0) is True
        assert await store_batch(connection, batch, 0.0) is False
        assert len(await _stored(connection, session)) == 2

    asyncio.run(_in_new_session(scenario))


def test_stored_events_have_time_gap_and_sequence_fields(database):
    async def scenario(connection, session):
        await store_batch(connection, _batch(_key(0, 100.0), _key(1, 250.0), session=session), 0.0)
        await store_batch(connection, _batch(_key(0, 1_000.0), batch_seq=1, session=session), 0.0)

        rows = await _stored(connection, session)
        assert [(row["batch_seq"], row["seq_in_batch"]) for row in rows] == [(0, 0), (0, 1), (1, 0)]
        assert [row["gap_ms"] for row in rows] == [None, 150.0, 750.0]
        assert rows[0]["time"] == datetime(1970, 1, 1, 0, 0, 0, 100_000, tzinfo=UTC)

    asyncio.run(_in_new_session(scenario))


def test_an_out_of_order_batch_is_stored_but_not_acked_early(database):
    async def scenario(connection, session):
        assert await acked_up_to(connection, session, 0) == -1

        await store_batch(connection, _batch(_key(0, 1.0), session=session), 0.0)
        await store_batch(connection, _batch(_key(0, 3.0), batch_seq=2, session=session), 0.0)
        assert await acked_up_to(connection, session, 0) == 0

        await store_batch(connection, _batch(_key(0, 2.0), batch_seq=1, session=session), 0.0)
        assert await acked_up_to(connection, session, 0) == 2
        assert [row["batch_seq"] for row in await _stored(connection, session)] == [0, 1, 2]

    asyncio.run(_in_new_session(scenario))


def test_a_batch_for_an_unknown_session_is_refused(database):
    async def scenario(connection, session):
        unknown = uuid4()
        # Refused by the foreign key on ingested_batches, before any event is written.
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await store_batch(connection, _batch(_key(0, 1.0), session=unknown), 0.0)
        assert await connection.fetchval(
            "SELECT count(*) FROM events WHERE session_id = $1", unknown
        ) == 0

    asyncio.run(_in_new_session(scenario))


def test_event_columns_match_the_events_table(database):
    # Fails when a migration adds an events column that Event does not list yet.
    async def scenario(connection, session):
        columns = await connection.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'events'"
        )
        assert {column["column_name"] for column in columns} == set(EVENT_COLUMNS)

    asyncio.run(_in_new_session(scenario))


def _sql(statement, *args):
    async def run():
        pool = await create_db_pool()
        try:
            return await pool.fetch(statement, *args)
        finally:
            await close_db_pool(pool)

    return asyncio.run(run())


# The endpoint runs with its own pool, so the session row must be committed before it connects.
@pytest.fixture
def session(database):
    session_id = uuid4()
    _sql(_NEW_SESSION, session_id, uuid4())
    yield session_id
    for table in _SESSION_TABLES:
        _sql(f"DELETE FROM {table} WHERE session_id = $1", session_id)


def _ack(up_to, load_seq=0):
    return {"ack": {"load_seq": load_seq, "up_to_batch_seq": up_to}}


class _Page:
    # One page load's browser clock: milliseconds since that page loaded.
    def __init__(self, loaded_ms_ago=0.0):
        self.start = time.perf_counter() * 1000 - loaded_ms_ago

    def now(self):
        return time.perf_counter() * 1000 - self.start


# What the browser does on connect: hello, the timed pings, then all samples in one message.
def _handshake(websocket, session, page=None):
    page = page or _Page()
    websocket.send_json({"hello": {"session_id": str(session)}})
    samples = []
    for _ in range(HANDSHAKE_SAMPLES):
        websocket.send_json({"ping": {"t0": page.now()}})
        pong = websocket.receive_json()["pong"]
        samples.append({**pong, "t3": page.now()})
    websocket.send_json({"clock": {"samples": samples}})
    return websocket.receive_json()["ready"]["load_seq"]


def _close_code(message, session=None):
    with TestClient(app) as client, client.websocket_connect("/ws/telemetry") as websocket:
        if session is not None:
            _handshake(websocket, session)
        websocket.send_text(message)
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
    return closed.value.code


def test_the_endpoint_acks_a_resent_batch_without_storing_it_again(session):
    batch = _batch(_key(0, 1.0), session=session).model_dump_json()
    with TestClient(app) as client, client.websocket_connect("/ws/telemetry") as websocket:
        assert _handshake(websocket, session) == 0
        websocket.send_text(batch)
        assert websocket.receive_json() == _ack(0)
        websocket.send_text(batch)
        assert websocket.receive_json() == _ack(0)

    assert _sql("SELECT count(*) FROM events WHERE session_id = $1", session)[0][0] == 1


def test_the_endpoint_never_acks_past_a_missing_batch(session):
    with TestClient(app) as client, client.websocket_connect("/ws/telemetry") as websocket:
        _handshake(websocket, session)
        for batch_seq, up_to in ((0, 0), (2, 0), (1, 2)):
            batch = _batch(_key(0, float(batch_seq)), batch_seq=batch_seq, session=session)
            websocket.send_text(batch.model_dump_json())
            assert websocket.receive_json() == _ack(up_to)


def test_an_invalid_batch_closes_the_connection(database):
    assert _close_code('{"events": []}') == 1008


def test_a_handshake_for_an_unknown_session_closes_the_connection(database):
    with TestClient(app) as client, client.websocket_connect("/ws/telemetry") as websocket:
        with pytest.raises(WebSocketDisconnect) as closed:
            _handshake(websocket, uuid4())
    assert closed.value.code == 1008


# modifiers breaks a CHECK; key_class is too big for SMALLINT and fails inside COPY.
@pytest.mark.parametrize("bad", [{"modifiers": 99}, {"key_class": 70_000}])
def test_a_batch_the_database_rejects_closes_the_connection(session, bad):
    message = _batch(_key(0, 1.0) | bad, session=session).model_dump_json()
    assert _close_code(message, session) == 1011


_SESSION_START = {
    "editor_profile": "research-v1",
    "editor_config_hash": "a" * 64,
    "user_agent": "pytest",
    "platform": "Linux x86_64",
    "screen_w": 1920,
    "screen_h": 1080,
    "viewport_w": 1280,
    "viewport_h": 720,
    "device_pixel_ratio": 1.5,
    "locale": "en-GB",
    "timezone": "Asia/Colombo",
}


# Deletes every session a test started through the endpoint, even on failure.
@pytest.fixture
def started(database):
    sessions = []
    yield sessions
    for session_id in sessions:
        for table in _SESSION_TABLES:
            _sql(f"DELETE FROM {table} WHERE session_id = $1", session_id)


def _start(client, started):
    response = client.post("/telemetry/sessions", json=_SESSION_START)
    assert response.status_code == 201
    ids = {key: UUID(value) for key, value in response.json().items()}
    started.append(ids["session_id"])
    return ids


def test_starting_a_session_creates_the_session_and_its_first_task(started):
    with TestClient(app) as client:
        ids = _start(client, started)

    session = _sql("SELECT * FROM sessions WHERE session_id = $1", ids["session_id"])[0]
    assert {key: session[key] for key in _SESSION_START} == _SESSION_START
    assert session["language"] == "python"

    task = _sql("SELECT * FROM tasks WHERE task_id = $1", ids["task_id"])[0]
    assert (task["session_id"], task["seq"], task["origin"], task["status"]) == (
        ids["session_id"], 0, "original", "active",
    )


def test_a_started_session_accepts_keystrokes(started):
    with TestClient(app) as client:
        ids = _start(client, started)
        batch = _batch(_key(0, 1.0, task=ids["task_id"]), session=ids["session_id"])
        with client.websocket_connect("/ws/telemetry") as websocket:
            assert _handshake(websocket, ids["session_id"]) == 0            
            websocket.send_text(batch.model_dump_json())
            assert websocket.receive_json() == _ack(0)


# The browser may not pick the language; a bad hash or size is refused before the database.
@pytest.mark.parametrize(
    "bad", [{"editor_config_hash": "not-a-hash"}, {"screen_w": 0}, {"language": "java"}]
)
def test_a_bad_session_start_is_refused(database, bad):
    with TestClient(app) as client:
        response = client.post("/telemetry/sessions", json=_SESSION_START | bad)
    assert response.status_code == 422


def test_a_message_before_hello_is_refused(database):
    assert _close_code('{"ping": {"t0": 1.0}}') == 1008


def test_a_batch_before_the_clock_handshake_is_refused(session):
    with TestClient(app) as client, client.websocket_connect("/ws/telemetry") as websocket:
        websocket.send_json({"hello": {"session_id": str(session)}})
        websocket.send_text(_batch(_key(0, 1.0), session=session).model_dump_json())
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_json()
    assert closed.value.code == 1008


def test_a_second_tab_is_refused_until_the_first_one_closes(session):
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as first:
            first.send_json({"hello": {"session_id": str(session)}})
            # A round trip proves the first hello was handled before the second tab arrives.
            first.send_json({"ping": {"t0": 1.0}})
            first.receive_json()

            with client.websocket_connect("/ws/telemetry") as second:
                second.send_json({"hello": {"session_id": str(session)}})
                with pytest.raises(WebSocketDisconnect) as closed:
                    second.receive_json()
            assert closed.value.code == 4409

        with client.websocket_connect("/ws/telemetry") as third:
            assert _handshake(third, session) == 0


def test_a_reload_starts_a_new_page_load_in_the_same_session(session):
    old_page = _Page(loaded_ms_ago=60_000)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/telemetry") as websocket:
            assert _handshake(websocket, session, old_page) == 0
            batch = _batch(_key(0, old_page.now()), _key(1, old_page.now(), kind=1), session=session)
            websocket.send_text(batch.model_dump_json())
            assert websocket.receive_json() == _ack(0)

        # The reload: a new connection, and a browser clock that starts again near 0.
        new_page = _Page()
        with client.websocket_connect("/ws/telemetry") as websocket:
            assert _handshake(websocket, session, new_page) == 1
            batch = _batch(
                _key(0, new_page.now()), _key(1, new_page.now(), kind=1), session=session, load_seq=1
            )
            websocket.send_text(batch.model_dump_json())
            assert websocket.receive_json() == _ack(0, load_seq=1)

    rows = _sql(
        "SELECT load_seq, batch_seq, perf_now, gap_ms, time FROM events "
        "WHERE session_id = $1 ORDER BY load_seq, batch_seq, seq_in_batch",
        session,
    )
    assert [(row["load_seq"], row["batch_seq"]) for row in rows] == [(0, 0), (0, 0), (1, 0), (1, 0)]

    last_before, first_after = rows[1], rows[2]
    assert first_after["gap_ms"] is None
    assert first_after["perf_now"] < last_before["perf_now"]
    assert first_after["time"] > last_before["time"]
    assert abs(first_after["time"] - datetime.now(UTC)) < timedelta(minutes=1)

    samples = _sql(
        "SELECT load_seq, count(*) AS stored, count(*) FILTER (WHERE accepted) AS used "
        "FROM clock_sync_samples WHERE session_id = $1 GROUP BY load_seq ORDER BY load_seq",
        session,
    )
    assert [(row["load_seq"], row["stored"], row["used"]) for row in samples] == [
        (0, HANDSHAKE_SAMPLES, HANDSHAKE_SAMPLES // 2),
        (1, HANDSHAKE_SAMPLES, HANDSHAKE_SAMPLES // 2),
    ]

def _submit_message(task_id, code="print(1)", version_id=3):
    return {"submit": {"task_id": str(task_id), "code_text": code, "version_id": version_id}}


def test_submitting_stores_the_code_and_closes_the_task(started):
    with TestClient(app) as client:
        ids = _start(client, started)
        with client.websocket_connect("/ws/telemetry") as websocket:
            assert _handshake(websocket, ids["session_id"]) == 0
            websocket.send_json(_submit_message(ids["task_id"]))
            assert websocket.receive_json() == {"submitted": {"task_id": str(ids["task_id"])}}

    submission = _sql("SELECT * FROM task_submissions WHERE task_id = $1", ids["task_id"])[0]
    assert (submission["code_text"], submission["version_id"], submission["load_seq"]) == ("print(1)", 3, 0)
    task = _sql("SELECT status, submitted_at FROM tasks WHERE task_id = $1", ids["task_id"])[0]
    assert task["status"] == "submitted"
    assert task["submitted_at"] == submission["submitted_at"]


def test_a_resent_submit_gets_the_same_answer(started):
    with TestClient(app) as client:
        ids = _start(client, started)
        with client.websocket_connect("/ws/telemetry") as websocket:
            _handshake(websocket, ids["session_id"])
            for _ in range(2):
                websocket.send_json(_submit_message(ids["task_id"]))
                assert websocket.receive_json() == {"submitted": {"task_id": str(ids["task_id"])}}

    assert _sql("SELECT count(*) FROM task_submissions WHERE task_id = $1", ids["task_id"])[0][0] == 1


def test_submitting_a_task_that_is_not_open_is_refused(started):
    with TestClient(app) as client:
        ids = _start(client, started)
        with client.websocket_connect("/ws/telemetry") as websocket:
            _handshake(websocket, ids["session_id"])
            websocket.send_json(_submit_message(uuid4()))
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
    assert closed.value.code == 1008
