import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
import pytest
from pydantic import ValidationError

from backend.app.db.pool import close_db_pool, create_db_pool
from backend.app.telemetry.ingest import EVENT_COLUMNS, Batch, event_rows
from backend.app.telemetry.router import acked_up_to, store_batch

SESSION = uuid4()
TASK = uuid4()
OTHER_TASK = uuid4()


def _batch(*events, batch_seq=0, session=SESSION):
    return Batch.model_validate(
        {"session_id": str(session), "load_seq": 0, "batch_seq": batch_seq, "events": list(events)}
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


# Each test gets its own session row and removes its rows afterwards, even on failure.
async def _in_new_session(scenario):
    pool = await create_db_pool()
    session = uuid4()
    try:
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO sessions (session_id, candidate_id, language, started_at) "
                "VALUES ($1, $2, 'python', now())",
                session,
                uuid4(),
            )
            try:
                await scenario(connection, session)
            finally:
                # sessions last: ingested_batches has a foreign key to it.
                for table in ("events", "ingested_batches", "sessions"):
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
