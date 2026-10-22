"""Feature passes: due windows, reading a task back, and the live and final passes."""

import json
from uuid import uuid4

import pandas as pd
import pytest

from backend.app.telemetry.features import due_window_starts, final_pass, live_pass, load_task
from backend.app.telemetry.ingest import Batch
from backend.app.telemetry.router import store_batch
from shared_features import FEATURE_NAMES, extract_features

from conftest import on_connection, run_sql

from backend.app.telemetry.clock_store import SessionLocks
from backend.app.telemetry.scheduler import run_tick, seconds_to_next_tick

# The grid starts at floor(10:00:05) - 20 s = 09:59:40, and that window ends at 10:00:10.
STARTED = pd.Timestamp("2026-10-22T10:00:05Z")


def _at(clock):
    return pd.Timestamp(f"2026-10-22T{clock}Z")


def test_a_window_is_due_only_after_its_grace():
    assert list(due_window_starts(STARTED, _at("10:00:11.999"), [])) == []
    assert list(due_window_starts(STARTED, _at("10:00:12"), [])) == [_at("09:59:40")]


def test_a_stored_window_is_not_due_again():
    assert list(due_window_starts(STARTED, _at("10:00:12"), [_at("09:59:40")])) == []


def test_missed_windows_are_all_due_together():
    # A minute without ticks, every window that ended by 10:01:00 is due, except the stored ones.
    due = due_window_starts(STARTED, _at("10:01:02"), [_at("09:59:40"), _at("10:00:00")])
    assert list(due) == [_at("09:59:50"), _at("10:00:10"), _at("10:00:20"), _at("10:00:30")]


def test_a_task_read_from_the_database_runs_through_the_extractor(database):
    session, task = uuid4(), uuid4()
    started = pd.Timestamp.now(tz="UTC").floor("10s")
    run_sql(
        "INSERT INTO sessions (session_id, candidate_id, language, started_at) "
        "VALUES ($1, $2, 'python', $3)",
        session, uuid4(), started.to_pydatetime(),
    )
    run_sql(
        "INSERT INTO tasks (task_id, session_id, seq, origin, status, created_at, started_at) "
        "VALUES ($1, $2, 0, 'original', 'active', $3, $3)",
        task, session, started.to_pydatetime(),
    )
    batch = Batch.model_validate({
        "session_id": str(session),
        "load_seq": 0,
        "batch_seq": 0,
        "events": [
            {"seq_in_batch": 0, "kind": 0, "perf_now": 1000.0, "task_id": str(task),
             "key_code": "KeyA", "key_class": 0},
            {"seq_in_batch": 1, "kind": 1, "perf_now": 1080.0, "task_id": str(task),
             "key_code": "KeyA", "key_class": 0},
        ],
    })

    try:
        # As if the page loaded exactly when the task started: the key press lands 1 s in.
        on_connection(lambda connection: store_batch(connection, batch, started.timestamp() * 1000))
        events, tasks = on_connection(lambda connection: load_task(connection, task))
        frame = extract_features(events, tasks, run_id="live")

        assert len(events) == 2
        assert list(frame["window_start"]) == [
            started - pd.Timedelta(seconds=20), started - pd.Timedelta(seconds=10), started,
        ]
        assert list(frame["keystroke_count"]) == [1, 1, 1]
    finally:
        for table in ("events", "ingested_batches", "tasks", "sessions"):
            run_sql(f"DELETE FROM {table} WHERE session_id = $1", session)


# A fixed past task: it started at 10:00:00, and its page loaded at that moment.
TASK_START = _at("10:00:00")


@pytest.fixture
def task(database):
    session, task_id = uuid4(), uuid4()
    run_sql(
        "INSERT INTO sessions (session_id, candidate_id, language, started_at) "
        "VALUES ($1, $2, 'python', $3)",
        session, uuid4(), TASK_START.to_pydatetime(),
    )
    run_sql(
        "INSERT INTO tasks (task_id, session_id, seq, origin, status, created_at, started_at) "
        "VALUES ($1, $2, 0, 'original', 'active', $3, $3)",
        task_id, session, TASK_START.to_pydatetime(),
    )
    yield session, task_id
    for table in ("feature_vectors", "events", "ingested_batches", "tasks", "sessions"):
        run_sql(f"DELETE FROM {table} WHERE session_id = $1", session)


def _press(session, task_id, batch_seq, *seconds):
    # One key press and release at each given second after the task start.
    events = []
    for second in seconds:
        for kind, perf_now in ((0, second * 1000.0), (1, second * 1000.0 + 80)):
            events.append({"seq_in_batch": len(events), "kind": kind, "perf_now": perf_now,
                           "task_id": str(task_id), "key_code": "KeyA", "key_class": 0})
    batch = Batch.model_validate(
        {"session_id": str(session), "load_seq": 0, "batch_seq": batch_seq, "events": events}
    )
    on_connection(lambda connection: store_batch(connection, batch, TASK_START.timestamp() * 1000))


def _vectors(session, run_id):
    rows = run_sql(
        "SELECT window_start, features FROM feature_vectors "
        "WHERE session_id = $1 AND run_id = $2 ORDER BY window_start",
        session, run_id,
    )
    return {row["window_start"].strftime("%H:%M:%S"): json.loads(row["features"]) for row in rows}


def _live(task_id, clock):
    return on_connection(lambda connection: live_pass(connection, task_id, _at(clock)))


def _submit(task_id, clock):
    run_sql(
        "UPDATE tasks SET status = 'submitted', submitted_at = $2 WHERE task_id = $1",
        task_id, _at(clock).to_pydatetime(),
    )


def test_the_live_pass_stores_every_due_window_including_quiet_ones(task):
    session, task_id = task
    _press(session, task_id, 0, 5)
    assert _live(task_id, "10:01:02") == 6

    live = _vectors(session, "live")
    assert list(live) == ["09:59:40", "09:59:50", "10:00:00", "10:00:10", "10:00:20", "10:00:30"]
    assert [vector["keystroke_count"] for vector in live.values()] == [1, 1, 1, 0, 0, 0]
    assert [vector["window_task_coverage_ms"] for vector in live.values()] == [
        10_000, 20_000, 30_000, 30_000, 30_000, 30_000,
    ]
    assert live["10:00:00"]["mean_dwell_ms"] == 80
    # A quiet window counts nothing and has no statistics, but still has every key.
    assert live["10:00:30"]["mean_dwell_ms"] is None
    assert len(live["10:00:30"]) == len(FEATURE_NAMES)


def test_a_late_event_never_changes_a_stored_live_vector(task):
    session, task_id = task
    _press(session, task_id, 0, 5)
    _live(task_id, "10:01:02")
    before = _vectors(session, "live")

    # A key from 10:00:15 arrives only now, after its windows were stored.
    _press(session, task_id, 1, 15)
    assert _live(task_id, "10:01:12") == 1

    after = _vectors(session, "live")
    assert {start: after[start] for start in before} == before
    assert list(after)[-1] == "10:00:40"


def test_the_live_pass_catches_up_after_missed_ticks(task):
    session, task_id = task
    _press(session, task_id, 0, 5)
    assert _live(task_id, "10:00:12") == 1
    # No tick for 50 s: the next one fills every window that became due meanwhile.
    assert _live(task_id, "10:01:02") == 5
    assert _live(task_id, "10:01:02") == 0
    assert len(_vectors(session, "live")) == 6


def test_the_final_pass_writes_beside_the_live_rows(task):
    session, task_id = task
    _press(session, task_id, 0, 5)
    _live(task_id, "10:01:02")
    _press(session, task_id, 1, 15)
    _submit(task_id, "10:01:00")
    live_before = _vectors(session, "live")

    assert on_connection(lambda connection: final_pass(connection, task_id)) == 9

    final = _vectors(session, "final")
    assert (list(final)[0], list(final)[-1]) == ("09:59:40", "10:01:00")
    assert final["10:00:00"]["keystroke_count"] == 2
    assert live_before["10:00:00"]["keystroke_count"] == 1
    assert _vectors(session, "live") == live_before


def test_a_submitted_task_gets_no_more_live_windows(task):
    session, task_id = task
    _submit(task_id, "10:00:30")
    assert _live(task_id, "10:01:02") == 0
    assert _vectors(session, "live") == {}


def test_the_final_pass_needs_a_submitted_task(task):
    _, task_id = task
    with pytest.raises(ValueError):
        on_connection(lambda connection: final_pass(connection, task_id))

def test_ticks_fall_two_seconds_after_each_window_boundary():
    assert seconds_to_next_tick(1_000.0) == 2.0
    assert seconds_to_next_tick(1_001.5) == 0.5
    assert seconds_to_next_tick(1_002.0) == 10.0


def test_a_tick_makes_windows_only_for_connected_sessions(task):
    session, task_id = task
    _press(session, task_id, 0, 5)
    locks = SessionLocks()

    def tick():
        return on_connection(lambda connection: run_tick(connection, locks, _at("10:01:02")))

    tick()
    assert _vectors(session, "live") == {}

    owner = object()
    locks.take(session, owner)
    tick()
    assert len(_vectors(session, "live")) == 6
