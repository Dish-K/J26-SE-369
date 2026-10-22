"""Feature passes: which windows are due, and reading a task back for the extractor."""

from uuid import uuid4

import pandas as pd

from backend.app.telemetry.features import due_window_starts, load_task
from backend.app.telemetry.ingest import Batch
from backend.app.telemetry.router import store_batch
from shared_features import extract_features

from conftest import on_connection, run_sql

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
