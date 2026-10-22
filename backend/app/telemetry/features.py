import asyncio
import json
from collections.abc import Iterable
from uuid import UUID

import asyncpg
import pandas as pd

from shared_features import (
    EVENT_SCHEMA,
    PARAMETERS,
    WINDOW_WIDTH,
    extract_features,
    feature_payload,
    window_grid,
)

WIDTH = pd.Timedelta(seconds=WINDOW_WIDTH)
GRACE = pd.Timedelta(milliseconds=PARAMETERS["window_grace_ms"])

# Exactly the columns the extractor reads, so the query and the schema cannot drift apart.
EXTRACTOR_COLUMNS = tuple(EVENT_SCHEMA.columns)
TASK_COLUMNS = ("task_id", "session_id", "started_at", "submitted_at")

_TASK = f"SELECT {', '.join(TASK_COLUMNS)} FROM tasks WHERE task_id = $1"

_TASK_EVENTS = f"""
SELECT {", ".join(EXTRACTOR_COLUMNS)}
FROM events
WHERE session_id = $1 AND task_id = $2
ORDER BY load_seq, batch_seq, seq_in_batch, change_index
"""

def due_window_starts(
    started_at: pd.Timestamp, now: pd.Timestamp, stored: Iterable[pd.Timestamp]
) -> pd.DatetimeIndex:
    # Due once the window has ended and the grace for late events has passed.
    grid = window_grid(started_at, now)
    due = grid[grid + WIDTH + GRACE <= now]
    return due.difference(pd.DatetimeIndex(list(stored), tz="UTC"))

async def load_task(
    connection: asyncpg.Connection, task_id: UUID
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = await connection.fetchrow(_TASK, task_id)
    if task is None:
        raise LookupError(f"no task {task_id}")

    rows = await connection.fetch(_TASK_EVENTS, task["session_id"], task_id)
    events = pd.DataFrame([dict(row) for row in rows], columns=list(EXTRACTOR_COLUMNS))
    tasks = pd.DataFrame([dict(task)], columns=list(TASK_COLUMNS))
    return events, tasks

_STORED_WINDOWS = """
SELECT window_start FROM feature_vectors
WHERE session_id = $1 AND task_id = $2 AND run_id = $3
"""

# The natural key makes a stored vector final: a later pass can add windows, never change one.
_INSERT_VECTOR = """
INSERT INTO feature_vectors
  (session_id, task_id, window_start, window_end, run_id, schema_version, features)
VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
ON CONFLICT (session_id, task_id, window_start, window_end, run_id) DO NOTHING
"""

async def store_vectors(connection: asyncpg.Connection, frame: pd.DataFrame) -> int:
    rows = []
    for record in frame.to_dict("records"):
        payload = feature_payload(record)
        rows.append((
            record["session_id"],
            record["task_id"],
            pd.Timestamp(record["window_start"]).to_pydatetime(),
            pd.Timestamp(record["window_end"]).to_pydatetime(),
            payload["run_id"],
            payload["schema_version"],
            json.dumps(payload["features"]),
        ))
    if rows:
        await connection.executemany(_INSERT_VECTOR, rows)
    return len(rows)

async def live_pass(connection: asyncpg.Connection, task_id: UUID, now: pd.Timestamp) -> int:
    events, tasks = await load_task(connection, task_id)
    task = tasks.iloc[0]
    if not pd.isna(task["submitted_at"]):
        return 0

    stored = await connection.fetch(_STORED_WINDOWS, task["session_id"], task_id, "live")
    due = due_window_starts(
        pd.Timestamp(task["started_at"]), now, [row["window_start"] for row in stored]
    )
    if due.empty:
        return 0

    # The extractor ends an unsubmitted task at its last event; in memory only, it runs to the latest due window.
    tasks = tasks.assign(submitted_at=due.max() + WIDTH)
    frame = await asyncio.to_thread(
        extract_features, events, tasks, run_id="live", window_starts=due
    )
    return await store_vectors(connection, frame)

async def final_pass(connection: asyncpg.Connection, task_id: UUID) -> int:
    events, tasks = await load_task(connection, task_id)
    if pd.isna(tasks.iloc[0]["submitted_at"]):
        raise ValueError(f"task {task_id} is not submitted")

    frame = await asyncio.to_thread(extract_features, events, tasks, run_id="final")
    return await store_vectors(connection, frame)
