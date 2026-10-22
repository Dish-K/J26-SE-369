from collections.abc import Iterable
from uuid import UUID

import asyncpg
import pandas as pd

from shared_features import EVENT_SCHEMA, PARAMETERS, WINDOW_WIDTH, window_grid

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
