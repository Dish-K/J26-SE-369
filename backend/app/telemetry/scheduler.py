import asyncio
import logging
import time

import asyncpg
import pandas as pd

from backend.app.telemetry.clock_store import SessionLocks
from backend.app.telemetry.features import GRACE, live_pass
from shared_features import WINDOW_STEP

logger = logging.getLogger(__name__)

_OPEN_TASKS = "SELECT task_id, session_id FROM tasks WHERE status = 'active'"


def seconds_to_next_tick(now: float) -> float:
    # A tick falls 2 s after every 10 s boundary, when the windows that just ended become due.
    grace = GRACE.total_seconds()
    next_tick = ((now - grace) // WINDOW_STEP + 1) * WINDOW_STEP + grace
    return next_tick - now


async def run_tick(connection: asyncpg.Connection, locks: SessionLocks, now: pd.Timestamp) -> int:
    stored = 0
    for task in await connection.fetch(_OPEN_TASKS):
        if not locks.is_held(task["session_id"]):
            continue
        try:
            stored += await live_pass(connection, task["task_id"], now)
        except Exception:
            # One broken task must not stop the others from getting their windows.
            logger.exception("Live pass failed for task %s", task["task_id"])
    return stored


async def run_scheduler(pool: asyncpg.Pool, locks: SessionLocks) -> None:
    while True:
        await asyncio.sleep(seconds_to_next_tick(time.time()))
        try:
            async with pool.acquire() as connection:
                await run_tick(connection, locks, pd.Timestamp.now(tz="UTC"))
        except Exception:
            # The scheduler must outlive a failed tick, for example while the database restarts.
            logger.exception("Live feature tick failed")
