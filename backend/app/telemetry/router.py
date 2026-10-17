import logging
from uuid import UUID

import asyncpg
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from backend.app.telemetry.ingest import EVENT_COLUMNS, TYPING_KINDS, Batch, event_rows

logger = logging.getLogger(__name__)

router = APIRouter()

# RETURNING gives a row only for a new batch, which is how a resend is spotted.
_RECORD_BATCH = """
INSERT INTO ingested_batches (session_id, load_seq, batch_seq)
VALUES ($1, $2, $3)
ON CONFLICT DO NOTHING
RETURNING batch_seq
"""

# Latest by sequence, never by time; served by events_extract_order_idx.
_PREVIOUS_TYPING = """
SELECT perf_now
FROM events
WHERE session_id = $1 AND task_id = $2 AND load_seq = $3
  AND batch_seq < $4 AND kind = ANY($5)
ORDER BY batch_seq DESC, seq_in_batch DESC
LIMIT 1
"""

# The first missing batch is 0 or one past a stored batch; everything below it is stored.
_ACKED_UP_TO = """
SELECT min(n) - 1
FROM (
  SELECT 0 AS n
  UNION ALL
  SELECT batch_seq + 1 FROM ingested_batches WHERE session_id = $1 AND load_seq = $2
) AS candidates
WHERE NOT EXISTS (
  SELECT 1 FROM ingested_batches
  WHERE session_id = $1 AND load_seq = $2 AND batch_seq = candidates.n
)
"""

# COPY raises a plain OverflowError, not a database error, for a number too big for its column.
_STORE_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OverflowError, OSError)

async def previous_typing(connection: asyncpg.Connection, batch: Batch) -> dict[UUID, float]:
    task_ids = {event.task_id for event in batch.events if event.kind in TYPING_KINDS}
    previous = {}

    for task_id in task_ids:
        perf_now = await connection.fetchval(
            _PREVIOUS_TYPING,
            batch.session_id,
            task_id,
            batch.load_seq,
            batch.batch_seq,
            sorted(TYPING_KINDS),
        )
        if perf_now is not None:
            previous[task_id] = perf_now

    return previous

async def store_batch(connection: asyncpg.Connection, batch: Batch, offset_ms: float) -> bool:
    # One transaction, so a batch is never recorded without its events or stored twice.
    async with connection.transaction():
        recorded = await connection.fetchval(
            _RECORD_BATCH, batch.session_id, batch.load_seq, batch.batch_seq
        )
        # Not "if not recorded": batch 0 returns 0, which is false.
        if recorded is None:
            return False

        rows = event_rows(batch, await previous_typing(connection, batch), offset_ms)
        await connection.copy_records_to_table("events", records=rows, columns=EVENT_COLUMNS)
        return True

async def acked_up_to(connection: asyncpg.Connection, session_id: UUID, load_seq: int) -> int:
    return await connection.fetchval(_ACKED_UP_TO, session_id, load_seq)

@router.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    pool = websocket.app.state.db_pool

    try:
        while True:
            message = await websocket.receive_text()

            try:
                batch = Batch.model_validate_json(message)
            except ValidationError as exc:
                # include_input=False keeps pasted text out of the logs.
                logger.warning(
                    "Refused telemetry batch: %s",
                    exc.errors(include_input=False, include_url=False),
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid batch")
                return

            try:
                async with pool.acquire() as connection:
                    # Zero until the Step 5 clock handshake gives a real offset.
                    await store_batch(connection, batch, offset_ms=0.0)
                    up_to = await acked_up_to(connection, batch.session_id, batch.load_seq)
            except asyncpg.ForeignKeyViolationError:
                logger.warning("Refused telemetry batch for unknown session %s", batch.session_id)
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="unknown session")
                return
            except _STORE_ERRORS:
                logger.exception(
                    "Could not store batch %s of session %s", batch.batch_seq, batch.session_id
                )
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="batch not stored")
                return

            if up_to >= 0:
                await websocket.send_json(
                    {"ack": {"load_seq": batch.load_seq, "up_to_batch_seq": up_to}}
                )
    except WebSocketDisconnect:
        pass
