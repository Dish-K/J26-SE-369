import json
import logging
from time import time_ns
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ValidationError

from backend.app.telemetry.clock_store import (
    NotSessionOwner,
    SessionLocks,
    UnknownSession,
    store_clock_samples,
    stored_offset,
)

from backend.app.telemetry.ingest import (
    EVENT_COLUMNS,
    TYPING_KINDS,
    Batch,
    ClockSamples,
    Hello,
    Ping,
    SessionStart,
    event_rows,
    Submit,
)

from backend.app.telemetry.tasks import submit_task

logger = logging.getLogger(__name__)

router = APIRouter()

# Close codes 4000-4999 are free for applications, the browser shows its blocked-tab message for this one.
SESSION_OPEN_ELSEWHERE = 4409

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

# Submissions are Python only, C3 parses them as Python.
_NEW_SESSION = """
INSERT INTO sessions (
  session_id, candidate_id, language, started_at,
  editor_profile, editor_config_hash, user_agent, platform,
  screen_w, screen_h, viewport_w, viewport_h, device_pixel_ratio, locale, timezone
)
VALUES ($1, $2, 'python', now(), $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
"""

_FIRST_TASK = """
INSERT INTO tasks (task_id, session_id, seq, origin, status, created_at, started_at)
VALUES ($1, $2, 0, 'original', 'active', now(), now())
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

@router.post("/telemetry/sessions", status_code=status.HTTP_201_CREATED)
async def start_session(start: SessionStart, request: Request) -> dict[str, UUID]:
    session_id = uuid4()
    task_id = uuid4()

    async with request.app.state.db_pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                _NEW_SESSION,
                session_id,
                # A placeholder until candidates log in.
                uuid4(),
                start.editor_profile,
                start.editor_config_hash,
                start.user_agent,
                start.platform,
                start.screen_w,
                start.screen_h,
                start.viewport_w,
                start.viewport_h,
                start.device_pixel_ratio,
                start.locale,
                start.timezone,
            )
            await connection.execute(_FIRST_TASK, task_id, session_id)

    return {"session_id": session_id, "task_id": task_id}

_MESSAGES: dict[str, type[BaseModel]] = {
    "hello": Hello,
    "ping": Ping,
    "clock": ClockSamples,
    "submit": Submit,
}

def _server_ms() -> float:
    return time_ns() / 1_000_000


def _parse(message: str) -> tuple[str, BaseModel]:
    data = json.loads(message)
    # A control message is one key wrapping its body; a batch is sent bare.
    if isinstance(data, dict) and len(data) == 1:
        ((kind, body),) = data.items()
        if kind in _MESSAGES:
            return kind, _MESSAGES[kind].model_validate(body)
    return "batch", Batch.model_validate(data)


class _Refused(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class _TelemetrySocket:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.pool = websocket.app.state.db_pool
        self.locks: SessionLocks = websocket.app.state.telemetry_session_locks
        self.session_id: UUID | None = None
        # One offset per page load, so every batch uses the offset of its own load_seq.
        self.offsets: dict[int, float] = {}
        self.load_seq: int | None = None

    async def handle(self, message: str, received_ms: float) -> None:
        try:
            kind, body = _parse(message)
        except ValueError as exc:
            # include_input=False keeps pasted text out of the logs.
            detail = (
                exc.errors(include_input=False, include_url=False)
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            logger.warning("Refused telemetry message: %s", detail)
            raise _Refused(status.WS_1008_POLICY_VIOLATION, "invalid message") from exc

        if kind == "hello":
            self.hello(body)
            return
        if self.session_id is None:
            raise _Refused(status.WS_1008_POLICY_VIOLATION, "hello first")

        try:
            if kind == "ping":
                await self.pong(body, received_ms)
            elif kind == "clock":
                await self.clock(body)
            elif kind == "submit":
                await self.submit(body)
            else:
                await self.batch(body)
        except UnknownSession as exc:
            raise _Refused(status.WS_1008_POLICY_VIOLATION, "unknown session") from exc
        except NotSessionOwner as exc:
            raise _Refused(SESSION_OPEN_ELSEWHERE, "session open elsewhere") from exc
        except _STORE_ERRORS as exc:
            logger.exception("Could not store telemetry for session %s", self.session_id)
            raise _Refused(status.WS_1011_INTERNAL_ERROR, "not stored") from exc

    def hello(self, hello: Hello) -> None:
        if self.session_id is not None:
            raise _Refused(status.WS_1008_POLICY_VIOLATION, "hello already sent")
        if not self.locks.take(hello.session_id, self.websocket):
            raise _Refused(SESSION_OPEN_ELSEWHERE, "session open elsewhere")
        self.session_id = hello.session_id

    async def pong(self, ping: Ping, received_ms: float) -> None:
        await self.websocket.send_json(
            {"pong": {"t0": ping.t0, "t1": received_ms, "t2": _server_ms()}}
        )

    async def clock(self, clock: ClockSamples) -> None:
        async with self.pool.acquire() as connection:
            load_seq, offset = await store_clock_samples(
                connection, self.locks, self.session_id, self.websocket, clock.samples
            )
        self.offsets[load_seq] = offset
        self.load_seq = load_seq
        await self.websocket.send_json({"ready": {"load_seq": load_seq}})

    async def batch(self, batch: Batch) -> None:
        if not self.locks.owns(batch.session_id, self.websocket):
            raise NotSessionOwner

        async with self.pool.acquire() as connection:
            offset = self.offsets.get(batch.load_seq)
            if offset is None:
                offset = await stored_offset(connection, batch.session_id, batch.load_seq)
            if offset is None:
                raise _Refused(
                    status.WS_1008_POLICY_VIOLATION, "no clock handshake for this page load"
                )
            self.offsets[batch.load_seq] = offset

            await store_batch(connection, batch, offset)
            up_to = await acked_up_to(connection, batch.session_id, batch.load_seq)

        if up_to >= 0:
            await self.websocket.send_json(
                {"ack": {"load_seq": batch.load_seq, "up_to_batch_seq": up_to}}
            )

    async def submit(self, submission: Submit) -> None:
        if self.load_seq is None:
            raise _Refused(status.WS_1008_POLICY_VIOLATION, "no clock handshake for this page load")

        async with self.pool.acquire() as connection:
            accepted = await submit_task(connection, self.session_id, self.load_seq, submission)
        if not accepted:
            raise _Refused(status.WS_1008_POLICY_VIOLATION, "task not open")

        await self.websocket.send_json({"submitted": {"task_id": str(submission.task_id)}})


    def release(self) -> None:
        if self.session_id is not None:
            self.locks.release(self.session_id, self.websocket)


@router.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    socket = _TelemetrySocket(websocket)

    try:
        while True:
            message = await websocket.receive_text()
            # Read first: anything done before this line adds error to the pong's t1.
            received_ms = _server_ms()
            try:
                await socket.handle(message, received_ms)
            except _Refused as refused:
                logger.warning("Closed telemetry connection: %s", refused.reason)
                await websocket.close(code=refused.code, reason=refused.reason)
                return
    except WebSocketDisconnect:
        pass
    finally:
        socket.release()

