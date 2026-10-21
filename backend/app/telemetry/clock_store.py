from datetime import UTC, datetime
from threading import Lock
from uuid import UUID

import asyncpg

from backend.app.telemetry.clock import (
    HANDSHAKE_SAMPLES,
    ClockSample,
    accepted,
    segment_offset,
)

class SessionLocks:
    def __init__(self) -> None:
        self._guard = Lock()
        self._owners: dict[UUID, object] = {}

    def take(self, session_id: UUID, owner: object) -> bool:
        with self._guard:
            if session_id in self._owners:
                return False
            self._owners[session_id] = owner
            return True

    def owns(self, session_id: UUID, owner: object) -> bool:
        with self._guard:
            return self._owners.get(session_id) is owner

    def release(self, session_id: UUID, owner: object) -> None:
        with self._guard:
            if self._owners.get(session_id) is owner:
                del self._owners[session_id]


class UnknownSession(Exception):
    pass


# An OSError, which the endpoint treats as "database down".
class NotSessionOwner(Exception):
    pass


_NEXT_LOAD_SEQ = """
SELECT GREATEST(
  COALESCE((SELECT max(load_seq) FROM clock_sync_samples WHERE session_id = $1), -1),
  COALESCE((SELECT max(load_seq) FROM ingested_batches WHERE session_id = $1), -1)
) + 1
"""

_INSERT_SAMPLE = """
INSERT INTO clock_sync_samples
  (time, session_id, load_seq, t0, t1, t2, t3, rtt_ms, offset_ms, accepted)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
"""


async def store_clock_samples(
    connection: asyncpg.Connection,
    locks: SessionLocks,
    session_id: UUID,
    owner: object,
    samples: list[ClockSample],
) -> tuple[int, float]:
    if len(samples) != HANDSHAKE_SAMPLES:
        raise ValueError(f"a handshake requires {HANDSHAKE_SAMPLES} clock samples")
    if not locks.owns(session_id, owner):
        raise NotSessionOwner

    flags = accepted(samples)
    offset_ms = segment_offset(samples)

    async with connection.transaction():
        # Serialize numbering with any other connection for this session.
        exists = await connection.fetchval(
            "SELECT session_id FROM sessions WHERE session_id = $1 FOR UPDATE",
            session_id,
        )
        if exists is None:
            raise UnknownSession

        load_seq = await connection.fetchval(_NEXT_LOAD_SEQ, session_id)
        if load_seq > 32767:
            raise ValueError("page-load sequence exhausted")

        await connection.executemany(
            _INSERT_SAMPLE,
            [
                (
                    # t1 is the server clock when the ping arrived.
                    datetime.fromtimestamp(sample.t1 / 1000, tz=UTC),
                    session_id, load_seq,
                    sample.t0, sample.t1, sample.t2, sample.t3,
                    sample.rtt_ms, sample.offset_ms, keep,
                )
                for sample, keep in zip(samples, flags)
            ],
        )
        await connection.execute(
            "UPDATE sessions SET final_offset_ms = $2 WHERE session_id = $1",
            session_id,
            offset_ms,
        )

    return load_seq, offset_ms
