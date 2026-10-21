import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.app.db.pool import close_db_pool, create_db_pool
from backend.app.telemetry.clock import HANDSHAKE_SAMPLES, ClockSample
from backend.app.telemetry.clock_store import (
    NotSessionOwner,
    SessionLocks,
    store_clock_samples,
)

# A realistic server clock reading, in milliseconds since 1970.
SERVER_MS = 1_760_000_000_000.0


def _samples(shift=0):
    round_trips = (1, 2, 3, 4, 50, 60, 70, 80)
    offsets = (100, 102, 104, 106, 5000, -5000, 9000, 0)
    return [
        ClockSample(
            t0=0.0,
            t1=SERVER_MS + offset + shift + rtt / 2,
            t2=SERVER_MS + offset + shift + rtt / 2,
            t3=float(rtt),
        )
        for rtt, offset in zip(round_trips, offsets)
    ]


def test_only_the_owner_can_release_a_session():
    locks = SessionLocks()
    session = uuid4()
    first, second = object(), object()

    assert locks.take(session, first)
    assert not locks.take(session, second)
    assert locks.owns(session, first)

    locks.release(session, second)
    assert not locks.take(session, second)

    locks.release(session, first)
    assert locks.take(session, second)


def test_a_handshake_needs_exactly_the_agreed_number_of_samples():
    # Refused before the database is touched, so no connection is needed.
    with pytest.raises(ValueError):
        asyncio.run(
            store_clock_samples(None, SessionLocks(), uuid4(), object(), _samples()[:-1])
        )


@pytest.fixture
def database():
    async def probe():
        pool = await create_db_pool()
        await close_db_pool(pool)

    try:
        asyncio.run(probe())
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"database not available: {exc}")


def test_samples_and_next_page_load_are_saved_together(database):
    async def scenario():
        pool = await create_db_pool()
        session = uuid4()
        locks = SessionLocks()
        first, second = object(), object()

        try:
            async with pool.acquire() as connection:
                await connection.execute(
                    "INSERT INTO sessions "
                    "(session_id, candidate_id, language, started_at) "
                    "VALUES ($1, $2, 'python', now())",
                    session,
                    uuid4(),
                )
                try:
                    # A batch from the old capture path already used load 0.
                    await connection.execute(
                        "INSERT INTO ingested_batches "
                        "(session_id, load_seq, batch_seq) VALUES ($1, 0, 0)",
                        session,
                    )

                    with pytest.raises(NotSessionOwner):
                        await store_clock_samples(connection, locks, session, first, _samples())

                    assert locks.take(session, first)
                    load_seq, offset = await store_clock_samples(
                        connection, locks, session, first, _samples()
                    )
                    assert (load_seq, offset) == (1, SERVER_MS + 103)

                    # A second tab cannot store while the first one owns the session.
                    with pytest.raises(NotSessionOwner):
                        await store_clock_samples(connection, locks, session, second, _samples())

                    locks.release(session, first)
                    assert locks.take(session, second)
                    load_seq, offset = await store_clock_samples(
                        connection, locks, session, second, _samples(200)
                    )
                    assert (load_seq, offset) == (2, SERVER_MS + 303)

                    rows = await connection.fetch(
                        "SELECT load_seq, accepted, time, t1 FROM clock_sync_samples "
                        "WHERE session_id = $1 ORDER BY load_seq, rtt_ms",
                        session,
                    )
                    assert len(rows) == 2 * HANDSHAKE_SAMPLES
                    assert [(row["load_seq"], row["accepted"]) for row in rows] == (
                        [(1, True)] * 4
                        + [(1, False)] * 4
                        + [(2, True)] * 4
                        + [(2, False)] * 4
                    )
                    assert all(
                        row["time"] == datetime.fromtimestamp(row["t1"] / 1000, tz=UTC)
                        for row in rows
                    )
                    assert await connection.fetchval(
                        "SELECT final_offset_ms FROM sessions WHERE session_id = $1",
                        session,
                    ) == SERVER_MS + 303
                finally:
                    await connection.execute(
                        "DELETE FROM clock_sync_samples WHERE session_id = $1", session
                    )
                    await connection.execute(
                        "DELETE FROM ingested_batches WHERE session_id = $1", session
                    )
                    await connection.execute(
                        "DELETE FROM sessions WHERE session_id = $1", session
                    )
        finally:
            await close_db_pool(pool)

    asyncio.run(scenario())
