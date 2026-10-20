import asyncio
from contextlib import asynccontextmanager

import asyncpg

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.flag_persistence import fetch_saved_flag
from backend.app.classifier.segment_lifecycle import SegmentLifecycle, persist_closed_segments
from backend.app.db.pool import close_db_pool, create_db_pool
from tests.classifier.smoke_flag_persistence import FailDuringLinks, scenario, seed


class BorrowedConnectionPool:
    """Use real SQL/savepoints inside the smoke test's rollback transaction."""
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


def coordinator(window):
    return SegmentLifecycle(session_id=window.session_id, task_id=window.task_id,
                            model_version=window.model_version)


async def main():
    pool = await create_db_pool()
    sessions = []
    try:
        async with pool.acquire() as connection:
            outer = connection.transaction()
            await outer.start()
            try:
                borrowed_pool = BorrowedConnectionPool(connection)
                a, b, clean = windows = scenario(
                    {"confidence": .7}, {"confidence": .9, "ranges": ((0,2,14,20),)},
                    {"category": "no_ai"},
                )
                sessions.append(a.session_id)
                await seed(connection, windows)
                flow = coordinator(a)
                first = flow.accept_window(a)
                flag_id = flag_id_for_segment(first.active_segments[0])
                assert await persist_closed_segments(flow,borrowed_pool) == ()
                assert await fetch_saved_flag(connection,flag_id) is None
                assert flow.snapshot() == first  # observation/interruption does not close
                grown = flow.accept_window(b)
                assert flag_id_for_segment(grown.active_segments[0]) == flag_id
                assert grown.active_segments[0].end_line == 20
                assert await persist_closed_segments(flow,borrowed_pool) == ()
                assert await fetch_saved_flag(connection,flag_id) is None
                closed = flow.accept_window(clean)
                assert len(closed.pending_segments) == 1
                assert await persist_closed_segments(flow,borrowed_pool) == (flag_id,)
                stored = await fetch_saved_flag(connection,flag_id)
                assert stored["detected_end_line"] == 20 and stored["confidence"] == .9
                assert stored["representative_prediction_id"] == b.prediction_id
                assert stored["window_count"] == 2
                assert set(stored["prediction_ids"]) == {a.prediction_id,b.prediction_id}
                assert await persist_closed_segments(flow,borrowed_pool) == ()
                assert await fetch_saved_flag(connection,flag_id) == stored
                print("PASS: active growth stays out of flag storage; closure saves final historical payload")
                print("PASS: stable identity, confidence, all prediction links and repeated-save safety")

                # Proves replay only when ORIGINAL window/region inputs are given.
                # Does not fetch/reconstruct evidence or implement restart recovery.
                replay = coordinator(a)
                for w in windows:
                    replay.accept_window(w)
                assert await persist_closed_segments(replay,borrowed_pool) == (flag_id,)
                assert await fetch_saved_flag(connection,flag_id) == stored
                print("PASS: original-input replay reuses existing stored flag without duplicates")

                windows = scenario({"ranges": ((0,2,10,15),(0,2,40,45))})
                a = windows[0]
                sessions.append(a.session_id)
                await seed(connection,windows)
                flow = coordinator(a)
                before = flow.accept_window(a)
                try:
                    flow.finish(pending_live_processing_complete=False)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Submission finished without a processing barrier")
                assert flow.snapshot() == before
                state = flow.finish(pending_live_processing_complete=True)
                assert not state.active_segments and len(state.pending_segments) == 2

                # Actual unique-constraint failure after partial writes, using
                # Task 12's failure injector. No state is acknowledged on failure.
                try:
                    await persist_closed_segments(flow,BorrowedConnectionPool(FailDuringLinks(connection)))
                except asyncpg.UniqueViolationError:
                    pass
                else:
                    raise AssertionError("Injected constraint failure did not occur")
                assert len(flow.snapshot().pending_segments) == 2
                assert flow.snapshot().persisted_flag_ids == ()
                assert await connection.fetchval(
                    "SELECT count(*) FROM classifier_flags WHERE task_id=$1",a.task_id,
                ) == 0
                saved = await persist_closed_segments(flow,borrowed_pool)
                assert len(saved) == 2 and len(set(saved)) == 2
                assert not flow.snapshot().pending_segments
                assert flow.finish(pending_live_processing_complete=True).persisted_flag_ids == saved
                assert await connection.fetchval(
                    "SELECT count(*) FROM classifier_flags WHERE task_id=$1",a.task_id,
                ) == 2
                assert await connection.fetchval(
                    "SELECT count(*) FROM classifier_flags_for_probing WHERE task_id=$1",a.task_id,
                ) == 0
                print("PASS: submission barrier closes remaining independent segments")
                print("PASS: database failure leaves pending flags retryable; successful retry saves both")
                print("PASS: historical flags remain outside the final-location C3 view")
            finally:
                await outer.rollback()
        async with pool.acquire() as connection:
            assert await connection.fetchval(
                "SELECT count(*) FROM sessions WHERE session_id=ANY($1::uuid[])",sessions,
            ) == 0
        print("PASS: all fixture rows rolled back")
    finally:
        await close_db_pool(pool)
    print("TASK 13 INCREMENT 1 LIFECYCLE DATABASE HANDOFF PASSED")


if __name__ == "__main__":
    asyncio.run(main())
