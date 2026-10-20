import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg

from backend.app.classifier.flag_persistence import fetch_saved_flag
from backend.app.classifier.submission_coordinator import SubmissionCoordinator, SubmissionPlan
from backend.app.db.pool import close_db_pool, create_db_pool
from tests.classifier.flag_persistence_support import make_window
from tests.classifier.smoke_flag_persistence import FailDuringLinks, seed
from tests.classifier.smoke_segment_lifecycle import BorrowedConnectionPool


async def main():
    pool = await create_db_pool()
    session_id, task_id = uuid4(), uuid4()
    try:
        async with pool.acquire() as connection:
            if not await connection.fetchval("SELECT to_regclass('classifier_lifecycle_checkpoints') IS NOT NULL"):
                raise RuntimeError("Apply migration 011 before this submission database test")
            outer = connection.transaction()
            await outer.start()
            try:
                base = (datetime.now(timezone.utc)-timedelta(minutes=5)).replace(microsecond=0)
                base -= timedelta(seconds=base.second % 10)
                windows = [make_window(n, base=base-timedelta(seconds=20),
                                       session_id=session_id, task_id=task_id,
                                       category="no_ai" if n < 4 else "external_ai")
                           for n in range(7)]
                await seed(connection, windows)
                await connection.execute("UPDATE tasks SET started_at=$2,submitted_at=$3 WHERE task_id=$1",
                                         task_id, base, base+timedelta(seconds=47))
                await connection.execute(
                    "INSERT INTO task_submissions(task_id,session_id,code_text,version_id,submitted_at,load_seq) "
                    "VALUES ($1,$2,'# synthetic submission fixture',2,$3,0)",
                    task_id, session_id, base+timedelta(seconds=47),
                )
                metadata = await connection.fetchrow(
                    "SELECT s.task_id,s.session_id,t.started_at,s.submitted_at FROM task_submissions s "
                    "JOIN tasks t ON t.task_id=s.task_id AND t.session_id=s.session_id WHERE s.task_id=$1",
                    task_id,
                )
                plan = SubmissionPlan(**dict(metadata), model_version=windows[0].model_version)
                borrowed = BorrowedConnectionPool(connection)
                coordinator = await SubmissionCoordinator.open(borrowed, plan, create_if_missing=True)
                try:
                    await coordinator.accept_window(windows[0], now=plan.required_windows[0].ready_at-timedelta(microseconds=1))
                except ValueError:
                    pass
                else:
                    raise AssertionError("Premature window was accepted")
                assert not await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM classifier_lifecycle_checkpoints WHERE task_id=$1)", task_id)
                print("PASS: normal readiness gate rejects early processing without checkpoint changes")

                for requirement, window in zip(plan.required_windows[:5], windows[:5]):
                    await coordinator.accept_window(window, now=requirement.ready_at)
                progress = await coordinator.advance(now=base+timedelta(seconds=53), telemetry_complete=True)
                assert progress.phase == "waiting_windows" and not progress.lifecycle.finished
                assert len(progress.pending_windows) == 2
                original = progress.lifecycle.active_segments[0]
                print("PASS: submission remains pending while final overlapping predictions are missing")

                # Recover a fresh coordinator from the real PostgreSQL checkpoint
                # and the same persisted submission metadata, not raw telemetry.
                del coordinator
                coordinator = await SubmissionCoordinator.open(borrowed, plan)
                for requirement, window in zip(plan.required_windows[5:], windows[5:]):
                    await coordinator.accept_window(window, now=requirement.ready_at)
                progress = await coordinator.advance(now=base+timedelta(seconds=72), telemetry_complete=False)
                assert progress.phase == "waiting_telemetry" and not progress.lifecycle.finished
                assert progress.lifecycle.active_segments[0].members[0] == original.members[0]
                print("PASS: recovered coordinator retains original evidence and requires delivery completion")

                failing = await SubmissionCoordinator.open(
                    BorrowedConnectionPool(FailDuringLinks(connection)), plan)
                try:
                    await failing.advance(now=base+timedelta(seconds=72), telemetry_complete=True)
                except asyncpg.UniqueViolationError:
                    pass
                else:
                    raise AssertionError("Actual provenance constraint failure did not occur")
                assert failing.progress(now=base+timedelta(seconds=72)).phase == "saving_flags"
                assert await connection.fetchval("SELECT count(*) FROM classifier_flags WHERE task_id=$1", task_id) == 0
                print("PASS: actual flag/link failure rolls back the flag and leaves recoverable pending saves")

                coordinator = await SubmissionCoordinator.open(borrowed, plan)
                progress = await coordinator.advance(now=base+timedelta(seconds=73), telemetry_complete=False)
                assert progress.phase == "complete" and not progress.lifecycle.pending_segments
                flag_id, = progress.lifecycle.persisted_flag_ids
                saved = await fetch_saved_flag(connection, flag_id)
                assert set(saved["prediction_ids"]) == {w.prediction_id for w in windows[4:]}
                assert saved["window_count"] == 3
                again = await SubmissionCoordinator.open(borrowed, plan)
                assert again.progress(now=base+timedelta(seconds=74)).phase == "saving_flags"
                assert (await again.advance(now=base+timedelta(seconds=74), telemetry_complete=False)).phase == "complete"
                assert await connection.fetchval("SELECT count(*) FROM classifier_flags WHERE task_id=$1", task_id) == 1
                assert await connection.fetchval("SELECT count(*) FROM classifier_flag_predictions WHERE flag_id=$1", flag_id) == 3
                print("PASS: retry and completed-state recovery preserve one flag and three provenance links")
            finally:
                await outer.rollback()
            assert not await connection.fetchval("SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id=$1)", session_id)
            assert not await connection.fetchval("SELECT EXISTS(SELECT 1 FROM task_submissions WHERE task_id=$1)", task_id)
            print("PASS: all submission-suite fixtures rolled back")
    finally:
        await close_db_pool(pool)
    print("TASK 13 INCREMENT 3 SUBMISSION DATABASE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
