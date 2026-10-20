import argparse
import asyncio
import json
import sys
from datetime import timedelta
from uuid import UUID

import asyncpg

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.lifecycle_checkpoint import CheckpointInvalid
from backend.app.classifier.lifecycle_recovery import CheckpointConflict,RecoverableSegmentLifecycle
from backend.app.db.pool import close_db_pool,create_db_pool
from tests.classifier.smoke_flag_persistence import scenario,seed
from tests.classifier.smoke_segment_lifecycle import BorrowedConnectionPool


async def open_flow(pool,w,**kwargs):
    return await RecoverableSegmentLifecycle.open(pool,session_id=w.session_id,
        task_id=w.task_id,model_version=w.model_version,**kwargs)


async def restore_only(args):
    pool=await create_db_pool()
    try:
        flow=await RecoverableSegmentLifecycle.open(pool,session_id=UUID(args.session_id),
            task_id=UUID(args.task_id),model_version=args.model_version)
        state=flow.snapshot()
        print(json.dumps({"ids":[str(flag_id_for_segment(s)) for s in state.active_segments],
            "ranges":[[s.start_line,s.end_line] for s in state.active_segments],
            "confidence":[s.confidence for s in state.active_segments],
            "count":state.processed_window_count,"revision":flow.revision}))
    finally:
        await close_db_pool(pool)


class FailedCheckpointConnection:
    def __init__(self,connection):self.connection=connection
    def __getattr__(self,name):return getattr(self.connection,name)
    async def fetchval(self,sql,*args):
        if "UPDATE classifier_lifecycle_checkpoints" in sql:
            await self.connection.execute("SELECT 1/0")
        return await self.connection.fetchval(sql,*args)


async def main():
    pool=await create_db_pool()
    async with pool.acquire() as connection:
        ready=await connection.fetchval("SELECT to_regclass('classifier_lifecycle_checkpoints') IS NOT NULL")
    if not ready:
        await close_db_pool(pool)
        raise RuntimeError("Apply migration 011 before running the recovery database test")
    a,b,clean=windows=scenario(
        {"confidence":.7},{"confidence":.9,"ranges":((0,2,14,20),)},
        {"category":"no_ai"},
    )
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():await seed(connection,windows)
        flow=await open_flow(pool,a,create_if_missing=True)
        await flow.accept_window(a)
        stale=await open_flow(pool,a)
        await flow.accept_window(b)
        expected=flow.snapshot()
        flag_id=flag_id_for_segment(expected.active_segments[0])
        assert flow.revision==2
        print("PASS: accepted inputs checkpointed before state is returned")

        # A new raw edit appears inside an already classified observation period.
        # Recovery must not query it and attach it to the original prediction.
        async with pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO events(time,session_id,task_id,load_seq,batch_seq,seq_in_batch,"
                "kind,perf_now,origin,version_id,change_index,start_line,start_col,end_line,end_col,"
                "inserted_len,removed_len,inserted_lines,doc_len) "
                "VALUES ($1,$2,$3,0,0,9,2,7000,0,2,0,40,1,40,1,20,0,5,200)",
                a.window_start+timedelta(seconds=7),a.session_id,a.task_id,
            )
        del flow
        child=await asyncio.create_subprocess_exec(
            sys.executable,"-m","tests.classifier.smoke_lifecycle_recovery","--restore-only",
            "--session-id",str(a.session_id),"--task-id",str(a.task_id),
            "--model-version",a.model_version,
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout,stderr=await asyncio.wait_for(child.communicate(),45)
        except BaseException:
            child.kill()
            await child.wait()
            raise
        assert child.returncode==0,stderr.decode()
        result=json.loads(stdout)
        assert result=={"ids":[str(flag_id)],"ranges":[[10,20]],"confidence":[.9],"count":2,"revision":2}
        print("PASS: separate Python process restores the same active flag, range, confidence and identity")
        print("PASS: late raw telemetry does not revise recovered observations")

        flow=await open_flow(pool,a)
        await flow.accept_window(clean)
        assert len(flow.snapshot().pending_segments)==1
        try:await stale.accept_window(b)
        except CheckpointConflict:pass
        else:raise AssertionError("Stale writer replaced a newer checkpoint")
        assert stale.snapshot().processed_window_count==1
        assert (await open_flow(pool,a)).snapshot().processed_window_count==3
        print("PASS: stale-writer conflict preserves the newer checkpoint and previous memory state")

        # Restore a temporary adapter on a borrowed real connection to inject a
        # database failure after validation. A failed finish must not be published.
        async with pool.acquire() as connection:
            failed=await open_flow(BorrowedConnectionPool(FailedCheckpointConnection(connection)),a)
            before=failed.snapshot()
            try:await failed.finish(pending_live_processing_complete=True)
            except asyncpg.DivisionByZeroError:pass
            else:raise AssertionError("Injected checkpoint failure did not occur")
            assert failed.snapshot()==before and not failed.snapshot().finished
        assert not (await open_flow(pool,a)).snapshot().finished
        print("PASS: checkpoint write failure rolls back without publishing finished state")

        await flow.finish(pending_live_processing_complete=True)
        saved=await flow.persist_closed()
        assert saved==(flag_id,)
        again=await open_flow(pool,a)
        assert again.snapshot().finished and not again.snapshot().active_segments
        assert len(again.snapshot().pending_segments)==1  # revalidate, never assume an acknowledgement
        assert await again.persist_closed()==(flag_id,)
        async with pool.acquire() as connection:
            assert await connection.fetchval("SELECT count(*) FROM classifier_flags WHERE task_id=$1",a.task_id)==1
            assert await connection.fetchval("SELECT count(*) FROM classifier_flag_predictions WHERE flag_id=$1",flag_id)==2
        print("PASS: finished-state restoration retries existing flags without duplicates")

        # Corrupt a checkpoint inside a rollback transaction; opening must fail
        # rather than silently returning an empty state or re-reading raw edits.
        async with pool.acquire() as connection:
            outer=connection.transaction()
            await outer.start()
            try:
                await connection.execute(
                    "UPDATE classifier_lifecycle_checkpoints SET checkpoint=jsonb_set("
                    "checkpoint,'{sha256}','\"corrupt\"'::jsonb) WHERE task_id=$1",a.task_id,
                )
                try:await open_flow(BorrowedConnectionPool(connection),a)
                except CheckpointInvalid:pass
                else:raise AssertionError("Corrupted checkpoint was accepted")
            finally:await outer.rollback()
        print("PASS: corrupted checkpoint is rejected and original checkpoint retained")

        async with pool.acquire() as connection:
            await connection.execute("DELETE FROM classifier_lifecycle_checkpoints WHERE session_id=$1",a.session_id)
            assert await connection.fetchval("SELECT count(*) FROM classifier_flags WHERE task_id=$1",a.task_id)==1
        try:await open_flow(pool,a)
        except LookupError:pass
        else:raise AssertionError("Missing recovery record was silently treated as an empty recovered stream")
        print("PASS: checkpoint expiry preserves historical flags; missing recovery state is explicit")
    finally:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("DELETE FROM classifier_lifecycle_checkpoints WHERE task_id=$1",a.task_id)
                await connection.execute("DELETE FROM classifier_flags WHERE task_id=$1",a.task_id)
                await connection.execute("DELETE FROM classifier_window_predictions WHERE task_id=$1",a.task_id)
                await connection.execute("DELETE FROM feature_vectors WHERE task_id=$1",a.task_id)
                await connection.execute("DELETE FROM events WHERE session_id=$1",a.session_id)
                await connection.execute("DELETE FROM tasks WHERE task_id=$1",a.task_id)
                await connection.execute("DELETE FROM sessions WHERE session_id=$1",a.session_id)
            assert not await connection.fetchval("SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id=$1)",a.session_id)
        await close_db_pool(pool)
    print("PASS: all UUID-scoped test fixtures removed")
    print("TASK 13 INCREMENT 2 RECOVERY DATABASE TEST PASSED")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-only",action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--task-id")
    parser.add_argument("--model-version")
    args=parser.parse_args()
    asyncio.run(restore_only(args) if args.restore_only else main())
