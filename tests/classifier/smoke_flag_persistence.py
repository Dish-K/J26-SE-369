from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.flag_persistence import (
    FlagPersistenceConflict, fetch_saved_flag, save_segment_flag,
    save_segment_flag_on_connection,
)
from backend.app.classifier.localization import (
    fetch_prediction_edit_evidence, localize_prediction_edit_evidence,
)
from backend.app.classifier.segment_builder import build_segment_drafts
from backend.app.db.pool import close_db_pool, create_db_pool
from tests.classifier.flag_persistence_support import make_window


def scenario(*specifications):
    session_id, task_id = uuid4(), uuid4()
    base = datetime.now(timezone.utc) - timedelta(minutes=5)
    return [make_window(n, session_id=session_id, task_id=task_id, base=base, **spec)
            for n, spec in enumerate(specifications)]


async def seed(connection, windows):
    first = windows[0]
    await connection.execute(
        "INSERT INTO sessions(session_id,candidate_id,language,started_at) "
        "VALUES ($1,$2,'python',$3)",
        first.session_id, uuid4(), first.window_start,
    )
    await connection.execute(
        "INSERT INTO tasks(task_id,session_id,seq,depth,origin,created_by,status,created_at,started_at) "
        "VALUES ($1,$2,0,0,'original','interviewer','active',$3,$3)",
        first.task_id, first.session_id, first.window_start,
    )
    for window in windows:
        await connection.execute(
            "INSERT INTO feature_vectors(feature_vector_id,session_id,task_id,window_start,"
            "window_end,run_id,schema_version,features) "
            "VALUES ($1,$2,$3,$4,$5,'live','task12-storage-fixture-v1','{}'::jsonb)",
            window.feature_vector_id, window.session_id, window.task_id,
            window.window_start, window.window_end,
        )
        await connection.execute(
            "INSERT INTO classifier_window_predictions(prediction_id,feature_vector_id,task_id,"
            "model_version,predicted_category,class_probabilities,confidence,ai_positive) "
            "VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8)",
            window.prediction_id, window.feature_vector_id, window.task_id,
            window.model_version, window.predicted_category,
            json.dumps(window.class_probabilities), window.confidence, window.ai_positive,
        )
        for region in window.regions:
            for event in region.edit_events:
                await connection.execute(
                    "INSERT INTO events(time,session_id,task_id,load_seq,batch_seq,seq_in_batch,"
                    "kind,perf_now,origin,version_id,change_index,start_line,start_col,end_line,end_col,"
                    "inserted_len,removed_len,inserted_lines,doc_len) "
                    "VALUES ($1,$2,$3,$4,$5,$6,2,$7,0,$8,0,$9,1,$9,1,20,0,$10,200)",
                    window.window_start + timedelta(seconds=5), window.session_id, window.task_id,
                    region.load_seq, event["batch_seq"], event["seq_in_batch"],
                    float(5000 + event["batch_seq"] * 10000), region.version_id,
                    region.start_line, region.end_line-region.start_line,
                )


async def verify_draft(connection, segment):
    flag_id = await save_segment_flag_on_connection(connection, segment)
    row = await fetch_saved_flag(connection, flag_id)
    assert row is not None
    assert flag_id == flag_id_for_segment(segment)
    assert (row["load_seq"], row["version_id"]) == (segment.load_seq, segment.version_id)
    assert (row["detected_start_line"], row["detected_end_line"]) == (segment.start_line, segment.end_line)
    assert row["class_probabilities"] == segment.class_probabilities
    assert row["confidence"] == segment.confidence
    assert row["representative_prediction_id"] == segment.representative_prediction_id
    assert row["detected_at"] == segment.first_window_start
    assert row["detected_until"] == segment.last_window_end
    assert row["category"] == segment.category
    assert row["segment_basis"] == "window_run" and row["evidence_kinds"] == ()
    assert row["window_count"] == segment.window_count
    assert set(row["prediction_ids"]) == set(segment.prediction_ids)
    assert await save_segment_flag_on_connection(connection, segment) == flag_id
    assert await fetch_saved_flag(connection, flag_id) == row
    return flag_id


class FailDuringLinks:
    """Inject a real unique-constraint error after the flag and first link exist."""
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    async def executemany(self, sql, pairs):
        await self.connection.execute(sql, *pairs[0])
        await self.connection.execute(sql, *pairs[0])  # duplicate link must fail


async def rollback_suite(pool):
    sessions = []
    async with pool.acquire() as connection:
        outer = connection.transaction()
        await outer.start()
        try:
            cases = [
                ("same-category disconnected regions", [{"ranges": ((0,2,10,15),(0,2,30,35))}], 2),
                ("mixed categories", [{}, {"category": "ide_ai"}], 2),
                ("bridging preserves independence", [{"ranges": ((0,2,10,10),(0,2,12,12))},
                                                    {"ranges": ((0,2,11,11),)}], 3),
                ("identical locations, distinct original observations",
                 [{"ranges": ((0,2,10,15),(0,2,10,15))}], 2),
                ("unknown version transition stays separate", [{}, {"ranges": ((0,3,10,15),)}], 2),
                ("page reload stays separate", [{}, {"ranges": ((1,2,10,15),)}], 2),
                ("clean gap ends segment", [{}, {"category": "no_ai"}, {}], 2),
                ("merged windows retain representative and provenance",
                 [{"confidence": .7}, {"confidence": .9, "ranges": ((0,2,14,20),)}], 1),
            ]
            for label, specs, count in cases:
                windows = scenario(*specs)
                sessions.append(windows[0].session_id)
                await seed(connection, windows)
                drafts = build_segment_drafts(windows)
                assert len(drafts) == count
                ids = [await verify_draft(connection, draft) for draft in drafts]
                assert len(set(ids)) == count
                assert await connection.fetchval(
                    "SELECT count(*) FROM classifier_flags WHERE task_id=$1", windows[0].task_id,
                ) == count
                # Historical flags must not be exposed as mapped final C3 ranges.
                assert await connection.fetchval(
                    "SELECT count(*) FROM classifier_flags_for_probing WHERE task_id=$1", windows[0].task_id,
                ) == 0
                print("PASS:", label)

            # Test the actual Task 10 -> Task 11 -> Task 12 handoff from stored edits.
            windows = scenario({"ranges": ((0,2,30,35),(0,2,50,51))})
            sessions.append(windows[0].session_id)
            await seed(connection, windows)
            evidence = await fetch_prediction_edit_evidence(connection, windows[0].prediction_id)
            located = localize_prediction_edit_evidence(evidence)
            assert len(located.regions) == 2
            drafts = build_segment_drafts([replace(windows[0], regions=located.regions)])
            for draft in drafts:
                await verify_draft(connection, draft)
                traced = await fetch_prediction_edit_evidence(connection, draft.prediction_ids[0])
                assert traced is not None and len(traced.edit_events) == 2
            print("PASS: database edits -> localization -> segments -> flags -> evidence trace")

            # Force failure after partial writes, verify nested savepoint rollback.
            windows = scenario({}, {"confidence": .9})
            sessions.append(windows[0].session_id)
            await seed(connection, windows)
            draft, = build_segment_drafts(windows)
            flag_id = flag_id_for_segment(draft)
            try:
                await save_segment_flag_on_connection(FailDuringLinks(connection), draft)
            except asyncpg.UniqueViolationError:
                pass
            else:
                raise AssertionError("Injected duplicate link was accepted")
            assert await fetch_saved_flag(connection, flag_id) is None
            assert await connection.fetchval(
                "SELECT count(*) FROM classifier_flag_predictions WHERE flag_id=$1", flag_id,
            ) == 0
            await verify_draft(connection, draft)
            print("PASS: actual constraint failure rolls back flag and links; retry succeeds")

            before = await fetch_saved_flag(connection, flag_id)
            try:
                await save_segment_flag_on_connection(connection, replace(draft, end_line=99))
            except FlagPersistenceConflict:
                pass
            else:
                raise AssertionError("Conflicting historical location was overwritten")
            assert await fetch_saved_flag(connection, flag_id) == before
            print("PASS: conflicting payload is rejected without overwriting history")
        finally:
            await outer.rollback()

    async with pool.acquire() as connection:
        assert await connection.fetchval(
            "SELECT count(*) FROM sessions WHERE session_id=ANY($1::uuid[])", sessions,
        ) == 0
    print("PASS: all rollback-suite fixtures removed")


async def concurrent_suite(pool):
    windows = scenario({}, {"confidence": .9})
    session_id, task_id = windows[0].session_id, windows[0].task_id
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                await seed(connection, windows)
        draft, = build_segment_drafts(windows)
        results = await asyncio.gather(*(save_segment_flag(pool, draft) for _ in range(4)))
        assert len(set(results)) == 1
        async with pool.acquire() as connection:
            row = await fetch_saved_flag(connection, results[0])
            assert set(row["prediction_ids"]) == set(draft.prediction_ids)
            assert await connection.fetchval(
                "SELECT count(*) FROM classifier_flags WHERE task_id=$1", task_id,
            ) == 1
        print("PASS: four concurrent pool saves produce one flag and two links")
    finally:
        # Only this test's fresh UUIDs; leave all other data untouched.
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("DELETE FROM classifier_flags WHERE task_id=$1", task_id)
                await connection.execute("DELETE FROM classifier_window_predictions WHERE task_id=$1", task_id)
                await connection.execute("DELETE FROM feature_vectors WHERE task_id=$1", task_id)
                await connection.execute("DELETE FROM events WHERE session_id=$1", session_id)
                await connection.execute("DELETE FROM tasks WHERE task_id=$1", task_id)
                await connection.execute("DELETE FROM sessions WHERE session_id=$1", session_id)
            assert not await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id=$1)", session_id,
            )
        print("PASS: concurrent-suite fixtures removed")


async def main(concurrency=False):
    pool = await create_db_pool()
    try:
        await rollback_suite(pool)
        if concurrency:
            await concurrent_suite(pool)
    finally:
        await close_db_pool(pool)
    print("TASK 12 INCREMENT 4 DATABASE PERSISTENCE PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", action="store_true")
    asyncio.run(main(parser.parse_args().concurrency))
