from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from backend.app.classifier.localization import fetch_prediction_edit_evidence, localize_prediction_edit_evidence
from backend.app.db.pool import close_db_pool, create_db_pool


async def _smoke() -> None:
    pool = await create_db_pool()
    try:
        async with pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()
            try:
                session_id = uuid4()
                task_id = uuid4()
                unrelated_task_id = uuid4()
                window_start = datetime.now(timezone.utc) - timedelta(minutes=2)
                window_end = window_start + timedelta(seconds=30)

                await connection.execute(
                    "INSERT INTO sessions(session_id,candidate_id,language,started_at) "
                    "VALUES ($1,$2,'python',$3)",
                    session_id, uuid4(), window_start,
                )
                for task, seq in ((task_id, 0), (unrelated_task_id, 1)):
                    await connection.execute(
                        "INSERT INTO tasks(task_id,session_id,seq,depth,origin,created_by,"
                        "status,created_at,started_at) "
                        "VALUES ($1,$2,$3,0,'original','interviewer','active',$4,$4)",
                        task, session_id, seq, window_start,
                    )

                fv_id = await connection.fetchval(
                    "INSERT INTO feature_vectors(session_id,task_id,window_start,"
                    "window_end,run_id,schema_version,features) "
                    "VALUES ($1,$2,$3,$4,'live','1.0.0',$5::jsonb) "
                    "RETURNING feature_vector_id",
                    session_id, task_id, window_start, window_end, json.dumps({}),
                )
                prediction_id = await connection.fetchval(
                    "INSERT INTO classifier_window_predictions("
                    "feature_vector_id,task_id,model_version,predicted_category,"
                    "class_probabilities,confidence,ai_positive) "
                    "VALUES ($1,$2,'task10-fixture-v1','ide_ai',$3::jsonb,0.90,TRUE) "
                    "RETURNING prediction_id",
                    fv_id, task_id,
                    json.dumps({"no_ai": 0.05, "ide_ai": 0.90, "external_ai": 0.05}),
                )

                async def insert_event(
                    *, task=task_id, kind=2, origin=None, line=8,
                    inserted_lines=0, seconds=5, seq=0, version=2
                ):
                    await connection.execute(
                        "INSERT INTO events(time,session_id,task_id,load_seq,batch_seq,"
                        "seq_in_batch,kind,perf_now,origin,version_id,change_index,"
                        "start_line,start_col,end_line,end_col,inserted_len,"
                        "removed_len,inserted_lines,doc_len) "
                        "VALUES ($1,$2,$3,0,0,$4,$5,$6,$7,$8,0,"
                        "$9,1,$9,1,5,0,$10,100)",
                        window_start + timedelta(seconds=seconds), session_id,
                        task, seq, kind, float(1000 + seconds * 1000), origin,
                        version, line, inserted_lines,
                    )

                await insert_event(seq=1, origin=0, line=8, seconds=5, version=2)
                await insert_event(seq=2, origin=1, line=30, inserted_lines=2, seconds=6, version=3)
                await insert_event(seq=3, origin=4, line=60, seconds=7, version=4)  # reset
                await insert_event(seq=4, kind=10, line=90, seconds=8)  # copy
                await insert_event(seq=5, origin=0, line=70, seconds=30)  # excluded end
                await insert_event(seq=6, task=unrelated_task_id, origin=0, line=99)  # other task

                evidence = await fetch_prediction_edit_evidence(connection, prediction_id)
                assert evidence is not None
                assert evidence.task_id == task_id
                assert evidence.prediction_id == prediction_id
                assert evidence.predicted_category == "ide_ai"
                assert len(evidence.edit_events) == 2, evidence.edit_events
                assert [e["touched_start_line"] for e in evidence.edit_events] == [8, 30]
                assert [e["touched_end_line"] for e in evidence.edit_events] == [8, 32]
                assert [e["version_id"] for e in evidence.edit_events] == [2, 3]
                assert all(e["session_id"] == session_id for e in evidence.edit_events)
                assert all(e["task_id"] == task_id for e in evidence.edit_events)

                located = localize_prediction_edit_evidence(evidence)
                assert [(r.start_line, r.end_line) for r in located.regions] == [(8, 8), (30, 32)]
                assert all(r.prediction_id == prediction_id for r in located.regions)
                assert [(r.load_seq, r.version_id) for r in located.regions] == [(0, 2), (0, 3)]
                assert located.unlocalized_events == ()

                clean_fv_id = await connection.fetchval(
                    "INSERT INTO feature_vectors(session_id,task_id,window_start,"
                    "window_end,run_id,schema_version,features) "
                    "VALUES ($1,$2,$3,$4,'live','1.0.0',$5::jsonb) "
                    "RETURNING feature_vector_id",
                    session_id, task_id,
                    window_start + timedelta(seconds=10),
                    window_end + timedelta(seconds=10),
                    json.dumps({}),
                )
                clean_id = await connection.fetchval(
                    "INSERT INTO classifier_window_predictions("
                    "feature_vector_id,task_id,model_version,predicted_category,"
                    "class_probabilities,confidence,ai_positive) "
                    "VALUES ($1,$2,'task10-fixture-v1','no_ai',$3::jsonb,0.80,FALSE) "
                    "RETURNING prediction_id",
                    clean_fv_id, task_id,
                    json.dumps({"no_ai": 0.80, "ide_ai": 0.10, "external_ai": 0.10}),
                )
                assert await fetch_prediction_edit_evidence(connection, clean_id) is None

                # Deletion-only evidence must not flag post-edit code.
                deletion_start = window_start + timedelta(seconds=40)
                deletion_fv_id = await connection.fetchval(
                    "INSERT INTO feature_vectors(session_id,task_id,window_start,"
                    "window_end,run_id,schema_version,features) "
                    "VALUES ($1,$2,$3,$4,'live','1.0.0',$5::jsonb) "
                    "RETURNING feature_vector_id",
                    session_id, task_id, deletion_start,
                    deletion_start + timedelta(seconds=30), json.dumps({}),
                )
                deletion_prediction_id = await connection.fetchval(
                    "INSERT INTO classifier_window_predictions("
                    "feature_vector_id,task_id,model_version,predicted_category,"
                    "class_probabilities,confidence,ai_positive) "
                    "VALUES ($1,$2,'task10-fixture-v1','external_ai',$3::jsonb,0.92,TRUE) "
                    "RETURNING prediction_id",
                    deletion_fv_id, task_id,
                    json.dumps({"no_ai": 0.04, "ide_ai": 0.04, "external_ai": 0.92}),
                )
                await connection.execute(
                    "INSERT INTO events(time,session_id,task_id,load_seq,batch_seq,"
                    "seq_in_batch,kind,perf_now,origin,version_id,change_index,"
                    "start_line,start_col,end_line,end_col,inserted_len,"
                    "removed_len,inserted_lines,doc_len) "
                    "VALUES ($1,$2,$3,0,0,7,2,45000,0,9,0,12,1,18,1,0,24,0,50)",
                    deletion_start + timedelta(seconds=5), session_id, task_id,
                )
                deletion_evidence = await fetch_prediction_edit_evidence(
                    connection, deletion_prediction_id,
                )
                assert deletion_evidence is not None
                assert deletion_evidence.predicted_category == "external_ai"
                assert len(deletion_evidence.edit_events) == 1
                deletion_result = localize_prediction_edit_evidence(deletion_evidence)
                assert deletion_result.regions == ()
                assert len(deletion_result.unlocalized_events) == 1
                assert deletion_result.unlocalized_reasons == ("deletion_only",)
                assert deletion_result.unlocalized_events[0]["replaced_start_line"] == 12
                assert deletion_result.unlocalized_events[0]["replaced_end_line"] == 18
                assert deletion_result.unlocalized_events[0]["touched_end_line"] == 12

                print("Deletion-only positive fixture:", deletion_prediction_id)
                print("Deletion-only evidence preserved, no new region:", deletion_result.unlocalized_reasons)

                print("Positive fixture prediction:", prediction_id)
                print("Matched edit evidence rows:", len(evidence.edit_events))
                print("Localized independent regions:", [(r.load_seq, r.version_id, r.start_line, r.end_line) for r in located.regions])
                print("Touched line regions:", [
                    (e["touched_start_line"], e["touched_end_line"])
                    for e in evidence.edit_events
                ])
                print("Reset, copy, other-task and window-end events: excluded")
                print("Clean window: no localization evidence returned")
                print("TASK 10 DATABASE EVIDENCE, REGIONS AND OPTION A PASSED")
            finally:
                await transaction.rollback()
                print("Test data rolled back")
    finally:
        await close_db_pool(pool)


if __name__ == "__main__":
    asyncio.run(_smoke())
