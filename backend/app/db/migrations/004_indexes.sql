-- 004_indexes.sql
-- Index only demonstrated project query paths. Constraints/primary keys
-- already create several B-tree indexes automatically; do not duplicate them.

-- Feature extraction and the degradation harness load one task's telemetry
-- and must walk events in true capture order. `time` alone is not sufficient:
-- two events may share perf_now, and a reload can produce a new clock offset.
CREATE INDEX events_extract_order_idx
  ON events (session_id, task_id, load_seq, batch_seq, seq_in_batch);

-- Clock reconciliation reads samples for one session/load segment.
CREATE INDEX clock_session_load_idx
  ON clock_sync_samples (session_id, load_seq, time);

-- Intentionally NOT added:
--   events(time)                      -> TimescaleDB adds it for the hypertable.
--   tasks(session_id, seq)            -> UNIQUE constraint already indexes it.
--   feature_vectors natural lookup    -> UNIQUE constraint already indexes
--                                        (session_id, task_id, window_start,
--                                         window_end, run_id). Its leftmost
--                                        prefix covers per-task/window fetch.
--   classifier_flags(task_id)        -> created in 005_classifier_output.sql
--                                        alongside the table itself.
--   classifier_task_scores(task_id)   -> UNIQUE (task_id, model_version)
--                                        already indexes it.
--   task_submissions(session_id)      -> table is tiny; add only if measured.
--   events(kind)                      -> no demonstrated selective query path.
