-- Verifies schema, C2 flag insertion and prediction membership.
-- All test rows are rolled back. Safe to rerun after successful migration.
BEGIN;

DO $$
DECLARE
  v_session UUID := uuid_generate_v4();
  v_task UUID := uuid_generate_v4();
  v_fv1 UUID;
  v_fv2 UUID;
  v_p1 UUID;
  v_p2 UUID;
  v_f1 UUID;
  v_f2 UUID;
  rejected BOOLEAN;
BEGIN
  IF (
    SELECT count(*)
      FROM information_schema.columns
     WHERE table_schema = current_schema()
       AND table_name = 'classifier_flags'
       AND (
         (column_name = 'load_seq' AND data_type = 'smallint' AND is_nullable = 'YES')
         OR (column_name = 'version_id' AND data_type = 'bigint' AND is_nullable = 'YES')
       )
  ) <> 2 THEN
    RAISE EXCEPTION 'Migration 009 columns missing or wrong types/nullability';
  END IF;

  IF (
    SELECT count(*)
      FROM pg_constraint
     WHERE conrelid = 'classifier_flags'::regclass
       AND conname IN (
         'classifier_flags_source_context_pair_chk',
         'classifier_flags_load_seq_chk',
         'classifier_flags_version_id_chk'
       )
  ) <> 3 THEN
    RAISE EXCEPTION 'Migration 009 source-context constraints missing';
  END IF;

  INSERT INTO sessions (session_id, candidate_id, language, started_at)
  VALUES (v_session, uuid_generate_v4(), 'python', now());

  INSERT INTO tasks (
    task_id, session_id, seq, depth, origin, created_by,
    status, created_at, started_at
  ) VALUES (
    v_task, v_session, 0, 0, 'original', 'interviewer',
    'active', now(), now()
  );

  INSERT INTO feature_vectors (
    session_id, task_id, window_start, window_end,
    run_id, schema_version, features
  ) VALUES (
    v_session, v_task, now() - interval '50 seconds',
    now() - interval '20 seconds', 'live', 'test-v1', '{}'::jsonb
  ) RETURNING feature_vector_id INTO v_fv1;

  INSERT INTO feature_vectors (
    session_id, task_id, window_start, window_end,
    run_id, schema_version, features
  ) VALUES (
    v_session, v_task, now() - interval '40 seconds',
    now() - interval '10 seconds', 'live', 'test-v1', '{}'::jsonb
  ) RETURNING feature_vector_id INTO v_fv2;

  -- Fixture probabilities only. No event type is being used to classify.
  INSERT INTO classifier_window_predictions (
    feature_vector_id, task_id, model_version, predicted_category,
    class_probabilities, confidence, ai_positive
  ) VALUES (
    v_fv1, v_task, 'verify-009-v1', 'external_ai',
    '{"no_ai":0.08,"ide_ai":0.06,"external_ai":0.86}'::jsonb,
    0.86, TRUE
  ) RETURNING prediction_id INTO v_p1;

  INSERT INTO classifier_window_predictions (
    feature_vector_id, task_id, model_version, predicted_category,
    class_probabilities, confidence, ai_positive
  ) VALUES (
    v_fv2, v_task, 'verify-009-v1', 'external_ai',
    '{"no_ai":0.05,"ide_ai":0.04,"external_ai":0.91}'::jsonb,
    0.91, TRUE
  ) RETURNING prediction_id INTO v_p2;

  -- Same historical line numbers, different page loads and detection episodes.
  -- Neither flag requires a matching code_snapshots row to exist.
  INSERT INTO classifier_flags (
    task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    load_seq, version_id, evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    v_task, 'verify-009-v1', 'external_ai', v_p1,
    '{"no_ai":0.08,"ide_ai":0.06,"external_ai":0.86}'::jsonb,
    0.86, 10, 15, 0, 4, ARRAY['typing_burst']::text[], 'window_run', 1,
    now() - interval '50 seconds', now() - interval '20 seconds'
  ) RETURNING flag_id INTO v_f1;

  INSERT INTO classifier_flags (
    task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    load_seq, version_id, evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    v_task, 'verify-009-v1', 'external_ai', v_p2,
    '{"no_ai":0.05,"ide_ai":0.04,"external_ai":0.91}'::jsonb,
    0.91, 10, 15, 1, 4, ARRAY['typing_burst']::text[], 'window_run', 1,
    now() - interval '40 seconds', now() - interval '10 seconds'
  ) RETURNING flag_id INTO v_f2;

  INSERT INTO classifier_flag_predictions (flag_id, prediction_id)
  VALUES (v_f1, v_p1), (v_f2, v_p2);

  IF (
    SELECT count(*)
      FROM classifier_flags f
      JOIN classifier_flag_predictions m ON m.flag_id = f.flag_id
      JOIN classifier_window_predictions p ON p.prediction_id = m.prediction_id
     WHERE f.task_id = v_task
       AND f.detected_start_line = 10 AND f.detected_end_line = 15
       AND f.version_id = 4
       AND f.load_seq IN (0, 1)
       AND p.task_id = f.task_id
       AND p.ai_positive
  ) <> 2 THEN
    RAISE EXCEPTION 'Historical identities or flag-prediction relationships did not survive';
  END IF;

  rejected := FALSE;
  BEGIN
    UPDATE classifier_flags SET version_id = NULL WHERE flag_id = v_f1;
  EXCEPTION WHEN check_violation THEN
    rejected := TRUE;
  END;
  IF NOT rejected THEN
    RAISE EXCEPTION 'A half-known document identity was accepted';
  END IF;

  rejected := FALSE;
  BEGIN
    UPDATE classifier_flags SET load_seq = -1 WHERE flag_id = v_f1;
  EXCEPTION WHEN check_violation THEN
    rejected := TRUE;
  END;
  IF NOT rejected THEN
    RAISE EXCEPTION 'Negative load sequence was accepted';
  END IF;

  rejected := FALSE;
  BEGIN
    UPDATE classifier_flags SET version_id = -1 WHERE flag_id = v_f1;
  EXCEPTION WHEN check_violation THEN
    rejected := TRUE;
  END;
  IF NOT rejected THEN
    RAISE EXCEPTION 'Negative document version was accepted';
  END IF;

  -- Legacy compatibility: NULL/NULL is allowed, never fabricated or backfilled.
  UPDATE classifier_flags
     SET load_seq = NULL, version_id = NULL
   WHERE flag_id = v_f1;
  IF NOT EXISTS (
    SELECT 1 FROM classifier_flags
     WHERE flag_id = v_f1 AND load_seq IS NULL AND version_id IS NULL
  ) THEN
    RAISE EXCEPTION 'Legacy unknown source context was rejected';
  END IF;

  RAISE NOTICE 'MIGRATION 009 VERIFIED: columns, identities, existing FKs, constraints, legacy NULLs';
END;
$$;

ROLLBACK;
