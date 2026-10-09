-- verify_contract.sql
-- Smoke-test the frozen database contract. Run AFTER migrations 001-006 on a
-- disposable/local database. Everything is wrapped in a transaction and rolled
-- back so no verification rows remain.

BEGIN;

DO $$
DECLARE
  s UUID := uuid_generate_v4();
  t UUID := uuid_generate_v4();
  clean_t UUID := uuid_generate_v4();
  fv1 UUID;
  fv2 UUID;
  p1 UUID;
  p2 UUID;
  f1 UUID;
  f2 UUID;
  duplicate_rejected BOOLEAN := FALSE;
BEGIN
  INSERT INTO sessions (
    session_id, candidate_id, language, started_at
  ) VALUES (
    s, uuid_generate_v4(), 'python', now()
  );

  INSERT INTO tasks (
    task_id, session_id, seq, depth, origin, created_by, status, created_at, started_at
  ) VALUES
    (t, s, 0, 0, 'original', 'interviewer', 'active', now(), now()),
    (clean_t, s, 1, 0, 'original', 'interviewer', 'active', now(), now());

  INSERT INTO feature_vectors (
    session_id, task_id, window_start, window_end, run_id, schema_version, features
  ) VALUES (
    s, t, now() - interval '40 seconds', now() - interval '10 seconds',
    'live', 'test-v1', '{}'::jsonb
  ) RETURNING feature_vector_id INTO fv1;

  INSERT INTO feature_vectors (
    session_id, task_id, window_start, window_end, run_id, schema_version, features
  ) VALUES (
    s, t, now() - interval '30 seconds', now(),
    'live', 'test-v1', '{}'::jsonb
  ) RETURNING feature_vector_id INTO fv2;

  INSERT INTO classifier_window_predictions (
    feature_vector_id, task_id, model_version, predicted_category,
    class_probabilities, confidence, ai_positive
  ) VALUES (
    fv1, t, 'verify-model-v1', 'external_ai',
    '{"no_ai":0.08,"ide_ai":0.07,"external_ai":0.85}'::jsonb,
    0.85, TRUE
  ) RETURNING prediction_id INTO p1;

  INSERT INTO classifier_window_predictions (
    feature_vector_id, task_id, model_version, predicted_category,
    class_probabilities, confidence, ai_positive
  ) VALUES (
    fv2, t, 'verify-model-v1', 'external_ai',
    '{"no_ai":0.06,"ide_ai":0.05,"external_ai":0.89}'::jsonb,
    0.89, TRUE
  ) RETURNING prediction_id INTO p2;

  INSERT INTO classifier_flags (
    task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence,
    detected_start_line, detected_end_line,
    evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    t, 'verify-model-v1', 'external_ai', p2,
    '{"no_ai":0.06,"ide_ai":0.05,"external_ai":0.89}'::jsonb,
    0.89,
    20, 42,
    ARRAY['paste','typing_burst']::text[], 'mixed', 2,
    now() - interval '30 seconds', now()
  ) RETURNING flag_id INTO f1;

  INSERT INTO classifier_flag_predictions (flag_id, prediction_id)
  VALUES (f1, p1), (f1, p2);

  -- Same category/range later is valid because detection time is different.
  INSERT INTO classifier_flags (
    task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence,
    detected_start_line, detected_end_line,
    evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    t, 'verify-model-v1', 'external_ai', p1,
    '{"no_ai":0.08,"ide_ai":0.07,"external_ai":0.85}'::jsonb,
    0.85,
    20, 42,
    ARRAY['typing_burst']::text[], 'window_run', 1,
    now() + interval '10 seconds', now() + interval '10 seconds'
  ) RETURNING flag_id INTO f2;

  INSERT INTO classifier_flag_predictions (flag_id, prediction_id)
  VALUES (f2, p1);

  -- Exact duplicate episode must be rejected.
  BEGIN
    INSERT INTO classifier_flags (
      task_id, model_version, category, representative_prediction_id,
      class_probabilities, confidence,
      detected_start_line, detected_end_line,
      evidence_kinds, segment_basis, window_count,
      detected_at, detected_until
    )
    SELECT
      task_id, model_version, category, representative_prediction_id,
      class_probabilities, confidence,
      detected_start_line, detected_end_line,
      evidence_kinds, segment_basis, window_count,
      detected_at, detected_until
    FROM classifier_flags
    WHERE flag_id = f2;
  EXCEPTION WHEN unique_violation THEN
    duplicate_rejected := TRUE;
  END;

  IF NOT duplicate_rejected THEN
    RAISE EXCEPTION 'duplicate classifier episode was not rejected';
  END IF;

  INSERT INTO task_submissions (
    task_id, session_id, code_text, version_id, submitted_at
  ) VALUES
    (t, s, 'line1\nline2\nline3', 100, now()),
    (clean_t, s, 'pass', 1, now());

  INSERT INTO classifier_flag_final_locations (
    flag_id, task_id, status, final_start_line, final_end_line
  ) VALUES
    (f1, t, 'mapped', 35, 57),
    (f2, t, 'deleted', NULL, NULL);

  INSERT INTO classifier_task_scores (
    task_id, model_version, verdict, ai_confidence
  ) VALUES (
    t, 'verify-model-v1', 'ai_detected', 0.89
  );

  INSERT INTO classifier_task_scores (
    task_id, model_version, verdict, ai_confidence
  ) VALUES (
    clean_t, 'verify-model-v1', 'no_ai_detected', NULL
  );

  IF NOT EXISTS (
    SELECT 1
    FROM classifier_flags_for_probing
    WHERE flag_id = f1 AND start_line = 35 AND end_line = 57
  ) THEN
    RAISE EXCEPTION 'mapped flag not exposed correctly through probing view';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM classifier_flags_for_probing
    WHERE flag_id = f2
  ) THEN
    RAISE EXCEPTION 'deleted flag must not appear in probing view';
  END IF;
END;
$$;

ROLLBACK;

-- If psql reaches this point without an ERROR, the contract smoke test passed.
