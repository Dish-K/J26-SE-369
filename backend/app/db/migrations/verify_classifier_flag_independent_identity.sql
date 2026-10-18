-- Verify that independent flag identities coexist and identical
-- flag_id retries remain rejected by the existing primary key.
-- All fixtures are rolled back. Run AFTER migration 010.
BEGIN;

DO $$
DECLARE
  v_session UUID := uuid_generate_v4();
  v_task UUID := uuid_generate_v4();
  v_vector UUID;
  v_prediction UUID;
  v_flag_a UUID := uuid_generate_v4();
  v_flag_b UUID := uuid_generate_v4();
  v_flag_c UUID := uuid_generate_v4();
  v_window_start TIMESTAMPTZ := now() - INTERVAL '40 seconds';
  v_window_end TIMESTAMPTZ := now() - INTERVAL '10 seconds';
  v_rejected BOOLEAN;
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'classifier_flags'::regclass
      AND conname = 'classifier_flags_episode_uk'
  ) THEN
    RAISE EXCEPTION 'Old episode uniqueness constraint is still present';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'classifier_flags'::regclass
      AND contype = 'p'
      AND conname = 'classifier_flags_pkey'
  ) THEN
    RAISE EXCEPTION 'Flag primary key is missing';
  END IF;

  IF (
    SELECT count(*) FROM pg_constraint
    WHERE conrelid = 'classifier_flags'::regclass
      AND conname IN (
        'classifier_flags_representative_fk',
        'classifier_flags_id_task_uk',
        'classifier_flags_source_context_pair_chk',
        'classifier_flags_load_seq_chk',
        'classifier_flags_version_id_chk'
      )
  ) <> 5 THEN
    RAISE EXCEPTION 'A required existing flag contract was changed';
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
    v_session, v_task, v_window_start, v_window_end,
    'live', 'verify-010-schema-v1', '{}'::jsonb
  ) RETURNING feature_vector_id INTO v_vector;

  -- Valid fixture prediction. This does NOT create a behavioral classification rule.
  INSERT INTO classifier_window_predictions (
    feature_vector_id, task_id, model_version, predicted_category,
    class_probabilities, confidence, ai_positive
  ) VALUES (
    v_vector, v_task, 'verify-010-v1', 'external_ai',
    '{"no_ai":0.05,"ide_ai":0.05,"external_ai":0.90}'::jsonb,
    0.90, TRUE
  ) RETURNING prediction_id INTO v_prediction;

  -- Flag A: known historical code location.
  INSERT INTO classifier_flags (
    flag_id, task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    load_seq, version_id, evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    v_flag_a, v_task, 'verify-010-v1', 'external_ai', v_prediction,
    '{"no_ai":0.05,"ide_ai":0.05,"external_ai":0.90}'::jsonb,
    0.90, 10, 15, 0, 4, ARRAY[]::TEXT[], 'window_run', 1,
    v_window_start, v_window_end
  );

  -- Flag B: every descriptive column matches A, but the independently
  -- established flag_id differs. This MUST now be allowed.
  INSERT INTO classifier_flags (
    flag_id, task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    load_seq, version_id, evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    v_flag_b, v_task, 'verify-010-v1', 'external_ai', v_prediction,
    '{"no_ai":0.05,"ide_ai":0.05,"external_ai":0.90}'::jsonb,
    0.90, 10, 15, 0, 4, ARRAY[]::TEXT[], 'window_run', 1,
    v_window_start, v_window_end
  );

  -- Flag C: another page load, same visible code range and detection window.
  INSERT INTO classifier_flags (
    flag_id, task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    load_seq, version_id, evidence_kinds, segment_basis, window_count,
    detected_at, detected_until
  ) VALUES (
    v_flag_c, v_task, 'verify-010-v1', 'external_ai', v_prediction,
    '{"no_ai":0.05,"ide_ai":0.05,"external_ai":0.90}'::jsonb,
    0.90, 10, 15, 1, 4, ARRAY[]::TEXT[], 'window_run', 1,
    v_window_start, v_window_end
  );

  INSERT INTO classifier_flag_predictions (flag_id, prediction_id)
  VALUES (v_flag_a, v_prediction), (v_flag_b, v_prediction), (v_flag_c, v_prediction);

  IF (
    SELECT count(*)
      FROM classifier_flags f
      JOIN classifier_flag_predictions fp ON fp.flag_id = f.flag_id
     WHERE f.flag_id IN (v_flag_a, v_flag_b, v_flag_c)
       AND fp.prediction_id = v_prediction
  ) <> 3 THEN
    RAISE EXCEPTION 'Independent flags or their prediction relationships were lost';
  END IF;

  -- Repeating the identical stable flag_id is an error, not a new flag.
  v_rejected := FALSE;
  BEGIN
    INSERT INTO classifier_flags (
      flag_id, task_id, model_version, category, representative_prediction_id,
      class_probabilities, confidence, detected_start_line, detected_end_line,
      load_seq, version_id, evidence_kinds, segment_basis, window_count,
      detected_at, detected_until
    ) VALUES (
      v_flag_a, v_task, 'verify-010-v1', 'external_ai', v_prediction,
      '{"no_ai":0.05,"ide_ai":0.05,"external_ai":0.90}'::jsonb,
      0.90, 10, 15, 0, 4, ARRAY[]::TEXT[], 'window_run', 1,
      v_window_start, v_window_end
    );
  EXCEPTION WHEN unique_violation THEN
    v_rejected := TRUE;
  END;
  IF NOT v_rejected THEN
    RAISE EXCEPTION 'Repeated save of the same flag_id was not rejected';
  END IF;

  -- Prediction relationships must still prevent duplicate members.
  v_rejected := FALSE;
  BEGIN
    INSERT INTO classifier_flag_predictions (flag_id, prediction_id)
    VALUES (v_flag_a, v_prediction);
  EXCEPTION WHEN unique_violation THEN
    v_rejected := TRUE;
  END;
  IF NOT v_rejected THEN
    RAISE EXCEPTION 'Duplicate prediction membership was not rejected';
  END IF;

  -- Submission-time mapping is a separate step; its view must remain.
  IF to_regclass('classifier_flags_for_probing') IS NULL THEN
    RAISE EXCEPTION 'C3 mapped-flags view missing';
  END IF;

  RAISE NOTICE 'MIGRATION 010 VERIFIED: independent identical flags, reload identity, stable-ID duplicate rejection, prediction FKs, C3 view';
END;
$$;

ROLLBACK;
