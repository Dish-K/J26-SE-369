-- Two-stage retention sweep check. Run after 001-008; everything rolls back.
-- The fixture must hold a row in every table, or a missing sweep step goes unseen.

BEGIN;

DO $$
DECLARE
  s   UUID := uuid_generate_v4();
  t   UUID := uuid_generate_v4();
  probe UUID := uuid_generate_v4();
  fv  UUID;
  p   UUID;
  f   UUID;
  bad_redaction_rejected BOOLEAN := FALSE;
  n INTEGER;
BEGIN
  INSERT INTO sessions (session_id, candidate_id, language, started_at)
  VALUES (s, uuid_generate_v4(), 'python', now());

  INSERT INTO tasks (task_id, session_id, seq, depth, origin, created_by, status, created_at, started_at)
  VALUES (t, s, 0, 0, 'original', 'interviewer', 'active', now(), now());

  INSERT INTO events (time, session_id, task_id, load_seq, batch_seq, seq_in_batch, kind, perf_now)
  VALUES (now(), s, t, 0, 0, 0, 0, 1000.0),
         (now(), s, t, 0, 0, 1, 1, 1080.0);

  INSERT INTO events (time, session_id, task_id, load_seq, batch_seq, seq_in_batch, kind, perf_now,
                      origin, inserted_len, removed_len, inserted_text)
  VALUES (now(), s, t, 0, 0, 2, 2, 1200.0, 1, 412, 0, 'secret pasted code');

  INSERT INTO clock_sync_samples (time, session_id, load_seq, t0, t1, t2, t3, rtt_ms, offset_ms, accepted)
  VALUES (now(), s, 0, 1.0, 2.0, 3.0, 4.0, 2.0, -38.2, TRUE);

  INSERT INTO ingested_batches (session_id, load_seq, batch_seq) VALUES (s, 0, 0);

  INSERT INTO feature_vectors (session_id, task_id, window_start, window_end, run_id, schema_version, features)
  VALUES (s, t, now() - interval '30 seconds', now(), 'live', 'test-v1', '{}'::jsonb)
  RETURNING feature_vector_id INTO fv;

  INSERT INTO classifier_window_predictions (
    feature_vector_id, task_id, model_version, predicted_category,
    class_probabilities, confidence, ai_positive)
  VALUES (fv, t, 'retention-v1', 'external_ai',
    '{"no_ai":0.06,"ide_ai":0.05,"external_ai":0.89}'::jsonb, 0.89, TRUE)
  RETURNING prediction_id INTO p;

  INSERT INTO classifier_flags (
    task_id, model_version, category, representative_prediction_id,
    class_probabilities, confidence, detected_start_line, detected_end_line,
    evidence_kinds, segment_basis, window_count, detected_at, detected_until)
  VALUES (t, 'retention-v1', 'external_ai', p,
    '{"no_ai":0.06,"ide_ai":0.05,"external_ai":0.89}'::jsonb, 0.89, 12, 41,
    ARRAY['paste']::text[], 'event', 1, now() - interval '30 seconds', now())
  RETURNING flag_id INTO f;

  INSERT INTO classifier_flag_predictions (flag_id, prediction_id) VALUES (f, p);

  INSERT INTO task_submissions (task_id, session_id, code_text, version_id,
                                load_seq, submitted_at)
  VALUES (t, s, 'def solve():\n    return 42', 100, 1, now());

  -- The same version_id on both sides of a reload, which 007 alone could not store.
  INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                              perf_now, load_seq, code_text, doc_len,
                              capture_reason)
  VALUES (s, t, 1,   now(), 1000.0,  0, 'print("before reload")',       22, 0),
         (s, t, 1,   now(), 1000.0,  1, 'print("after reload")',        21, 0),
         (s, t, 100, now(), 20000.0, 1, 'def solve():' || chr(10) ||
                                        '    return 42',                31, 1);

  INSERT INTO classifier_flag_final_locations (flag_id, task_id, status, final_start_line, final_end_line)
  VALUES (f, t, 'mapped', 50, 79);

  INSERT INTO classifier_task_scores (task_id, model_version, verdict, ai_confidence)
  VALUES (t, 'retention-v1', 'ai_detected', 0.89);

  -- A probe task pointing back at the flag closes the tasks <-> flags loop.
  INSERT INTO tasks (task_id, session_id, seq, depth, origin, parent_task_id,
                     parent_flag_id, created_by, status, created_at)
  VALUES (probe, s, 1, 1, 'probe', t, f, 'auto', 'pending', now());

  BEGIN
    UPDATE task_submissions SET code_redacted_at = now() WHERE task_id = t;
  EXCEPTION WHEN check_violation THEN
    bad_redaction_rejected := TRUE;
  END;

  IF NOT bad_redaction_rejected THEN
    RAISE EXCEPTION 'code_redacted_at was accepted without emptying code_text';
  END IF;

  -- Stage 1: sensitive-data expiry.
  DELETE FROM events             WHERE session_id = s;
  DELETE FROM clock_sync_samples WHERE session_id = s;
  DELETE FROM ingested_batches   WHERE session_id = s;

  UPDATE task_submissions
     SET code_text = '', code_redacted_at = now()
   WHERE session_id = s;

  UPDATE code_snapshots
     SET code_text = '', code_redacted_at = now()
   WHERE session_id = s AND code_redacted_at IS NULL;

  SELECT count(*) INTO n FROM events WHERE session_id = s;
  IF n <> 0 THEN RAISE EXCEPTION 'stage 1 left % event rows', n; END IF;

  SELECT count(*) INTO n FROM task_submissions
   WHERE session_id = s AND code_text = '' AND code_redacted_at IS NOT NULL;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 did not redact the submission'; END IF;

  SELECT count(*) INTO n FROM code_snapshots
   WHERE session_id = s AND (code_text <> '' OR code_redacted_at IS NULL);
  IF n <> 0 THEN
    RAISE EXCEPTION 'stage 1 left % snapshot bodies unredacted', n;
  END IF;

  SELECT count(*) INTO n FROM code_snapshots WHERE session_id = s;
  IF n <> 3 THEN
    RAISE EXCEPTION 'stage 1 deleted snapshot rows (found %) -- a flag that '
                    'names a document version loses its anchor', n;
  END IF;

  SELECT count(*) INTO n FROM code_snapshots
   WHERE session_id = s AND version_id = 1;
  IF n <> 2 THEN
    RAISE EXCEPTION 'the pre/post-reload version_id=1 pair did not survive';
  END IF;

  SELECT count(*) INTO n FROM classifier_flags WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed classifier_flags'; END IF;
  SELECT count(*) INTO n FROM classifier_flag_final_locations WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed final locations'; END IF;
  SELECT count(*) INTO n FROM classifier_window_predictions WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed window predictions'; END IF;
  SELECT count(*) INTO n FROM feature_vectors WHERE session_id = s;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed feature vectors'; END IF;

  SELECT count(*) INTO n FROM classifier_flags_for_probing WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'probing view broken after stage 1'; END IF;

  RAISE NOTICE 'STAGE 1 OK - telemetry removed, submission and snapshots redacted, C2 results intact';

  -- Stage 2: research-data deletion. parent_flag_id's SET NULL lets flags go before the probe task.
  DELETE FROM classifier_flag_final_locations WHERE task_id = t;
  DELETE FROM classifier_flag_predictions     WHERE flag_id = f;
  DELETE FROM classifier_flags                WHERE task_id = t;

  SELECT count(*) INTO n FROM tasks WHERE task_id = probe AND parent_flag_id IS NULL;
  IF n <> 1 THEN RAISE EXCEPTION 'ON DELETE SET NULL did not clear parent_flag_id'; END IF;

  DELETE FROM classifier_task_scores       WHERE task_id = t;
  DELETE FROM classifier_window_predictions WHERE task_id = t;
  DELETE FROM feature_vectors               WHERE session_id = s;
  DELETE FROM task_submissions              WHERE session_id = s;

  -- Must run before tasks: the snapshot FK has no ON DELETE action, so skipping
  -- this aborts the whole sweep.
  DELETE FROM code_snapshots                WHERE session_id = s;

  DELETE FROM tasks                         WHERE session_id = s;
  DELETE FROM sessions                      WHERE session_id = s;

  SELECT count(*) INTO n FROM sessions WHERE session_id = s;
  IF n <> 0 THEN RAISE EXCEPTION 'stage 2 did not complete'; END IF;

  SELECT count(*) INTO n FROM code_snapshots WHERE session_id = s;
  IF n <> 0 THEN RAISE EXCEPTION 'stage 2 left % snapshot rows', n; END IF;

  RAISE NOTICE 'STAGE 2 OK - all data removed in dependency order';
  RAISE NOTICE 'RETENTION CONTRACT VERIFIED';
END
$$;

ROLLBACK;
