-- verify_retention.sql
-- Exercise the two-stage retention procedure from specification §17.1.
--
-- Run AFTER migrations 001-006, on a disposable/local database. Everything is
-- wrapped in a transaction and rolled back, so no rows remain.
--
-- What this proves:
--   * stage 1 removes raw telemetry and redacts submitted code while leaving
--     every derived C2 result intact;
--   * code_redacted_at distinguishes a redacted submission from an empty one,
--     and the CHECK rejects a timestamp without an emptied body;
--   * stage 2 deletes the derived data in the documented order without the
--     tasks <-> classifier_flags loop blocking the transaction.

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
  ---------------------------------------------------------------------------
  -- Build one complete session: capture, features, C2 output, submission.
  ---------------------------------------------------------------------------
  INSERT INTO sessions (session_id, candidate_id, language, started_at)
  VALUES (s, uuid_generate_v4(), 'python', now());

  INSERT INTO tasks (task_id, session_id, seq, depth, origin, created_by, status, created_at, started_at)
  VALUES (t, s, 0, 0, 'original', 'interviewer', 'active', now(), now());

  -- raw telemetry: one keystroke pair and one paste carrying text
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

  INSERT INTO task_submissions (task_id, session_id, code_text, version_id, submitted_at)
  VALUES (t, s, 'def solve():\n    return 42', 100, now());

  INSERT INTO classifier_flag_final_locations (flag_id, task_id, status, final_start_line, final_end_line)
  VALUES (f, t, 'mapped', 50, 79);

  INSERT INTO classifier_task_scores (task_id, model_version, verdict, ai_confidence)
  VALUES (t, 'retention-v1', 'ai_detected', 0.89);

  -- a probe task pointing back at the flag: this is the dependency loop
  INSERT INTO tasks (task_id, session_id, seq, depth, origin, parent_task_id,
                     parent_flag_id, created_by, status, created_at)
  VALUES (probe, s, 1, 1, 'probe', t, f, 'auto', 'pending', now());

  ---------------------------------------------------------------------------
  -- The CHECK must reject a redaction timestamp without an emptied body.
  ---------------------------------------------------------------------------
  BEGIN
    UPDATE task_submissions SET code_redacted_at = now() WHERE task_id = t;
  EXCEPTION WHEN check_violation THEN
    bad_redaction_rejected := TRUE;
  END;

  IF NOT bad_redaction_rejected THEN
    RAISE EXCEPTION 'code_redacted_at was accepted without emptying code_text';
  END IF;

  ---------------------------------------------------------------------------
  -- STAGE 1 - sensitive-data expiry (spec §17.1)
  ---------------------------------------------------------------------------
  DELETE FROM events             WHERE session_id = s;
  DELETE FROM clock_sync_samples WHERE session_id = s;
  DELETE FROM ingested_batches   WHERE session_id = s;

  UPDATE task_submissions
     SET code_text = '', code_redacted_at = now()
   WHERE session_id = s;

  -- raw telemetry gone
  SELECT count(*) INTO n FROM events WHERE session_id = s;
  IF n <> 0 THEN RAISE EXCEPTION 'stage 1 left % event rows', n; END IF;

  -- submitted code redacted, and provably redacted rather than empty
  SELECT count(*) INTO n FROM task_submissions
   WHERE session_id = s AND code_text = '' AND code_redacted_at IS NOT NULL;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 did not redact the submission'; END IF;

  -- every derived C2 result survives
  SELECT count(*) INTO n FROM classifier_flags WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed classifier_flags'; END IF;
  SELECT count(*) INTO n FROM classifier_flag_final_locations WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed final locations'; END IF;
  SELECT count(*) INTO n FROM classifier_window_predictions WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed window predictions'; END IF;
  SELECT count(*) INTO n FROM feature_vectors WHERE session_id = s;
  IF n <> 1 THEN RAISE EXCEPTION 'stage 1 destroyed feature vectors'; END IF;

  -- and C3 can still read the flag through its view
  SELECT count(*) INTO n FROM classifier_flags_for_probing WHERE task_id = t;
  IF n <> 1 THEN RAISE EXCEPTION 'probing view broken after stage 1'; END IF;

  RAISE NOTICE 'STAGE 1 OK - telemetry removed, code redacted, C2 results intact';

  ---------------------------------------------------------------------------
  -- STAGE 2 - final research-data deletion, in the documented order.
  -- tasks.parent_flag_id uses ON DELETE SET NULL, so deleting the flags does
  -- not deadlock against the probe task that references them.
  ---------------------------------------------------------------------------
  DELETE FROM classifier_flag_final_locations WHERE task_id = t;
  DELETE FROM classifier_flag_predictions     WHERE flag_id = f;
  DELETE FROM classifier_flags                WHERE task_id = t;

  -- the loop-breaker actually fired
  SELECT count(*) INTO n FROM tasks WHERE task_id = probe AND parent_flag_id IS NULL;
  IF n <> 1 THEN RAISE EXCEPTION 'ON DELETE SET NULL did not clear parent_flag_id'; END IF;

  DELETE FROM classifier_task_scores       WHERE task_id = t;
  DELETE FROM classifier_window_predictions WHERE task_id = t;
  DELETE FROM feature_vectors               WHERE session_id = s;
  DELETE FROM task_submissions              WHERE session_id = s;
  DELETE FROM tasks                         WHERE session_id = s;
  DELETE FROM sessions                      WHERE session_id = s;

  SELECT count(*) INTO n FROM sessions WHERE session_id = s;
  IF n <> 0 THEN RAISE EXCEPTION 'stage 2 did not complete'; END IF;

  RAISE NOTICE 'STAGE 2 OK - all data removed in dependency order';
  RAISE NOTICE 'RETENTION CONTRACT VERIFIED';
END
$$;

ROLLBACK;
