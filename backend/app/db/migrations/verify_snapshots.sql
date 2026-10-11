-- code_snapshots contract checks. Run after 001-008; everything rolls back.
-- Fails against 001-007 alone on purpose: the reload case is the defect 008 fixes.

BEGIN;

DO $$
DECLARE
  s      UUID := uuid_generate_v4();
  other  UUID := uuid_generate_v4();
  t      UUID := uuid_generate_v4();
  n                        INTEGER;
  dup_rejected             BOOLEAN := FALSE;
  cross_session_rejected   BOOLEAN := FALSE;
  bad_redaction_rejected   BOOLEAN := FALSE;
BEGIN
  INSERT INTO sessions (session_id, candidate_id, language, started_at)
  VALUES (s,     uuid_generate_v4(), 'python', now()),
         (other, uuid_generate_v4(), 'python', now());

  INSERT INTO tasks (task_id, session_id, seq, depth, origin, created_by,
                     status, created_at, started_at)
  VALUES (t, s, 0, 0, 'original', 'interviewer', 'active', now(), now());

  INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                              perf_now, load_seq, code_text, doc_len,
                              capture_reason)
  VALUES (s, t, 100, now(), 10000.0, 0, 'def solve():' || chr(10) || '    pass', 25, 0),
         (s, t, 137, now(), 20000.0, 0, 'def solve():' || chr(10) || '    return 42', 31, 1);

  SELECT count(*) INTO n FROM code_snapshots WHERE task_id = t;
  IF n <> 2 THEN RAISE EXCEPTION 'expected 2 snapshots, found %', n; END IF;

  -- Regression: the same version_id before and after a reload is two documents.
  INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                              perf_now, load_seq, code_text, doc_len,
                              capture_reason)
  VALUES (s, t, 1, now(), 100.0, 0, 'print("before reload")', 22, 0);

  INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                              perf_now, load_seq, code_text, doc_len,
                              capture_reason)
  VALUES (s, t, 1, now(), 100.0, 1, 'print("after reload")', 21, 0)
  ON CONFLICT (task_id, load_seq, version_id) DO NOTHING;

  SELECT count(*) INTO n FROM code_snapshots WHERE task_id = t AND version_id = 1;
  IF n <> 2 THEN
    RAISE EXCEPTION
      'reload regression: expected both version_id=1 snapshots, found % -- a '
      'post-reload document was silently dropped', n;
  END IF;

  SELECT count(*) INTO n
    FROM code_snapshots
   WHERE task_id = t AND version_id = 1 AND load_seq = 1
     AND code_text = 'print("after reload")';
  IF n <> 1 THEN
    RAISE EXCEPTION 'the post-reload document is not retrievable by load_seq';
  END IF;

  RAISE NOTICE 'RELOAD OK - version_id is scoped to load_seq, both documents kept';

  SELECT count(*) INTO n
    FROM code_snapshots WHERE task_id = t AND capture_reason = 1;
  IF n <> 1 THEN RAISE EXCEPTION 'capture_reason not stored'; END IF;

  BEGIN
    INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                                perf_now, load_seq, code_text, doc_len,
                                capture_reason)
    VALUES (s, t, 900, now(), 50000.0, 0, 'x = 1', 5, 7);
    RAISE EXCEPTION 'an out-of-range capture_reason was accepted';
  EXCEPTION WHEN check_violation THEN
    NULL;  -- expected
  END;

  -- NULL means unknown, never a defaulted tick.
  INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                              perf_now, load_seq, code_text, doc_len)
  VALUES (s, t, 901, now(), 51000.0, 0, 'y = 2', 5);

  SELECT count(*) INTO n
    FROM code_snapshots WHERE task_id = t AND capture_reason IS NULL;
  IF n <> 1 THEN RAISE EXCEPTION 'unknown capture_reason was not preserved as NULL'; END IF;

  RAISE NOTICE 'CAPTURE REASON OK - recorded, range-checked, unknown kept unknown';

  -- The writer ticks every 10s even when nothing changed.
  BEGIN
    INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                                perf_now, load_seq, code_text, doc_len,
                                capture_reason)
    VALUES (s, t, 137, now(), 30000.0, 0, 'def solve():' || chr(10) || '    return 42', 31, 0);
  EXCEPTION WHEN unique_violation THEN
    dup_rejected := TRUE;
  END;

  IF NOT dup_rejected THEN
    RAISE EXCEPTION 'a duplicate (task_id, load_seq, version_id) snapshot was accepted';
  END IF;

  INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                              perf_now, load_seq, code_text, doc_len,
                              capture_reason)
  VALUES (s, t, 137, now(), 30000.0, 0, 'def solve():' || chr(10) || '    return 42', 31, 0)
  ON CONFLICT (task_id, load_seq, version_id) DO NOTHING;

  SELECT count(*) INTO n FROM code_snapshots WHERE task_id = t AND version_id = 137;
  IF n <> 1 THEN RAISE EXCEPTION 'idempotent tick changed the row count to %', n; END IF;

  RAISE NOTICE 'DEDUP OK - one row per (task, document version), tick is idempotent';

  BEGIN
    INSERT INTO code_snapshots (session_id, task_id, version_id, captured_at,
                                perf_now, load_seq, code_text, doc_len)
    VALUES (other, t, 200, now(), 40000.0, 0, 'x = 1', 5);
  EXCEPTION WHEN foreign_key_violation THEN
    cross_session_rejected := TRUE;
  END;

  IF NOT cross_session_rejected THEN
    RAISE EXCEPTION 'a snapshot was accepted under a session that does not own the task';
  END IF;

  RAISE NOTICE 'SCOPING OK - composite FK rejects a task/session mismatch';

  BEGIN
    UPDATE code_snapshots
       SET code_redacted_at = now()
     WHERE task_id = t AND version_id = 100;
  EXCEPTION WHEN check_violation THEN
    bad_redaction_rejected := TRUE;
  END;

  IF NOT bad_redaction_rejected THEN
    RAISE EXCEPTION 'redaction timestamp accepted while code_text still populated';
  END IF;

  -- Retention stage 1 redacts rather than deletes, so version anchors survive.
  UPDATE code_snapshots
     SET code_text = '', code_redacted_at = now()
   WHERE task_id = t;

  SELECT count(*) INTO n
    FROM code_snapshots
   WHERE task_id = t AND (code_redacted_at IS NULL OR code_text <> '');
  IF n <> 0 THEN RAISE EXCEPTION '% snapshots survived redaction unredacted', n; END IF;

  SELECT count(*) INTO n FROM code_snapshots WHERE task_id = t AND version_id = 137;
  IF n <> 1 THEN RAISE EXCEPTION 'redaction removed the snapshot anchor'; END IF;

  RAISE NOTICE 'REDACTION OK - bodies emptied, version anchors retained';
  RAISE NOTICE 'SNAPSHOT CONTRACT VERIFIED';
END
$$;

ROLLBACK;
