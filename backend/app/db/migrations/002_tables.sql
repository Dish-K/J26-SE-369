-- 002_tables.sql
-- Stable C1 telemetry + shared feature-vector foundation for CodeTrace.

-- one interview session can contain an original task and later probes.
-- task-scoped coding events and session-scoped environment events share one ordered telemetry stream.
-- final submitted code is stored once per task.
-- clock-sync samples and batch-ingest acknowledgements are retained.
-- feature vectors are validated by the shared Python contract before write.
-- the normal live extractor emits 30 s windows every 10 s, while the schema also permits alternate window sizes for C1's window-size study.

-- candidate_id is a pseudonymous research ID, never a name or email.
CREATE TABLE sessions (
  session_id          UUID PRIMARY KEY,
  candidate_id        UUID NOT NULL,
  language            TEXT NOT NULL,
  started_at          TIMESTAMPTZ NOT NULL,
  ended_at            TIMESTAMPTZ,

  -- final browser's server clock offset retained for audit/reconstruction
  final_offset_ms     DOUBLE PRECISION,

  capture_profile     TEXT NOT NULL DEFAULT 'research'
                        CHECK (capture_profile IN ('research')),

  -- browser / machine context used to characterize capture conditions
  user_agent          TEXT,
  platform            TEXT,
  screen_w            INTEGER,
  screen_h            INTEGER,
  viewport_w          INTEGER,
  viewport_h          INTEGER,
  device_pixel_ratio  REAL,
  locale              TEXT,
  timezone            TEXT,

  -- Stored so editor parity between training and inference can be proven.
  editor_profile      TEXT,
  editor_config_hash  TEXT,

  -- data-quality summary
  ime_detected        BOOLEAN NOT NULL DEFAULT FALSE,
  dropped_event_count INTEGER NOT NULL DEFAULT 0,
  degraded            BOOLEAN NOT NULL DEFAULT FALSE,

  -- NULL means not checked; never record unknown as FALSE.
  multi_screen        BOOLEAN,

  CONSTRAINT sessions_time_order_chk
    CHECK (ended_at IS NULL OR ended_at >= started_at),
  CONSTRAINT sessions_dropped_event_count_chk
    CHECK (dropped_event_count >= 0),
  CONSTRAINT sessions_screen_w_chk
    CHECK (screen_w IS NULL OR screen_w > 0),
  CONSTRAINT sessions_screen_h_chk
    CHECK (screen_h IS NULL OR screen_h > 0),
  CONSTRAINT sessions_viewport_w_chk
    CHECK (viewport_w IS NULL OR viewport_w > 0),
  CONSTRAINT sessions_viewport_h_chk
    CHECK (viewport_h IS NULL OR viewport_h > 0),
  CONSTRAINT sessions_device_pixel_ratio_chk
    CHECK (device_pixel_ratio IS NULL OR device_pixel_ratio > 0)
);

-- tasks.
-- condition is per task because participants can be instructed differently per task.
CREATE TABLE tasks (
  task_id         UUID PRIMARY KEY,
  session_id      UUID NOT NULL REFERENCES sessions(session_id),
  seq             INTEGER NOT NULL,
  depth           INTEGER NOT NULL DEFAULT 0,
  origin          TEXT NOT NULL CHECK (origin IN ('original', 'probe')),
  parent_task_id  UUID REFERENCES tasks(task_id),

  -- FK added in 005, once classifier_flags exists.
  parent_flag_id  UUID,

  created_by      TEXT CHECK (created_by IN ('interviewer', 'auto')),
  status          TEXT NOT NULL
                    CHECK (status IN ('pending', 'active', 'submitted', 'expired')),

  -- Distinct task lifecycle moments.
  created_at      TIMESTAMPTZ NOT NULL,
  delivered_at    TIMESTAMPTZ,
  started_at      TIMESTAMPTZ,
  submitted_at    TIMESTAMPTZ,

  condition       TEXT CHECK (condition IN ('no_ai', 'ide_ai', 'external_ai')),

  CONSTRAINT tasks_seq_chk CHECK (seq >= 0),
  CONSTRAINT tasks_depth_chk CHECK (depth >= 0),
  CONSTRAINT tasks_parent_shape_chk CHECK (
    (origin = 'original' AND parent_task_id IS NULL)
    OR
    (origin = 'probe' AND parent_task_id IS NOT NULL)
  ),

  -- seq gives deterministic task order inside one session.
  UNIQUE (session_id, seq),

  -- Composite FK target, so a task_id cannot be paired with the wrong session.
  UNIQUE (task_id, session_id)
);

-- events
-- Unified high-volume telemetry stream. This becomes the only hypertable in 003_hypertables.sql.
-- kind:
--   0 keydown        1 keyup          2 edit
--   3 blur           4 focus          5 visibility change
--   6 fullscreen     7 resize
--   8 run started    9 run finished
--  10 copy          11 cut           12 page hide
--  13 network state 14 display connected/disconnected
--
-- Task-scoped events (0,1,2,8,9) require task_id. Session-scoped environment
-- events may occur before/after/between tasks and may therefore have NULL
-- task_id. Those NULL-task events must never be included in behavioural
-- feature windows.
CREATE TABLE events (
  time          TIMESTAMPTZ      NOT NULL,   -- server-aligned; partition column
  session_id    UUID             NOT NULL,

  -- NULL outside a task; such events never enter a feature window.
  task_id       UUID,

  load_seq      SMALLINT         NOT NULL,   -- increments per page load
  batch_seq     BIGINT           NOT NULL,   -- delivery only; resets on reload

  -- (load_seq, batch_seq, seq_in_batch) orders a session's events. time can tie
  -- or run backwards after a reload, so never order by it.
  seq_in_batch  SMALLINT         NOT NULL,
  kind          SMALLINT         NOT NULL CHECK (kind BETWEEN 0 AND 14),

  -- Never rewritten after capture; the degradation study perturbs it.
  perf_now      DOUBLE PRECISION NOT NULL,

  -- Since the previous kind 0-2 event in the same (session_id, load_seq, task_id).
  gap_ms        DOUBLE PRECISION,

  -- keystrokes, kinds 0-1
  key_class     SMALLINT,
  key_code      TEXT,
  modifiers     SMALLINT,
  is_repeat     BOOLEAN,
  is_trusted    BOOLEAN,

  -- edit fields, kind 2
  -- origin: 0 typed, 1 pasted, 2 undo, 3 redo, 4 reset, 5 autocomplete_accepted (reserved until that capture profile is implemented).
  origin         SMALLINT CHECK (origin BETWEEN 0 AND 5),
  version_id     BIGINT,
  change_index   SMALLINT,
  start_line     INTEGER,
  start_col      INTEGER,
  end_line       INTEGER,
  end_col        INTEGER,
  range_offset   INTEGER,
  inserted_len   INTEGER,
  removed_len    INTEGER,
  inserted_lines SMALLINT,
  doc_len        INTEGER,
  paste_origin   SMALLINT,

  -- Pasted text only, gated by ethics approval. Typed and deleted text is never stored.
  inserted_text  TEXT,

  -- environment / network / display fields
  env_state      SMALLINT,
  viewport_w     INTEGER,
  viewport_h     INTEGER,

  -- Sandbox execution id; unrelated to feature_vectors.run_id.
  run_id              UUID,
  exit_status         SMALLINT,
  browser_duration_ms DOUBLE PRECISION,
  sandbox_time_ms     DOUBLE PRECISION,
  sandbox_memory_kb   INTEGER,

  -- Copy and cut store a length, never the text, and reuse start_line..end_col
  -- for the copied range. A later paste is its own kind 2 event.
  copied_len    INTEGER,
  source_pane   SMALLINT,

  CONSTRAINT events_task_scoped_have_task_chk
    CHECK (kind NOT IN (0, 1, 2, 8, 9) OR task_id IS NOT NULL),
  CONSTRAINT events_sequence_nonnegative_chk
    CHECK (load_seq >= 0 AND batch_seq >= 0 AND seq_in_batch >= 0),
  CONSTRAINT events_gap_nonnegative_chk
    CHECK (gap_ms IS NULL OR gap_ms >= 0),
  CONSTRAINT events_modifiers_chk
    CHECK (modifiers IS NULL OR modifiers BETWEEN 0 AND 15),
  CONSTRAINT events_change_index_chk
    CHECK (change_index IS NULL OR change_index >= 0),
  CONSTRAINT events_start_line_chk
    CHECK (start_line IS NULL OR start_line >= 1),
  CONSTRAINT events_start_col_chk
    CHECK (start_col IS NULL OR start_col >= 1),
  CONSTRAINT events_end_line_chk
    CHECK (end_line IS NULL OR end_line >= 1),
  CONSTRAINT events_end_col_chk
    CHECK (end_col IS NULL OR end_col >= 1),
  CONSTRAINT events_range_offset_chk
    CHECK (range_offset IS NULL OR range_offset >= 0),
  CONSTRAINT events_inserted_len_chk
    CHECK (inserted_len IS NULL OR inserted_len >= 0),
  CONSTRAINT events_removed_len_chk
    CHECK (removed_len IS NULL OR removed_len >= 0),
  CONSTRAINT events_inserted_lines_chk
    CHECK (inserted_lines IS NULL OR inserted_lines >= 0),
  CONSTRAINT events_doc_len_chk
    CHECK (doc_len IS NULL OR doc_len >= 0),
  CONSTRAINT events_inserted_text_only_paste_chk
    CHECK (inserted_text IS NULL OR (kind = 2 AND origin = 1)),
  CONSTRAINT events_viewport_w_chk
    CHECK (viewport_w IS NULL OR viewport_w > 0),
  CONSTRAINT events_viewport_h_chk
    CHECK (viewport_h IS NULL OR viewport_h > 0),
  CONSTRAINT events_exit_status_chk
    CHECK (exit_status IS NULL OR exit_status BETWEEN 0 AND 2),
  CONSTRAINT events_browser_duration_chk
    CHECK (browser_duration_ms IS NULL OR browser_duration_ms >= 0),
  CONSTRAINT events_sandbox_time_chk
    CHECK (sandbox_time_ms IS NULL OR sandbox_time_ms >= 0),
  CONSTRAINT events_sandbox_memory_chk
    CHECK (sandbox_memory_kb IS NULL OR sandbox_memory_kb >= 0),
  CONSTRAINT events_copied_len_chk
    CHECK (copied_len IS NULL OR copied_len >= 0),
  CONSTRAINT events_source_pane_chk
    CHECK (source_pane IS NULL OR source_pane BETWEEN 0 AND 3)
);

CREATE TABLE task_submissions (
  task_id      UUID PRIMARY KEY,
  session_id   UUID NOT NULL,
  code_text    TEXT NOT NULL,
  version_id   BIGINT,
  submitted_at TIMESTAMPTZ NOT NULL,

  -- Retention stage 1 empties code_text but keeps the row, because flag mappings
  -- reference it. This tells a redacted file apart from an empty submission.
  code_redacted_at TIMESTAMPTZ,

  CONSTRAINT task_submissions_task_session_fk
    FOREIGN KEY (task_id, session_id)
    REFERENCES tasks(task_id, session_id),

  -- One direction only: a candidate may genuinely submit an empty file.
  CONSTRAINT task_submissions_redaction_chk
    CHECK (code_redacted_at IS NULL OR code_text = '')
);

-- Rejected samples are kept too; they are evidence for the reliability study.
CREATE TABLE clock_sync_samples (
  time       TIMESTAMPTZ NOT NULL,
  session_id UUID NOT NULL REFERENCES sessions(session_id),
  load_seq   SMALLINT NOT NULL,
  t0         DOUBLE PRECISION NOT NULL,
  t1         DOUBLE PRECISION NOT NULL,
  t2         DOUBLE PRECISION NOT NULL,
  t3         DOUBLE PRECISION NOT NULL,
  rtt_ms     DOUBLE PRECISION NOT NULL,
  offset_ms  DOUBLE PRECISION NOT NULL,
  accepted   BOOLEAN NOT NULL,

  CONSTRAINT clock_sync_load_seq_chk CHECK (load_seq >= 0),
  CONSTRAINT clock_sync_rtt_chk CHECK (rtt_ms >= 0)
);

-- Write the ledger row and the batch's events in one transaction, so a retried
-- batch is detected here instead of by making event rows unique.
CREATE TABLE ingested_batches (
  session_id  UUID NOT NULL REFERENCES sessions(session_id),
  load_seq    SMALLINT NOT NULL,
  batch_seq   BIGINT NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT ingested_batches_load_seq_chk CHECK (load_seq >= 0),
  CONSTRAINT ingested_batches_batch_seq_chk CHECK (batch_seq >= 0),
  PRIMARY KEY (session_id, load_seq, batch_seq)
);

-- The keys and types inside features are owned by shared_features, not by SQL.
CREATE TABLE feature_vectors (
  feature_vector_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  session_id        UUID        NOT NULL,
  task_id           UUID        NOT NULL,
  window_start      TIMESTAMPTZ NOT NULL,
  window_end        TIMESTAMPTZ NOT NULL,

  -- 'live', 'final' or a degradation label; unrelated to events.run_id.
  run_id             TEXT NOT NULL DEFAULT 'live',
  schema_version     TEXT NOT NULL,
  features           JSONB NOT NULL,

  CONSTRAINT feature_vectors_task_session_fk
    FOREIGN KEY (task_id, session_id)
    REFERENCES tasks(task_id, session_id),
  CONSTRAINT feature_vectors_window_chk
    CHECK (window_end > window_start),
  CONSTRAINT feature_vectors_run_id_chk
    CHECK (btrim(run_id) <> ''),
  CONSTRAINT feature_vectors_schema_version_chk
    CHECK (btrim(schema_version) <> ''),
  CONSTRAINT feature_vectors_features_object_chk
    CHECK (jsonb_typeof(features) = 'object'),

  -- window_end is in the key because the window-size study can start two sizes
  -- at the same instant.
  CONSTRAINT feature_vectors_natural_uk
    UNIQUE (session_id, task_id, window_start, window_end, run_id),

  -- Composite FK target for classifier_window_predictions.
  CONSTRAINT feature_vectors_id_task_uk
    UNIQUE (feature_vector_id, task_id)
);
