-- 002_tables.sql
-- Stable C1 telemetry + shared feature-vector foundation for CodeTrace.
--
-- Freeze this file before participant collection begins. After it has been
-- applied to any shared/pilot database, do not edit it in place; later schema
-- changes must be new numbered migrations.
--
-- Current authoritative behaviour represented here:
--   * one interview session can contain an original task and later probes;
--   * task-scoped coding events and session-scoped environment events share
--     one ordered telemetry stream;
--   * final submitted code is stored once per task;
--   * clock-sync samples and batch-ingest acknowledgements are retained;
--   * feature vectors are validated by the shared Python contract before write;
--   * the normal live extractor emits 30 s windows every 10 s, while the
--     schema also permits alternate window sizes for C1's window-size study.

-- ---------------------------------------------------------------------------
-- sessions
-- One browser interview session. candidate_id is a pseudonymous research ID,
-- not a name/email/student number.
-- ---------------------------------------------------------------------------
CREATE TABLE sessions (
  session_id          UUID PRIMARY KEY,
  candidate_id        UUID NOT NULL,
  language            TEXT NOT NULL,
  started_at          TIMESTAMPTZ NOT NULL,
  ended_at            TIMESTAMPTZ,

  -- final browser->server clock offset retained for audit/reconstruction
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

  -- editor configuration, retained so capture/train/inference parity can be
  -- checked later rather than assumed
  editor_profile      TEXT,
  editor_config_hash  TEXT,

  -- data-quality summary
  ime_detected        BOOLEAN NOT NULL DEFAULT FALSE,
  dropped_event_count INTEGER NOT NULL DEFAULT 0,
  degraded            BOOLEAN NOT NULL DEFAULT FALSE,

  -- TRUE = multiple displays observed; FALSE = one display observed;
  -- NULL = unavailable / not checked. Unknown must not be written as FALSE.
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

-- ---------------------------------------------------------------------------
-- tasks
-- A session contains one original task and may contain follow-up probe tasks.
-- Ground-truth behaviour condition belongs to the task because participants
-- can be instructed differently per task.
-- ---------------------------------------------------------------------------
CREATE TABLE tasks (
  task_id         UUID PRIMARY KEY,
  session_id      UUID NOT NULL REFERENCES sessions(session_id),
  seq             INTEGER NOT NULL,
  depth           INTEGER NOT NULL DEFAULT 0,
  origin          TEXT NOT NULL CHECK (origin IN ('original', 'probe')),
  parent_task_id  UUID REFERENCES tasks(task_id),

  -- Points to the C2 prediction/flag that caused a probe. The FK is added in
  -- 005_classifier_output.sql after classifier_flags exists.
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

  -- Supports composite FKs from small downstream tables so a task_id cannot
  -- accidentally be paired with the wrong session_id.
  UNIQUE (task_id, session_id)
);

-- ---------------------------------------------------------------------------
-- events
-- Unified high-volume telemetry stream. This becomes the only hypertable in
-- 003_hypertables.sql.
--
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
-- ---------------------------------------------------------------------------
CREATE TABLE events (
  -- shared identity / timing -------------------------------------------------
  time          TIMESTAMPTZ      NOT NULL,   -- server-aligned; partition key
  session_id    UUID             NOT NULL,

  -- TASK SCOPING. Two rules, enforced by events_task_scoped_have_task below:
  --   Task-scoped events (kinds 0,1,2,8,9)    -> task_id REQUIRED
  --   Session-scoped events (3-7, 10-14)      -> task_id MAY be NULL
  -- NULL means the event happened outside any task: before the first one
  -- starts, or during the wait while C2 scores and C3 generates a probe.
  -- Those events are recorded but NEVER enter a behavioural window (spec §3.5).
  task_id       UUID,

  load_seq      SMALLINT         NOT NULL,   -- increments per page load
  batch_seq     BIGINT           NOT NULL,   -- delivery batch; resets on reload

  -- EVENT ORDERING. Zero-based position of this event inside its batch,
  -- assigned by the client from the order of the batch's events array, which
  -- is capture order.
  --
  -- batch_seq identifies the DELIVERY BATCH and is shared by every event in
  -- it. seq_in_batch identifies the EVENT within that batch. The two solve
  -- different problems: batch_seq is delivery/idempotency, seq_in_batch is
  -- ordering. Do not use batch_seq to order events.
  --
  -- Within ONE session, (load_seq, batch_seq, seq_in_batch) totally orders
  -- events: load_seq increases per page load, batch_seq is monotonic within a
  -- load_seq (required by the cumulative ack in spec §6.2), and seq_in_batch
  -- is the position within the batch. This is NOT a global ordering key --
  -- session_id is required.
  --
  -- Needed because perf_now can be coarsened by the browser, so two events
  -- can legitimately share a timestamp:
  --     keydown  perf_now = 15342
  --     edit     perf_now = 15342
  -- Timestamp order alone cannot then recover which happened first, and
  -- gap_ms depends on walking typing events in the true order.
  --
  -- WARNING: `time` is NOT monotonic across a load_seq boundary. A reload
  -- produces a new clock offset, so a later event can carry an earlier
  -- wall-clock time. Order by the sequence triple, not by time.
  seq_in_batch  SMALLINT         NOT NULL,
  kind          SMALLINT         NOT NULL CHECK (kind BETWEEN 0 AND 14),

  -- Raw browser monotonic timestamp. Never rewrite this value after capture.
  perf_now      DOUBLE PRECISION NOT NULL,

  -- Milliseconds since the previous typing event (kinds 0-2) within the same
  -- (session_id, load_seq, task_id). NULL on non-typing events and the first
  -- typing event of a segment. Degradation experiments recompute this value
  -- from perturbed raw timings rather than trusting the stored live value.
  gap_ms        DOUBLE PRECISION,

  -- keystroke fields, kinds 0-1 ---------------------------------------------
  key_class     SMALLINT,
  key_code      TEXT,
  modifiers     SMALLINT,
  is_repeat     BOOLEAN,
  is_trusted    BOOLEAN,

  -- edit fields, kind 2 ------------------------------------------------------
  -- origin: 0 typed, 1 pasted, 2 undo, 3 redo, 4 reset,
  --         5 autocomplete_accepted (reserved until that capture profile is
  --         implemented).
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

  -- Paste text is the only in-progress source text retained. Population of
  -- this column remains gated by ethics/consent approval. Typed characters
  -- and deleted text are never stored.
  inserted_text  TEXT,

  -- environment / network / display fields ---------------------------------
  env_state      SMALLINT,
  viewport_w     INTEGER,
  viewport_h     INTEGER,

  -- candidate code-execution fields, kinds 8-9 -----------------------------
  -- This run_id identifies a sandbox execution. It is unrelated to the
  -- feature_vectors.run_id label used for live/degraded extraction variants.
  run_id              UUID,
  exit_status         SMALLINT,
  browser_duration_ms DOUBLE PRECISION,
  sandbox_time_ms     DOUBLE PRECISION,
  sandbox_memory_kb   INTEGER,

  -- clipboard fields, kinds 10-11 -------------------------------------------
  -- Length and source location only; copied/cut text itself is never stored.
  -- ⚠️ Copy and cut REUSE start_line / start_col / end_line / end_col above.
  -- On those events those columns describe the SOURCE RANGE the candidate
  -- selected and copied. They do NOT describe where the content was later
  -- inserted -- a paste elsewhere is a separate kind 2 edit event carrying
  -- its own range. Spec §10.1.
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

-- ---------------------------------------------------------------------------
-- task_submissions
-- Exactly one final submitted source file per task. No periodic snapshots and
-- no typed-then-deleted text are stored here.
-- ---------------------------------------------------------------------------
CREATE TABLE task_submissions (
  task_id      UUID PRIMARY KEY,
  session_id   UUID NOT NULL,
  code_text    TEXT NOT NULL,
  version_id   BIGINT,
  submitted_at TIMESTAMPTZ NOT NULL,

  -- Set when code_text is emptied by the stage-1 retention sweep (spec §17.1).
  --
  -- The row itself is NOT deleted at stage 1, because C2's
  -- classifier_flag_final_locations references it: deleting the submission
  -- would take the flag-to-final-code mapping with it, and that mapping is
  -- derived research data with a longer lifetime than the source code.
  -- Emptying code_text removes the sensitive content while every foreign key
  -- stays satisfied.
  --
  -- This column exists because an empty code_text is otherwise ambiguous --
  -- a candidate may legitimately submit an empty file. The timestamp
  -- distinguishes "redacted on this date" from "submitted empty".
  code_redacted_at TIMESTAMPTZ,

  CONSTRAINT task_submissions_task_session_fk
    FOREIGN KEY (task_id, session_id)
    REFERENCES tasks(task_id, session_id),

  -- Redaction implies empty. The converse does not hold, so this is only
  -- checked in one direction.
  CONSTRAINT task_submissions_redaction_chk
    CHECK (code_redacted_at IS NULL OR code_text = '')
);

-- ---------------------------------------------------------------------------
-- clock_sync_samples
-- Keep both accepted and rejected samples. Rejected samples are still useful
-- evidence about network/capture conditions in C1's reliability study.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- ingested_batches
-- Persistent idempotency ledger for acknowledged WebSocket batches. The
-- ledger row and all events in that batch must be inserted in one DB
-- transaction. A retry of an already committed batch is therefore detected
-- without trying to make individual event rows globally unique.
-- ---------------------------------------------------------------------------
CREATE TABLE ingested_batches (
  session_id  UUID NOT NULL REFERENCES sessions(session_id),
  load_seq    SMALLINT NOT NULL,
  batch_seq   BIGINT NOT NULL,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT ingested_batches_load_seq_chk CHECK (load_seq >= 0),
  CONSTRAINT ingested_batches_batch_seq_chk CHECK (batch_seq >= 0),
  PRIMARY KEY (session_id, load_seq, batch_seq)
);

-- ---------------------------------------------------------------------------
-- feature_vectors
-- Persisted Contract-A payloads. The shared Python/Pandera schema remains the
-- authority for feature keys, types, units, valid ranges, missing-value policy
-- and actionable metadata inside `features`.
--
-- Normal live operation: 30 s windows emitted every 10 s.
-- Research operation: alternate window sizes are allowed. For that reason,
-- window_end is part of the natural uniqueness rule; two different window
-- sizes may legitimately start at the same instant.
--
-- Each row is an independently addressable model input. C2 performs incremental
-- per-window prediction and stores the corresponding prediction in
-- classifier_window_predictions (005_classifier_output.sql).
-- ---------------------------------------------------------------------------
CREATE TABLE feature_vectors (
  feature_vector_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  session_id        UUID        NOT NULL,
  task_id           UUID        NOT NULL,
  window_start      TIMESTAMPTZ NOT NULL,
  window_end        TIMESTAMPTZ NOT NULL,

  -- 'live' for the unmodified vector; another non-empty label for controlled
  -- degradation variants. This value is unrelated to events.run_id.
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

  -- The UUID is the stable cross-component identifier. The natural key still
  -- prevents duplicate extraction of the same task/window/run combination.
  CONSTRAINT feature_vectors_natural_uk
    UNIQUE (session_id, task_id, window_start, window_end, run_id),

  -- Supports the composite FK used by classifier_window_predictions so a
  -- prediction cannot pair a feature vector with the wrong task.
  CONSTRAINT feature_vectors_id_task_uk
    UNIQUE (feature_vector_id, task_id)
);
