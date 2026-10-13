-- Periodic full-document snapshots, so a live flag can be resolved to code before submission.
-- Most sensitive store in C1: text present at a snapshot is kept even if deleted later.
CREATE TABLE code_snapshots (
  snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  session_id  UUID NOT NULL,
  task_id     UUID NOT NULL,

  -- Matches events.version_id, so a flag resolves to an exact document, not a nearby time.
  version_id  BIGINT NOT NULL,

  -- Same derivation as events.time.
  captured_at TIMESTAMPTZ NOT NULL,

  -- Kept raw for the degradation study, as in events.
  perf_now    DOUBLE PRECISION NOT NULL,

  -- version_id restarts on every page load, so it only means something with load_seq.
  load_seq    SMALLINT NOT NULL,

  code_text   TEXT    NOT NULL,
  doc_len     INTEGER NOT NULL,

  -- Retention stage 1 empties code_text but keeps the row, as task_submissions does.
  code_redacted_at TIMESTAMPTZ,

  CONSTRAINT code_snapshots_task_session_fk
    FOREIGN KEY (task_id, session_id)
    REFERENCES tasks(task_id, session_id),

  CONSTRAINT code_snapshots_version_chk
    CHECK (version_id >= 0),

  CONSTRAINT code_snapshots_doc_len_chk
    CHECK (doc_len >= 0),

  CONSTRAINT code_snapshots_load_seq_chk
    CHECK (load_seq >= 0),

  -- One direction only: a document can genuinely be empty.
  CONSTRAINT code_snapshots_redaction_chk
    CHECK (code_redacted_at IS NULL OR code_text = ''),

  -- The writer inserts ON CONFLICT DO NOTHING, so an unchanged document is stored once.
  -- Replaced in 008: this key collides across a page reload.
  CONSTRAINT code_snapshots_task_version_uk
    UNIQUE (task_id, version_id)
);
