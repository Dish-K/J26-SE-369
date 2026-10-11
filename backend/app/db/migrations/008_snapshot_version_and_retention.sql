-- version_id restarts on every page load, so snapshot and submission identity must
-- include load_seq. Part of the 001-008 sequence, not an optional upgrade.

-- Writers must use ON CONFLICT (task_id, load_seq, version_id) DO NOTHING.
ALTER TABLE code_snapshots
  DROP CONSTRAINT code_snapshots_task_version_uk;

ALTER TABLE code_snapshots
  ADD CONSTRAINT code_snapshots_task_load_version_uk
  UNIQUE (task_id, load_seq, version_id);

-- Codes live in shared_features/registry.yaml (enums.CaptureReason). NULL means
-- unknown, never a guessed 0.
ALTER TABLE code_snapshots
  ADD COLUMN capture_reason SMALLINT;

ALTER TABLE code_snapshots
  ADD CONSTRAINT code_snapshots_capture_reason_chk
  CHECK (capture_reason IS NULL OR capture_reason BETWEEN 0 AND 2);

-- The load_seq of the document actually read at submission. NULL means unknown.
ALTER TABLE task_submissions
  ADD COLUMN load_seq SMALLINT;

ALTER TABLE task_submissions
  ADD CONSTRAINT task_submissions_load_seq_chk
  CHECK (load_seq IS NULL OR load_seq >= 0);

-- No ON DELETE CASCADE on code_snapshots: the retention sweep deletes it before
-- tasks by name (see verify_retention.sql).
