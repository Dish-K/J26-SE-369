-- C2: preserve the historical document identity of a flag.
-- This alters the existing table created by migration 005.
-- Older flags have unknown source identity; never infer it from a later snapshot.

BEGIN;

ALTER TABLE classifier_flags
  ADD COLUMN load_seq SMALLINT,
  ADD COLUMN version_id BIGINT;

-- Existing flags retain (NULL, NULL). New C2 writes must supply both fields.
ALTER TABLE classifier_flags
  ADD CONSTRAINT classifier_flags_source_context_pair_chk
  CHECK (
    (load_seq IS NULL AND version_id IS NULL)
    OR (load_seq IS NOT NULL AND version_id IS NOT NULL)
  ),
  ADD CONSTRAINT classifier_flags_load_seq_chk
  CHECK (load_seq IS NULL OR load_seq >= 0),
  ADD CONSTRAINT classifier_flags_version_id_chk
  CHECK (version_id IS NULL OR version_id >= 0);

-- No FK to code_snapshots: C1 does not store a snapshot for every edit version.
-- Existing task/prediction FKs, episode uniqueness, and C3 view remain unchanged.

COMMIT;
