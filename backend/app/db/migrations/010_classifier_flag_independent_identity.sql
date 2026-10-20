-- Identity-based uniqueness for independently formed classifier flags.
-- Prerequisites: migrations 001-009, including source document context in 009.
-- Every new flag is persisted with a stable flag_id from flag_identity.py.
-- Episode attributes do not uniquely identify a flag: independent segments can
-- share the task, model, category, line range, and detection timestamps.
-- The flag_id primary key remains the database's duplicate-save guard.
-- Existing FKs, identity checks, prediction memberships, and C3 view stay intact.

BEGIN;

ALTER TABLE classifier_flags
    DROP CONSTRAINT classifier_flags_episode_uk;

COMMIT;