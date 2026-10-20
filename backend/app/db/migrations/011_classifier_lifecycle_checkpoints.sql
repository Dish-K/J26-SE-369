-- Classifier restart recovery.
-- Freeze accepted localized observations and edit metadata.
-- Raw source-code text is rejected by the Python checkpoint codec.
-- Purge these recovery inputs when source telemetry expires.
BEGIN;

CREATE TABLE classifier_lifecycle_checkpoints (
    task_id UUID NOT NULL,
    session_id UUID NOT NULL,
    model_version TEXT NOT NULL CHECK (btrim(model_version) <> ''),
    revision BIGINT NOT NULL CHECK (revision >= 1),
    checkpoint JSONB NOT NULL CHECK (jsonb_typeof(checkpoint) = 'object'),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (task_id, model_version),
    FOREIGN KEY (task_id, session_id)
        REFERENCES tasks(task_id, session_id) ON DELETE CASCADE
);

CREATE INDEX classifier_lifecycle_checkpoints_session_idx
    ON classifier_lifecycle_checkpoints(session_id);

COMMIT;
