-- 006_classifier_flag_final_locations.sql
-- Submission-time mapping for immutable historical C2 flags.
--
-- classifier_flags keeps the line range exactly as it existed for the live
-- detection episode. After the candidate submits, the application resolves
-- each historical flag against task_submissions and writes exactly one row here.
--
-- status semantics:
--   mapped     -> the same flagged code can be located in the submitted file;
--   deleted    -> the flagged code no longer exists in the submitted file;
--   unmappable -> the system cannot safely determine a final location.
--
-- C3 must use final_start_line/final_end_line only when status='mapped'. It must
-- never treat the immutable historical coordinates as final submitted-code
-- coordinates.

CREATE TABLE classifier_flag_final_locations (
  flag_id UUID PRIMARY KEY,
  task_id UUID NOT NULL,

  status TEXT NOT NULL
    CHECK (status IN ('mapped', 'deleted', 'unmappable')),

  final_start_line INTEGER,
  final_end_line   INTEGER,

  resolved_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT classifier_flag_final_locations_flag_task_fk
    FOREIGN KEY (flag_id, task_id)
    REFERENCES classifier_flags(flag_id, task_id)
    ON DELETE CASCADE,

  -- A final location may only be resolved after a final submission exists.
  CONSTRAINT classifier_flag_final_locations_submission_fk
    FOREIGN KEY (task_id)
    REFERENCES task_submissions(task_id),

  CONSTRAINT classifier_flag_final_locations_shape_chk
    CHECK (
      (status = 'mapped'
       AND final_start_line IS NOT NULL
       AND final_end_line IS NOT NULL
       AND final_start_line >= 1
       AND final_end_line >= final_start_line)
      OR
      (status IN ('deleted', 'unmappable')
       AND final_start_line IS NULL
       AND final_end_line IS NULL)
    )
);

CREATE INDEX classifier_flag_final_locations_task_idx
  ON classifier_flag_final_locations (task_id, status);

-- Safe C3-facing view. Historical coordinates are exposed for audit, while the
-- final mapped coordinates are the only ranges to use for submitted-code fetch.
CREATE VIEW classifier_flags_for_probing AS
SELECT
  f.flag_id,
  f.task_id,
  f.model_version,
  f.category,
  f.confidence,
  f.class_probabilities,
  f.evidence_kinds,
  f.segment_basis,
  f.window_count,
  f.detected_start_line,
  f.detected_end_line,
  f.detected_at,
  f.detected_until,
  l.final_start_line AS start_line,
  l.final_end_line   AS end_line,
  l.resolved_at
FROM classifier_flags AS f
JOIN classifier_flag_final_locations AS l
  ON l.flag_id = f.flag_id
WHERE l.status = 'mapped';
