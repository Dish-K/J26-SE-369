-- C2 output. The frozen segment and scoring rules are listed in README.md.

CREATE TABLE classifier_window_predictions (
  prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  feature_vector_id UUID NOT NULL,
  task_id           UUID NOT NULL,
  model_version     TEXT NOT NULL,

  predicted_category TEXT NOT NULL
    CHECK (predicted_category IN ('no_ai', 'ide_ai', 'external_ai')),

  class_probabilities JSONB NOT NULL,

  confidence DOUBLE PRECISION NOT NULL,

  -- An AI class below the threshold in the model artifact stays FALSE.
  ai_positive BOOLEAN NOT NULL,

  predicted_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT classifier_window_predictions_feature_task_fk
    FOREIGN KEY (feature_vector_id, task_id)
    REFERENCES feature_vectors(feature_vector_id, task_id),

  CONSTRAINT classifier_window_predictions_model_version_chk
    CHECK (btrim(model_version) <> ''),

  CONSTRAINT classifier_window_predictions_prob_object_chk
    CHECK (jsonb_typeof(class_probabilities) = 'object'),
  CONSTRAINT classifier_window_predictions_prob_keys_chk
    CHECK (class_probabilities ? 'no_ai'
           AND class_probabilities ? 'ide_ai'
           AND class_probabilities ? 'external_ai'),
  CONSTRAINT classifier_window_predictions_prob_types_chk
    CHECK (jsonb_typeof(class_probabilities -> 'no_ai') = 'number'
           AND jsonb_typeof(class_probabilities -> 'ide_ai') = 'number'
           AND jsonb_typeof(class_probabilities -> 'external_ai') = 'number'),
  CONSTRAINT classifier_window_predictions_prob_ranges_chk
    CHECK (
      (class_probabilities ->> 'no_ai')::DOUBLE PRECISION BETWEEN 0.0 AND 1.0
      AND (class_probabilities ->> 'ide_ai')::DOUBLE PRECISION BETWEEN 0.0 AND 1.0
      AND (class_probabilities ->> 'external_ai')::DOUBLE PRECISION BETWEEN 0.0 AND 1.0
    ),
  CONSTRAINT classifier_window_predictions_prob_sum_chk
    CHECK (abs((class_probabilities ->> 'no_ai')::DOUBLE PRECISION
             + (class_probabilities ->> 'ide_ai')::DOUBLE PRECISION
             + (class_probabilities ->> 'external_ai')::DOUBLE PRECISION
             - 1.0) <= 0.000001),
  CONSTRAINT classifier_window_predictions_confidence_chk
    CHECK (confidence BETWEEN 0.0 AND 1.0),
  CONSTRAINT classifier_window_predictions_confidence_matches_class_chk
    CHECK (
      abs(
        confidence -
        CASE predicted_category
          WHEN 'no_ai'       THEN (class_probabilities ->> 'no_ai')::DOUBLE PRECISION
          WHEN 'ide_ai'      THEN (class_probabilities ->> 'ide_ai')::DOUBLE PRECISION
          WHEN 'external_ai' THEN (class_probabilities ->> 'external_ai')::DOUBLE PRECISION
        END
      ) <= 0.000001
    ),
  CONSTRAINT classifier_window_predictions_positive_class_chk
    CHECK (NOT ai_positive OR predicted_category IN ('ide_ai', 'external_ai')),

  CONSTRAINT classifier_window_predictions_feature_model_uk
    UNIQUE (feature_vector_id, model_version),

  -- Composite FK target, so a flag's representative matches its task, model and category.
  CONSTRAINT classifier_window_predictions_rep_identity_uk
    UNIQUE (prediction_id, task_id, model_version, predicted_category)
);

CREATE INDEX classifier_window_predictions_task_idx
  ON classifier_window_predictions (task_id, model_version, predicted_at);

-- Line ranges are historical, from detection time; 006 maps them to the final file.
CREATE TABLE classifier_flags (
  flag_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  task_id       UUID NOT NULL REFERENCES tasks(task_id),
  model_version TEXT NOT NULL,

  category TEXT NOT NULL
    CHECK (category IN ('external_ai', 'ide_ai')),

  -- The highest-confidence member window.
  representative_prediction_id UUID NOT NULL,

  -- Copied from the representative, never averaged; the trigger below enforces it.
  class_probabilities JSONB NOT NULL,
  confidence          DOUBLE PRECISION NOT NULL,

  -- Never rebased as the candidate keeps editing.
  detected_start_line INTEGER NOT NULL,
  detected_end_line   INTEGER NOT NULL,

  -- Descriptive evidence only, never a classification rule.
  evidence_kinds TEXT[] NOT NULL,

  -- event: anchored on an exact edit range; window_run: built from windows; mixed: both.
  segment_basis TEXT NOT NULL
    CHECK (segment_basis IN ('event', 'window_run', 'mixed')),

  window_count SMALLINT NOT NULL DEFAULT 1,

  detected_at    TIMESTAMPTZ NOT NULL,
  detected_until TIMESTAMPTZ NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT classifier_flags_model_version_chk
    CHECK (btrim(model_version) <> ''),
  CONSTRAINT classifier_flags_prob_object_chk
    CHECK (jsonb_typeof(class_probabilities) = 'object'),
  CONSTRAINT classifier_flags_prob_keys_chk
    CHECK (class_probabilities ? 'no_ai'
           AND class_probabilities ? 'ide_ai'
           AND class_probabilities ? 'external_ai'),
  CONSTRAINT classifier_flags_prob_types_chk
    CHECK (jsonb_typeof(class_probabilities -> 'no_ai') = 'number'
           AND jsonb_typeof(class_probabilities -> 'ide_ai') = 'number'
           AND jsonb_typeof(class_probabilities -> 'external_ai') = 'number'),
  CONSTRAINT classifier_flags_prob_ranges_chk
    CHECK (
      (class_probabilities ->> 'no_ai')::DOUBLE PRECISION BETWEEN 0.0 AND 1.0
      AND (class_probabilities ->> 'ide_ai')::DOUBLE PRECISION BETWEEN 0.0 AND 1.0
      AND (class_probabilities ->> 'external_ai')::DOUBLE PRECISION BETWEEN 0.0 AND 1.0
    ),
  CONSTRAINT classifier_flags_prob_sum_chk
    CHECK (abs((class_probabilities ->> 'no_ai')::DOUBLE PRECISION
             + (class_probabilities ->> 'ide_ai')::DOUBLE PRECISION
             + (class_probabilities ->> 'external_ai')::DOUBLE PRECISION
             - 1.0) <= 0.000001),
  CONSTRAINT classifier_flags_confidence_chk
    CHECK (confidence BETWEEN 0.0 AND 1.0),
  CONSTRAINT classifier_flags_confidence_matches_category_chk
    CHECK (
      abs(
        confidence -
        CASE category
          WHEN 'ide_ai'      THEN (class_probabilities ->> 'ide_ai')::DOUBLE PRECISION
          WHEN 'external_ai' THEN (class_probabilities ->> 'external_ai')::DOUBLE PRECISION
        END
      ) <= 0.000001
    ),
  CONSTRAINT classifier_flags_line_range_chk
    CHECK (detected_start_line >= 1 AND detected_end_line >= detected_start_line),
  CONSTRAINT classifier_flags_window_count_chk
    CHECK (window_count >= 1),
  CONSTRAINT classifier_flags_event_basis_single_window_chk
    CHECK (segment_basis <> 'event' OR window_count = 1),
  CONSTRAINT classifier_flags_detected_order_chk
    CHECK (detected_until >= detected_at),
  CONSTRAINT classifier_flags_evidence_nonempty_chk
    CHECK (array_length(evidence_kinds, 1) >= 1),
  CONSTRAINT classifier_flags_evidence_no_nulls_chk
    CHECK (array_position(evidence_kinds, NULL) IS NULL),
  CONSTRAINT classifier_flags_evidence_valid_chk
    CHECK (evidence_kinds <@ ARRAY[
      'paste',
      'suggestion_accepted',
      'typing_burst',
      'uniform_intervals',
      'no_typing_errors'
    ]::TEXT[]),

  CONSTRAINT classifier_flags_representative_fk
    FOREIGN KEY (
      representative_prediction_id,
      task_id,
      model_version,
      category
    ) REFERENCES classifier_window_predictions (
      prediction_id,
      task_id,
      model_version,
      predicted_category
    ),

  -- Composite FK target for 006.
  CONSTRAINT classifier_flags_id_task_uk
    UNIQUE (flag_id, task_id),

  -- Includes detection time: separate episodes may later hit the same lines.
  CONSTRAINT classifier_flags_episode_uk
    UNIQUE (
      task_id,
      model_version,
      category,
      detected_start_line,
      detected_end_line,
      detected_at,
      detected_until
    )
);

CREATE INDEX classifier_flags_task_idx
  ON classifier_flags (task_id, model_version, detected_at);

-- The representative must be AI-positive and the flag must copy its numbers exactly.
CREATE OR REPLACE FUNCTION classifier_validate_flag_representative()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  p classifier_window_predictions%ROWTYPE;
BEGIN
  SELECT *
    INTO p
    FROM classifier_window_predictions
   WHERE prediction_id = NEW.representative_prediction_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'representative prediction % does not exist',
      NEW.representative_prediction_id;
  END IF;

  IF p.task_id <> NEW.task_id
     OR p.model_version <> NEW.model_version
     OR p.predicted_category <> NEW.category THEN
    RAISE EXCEPTION 'representative prediction does not match flag task/model/category';
  END IF;

  IF NOT p.ai_positive THEN
    RAISE EXCEPTION 'representative prediction must be AI-positive';
  END IF;

  IF p.class_probabilities <> NEW.class_probabilities THEN
    RAISE EXCEPTION 'flag class_probabilities must equal representative prediction probabilities';
  END IF;

  IF abs(p.confidence - NEW.confidence) > 0.000001 THEN
    RAISE EXCEPTION 'flag confidence must equal representative prediction confidence';
  END IF;

  RETURN NEW;
END;
$$;

CREATE TRIGGER classifier_flags_validate_representative_trg
BEFORE INSERT OR UPDATE OF
  representative_prediction_id,
  task_id,
  model_version,
  category,
  class_probabilities,
  confidence
ON classifier_flags
FOR EACH ROW
EXECUTE FUNCTION classifier_validate_flag_representative();

-- One prediction can feed several flags when its window has disconnected edit regions.
CREATE TABLE classifier_flag_predictions (
  flag_id       UUID NOT NULL REFERENCES classifier_flags(flag_id) ON DELETE CASCADE,
  prediction_id UUID NOT NULL REFERENCES classifier_window_predictions(prediction_id),

  PRIMARY KEY (flag_id, prediction_id)
);

CREATE INDEX classifier_flag_predictions_prediction_idx
  ON classifier_flag_predictions (prediction_id, flag_id);

CREATE OR REPLACE FUNCTION classifier_validate_flag_prediction_member()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  f classifier_flags%ROWTYPE;
  p classifier_window_predictions%ROWTYPE;
BEGIN
  SELECT * INTO f FROM classifier_flags WHERE flag_id = NEW.flag_id;
  SELECT * INTO p FROM classifier_window_predictions WHERE prediction_id = NEW.prediction_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'prediction % does not exist', NEW.prediction_id;
  END IF;

  IF p.task_id <> f.task_id
     OR p.model_version <> f.model_version
     OR p.predicted_category <> f.category
     OR NOT p.ai_positive THEN
    RAISE EXCEPTION 'flag member prediction must be AI-positive and match flag task/model/category';
  END IF;

  RETURN NEW;
END;
$$;

CREATE TRIGGER classifier_flag_predictions_validate_trg
BEFORE INSERT OR UPDATE
ON classifier_flag_predictions
FOR EACH ROW
EXECUTE FUNCTION classifier_validate_flag_prediction_member();

-- ai_confidence is the max flag confidence; a clean task is no_ai_detected with NULL.
CREATE TABLE classifier_task_scores (
  task_score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  task_id       UUID NOT NULL REFERENCES tasks(task_id),
  model_version TEXT NOT NULL,

  verdict TEXT NOT NULL
    CHECK (verdict IN ('ai_detected', 'no_ai_detected')),

  ai_confidence DOUBLE PRECISION,
  scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT classifier_task_scores_confidence_chk
    CHECK (ai_confidence IS NULL OR ai_confidence BETWEEN 0.0 AND 1.0),
  CONSTRAINT classifier_task_scores_model_version_chk
    CHECK (btrim(model_version) <> ''),
  CONSTRAINT classifier_task_scores_confidence_matches_verdict_chk
    CHECK (
      (verdict = 'ai_detected'    AND ai_confidence IS NOT NULL)
      OR
      (verdict = 'no_ai_detected' AND ai_confidence IS NULL)
    ),

  UNIQUE (task_id, model_version)
);

CREATE OR REPLACE FUNCTION classifier_validate_task_score()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
  flag_count BIGINT;
  max_flag_confidence DOUBLE PRECISION;
BEGIN
  SELECT count(*), max(confidence)
    INTO flag_count, max_flag_confidence
    FROM classifier_flags
   WHERE task_id = NEW.task_id
     AND model_version = NEW.model_version;

  IF NEW.verdict = 'no_ai_detected' THEN
    IF flag_count <> 0 THEN
      RAISE EXCEPTION 'no_ai_detected task score cannot coexist with classifier flags';
    END IF;
  ELSE
    IF flag_count = 0 THEN
      RAISE EXCEPTION 'ai_detected task score requires at least one classifier flag';
    END IF;

    IF abs(NEW.ai_confidence - max_flag_confidence) > 0.000001 THEN
      RAISE EXCEPTION 'ai_confidence must equal maximum segment confidence (%)',
        max_flag_confidence;
    END IF;
  END IF;

  RETURN NEW;
END;
$$;

CREATE TRIGGER classifier_task_scores_validate_trg
BEFORE INSERT OR UPDATE OF task_id, model_version, verdict, ai_confidence
ON classifier_task_scores
FOR EACH ROW
EXECUTE FUNCTION classifier_validate_task_score();

-- SET NULL breaks the tasks <-> classifier_flags loop, which would otherwise block retention.
ALTER TABLE tasks
  ADD CONSTRAINT tasks_parent_flag_fk
  FOREIGN KEY (parent_flag_id)
  REFERENCES classifier_flags(flag_id)
  ON DELETE SET NULL;
