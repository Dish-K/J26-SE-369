-- 005_classifier_output.sql
-- Durable Behavioral Classifier (C2) output and prediction audit trail.
-- Consumed downstream by Adaptive Probing (C3) and Explainability (C4).
--
-- Runtime decisions frozen for database-contract-v1:
--   * C1 emits 30 s feature windows every 10 s in normal live operation.
--   * C2 predicts one of no_ai / ide_ai / external_ai for each completed
--     feature-vector row and stores the full three-class probability vector.
--   * An AI category becomes AI-positive only when its predicted-category
--     probability meets the model-version threshold. The threshold itself
--     lives in the versioned model artifact, not in this database.
--   * Evidence such as paste, accepted suggestions, typing bursts, uniform
--     timing and low correction activity is descriptive evidence only. It must
--     never be used as a deterministic class rule.
--   * AI-positive observations are localized to one or more code regions.
--     Disconnected edit ranges inside one window stay separate.
--   * A flagged segment is a maximal connected run of AI-positive observations
--     that share task, model version and AI category, are consecutive in the
--     10 s window stream, and whose localized ranges overlap or directly touch.
--   * A no_ai/below-threshold window, category change, task change,
--     model-version change, non-consecutive window, or separated code range
--     breaks the segment.
--   * A segment formed from several windows inherits confidence and the full
--     probability vector from the highest-confidence member window for the
--     segment's category. Probabilities are not averaged across windows.
--   * classifier_flags stores immutable HISTORICAL detection coordinates.
--     Final submitted-code coordinates are resolved separately in migration 006.
--   * At submission, answer-level AI confidence is the maximum confidence among
--     that task/model's AI-associated flags. If there are no flags, the verdict
--     is no_ai_detected and ai_confidence is NULL.

-- ---------------------------------------------------------------------------
-- classifier_window_predictions
-- One durable prediction for one feature vector under one model version.
-- This table is the audit bridge between Contract-A input and final segments.
-- ---------------------------------------------------------------------------
CREATE TABLE classifier_window_predictions (
  prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

  feature_vector_id UUID NOT NULL,
  task_id           UUID NOT NULL,
  model_version     TEXT NOT NULL,

  predicted_category TEXT NOT NULL
    CHECK (predicted_category IN ('no_ai', 'ide_ai', 'external_ai')),

  class_probabilities JSONB NOT NULL,

  -- Always equals the probability of predicted_category.
  confidence DOUBLE PRECISION NOT NULL,

  -- TRUE only when the winning class is ide_ai/external_ai AND its confidence
  -- meets the threshold stored in the referenced model artifact. An AI class
  -- may therefore have ai_positive=FALSE when it is below threshold.
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

  -- The same feature input may be re-scored by a later model version, but not
  -- twice by the same model version.
  CONSTRAINT classifier_window_predictions_feature_model_uk
    UNIQUE (feature_vector_id, model_version),

  -- Composite identity used by classifier_flags so the representative
  -- prediction is guaranteed to match the flag's task/model/category.
  CONSTRAINT classifier_window_predictions_rep_identity_uk
    UNIQUE (prediction_id, task_id, model_version, predicted_category)
);

CREATE INDEX classifier_window_predictions_task_idx
  ON classifier_window_predictions (task_id, model_version, predicted_at);

-- ---------------------------------------------------------------------------
-- SEGMENT FORMATION / LOCALIZATION CONTRACT
--
-- Localization happens only AFTER the trained model has produced an AI-positive
-- prediction. An edit event never creates an AI class by itself.
--
-- For each AI-positive window, C2 uses the underlying edit-event ranges where
-- available. Ranges inside one window are combined only when they overlap or
-- directly touch. Disconnected regions are independent inputs to the segment
-- builder. min_line/max_line style window summaries are fallback localization,
-- not permission to invent a single span across scattered edits.
--
-- Two localized observations connect into one segment only when ALL hold:
--   1. same task_id
--   2. same model_version
--   3. same AI category
--   4. consecutive in the 10-second window stream
--   5. ranges overlap or directly touch (zero-line gap)
--
-- Range connectivity for inclusive line ranges A and B:
--   A.start <= B.end + 1 AND B.start <= A.end + 1
--
-- A no_ai/below-threshold window breaks temporal continuity. A later AI-positive
-- observation is a new episode even when it modifies the same lines.
-- Evidence kind is NOT part of segment identity.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- classifier_flags
-- One row per independently identified suspicious code segment.
-- Coordinates here are immutable historical coordinates from the detection
-- episode, not final-submission coordinates. Migration 006 resolves each flag
-- against task_submissions for C3.
-- ---------------------------------------------------------------------------
CREATE TABLE classifier_flags (
  flag_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  task_id       UUID NOT NULL REFERENCES tasks(task_id),
  model_version TEXT NOT NULL,

  category TEXT NOT NULL
    CHECK (category IN ('external_ai', 'ide_ai')),

  -- Highest-confidence member window for this segment. The composite FK below
  -- guarantees same task, model version and category.
  representative_prediction_id UUID NOT NULL,

  -- Copied from the representative prediction. The validation trigger below
  -- prevents these values from drifting from that source prediction.
  class_probabilities JSONB NOT NULL,
  confidence          DOUBLE PRECISION NOT NULL,

  -- Historical line range at detection/finalization time. Do not silently
  -- rebase these values as the candidate later edits the document.
  detected_start_line INTEGER NOT NULL,
  detected_end_line   INTEGER NOT NULL,

  -- Descriptive evidence associated with the learned prediction. This field is
  -- never a deterministic classifier rule. Treat as a set; order is irrelevant.
  evidence_kinds TEXT[] NOT NULL,

  -- How the historical range was localized:
  --   event      -> exact edit-event range provided the localization anchor;
  --   window_run -> built from one/more AI-positive windows using exact edit
  --                 ranges where available, with a window summary only as a
  --                 documented fallback;
  --   mixed      -> both event-anchored and window-derived evidence contributed.
  segment_basis TEXT NOT NULL
    CHECK (segment_basis IN ('event', 'window_run', 'mixed')),

  -- Number of AI-positive window predictions treated as contributors to this
  -- segment. For an event-basis segment this remains 1 by contract.
  window_count SMALLINT NOT NULL DEFAULT 1,

  -- Time span of the historical detection episode.
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

  -- Supports migration 006's composite FK and prevents task mismatches there.
  CONSTRAINT classifier_flags_id_task_uk
    UNIQUE (flag_id, task_id),

  -- Duplicate protection includes detection time so two genuinely separate
  -- episodes may affect exactly the same lines later in the same task.
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

-- Validate that the representative prediction is AI-positive and that the
-- copied probability vector/confidence exactly come from that prediction.
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

-- ---------------------------------------------------------------------------
-- classifier_flag_predictions
-- Provenance membership: which AI-positive window predictions contributed to
-- each final historical segment. One prediction may contribute to multiple
-- segments when one positive window localizes to disconnected code regions.
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- classifier_task_scores
-- One result per task/model version, produced at submission.
-- ai_confidence = MAX(confidence) across that task/model's classifier_flags.
-- A clean answer has no flags, verdict=no_ai_detected and ai_confidence=NULL.
-- ---------------------------------------------------------------------------
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

-- A probe task is generated from one specific historical classifier flag.
--
-- ON DELETE SET NULL breaks a dependency loop: tasks.parent_flag_id points at
-- classifier_flags, and classifier_flags.task_id points back at tasks. Without
-- it, neither table can be deleted first and the stage-2 retention sweep
-- (spec §17.1) deadlocks -- the whole transaction rolls back and nothing is
-- deleted at all.
--
-- parent_flag_id is already nullable, since an original task has no parent
-- flag, so clearing it loses no information that survives the deletion anyway.
ALTER TABLE tasks
  ADD CONSTRAINT tasks_parent_flag_fk
  FOREIGN KEY (parent_flag_id)
  REFERENCES classifier_flags(flag_id)
  ON DELETE SET NULL;

-- ZERO FLAGS IS A valid result when classifier_task_scores says no_ai_detected.
