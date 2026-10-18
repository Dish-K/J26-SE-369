"""Typed views of the C2 output C4 consumes, plus the Phase 1 AnalysisContext.

Field names and types follow migrations 002 (``feature_vectors``) and 005
(``classifier_window_predictions``, ``classifier_flags``). The intrinsic
validators below restate the CHECK constraints of those migrations so a row
that somehow bypassed the database (fixtures, exports) is rejected the same
way. Cross-record consistency lives in ``validation.py``, not here.
"""

import math
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.explainability.config import C2_CONTRACT_TOLERANCE

ClassName = Literal["no_ai", "ide_ai", "external_ai"]
AICategory = Literal["ide_ai", "external_ai"]
EvidenceKind = Literal[
    "paste",
    "suggestion_accepted",
    "typing_burst",
    "uniform_intervals",
    "no_typing_errors",
]
SegmentBasis = Literal["event", "window_run", "mixed"]

C2_CLASS_NAMES: frozenset[str] = frozenset({"no_ai", "ide_ai", "external_ai"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _validate_probability_vector(value: dict[str, float]) -> dict[str, float]:
    keys = set(value)
    if keys != C2_CLASS_NAMES:
        raise ValueError(
            f"class_probabilities must have exactly the keys {sorted(C2_CLASS_NAMES)}, "
            f"got {sorted(keys)}"
        )
    for name, probability in value.items():
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability for {name!r} must be in [0, 1], got {probability}")
    total = sum(value.values())
    if abs(total - 1.0) > C2_CONTRACT_TOLERANCE:
        raise ValueError(f"class_probabilities must sum to 1, got {total}")
    return value


def _check_confidence_matches(
    probabilities: dict[str, float], category: str, confidence: float
) -> None:
    if abs(probabilities[category] - confidence) > C2_CONTRACT_TOLERANCE:
        raise ValueError(
            f"confidence {confidence} does not equal the probability of "
            f"{category!r} ({probabilities[category]})"
        )


class ClassifierFlag(_Frozen):
    """One row of ``classifier_flags``: a historical AI-associated segment."""

    flag_id: UUID
    task_id: UUID
    model_version: str = Field(min_length=1)
    category: AICategory
    representative_prediction_id: UUID
    class_probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    detected_start_line: int = Field(ge=1)
    detected_end_line: int = Field(ge=1)
    evidence_kinds: tuple[EvidenceKind, ...] = Field(min_length=1)
    segment_basis: SegmentBasis
    window_count: int = Field(ge=1)
    detected_at: datetime
    detected_until: datetime
    created_at: datetime | None = None

    check_probabilities = field_validator("class_probabilities")(_validate_probability_vector)

    @model_validator(mode="after")
    def _check_row(self) -> "ClassifierFlag":
        _check_confidence_matches(self.class_probabilities, self.category, self.confidence)
        if self.detected_end_line < self.detected_start_line:
            raise ValueError("detected_end_line must be >= detected_start_line")
        if self.detected_until < self.detected_at:
            raise ValueError("detected_until must be >= detected_at")
        if self.segment_basis == "event" and self.window_count != 1:
            raise ValueError("an event-basis segment must have window_count = 1")
        return self


class WindowPrediction(_Frozen):
    """One row of ``classifier_window_predictions``."""

    prediction_id: UUID
    feature_vector_id: UUID
    task_id: UUID
    model_version: str = Field(min_length=1)
    predicted_category: ClassName
    class_probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    ai_positive: bool
    predicted_at: datetime | None = None

    check_probabilities = field_validator("class_probabilities")(_validate_probability_vector)

    @model_validator(mode="after")
    def _check_row(self) -> "WindowPrediction":
        _check_confidence_matches(
            self.class_probabilities, self.predicted_category, self.confidence
        )
        if self.ai_positive and self.predicted_category == "no_ai":
            raise ValueError("a no_ai prediction can never be ai_positive")
        return self


class FeatureVector(_Frozen):
    """One row of ``feature_vectors``. ``features`` is the Contract-A payload."""

    feature_vector_id: UUID
    session_id: UUID
    task_id: UUID
    window_start: datetime
    window_end: datetime
    run_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    features: dict[str, Any]

    @model_validator(mode="after")
    def _check_window(self) -> "FeatureVector":
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self


class BackgroundReference(_Frozen):
    """Identifies the versioned XAI background/reference sample.

    The background is a C4-owned resource. ``dataset_version`` is its version;
    ``uri`` locates the sample (CSV with one column per model feature).
    ``compatible_model_version``, when set, restricts it to one C2 model.
    """

    reference_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    split: str | None = None
    sample_count: int | None = Field(default=None, ge=1)
    uri: str | None = None
    compatible_model_version: str | None = None
    training_dataset_version: str | None = None


class ModelArtifactMetadata(_Frozen):
    """Everything in a C2 artifact except the fitted model object itself.

    Mirrors the artifact contract in the C2 implementation plan §13.
    """

    model_version: str = Field(min_length=1)
    model_family: str | None = None
    feature_schema_version: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    class_order: tuple[ClassName, ...]
    classification_threshold: float = Field(ge=0.0, le=1.0)
    training_dataset_version: str | None = None
    generator_version: str | None = None
    training_timestamp: str | None = None
    xai_background_ref: BackgroundReference | None = None

    @field_validator("feature_names")
    @classmethod
    def _unique_feature_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("feature_names must not contain duplicates")
        if any(not name for name in value):
            raise ValueError("feature_names must not contain empty names")
        return value

    @field_validator("class_order")
    @classmethod
    def _complete_class_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(C2_CLASS_NAMES) or set(value) != C2_CLASS_NAMES:
            raise ValueError(
                f"class_order must be a permutation of {sorted(C2_CLASS_NAMES)}, got {list(value)}"
            )
        return value


class ReproductionResult(_Frozen):
    """Stored C2 probabilities next to C4's independent re-run of the model."""

    stored_probabilities: dict[str, float]
    reproduced_probabilities: dict[str, float]
    max_abs_difference: float
    tolerance: float
    stored_category: ClassName
    reproduced_category: ClassName


class AnalysisContext(_Frozen):
    """The validated input every later explanation phase starts from.

    Built only after every provenance, schema and reproduction check passed.
    ``model_input`` holds the exact values, in exact model order, that
    reproduced the stored C2 prediction.
    """

    flag: ClassifierFlag
    representative_prediction: WindowPrediction
    contributing_prediction_ids: tuple[UUID, ...]
    feature_vector: FeatureVector
    artifact: ModelArtifactMetadata
    model_version: str
    model_family: str
    feature_schema_version: str
    target_class: AICategory
    target_class_index: int = Field(ge=0)
    feature_names: tuple[str, ...]
    class_names: tuple[ClassName, ...]
    model_input: dict[str, float | None]
    feature_metadata: dict[str, dict[str, Any]]
    reproduction: ReproductionResult
    background_ref: BackgroundReference | None = None
    background_ref_source: Literal["artifact", "configured_dev"] | None = None

    @model_validator(mode="after")
    def _check_alignment(self) -> "AnalysisContext":
        if self.class_names[self.target_class_index] != self.target_class:
            raise ValueError("target_class_index does not point at target_class")
        if tuple(self.model_input) != self.feature_names:
            raise ValueError("model_input must be keyed in exact feature_names order")
        if self.representative_prediction.prediction_id != self.flag.representative_prediction_id:
            raise ValueError("representative_prediction is not the flag's representative")
        return self


# ---------------------------------------------------------------------------
# Phase 2: SHAP / LIME results
#
# Every result names the exact prediction, feature vector, model version and
# target class it explains, so SHAP and LIME can be checked for alignment
# before any comparison (Phase 3). Feature values are the ORIGINAL Contract-A
# inputs, never preprocessed/scaled values.
# ---------------------------------------------------------------------------

Direction = Literal["toward_target", "away_from_target", "neutral"]
ShapOutputSpace = Literal["raw", "log_odds", "probability"]
ShapStrategyKind = Literal["linear", "tree", "model_agnostic"]


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _check_ranked(names: list[str], ranks: list[int]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("ranks must be exactly 1..n")
    if ranks != sorted(ranks):
        raise ValueError("features must be listed in rank order")


class ShapFeatureContribution(_Frozen):
    name: str = Field(min_length=1)
    value: float | None
    shap_value: float
    absolute_shap_value: float = Field(ge=0.0)
    rank: int = Field(ge=1)
    direction: Direction

    @field_validator("shap_value")
    @classmethod
    def _finite_value(cls, value: float) -> float:
        return _finite(value, "shap_value")


class ShapExplanationResult(_Frozen):
    """Local SHAP explanation of one representative C2 prediction."""

    prediction_id: UUID
    feature_vector_id: UUID
    model_version: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    target_class: AICategory
    target_class_index: int = Field(ge=0)
    target_probability: float = Field(ge=0.0, le=1.0)
    strategy: ShapStrategyKind
    strategy_reason: str
    explainer_type: str = Field(min_length=1)
    output_space: ShapOutputSpace
    output_description: str
    background_id: str
    background_version: str
    base_value: float
    model_output: float
    reconstructed_output: float
    reconstruction_error: float = Field(ge=0.0)
    reconstruction_tolerance: float = Field(gt=0.0)
    features: tuple[ShapFeatureContribution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_result(self) -> "ShapExplanationResult":
        for name in ("base_value", "model_output", "reconstructed_output"):
            _finite(getattr(self, name), name)
        _check_ranked([f.name for f in self.features], [f.rank for f in self.features])
        if self.reconstruction_error > self.reconstruction_tolerance:
            raise ValueError(
                f"reconstruction_error {self.reconstruction_error} exceeds tolerance "
                f"{self.reconstruction_tolerance}"
            )
        return self


class GlobalShapFeatureImportance(_Frozen):
    name: str = Field(min_length=1)
    mean_abs_shap: float = Field(ge=0.0)
    mean_shap: float
    rank: int = Field(ge=1)


class GlobalShapBeeswarm(_Frozen):
    """Per-sample data for a future beeswarm plot (rows = samples)."""

    feature_names: tuple[str, ...]
    feature_values: tuple[tuple[float, ...], ...]
    shap_values: tuple[tuple[float, ...], ...]


class GlobalShapResult(_Frozen):
    """Class-specific SHAP summary over many samples.

    Research/dashboard summary only; never a substitute for the local
    explanation of a representative prediction.
    """

    model_version: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    target_class: ClassName
    target_class_index: int = Field(ge=0)
    strategy: ShapStrategyKind
    explainer_type: str
    output_space: ShapOutputSpace
    background_id: str
    background_version: str
    dataset_reference: str = Field(min_length=1)
    sample_count: int = Field(ge=1)
    max_reconstruction_error: float = Field(ge=0.0)
    feature_importance: tuple[GlobalShapFeatureImportance, ...] = Field(min_length=1)
    beeswarm: GlobalShapBeeswarm | None = None

    @model_validator(mode="after")
    def _check_result(self) -> "GlobalShapResult":
        _check_ranked(
            [f.name for f in self.feature_importance], [f.rank for f in self.feature_importance]
        )
        if self.beeswarm is not None and len(self.beeswarm.shap_values) != self.sample_count:
            raise ValueError("beeswarm rows must equal sample_count")
        return self


class LimeFeatureContribution(_Frozen):
    name: str = Field(min_length=1)
    feature_index: int = Field(ge=0)
    value: float | None
    condition: str
    weight: float
    absolute_weight: float = Field(ge=0.0)
    rank: int = Field(ge=1)
    direction: Direction

    @field_validator("weight")
    @classmethod
    def _finite_weight(cls, value: float) -> float:
        return _finite(value, "weight")


class LimeExplanationResult(_Frozen):
    """One seeded LIME explanation of one representative C2 prediction."""

    prediction_id: UUID
    feature_vector_id: UUID
    model_version: str = Field(min_length=1)
    target_class: AICategory
    target_class_index: int = Field(ge=0)
    target_probability: float = Field(ge=0.0, le=1.0)
    background_id: str
    background_version: str
    seed: int
    num_samples: int = Field(ge=1)
    discretize_continuous: bool
    intercept: float | None = None
    local_prediction: float | None = None
    score: float | None = None
    features: tuple[LimeFeatureContribution, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_result(self) -> "LimeExplanationResult":
        _check_ranked([f.name for f in self.features], [f.rank for f in self.features])
        return self


# ---------------------------------------------------------------------------
# Phase 3: explanation reliability results
#
# Every result carries the provenance of the one prediction it evaluates and
# keeps its components, so no composite hides the evidence behind it. A
# metric that is mathematically undefined for its inputs has
# status="not_computable" and a reason; its score is None, never 0 or 0.5.
# None of these scores is a probability that an explanation is correct.
# ---------------------------------------------------------------------------

MetricStatus = Literal["computed", "not_computable"]
RankingSource = Literal["shap", "lime", "consensus"]


def _unit_or_none(value: float | None, name: str) -> float | None:
    if value is not None and not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _check_status(status: str, score: float | None, reason: str | None, what: str) -> None:
    if status == "computed" and score is None:
        raise ValueError(f"{what}: computed status requires a score")
    if status == "not_computable" and (score is not None or not reason):
        raise ValueError(f"{what}: not_computable requires no score and a reason")


class _Provenance(_Frozen):
    prediction_id: UUID
    feature_vector_id: UUID
    model_version: str = Field(min_length=1)
    target_class: AICategory
    target_class_index: int = Field(ge=0)


class AgreementAtKResult(_Frozen):
    k: int = Field(ge=1)
    shap_top_features: tuple[str, ...]
    lime_top_features: tuple[str, ...]
    intersection: tuple[str, ...]
    overlap_at_k: float
    jaccard: float
    feature_set_agreement: float
    comparable_direction_features: tuple[str, ...]
    same_direction_features: tuple[str, ...]
    different_direction_features: tuple[str, ...]
    direction_status: Literal["computed", "no_common_top_k_features"]
    direction_agreement: float
    status: MetricStatus
    agreement_score: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "AgreementAtKResult":
        for name in (
            "overlap_at_k",
            "jaccard",
            "feature_set_agreement",
            "direction_agreement",
            "agreement_score",
        ):
            _unit_or_none(getattr(self, name), name)
        _check_status(self.status, self.agreement_score, self.reason, f"agreement@{self.k}")
        return self


class AgreementResult(_Provenance):
    """SHAP-LIME agreement (agreement_v1_hierarchical) at every configured K.

    Rank agreement uses the full canonical feature universe and is therefore
    shared by all K; feature-set and direction agreement are per K.
    """

    version: str
    feature_count: int = Field(ge=1)
    spearman_raw: float | None = None
    spearman_normalized: float | None = None
    kendall_raw: float | None = None
    kendall_normalized: float | None = None
    rank_agreement: float | None = None
    rank_status: MetricStatus
    rank_reason: str | None = None
    at_k: tuple[AgreementAtKResult, ...] = Field(min_length=1)
    primary_top_k: int
    status: MetricStatus
    score: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "AgreementResult":
        for name in ("spearman_normalized", "kendall_normalized", "rank_agreement", "score"):
            _unit_or_none(getattr(self, name), name)
        _check_status(self.rank_status, self.rank_agreement, self.rank_reason, "rank agreement")
        _check_status(self.status, self.score, self.reason, "agreement")
        if self.primary_top_k not in {r.k for r in self.at_k}:
            raise ValueError("primary_top_k has no result")
        return self


class LimeFitSummary(_Frozen):
    """LIME local-surrogate fit (its R^2 score) across runs; reported, never used to filter."""

    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    std: float | None = None


class FailedLimeRun(_Frozen):
    seed: int
    error_type: str
    message: str


class StabilityAtKResult(_Frozen):
    k: int = Field(ge=1)
    pair_count: int = Field(ge=0)
    set_stability: float | None = None
    min_pairwise_jaccard: float | None = None
    max_pairwise_jaccard: float | None = None
    std_pairwise_jaccard: float | None = None
    status: MetricStatus
    stability_score: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "StabilityAtKResult":
        for name in (
            "set_stability",
            "min_pairwise_jaccard",
            "max_pairwise_jaccard",
            "stability_score",
        ):
            _unit_or_none(getattr(self, name), name)
        _check_status(self.status, self.stability_score, self.reason, f"stability@{self.k}")
        return self


class StabilityResult(_Provenance):
    """Repeated-seed LIME stability (stability_v1). Keeps every run."""

    version: str
    repetitions_requested: int = Field(ge=2)
    successful_runs: int = Field(ge=0)
    failed_runs: tuple[FailedLimeRun, ...]
    seeds: tuple[int, ...]
    lime_num_samples: int = Field(ge=1)
    lime_discretize_continuous: bool
    background_id: str
    background_version: str
    pair_count: int = Field(ge=0)
    mean_spearman_normalized: float | None = None
    mean_kendall_normalized: float | None = None
    rank_stability: float | None = None
    rank_status: MetricStatus
    rank_reason: str | None = None
    at_k: tuple[StabilityAtKResult, ...] = Field(min_length=1)
    primary_top_k: int
    status: MetricStatus
    score: float | None = None
    reason: str | None = None
    lime_fit: LimeFitSummary
    runs: tuple[LimeExplanationResult, ...]

    @model_validator(mode="after")
    def _check(self) -> "StabilityResult":
        for name in (
            "mean_spearman_normalized",
            "mean_kendall_normalized",
            "rank_stability",
            "score",
        ):
            _unit_or_none(getattr(self, name), name)
        _check_status(self.rank_status, self.rank_stability, self.rank_reason, "rank stability")
        _check_status(self.status, self.score, self.reason, "stability")
        if self.successful_runs + len(self.failed_runs) != self.repetitions_requested:
            raise ValueError("successful + failed runs must equal repetitions requested")
        if len(self.runs) != self.successful_runs or len(self.seeds) != self.repetitions_requested:
            raise ValueError("runs/seeds do not match the run counts")
        return self


class FaithfulnessAtKResult(_Frozen):
    k: int = Field(ge=1)
    top_features: tuple[str, ...]
    baseline_features: tuple[str, ...]
    p_top: float = Field(ge=0.0, le=1.0)
    p_base: float = Field(ge=0.0, le=1.0)
    raw_delta_top: float  # signed p0 - p_top; may be negative
    raw_delta_base: float
    d_top: float = Field(ge=0.0)
    d_base: float = Field(ge=0.0)
    top_probabilities: tuple[float, ...]
    baseline_probabilities: tuple[float, ...]
    status: MetricStatus
    score: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "FaithfulnessAtKResult":
        _unit_or_none(self.score, "faithfulness")
        _check_status(self.status, self.score, self.reason, f"faithfulness@{self.k}")
        if set(self.top_features) & set(self.baseline_features):
            raise ValueError("top and baseline feature sets must be disjoint")
        return self


class FaithfulnessResult(_Provenance):
    """Perturbation faithfulness (faithfulness_v1) in target-class probability space."""

    version: str
    ranking_source: RankingSource
    ranking: tuple[str, ...]
    perturbation_method: str
    perturbation_samples: int = Field(ge=1)
    perturbation_seed: int
    replacement_row_indices: tuple[int, ...]
    background_id: str
    background_version: str
    epsilon: float = Field(gt=0.0)
    min_total_effect: float = Field(ge=0.0)
    p0: float = Field(ge=0.0, le=1.0)
    at_k: tuple[FaithfulnessAtKResult, ...] = Field(min_length=1)
    primary_top_k: int
    status: MetricStatus
    score: float | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "FaithfulnessResult":
        _unit_or_none(self.score, "faithfulness")
        _check_status(self.status, self.score, self.reason, "faithfulness")
        if len(self.replacement_row_indices) != self.perturbation_samples:
            raise ValueError("replacement_row_indices must have perturbation_samples entries")
        return self


class EcsResult(_Provenance):
    """Explanation Confidence Score: a composite reliability INDICATOR under the
    defined agreement/stability/faithfulness framework. Not a probability."""

    version: str
    status: MetricStatus
    agreement_score: float | None = None
    stability_score: float | None = None
    faithfulness_score: float | None = None
    weights: dict[str, float]
    score: float | None = None
    missing_components: tuple[str, ...] = ()
    reason: str | None = None
    agreement_version: str
    agreement_top_k: int
    stability_version: str
    stability_top_k: int
    stability_repetitions: int
    faithfulness_version: str
    faithfulness_top_k: int
    faithfulness_ranking_source: RankingSource

    @model_validator(mode="after")
    def _check(self) -> "EcsResult":
        for name in ("agreement_score", "stability_score", "faithfulness_score", "score"):
            _unit_or_none(getattr(self, name), name)
        _check_status(self.status, self.score, self.reason, "ECS")
        return self


class EcsSensitivityScenario(_Frozen):
    name: str = Field(min_length=1)
    weights: dict[str, float]
    status: MetricStatus
    score: float | None = None

    @model_validator(mode="after")
    def _check(self) -> "EcsSensitivityScenario":
        _unit_or_none(self.score, "scenario score")
        if self.status == "computed" and self.score is None:
            raise ValueError("computed scenario requires a score")
        return self


class EcsSensitivityResult(_Provenance):
    baseline_version: str
    scenarios: tuple[EcsSensitivityScenario, ...] = Field(min_length=1)
