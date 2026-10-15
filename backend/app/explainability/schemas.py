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
