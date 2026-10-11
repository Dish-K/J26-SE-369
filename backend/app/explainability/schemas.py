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
    """Identifies the versioned XAI background/reference sample for Phase 2.

    Phase 1 only records which sample would be used; nothing is loaded.
    """

    reference_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    split: str | None = None
    sample_count: int | None = Field(default=None, ge=1)
    uri: str | None = None


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
