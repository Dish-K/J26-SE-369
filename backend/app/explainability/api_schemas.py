"""Response shapes for the C4 HTTP API (PP1 demo endpoints).

They compose the validated Phase 1-3 result schemas rather than re-describing
them, so the API cannot drift from what the pipeline computed.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from backend.app.explainability.schemas import (
    AgreementResult,
    ClassName,
    ClassifierFlag,
    EcsResult,
    EcsSensitivityResult,
    FaithfulnessResult,
    LimeExplanationResult,
    ShapExplanationResult,
    WindowPrediction,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class DataSource(_Frozen):
    """Always shown with demo results; synthetic output must never pass as real."""

    kind: str
    label: str
    note: str


class DemoFlagSummary(_Frozen):
    flag_id: UUID
    title: str
    task_id: UUID
    category: str
    confidence: float
    window_count: int
    model_version: str


class DemoFlagList(_Frozen):
    data_source: DataSource
    flags: tuple[DemoFlagSummary, ...]


class ModelInfo(_Frozen):
    model_version: str
    model_family: str
    feature_schema_version: str
    classification_threshold: float
    class_names: tuple[ClassName, ...]
    target_class: str
    target_class_index: int


class FeatureVectorInfo(_Frozen):
    feature_vector_id: UUID
    session_id: UUID
    task_id: UUID
    window_start: datetime
    window_end: datetime
    run_id: str
    schema_version: str


class FeatureInfo(_Frozen):
    """One model feature with its shared-registry display metadata."""

    name: str
    value: float | None
    label: str | None = None
    unit: str | None = None
    display_as: str | None = None
    description: str | None = None


class TimelineWindow(_Frozen):
    """Every C2 window prediction of the task, not just the flagged ones."""

    prediction_id: UUID
    window_start: datetime
    window_end: datetime
    predicted_category: ClassName
    confidence: float
    ai_positive: bool
    in_segment: bool
    is_representative: bool


class AnalysisResponse(_Frozen):
    data_source: DataSource
    flag: ClassifierFlag
    representative_prediction: WindowPrediction
    feature_vector: FeatureVectorInfo
    model: ModelInfo
    features: tuple[FeatureInfo, ...]
    timeline: tuple[TimelineWindow, ...]
    shap: ShapExplanationResult
    lime: LimeExplanationResult
    agreement: AgreementResult
    # The full StabilityResult keeps all 20 LIME runs; the API sends the
    # summary (scores, seeds, fit statistics) without the per-run detail.
    stability: dict[str, Any]
    faithfulness: FaithfulnessResult
    ecs: EcsResult
    ecs_sensitivity: EcsSensitivityResult
    planned: dict[str, str]
