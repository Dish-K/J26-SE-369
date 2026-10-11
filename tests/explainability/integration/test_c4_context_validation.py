"""Every broken C2 -> C4 link must stop context building with a specific error."""

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.explainability.errors import (
    C2RecordInvalidError,
    FeatureSchemaMismatchError,
    FeatureVectorNotFoundError,
    FeatureVectorProvenanceError,
    FlagNotFoundError,
    MissingFeatureError,
    ModelArtifactNotFoundError,
    ModelVersionMismatchError,
    PredictionNotFoundError,
    ProbabilityMismatchError,
    RepresentativePredictionMismatchError,
)
from backend.app.explainability.feature_registry import FeatureRegistry, parse_feature_registry
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from backend.app.explainability.schemas import BackgroundReference
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import (
    CLASS_ORDER,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    C2Records,
    InMemoryC2Reader,
    _training_frame,
    artifact_dict,
    write_artifact,
)


def _external_ids(records: C2Records) -> tuple[UUID, UUID, UUID]:
    flag_id = records.flag_ids_by_name["external_ai_segment"]
    prediction_id = records.flags[flag_id]["representative_prediction_id"]
    return flag_id, prediction_id, records.predictions[prediction_id]["feature_vector_id"]


def _build(service: AnalysisService, flag_id: UUID) -> Any:
    return asyncio.run(service.build_context(flag_id))


def _service(records: C2Records, artifact_dir: Path, registry: FeatureRegistry, **kw: Any):
    return AnalysisService(InMemoryC2Reader(records), ArtifactLoader(artifact_dir), registry, **kw)


# --- missing records ---------------------------------------------------------


def test_unknown_flag(service: AnalysisService) -> None:
    with pytest.raises(FlagNotFoundError):
        _build(service, uuid4())


def test_missing_representative_prediction(records: C2Records, service: AnalysisService) -> None:
    flag_id, prediction_id, _ = _external_ids(records)
    del records.predictions[prediction_id]
    with pytest.raises(PredictionNotFoundError):
        _build(service, flag_id)


def test_missing_feature_vector(records: C2Records, service: AnalysisService) -> None:
    flag_id, _, vector_id = _external_ids(records)
    del records.feature_vectors[vector_id]
    with pytest.raises(FeatureVectorNotFoundError):
        _build(service, flag_id)


def test_missing_artifact_for_flag_model_version(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, _, _ = _external_ids(records)
    records.flags[flag_id]["model_version"] = "synthetic-prototype-v1"
    for pid in records.members[flag_id]:
        records.predictions[pid]["model_version"] = "synthetic-prototype-v1"
    with pytest.raises(ModelArtifactNotFoundError, match="synthetic-prototype-v1"):
        _build(service, flag_id)


# --- representative provenance ---------------------------------------------


def test_model_version_mismatch(records: C2Records, service: AnalysisService) -> None:
    flag_id, prediction_id, _ = _external_ids(records)
    records.predictions[prediction_id]["model_version"] = "synthetic-prototype-v1"
    with pytest.raises(ModelVersionMismatchError):
        _build(service, flag_id)


def test_flag_prediction_category_mismatch(records: C2Records, service: AnalysisService) -> None:
    flag_id, _, _ = _external_ids(records)
    flag = records.flags[flag_id]
    flag["category"] = "ide_ai"
    flag["confidence"] = flag["class_probabilities"]["ide_ai"]  # keep the row itself valid
    with pytest.raises(RepresentativePredictionMismatchError, match="category"):
        _build(service, flag_id)


def test_flag_probability_vector_mismatch(records: C2Records, service: AnalysisService) -> None:
    flag_id, _, _ = _external_ids(records)
    probabilities = records.flags[flag_id]["class_probabilities"]
    shifted = probabilities["no_ai"] / 2
    probabilities["no_ai"] -= shifted
    probabilities["ide_ai"] += shifted
    with pytest.raises(ProbabilityMismatchError):
        _build(service, flag_id)


def test_flag_confidence_mismatch(records: C2Records, service: AnalysisService) -> None:
    flag_id, prediction_id, _ = _external_ids(records)
    # Move the prediction's mass so its own row stays valid but differs from the flag.
    prediction = records.predictions[prediction_id]
    probabilities = prediction["class_probabilities"]
    probabilities["external_ai"] -= 0.001
    probabilities["no_ai"] += 0.001
    prediction["confidence"] = probabilities["external_ai"]
    with pytest.raises(ProbabilityMismatchError, match="confidence"):
        _build(service, flag_id)


def test_invalid_probability_sum_is_rejected(records: C2Records, service: AnalysisService) -> None:
    flag_id, prediction_id, _ = _external_ids(records)
    records.predictions[prediction_id]["class_probabilities"]["no_ai"] += 0.1
    with pytest.raises(C2RecordInvalidError, match="sum"):
        _build(service, flag_id)


def test_representative_task_mismatch(records: C2Records, service: AnalysisService) -> None:
    flag_id, prediction_id, _ = _external_ids(records)
    records.predictions[prediction_id]["task_id"] = uuid4()
    with pytest.raises(RepresentativePredictionMismatchError, match="task"):
        _build(service, flag_id)


def test_representative_must_be_ai_positive(records: C2Records, service: AnalysisService) -> None:
    flag_id, prediction_id, _ = _external_ids(records)
    records.predictions[prediction_id]["ai_positive"] = False
    with pytest.raises(RepresentativePredictionMismatchError, match="ai_positive"):
        _build(service, flag_id)


def test_representative_must_be_highest_confidence_member(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, representative_id, _ = _external_ids(records)
    weaker = next(p for p in records.members[flag_id] if p != representative_id)
    flag = records.flags[flag_id]
    flag["representative_prediction_id"] = weaker
    flag["class_probabilities"] = dict(records.predictions[weaker]["class_probabilities"])
    flag["confidence"] = records.predictions[weaker]["confidence"]
    with pytest.raises(RepresentativePredictionMismatchError, match="highest"):
        _build(service, flag_id)


def test_representative_must_be_a_recorded_member(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, representative_id, _ = _external_ids(records)
    records.members[flag_id].remove(representative_id)
    records.flags[flag_id]["window_count"] = len(records.members[flag_id])
    with pytest.raises(RepresentativePredictionMismatchError, match="not among"):
        _build(service, flag_id)


def test_window_count_must_match_recorded_members(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, _, _ = _external_ids(records)
    records.flags[flag_id]["window_count"] = 2
    with pytest.raises(RepresentativePredictionMismatchError, match="window_count"):
        _build(service, flag_id)


# --- feature vector provenance -----------------------------------------------


def test_feature_vector_from_another_task(records: C2Records, service: AnalysisService) -> None:
    flag_id, _, vector_id = _external_ids(records)
    records.feature_vectors[vector_id]["task_id"] = uuid4()
    with pytest.raises(FeatureVectorProvenanceError, match="task"):
        _build(service, flag_id)


def test_degraded_feature_vector_is_not_explained(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, _, vector_id = _external_ids(records)
    records.feature_vectors[vector_id]["run_id"] = "drop-0.2"
    with pytest.raises(FeatureVectorProvenanceError, match="run_id"):
        _build(service, flag_id)


# --- schema / features -------------------------------------------------------


def test_feature_vector_schema_version_mismatch(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, _, vector_id = _external_ids(records)
    records.feature_vectors[vector_id]["schema_version"] = "contract-a-v2"
    with pytest.raises(FeatureSchemaMismatchError):
        _build(service, flag_id)


def test_registry_schema_version_mismatch(records: C2Records, artifact_dir: Path) -> None:
    registry = parse_feature_registry(
        {"schema_version": "contract-a-v2", "features": {name: {} for name in FEATURE_NAMES}}
    )
    flag_id, _, _ = _external_ids(records)
    with pytest.raises(FeatureSchemaMismatchError):
        _build(_service(records, artifact_dir, registry), flag_id)


def test_model_feature_missing_from_registry(records: C2Records, artifact_dir: Path) -> None:
    registry = parse_feature_registry(
        {"schema_version": SCHEMA_VERSION, "features": {n: {} for n in FEATURE_NAMES[1:]}}
    )
    flag_id, _, _ = _external_ids(records)
    with pytest.raises(MissingFeatureError, match=FEATURE_NAMES[0]):
        _build(_service(records, artifact_dir, registry), flag_id)


def test_model_feature_missing_from_vector(records: C2Records, service: AnalysisService) -> None:
    flag_id, _, vector_id = _external_ids(records)
    del records.feature_vectors[vector_id]["features"]["paste_ratio"]
    with pytest.raises(MissingFeatureError, match="paste_ratio"):
        _build(service, flag_id)


# --- reproduction ------------------------------------------------------------


def test_stale_feature_vector_does_not_reproduce(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id, _, vector_id = _external_ids(records)
    records.feature_vectors[vector_id]["features"]["pause_ms"] /= 2
    with pytest.raises(ProbabilityMismatchError, match="does not reproduce"):
        _build(service, flag_id)


def test_wrong_class_order_does_not_reproduce(
    records: C2Records, tmp_path: Path, registry: FeatureRegistry
) -> None:
    """An integer-label model whose class_order names columns wrongly is caught."""
    frame, labels = _training_frame()
    encoded = np.array([CLASS_ORDER.index(label) for label in labels])
    model = Pipeline(
        [("scale", StandardScaler()), ("classify", LogisticRegression(max_iter=2000))]
    ).fit(frame, encoded)
    wrong_order = ["external_ai", "ide_ai", "no_ai"]  # true encoding is CLASS_ORDER
    write_artifact(tmp_path, artifact_dict(model, class_order=wrong_order))

    flag_id, _, _ = _external_ids(records)
    with pytest.raises(ProbabilityMismatchError, match="does not reproduce"):
        _build(_service(records, tmp_path, registry), flag_id)


# --- background reference ------------------------------------------------------


def _background(schema_version: str = SCHEMA_VERSION) -> BackgroundReference:
    return BackgroundReference(
        reference_id="prototype-background",
        dataset_version="synthetic-prototype-fixture",
        schema_version=schema_version,
        source="test-fixture",
        split="training",
        sample_count=200,
    )


def test_background_ref_from_artifact_is_preserved(
    records: C2Records, tmp_path: Path, prototype_model: Any, registry: FeatureRegistry
) -> None:
    write_artifact(
        tmp_path, artifact_dict(prototype_model, xai_background_ref=_background().model_dump())
    )
    flag_id, _, _ = _external_ids(records)
    context = _build(_service(records, tmp_path, registry), flag_id)
    assert context.background_ref == _background()
    assert context.background_ref_source == "artifact"


def test_configured_dev_background_is_used_and_labelled(
    records: C2Records, artifact_dir: Path, registry: FeatureRegistry
) -> None:
    flag_id, _, _ = _external_ids(records)
    service = _service(records, artifact_dir, registry, dev_background_ref=_background())
    context = _build(service, flag_id)
    assert context.background_ref_source == "configured_dev"


def test_background_with_other_schema_is_rejected(
    records: C2Records, artifact_dir: Path, registry: FeatureRegistry
) -> None:
    flag_id, _, _ = _external_ids(records)
    service = _service(
        records, artifact_dir, registry, dev_background_ref=_background("contract-a-v2")
    )
    with pytest.raises(FeatureSchemaMismatchError):
        _build(service, flag_id)
