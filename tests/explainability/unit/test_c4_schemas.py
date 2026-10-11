from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.app.explainability.schemas import (
    AnalysisContext,
    ClassifierFlag,
    FeatureVector,
    ModelArtifactMetadata,
    WindowPrediction,
)
from tests.explainability.fixtures.prototype_c2 import C2Records


def _flag_row(records: C2Records) -> dict[str, Any]:
    return dict(records.flags[records.flag_ids_by_name["external_ai_segment"]])


def _prediction_row(records: C2Records) -> dict[str, Any]:
    flag = _flag_row(records)
    return dict(records.predictions[flag["representative_prediction_id"]])


def test_valid_classifier_flag_parses(records: C2Records) -> None:
    flag = ClassifierFlag.model_validate(_flag_row(records))
    assert flag.category == "external_ai"
    assert flag.window_count == 3


@pytest.mark.parametrize(
    "change",
    [
        {"category": "no_ai"},  # no_ai never produces a flag row
        {"evidence_kinds": []},
        {"evidence_kinds": ["telepathy"]},
        {"confidence": 0.5},  # no longer the category's probability
        {"detected_end_line": 1},
        {"segment_basis": "event"},  # event basis requires window_count = 1
    ],
)
def test_malformed_classifier_flag_fails(records: C2Records, change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ClassifierFlag.model_validate({**_flag_row(records), **change})


def test_valid_window_prediction_parses(records: C2Records) -> None:
    prediction = WindowPrediction.model_validate(_prediction_row(records))
    assert prediction.ai_positive


@pytest.mark.parametrize(
    "probabilities",
    [
        {"no_ai": 0.2, "ide_ai": 0.2, "external_ai": 0.2},  # sum != 1
        {"no_ai": -0.1, "ide_ai": 0.1, "external_ai": 1.0},  # out of range
        {"no_ai": 0.5, "external_ai": 0.5},  # missing class
        {"no_ai": 0.5, "ide_ai": 0.0, "external_ai": 0.5, "other": 0.0},  # extra class
        {"no_ai": float("nan"), "ide_ai": 0.5, "external_ai": 0.5},
    ],
)
def test_malformed_probability_vector_fails(
    records: C2Records, probabilities: dict[str, float]
) -> None:
    with pytest.raises(ValidationError):
        WindowPrediction.model_validate(
            {**_prediction_row(records), "class_probabilities": probabilities}
        )


def test_no_ai_prediction_cannot_be_ai_positive(records: C2Records) -> None:
    row = _prediction_row(records)
    probabilities = {"no_ai": 0.9, "ide_ai": 0.05, "external_ai": 0.05}
    with pytest.raises(ValidationError):
        WindowPrediction.model_validate(
            {
                **row,
                "predicted_category": "no_ai",
                "class_probabilities": probabilities,
                "confidence": 0.9,
                "ai_positive": True,
            }
        )


def test_valid_feature_vector_parses(records: C2Records) -> None:
    row = next(iter(records.feature_vectors.values()))
    vector = FeatureVector.model_validate(row)
    assert vector.run_id == "live"


def test_feature_vector_rejects_inverted_window(records: C2Records) -> None:
    row = dict(next(iter(records.feature_vectors.values())))
    row["window_end"] = row["window_start"]
    with pytest.raises(ValidationError):
        FeatureVector.model_validate(row)


@pytest.mark.parametrize(
    "change",
    [
        {"class_order": ["no_ai", "ide_ai"]},
        {"class_order": ["no_ai", "ide_ai", "ide_ai"]},
        {"feature_names": []},
        {"feature_names": ["a", "a"]},
        {"classification_threshold": 1.5},
    ],
)
def test_malformed_artifact_metadata_fails(change: dict[str, Any]) -> None:
    base = {
        "model_version": "m",
        "feature_schema_version": "s",
        "feature_names": ["a", "b"],
        "class_order": ["no_ai", "ide_ai", "external_ai"],
        "classification_threshold": 0.7,
    }
    with pytest.raises(ValidationError):
        ModelArtifactMetadata.model_validate({**base, **change})


def test_analysis_context_rejects_misaligned_target(records: C2Records) -> None:
    flag = ClassifierFlag.model_validate(_flag_row(records))
    prediction = WindowPrediction.model_validate(_prediction_row(records))
    vector = FeatureVector.model_validate(records.feature_vectors[prediction.feature_vector_id])
    metadata = ModelArtifactMetadata(
        model_version=flag.model_version,
        feature_schema_version=vector.schema_version,
        feature_names=("a",),
        class_order=("no_ai", "ide_ai", "external_ai"),
        classification_threshold=0.7,
    )
    valid = dict(
        flag=flag,
        representative_prediction=prediction,
        contributing_prediction_ids=(prediction.prediction_id,),
        feature_vector=vector,
        artifact=metadata,
        model_version=flag.model_version,
        model_family="LogisticRegression",
        feature_schema_version=vector.schema_version,
        target_class="external_ai",
        target_class_index=2,
        feature_names=("a",),
        class_names=("no_ai", "ide_ai", "external_ai"),
        model_input={"a": 1.0},
        feature_metadata={},
        reproduction={
            "stored_probabilities": prediction.class_probabilities,
            "reproduced_probabilities": prediction.class_probabilities,
            "max_abs_difference": 0.0,
            "tolerance": 1e-6,
            "stored_category": "external_ai",
            "reproduced_category": "external_ai",
        },
    )
    AnalysisContext.model_validate(valid)

    with pytest.raises(ValidationError):
        AnalysisContext.model_validate({**valid, "target_class_index": 0})
    with pytest.raises(ValidationError):
        AnalysisContext.model_validate({**valid, "model_input": {"b": 1.0}})
    with pytest.raises(ValidationError):
        other = prediction.model_copy(update={"prediction_id": uuid4()})
        AnalysisContext.model_validate({**valid, "representative_prediction": other})
