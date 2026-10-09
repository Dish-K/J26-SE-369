from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping
from uuid import UUID

from .artifact import CLASS_ORDER, ModelArtifact


AI_CATEGORIES = frozenset({"ide_ai", "external_ai"})

PROBABILITY_SUM_TOLERANCE = 0.000001

@dataclass(frozen=True)
class WindowPrediction:
    """
    Internal Classifier result for one completed feature-vector window.

    This object maps directly to the fields later persisted in classifier_window_predictions.
    """

    feature_vector_id: UUID
    task_id: UUID
    model_version: str

    predicted_category: str
    class_probabilities: dict[str, float]
    confidence: float
    ai_positive: bool

def _validate_feature_keys(
        features: Mapping[str, Any], 
        expected_feature_names: list[str],
) -> None:
    """
    Ensure runtime features match the feature contract expected by the trained artifact.
    """ 

    expected = set(expected_feature_names)
    received = set(features.keys())

    missing = expected - received
    extra = received - expected

    if missing:
        raise ValueError("Feature vector is missing required features:" f"{sorted(missing)}")

    if extra:
        raise ValueError("Feature vector contains unexpected features: " f"{sorted(extra)}")


def _map_probabilities(
        raw_probabilities: Any,
        model_class_order: list[str],
) -> dict[str, float]:
    """
    Convert the estimator's probability array into the canonical
    Classifier probability object.

    Example:

        model_class_order:
            external_ai, ide_ai, no_ai

        raw probabilities:
            0.10, 0.20, 0.70

        returned mapping:
            no_ai       -> 0.70
            ide_ai      -> 0.20
            external_ai -> 0.10
    """

    probabilities = [float(value) for value in raw_probabilities]

    if len(probabilities) != len(model_class_order):
        raise ValueError("Probability count does not match model class order")

    probability_by_model_class = dict(
        zip(
            model_class_order, 
            probabilities, 
            strict=True
        )
    )

    # Always expose probabilities using the canonical classifier names/order
    probability_by_class = {
        class_name: probability_by_model_class[class_name]
        for class_name in CLASS_ORDER
    }

    for class_name, probability in probability_by_class.items():
        if not isfinite(probability):
            raise ValueError(f"Probability for {class_name} is not finite")

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Probability for {class_name} must be between"
                f"0.0 and 1.0, got {probability}"
            )

    probability_sum = sum(probability_by_class.values())

    if abs(probability_sum - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(
            "Class probabilities must sum to 1.0; "
            f"got {probability_sum}"
        )

    return probability_by_class

def _choose_predicted_category(
        class_probabilities: Mapping[str, float]
) -> str:
    """
    Choose the class with the highest probability.

    Exact ties use the project's canonical tie-break order:

        no_ai -> ide_ai -> external_ai
    """

    maximum_probability = max(class_probabilities.values())

    for class_name in CLASS_ORDER:
        if class_probabilities[class_name] == maximum_probability:
            return class_name

    raise RuntimeError("Unable to determine predicted category")

def predict_window(
        *,
        artifact: ModelArtifact,
        feature_vector_id: UUID,
        task_id: UUID,
        features: Mapping[str, Any],
) -> WindowPrediction:
    """
    Score one completed Telemetry feature vector using one model artifact.
    """

    # Make sure the artifact isself is valid before trusting it.
    artifact.validate()

    # Make sure Telemetry supplied exactly the features this model expects.
    _validate_feature_keys(features, artifact.feature_names,)

    ordered_feature_values = [
        features[feature_name]
        for feature_name in artifact.feature_names
    ]

    # One feature vector = one model input row.
    prediction_matrix = artifact.model.predict_proba([ordered_feature_values])

    if len(prediction_matrix) != 1:
        raise ValueError("Expected predict_proba() to return exactly one raw")

    class_probabilities = _map_probabilities(
        prediction_matrix[0],
        artifact.class_order,
    )

    predict_category = _choose_predicted_category(class_probabilities)

    confidence = class_probabilities[predict_category]

    ai_positive = (
        predict_category in AI_CATEGORIES 
        and confidence >= artifact.threshold
    )

    return WindowPrediction(
        feature_vector_id=feature_vector_id,
        task_id=task_id,
        model_version=artifact.model_version,
        predicted_category=predict_category,
        class_probabilities=class_probabilities,
        confidence=confidence,
        ai_positive=ai_positive,
    )