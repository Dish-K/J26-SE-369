"""Checks every explainer runs before explaining an AnalysisContext.

They guarantee SHAP and LIME explain the representative C2 prediction that
Phase 1 validated: same model version, same feature list and order, same
class order, same target class, and the background the context resolved.
"""

import math
from collections.abc import Mapping

import numpy as np

from backend.app.explainability.config import C2_CONTRACT_TOLERANCE
from backend.app.explainability.errors import (
    ExplanationBackgroundMismatchError,
    ExplanationError,
    ModelVersionMismatchError,
    TargetClassNotFoundError,
)
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import AnalysisContext


def check_alignment(
    context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    if adapter.get_model_version() != context.model_version:
        raise ModelVersionMismatchError(
            f"adapter model {adapter.get_model_version()!r} != context model "
            f"{context.model_version!r}"
        )
    if adapter.get_feature_names() != context.feature_names:
        raise ExplanationError("adapter feature order differs from the analysis context")
    if adapter.get_class_names() != context.class_names:
        raise ExplanationError("adapter class order differs from the analysis context")

    prediction = context.representative_prediction
    if not (context.target_class == prediction.predicted_category == context.flag.category):
        raise TargetClassNotFoundError(
            f"target class {context.target_class!r} is not the representative predicted "
            f"category {prediction.predicted_category!r} / flag category {context.flag.category!r}"
        )
    target_index(adapter, context.target_class, expected_index=context.target_class_index)

    if context.background_ref is None:
        raise ExplanationBackgroundMismatchError(
            f"analysis context for flag {context.flag.flag_id} resolved no XAI background"
        )
    if background.reference != context.background_ref:
        raise ExplanationBackgroundMismatchError(
            f"background {background.background_id!r}@{background.background_version!r} is not "
            f"the one resolved for this context ({context.background_ref.reference_id!r}"
            f"@{context.background_ref.dataset_version!r})"
        )
    if background.feature_names != context.feature_names:
        raise ExplanationBackgroundMismatchError("background columns are not the model features")


def target_index(
    adapter: ClassifierAdapter, target_class: str, *, expected_index: int | None = None
) -> int:
    """Column of ``target_class`` in the model's own probability output."""
    class_names = adapter.get_class_names()
    if target_class not in class_names:
        raise TargetClassNotFoundError(
            f"target class {target_class!r} not in model classes {list(class_names)}"
        )
    index = class_names.index(target_class)
    if expected_index is not None and index != expected_index:
        raise TargetClassNotFoundError(
            f"target class {target_class!r} is column {index}, context says {expected_index}"
        )
    return index


def instance_row(context: AnalysisContext, adapter: ClassifierAdapter) -> np.ndarray:
    """The representative feature vector as one raw row in artifact order."""
    return adapter.prepare_features(context.model_input).to_numpy(dtype=float)


def checked_target_probability(
    context: AnalysisContext, adapter: ClassifierAdapter, probabilities: Mapping[str, float]
) -> float:
    """Target probability, which must still equal the Phase 1 reproduction."""
    probability = float(probabilities[context.target_class])
    reproduced = context.reproduction.reproduced_probabilities[context.target_class]
    if not math.isclose(probability, reproduced, rel_tol=0.0, abs_tol=C2_CONTRACT_TOLERANCE):
        raise ExplanationError(
            f"model {adapter.get_model_version()!r} now gives {probability} for "
            f"{context.target_class!r}; Phase 1 reproduced {reproduced} (inference drift)"
        )
    return probability
