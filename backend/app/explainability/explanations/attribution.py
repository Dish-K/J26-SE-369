"""Shared helpers for turning explainer output into per-feature attributions.

``select_class_attributions`` is the single place that interprets the array
layouts SHAP returns. The layout contract (SHAP >= 0.45, both pinned versions
0.51 and 0.53) is instances x features x outputs, with outputs last. Older
SHAP returned a list with one instances x features array per output; that is
accepted too. Any other layout raises rather than being transposed on a guess.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.app.explainability.config import ATTRIBUTION_NEUTRAL_EPSILON
from backend.app.explainability.errors import ExplanationShapeError


def select_class_attributions(
    values: Any,
    *,
    n_instances: int,
    n_features: int,
    n_classes: int,
    class_index: int,
) -> np.ndarray:
    """Return an ``(n_instances, n_features)`` array for one output class."""
    if not 0 <= class_index < n_classes:
        raise ExplanationShapeError(f"class index {class_index} outside 0..{n_classes - 1}")

    if isinstance(values, (list, tuple)):
        if len(values) != n_classes:
            raise ExplanationShapeError(
                f"per-class attribution list has {len(values)} entries, expected {n_classes}"
            )
        selected = np.asarray(values[class_index], dtype=float)
        if selected.ndim == 1:
            selected = selected.reshape(1, -1)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim != 3:
            raise ExplanationShapeError(
                f"expected instances x features x classes, got shape {array.shape}"
            )
        if array.shape[1:] != (n_features, n_classes):
            raise ExplanationShapeError(
                f"attribution shape {array.shape} does not match "
                f"(instances, {n_features} features, {n_classes} classes)"
            )
        selected = array[:, :, class_index]

    if selected.shape != (n_instances, n_features):
        raise ExplanationShapeError(
            f"selected attributions have shape {selected.shape}, "
            f"expected ({n_instances}, {n_features})"
        )
    if not np.isfinite(selected).all():
        raise ExplanationShapeError("attributions contain non-finite values")
    return selected


def select_class_base_values(
    base_values: Any, *, n_instances: int, n_classes: int, class_index: int
) -> np.ndarray:
    """Return the expected value of one class for each instance, shape ``(n_instances,)``.

    Accepts one value per class (shared by all instances) or an
    ``(n_instances, n_classes)`` array.
    """
    array = np.asarray(base_values, dtype=float)
    if array.shape == (n_classes,):
        selected = np.full(n_instances, array[class_index])
    elif array.shape == (n_instances, n_classes):
        selected = array[:, class_index]
    else:
        raise ExplanationShapeError(
            f"base values shape {array.shape} is neither ({n_classes},) nor "
            f"({n_instances}, {n_classes})"
        )
    if not np.isfinite(selected).all():
        raise ExplanationShapeError("base values contain non-finite values")
    return selected


def direction_of(value: float, epsilon: float = ATTRIBUTION_NEUTRAL_EPSILON) -> str:
    if value > epsilon:
        return "toward_target"
    if value < -epsilon:
        return "away_from_target"
    return "neutral"


@dataclass(frozen=True)
class RankedAttribution:
    name: str
    index: int
    attribution: float
    rank: int
    direction: str

    @property
    def magnitude(self) -> float:
        return abs(self.attribution)


def rank_attributions(
    feature_names: Sequence[str], attributions: Sequence[float]
) -> list[RankedAttribution]:
    """Rank by absolute attribution, descending; ties keep model feature order."""
    if len(feature_names) != len(attributions):
        raise ExplanationShapeError(
            f"{len(attributions)} attributions for {len(feature_names)} features"
        )
    order = sorted(range(len(feature_names)), key=lambda i: (-abs(attributions[i]), i))
    return [
        RankedAttribution(
            name=feature_names[i],
            index=i,
            attribution=float(attributions[i]),
            rank=rank,
            direction=direction_of(float(attributions[i])),
        )
        for rank, i in enumerate(order, start=1)
    ]
