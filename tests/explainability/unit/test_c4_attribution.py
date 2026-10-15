"""SHAP output-shape normalization, ranking and direction (no SHAP needed)."""

import numpy as np
import pytest

from backend.app.explainability.errors import ExplanationShapeError
from backend.app.explainability.explanations.attribution import (
    direction_of,
    rank_attributions,
    select_class_attributions,
    select_class_base_values,
)

N, F, C = 2, 4, 3


def _values() -> np.ndarray:
    # value encodes (instance, feature, class) so any axis mix-up is visible
    i, f, c = np.meshgrid(np.arange(N), np.arange(F), np.arange(C), indexing="ij")
    return (100 * i + 10 * f + c).astype(float)


def test_instances_features_classes_layout() -> None:
    selected = select_class_attributions(
        _values(), n_instances=N, n_features=F, n_classes=C, class_index=2
    )
    assert selected.shape == (N, F)
    assert selected[1].tolist() == [102.0, 112.0, 122.0, 132.0]


def test_legacy_per_class_list_layout() -> None:
    per_class = [_values()[:, :, c] for c in range(C)]
    selected = select_class_attributions(
        per_class, n_instances=N, n_features=F, n_classes=C, class_index=1
    )
    assert selected[0].tolist() == [1.0, 11.0, 21.0, 31.0]


def test_single_instance_legacy_vectors() -> None:
    per_class = [np.arange(F, dtype=float) + c for c in range(C)]
    selected = select_class_attributions(
        per_class, n_instances=1, n_features=F, n_classes=C, class_index=2
    )
    assert selected.tolist() == [[2.0, 3.0, 4.0, 5.0]]


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((N, C, F)),  # classes before features: not the contract layout
        np.zeros((N, F)),  # no class axis for a multiclass explanation
        np.zeros((N, F, C + 1)),
        np.zeros((N, F - 1, C)),
        np.zeros((F, C)),
        [np.zeros((N, F))] * (C - 1),
    ],
)
def test_malformed_layouts_fail(values: object) -> None:
    with pytest.raises(ExplanationShapeError):
        select_class_attributions(values, n_instances=N, n_features=F, n_classes=C, class_index=0)


def test_wrong_instance_count_fails() -> None:
    with pytest.raises(ExplanationShapeError):
        select_class_attributions(_values(), n_instances=1, n_features=F, n_classes=C, class_index=0)


def test_non_finite_attributions_fail() -> None:
    values = _values()
    values[0, 0, 0] = np.nan
    with pytest.raises(ExplanationShapeError):
        select_class_attributions(values, n_instances=N, n_features=F, n_classes=C, class_index=0)


def test_class_index_out_of_range_fails() -> None:
    with pytest.raises(ExplanationShapeError):
        select_class_attributions(_values(), n_instances=N, n_features=F, n_classes=C, class_index=3)


def test_base_values_shared_or_per_instance() -> None:
    shared = select_class_base_values([0.1, 0.2, 0.7], n_instances=N, n_classes=C, class_index=2)
    assert shared.tolist() == [0.7, 0.7]
    per_row = select_class_base_values(
        [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], n_instances=N, n_classes=C, class_index=0
    )
    assert per_row.tolist() == [0.1, 0.3]
    with pytest.raises(ExplanationShapeError):
        select_class_base_values(0.5, n_instances=N, n_classes=C, class_index=0)


def test_ranking_uses_absolute_magnitude_with_stable_ties() -> None:
    ranked = rank_attributions(["a", "b", "c", "d"], [0.1, -0.5, 0.3, -0.1])
    assert [(r.name, r.rank) for r in ranked] == [("b", 1), ("c", 2), ("a", 3), ("d", 4)]
    assert ranked[0].magnitude == 0.5


@pytest.mark.parametrize(
    ("value", "direction"),
    [(0.2, "toward_target"), (-0.2, "away_from_target"), (0.0, "neutral"), (1e-15, "neutral")],
)
def test_direction(value: float, direction: str) -> None:
    assert direction_of(value) == direction
