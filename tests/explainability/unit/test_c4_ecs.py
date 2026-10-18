"""Explanation Confidence Score (ecs_v1_equal_weight) and sensitivity."""

import pytest

from backend.app.explainability.config import EcsConfig, EcsWeights
from backend.app.explainability.confidence.ecs import EcsCalculator
from backend.app.explainability.errors import (
    ExplanationAlignmentError,
    InvalidMetricConfigurationError,
)
from tests.explainability.fixtures.explanation_builders import (
    PREDICTION_ID,
    agreement_result,
    faithfulness_result,
    stability_result,
)


def _ecs(a, s, f, config=None):  # type: ignore[no-untyped-def]
    return EcsCalculator(config).compute(agreement_result(a), stability_result(s), faithfulness_result(f))


@pytest.mark.parametrize(("a", "s", "f", "expected"), [(1, 1, 1, 1.0), (0, 0, 0, 0.0)])
def test_extremes(a: float, s: float, f: float, expected: float) -> None:
    assert _ecs(a, s, f).score == pytest.approx(expected)


def test_equal_weight_is_the_arithmetic_mean() -> None:
    result = _ecs(0.9, 0.6, 0.3)
    assert result.score == pytest.approx((0.9 + 0.6 + 0.3) / 3)
    assert result.status == "computed"
    assert 0.0 <= result.score <= 1.0


def test_version_components_and_provenance_are_kept() -> None:
    result = _ecs(0.9, 0.6, 0.3)
    assert result.version == "ecs_v1_equal_weight"
    assert result.weights == {"agreement": 1 / 3, "stability": 1 / 3, "faithfulness": 1 / 3}
    assert (result.agreement_score, result.stability_score, result.faithfulness_score) == (0.9, 0.6, 0.3)
    assert result.prediction_id == PREDICTION_ID
    assert result.model_version == "golden-model-v0"
    assert result.target_class == "external_ai" and result.target_class_index == 0
    assert result.agreement_version == "agreement_v1_hierarchical"
    assert result.faithfulness_ranking_source == "shap"


@pytest.mark.parametrize(
    ("scores", "missing"),
    [((None, 0.6, 0.3), "agreement"), ((0.9, None, 0.3), "stability"), ((0.9, 0.6, None), "faithfulness")],
)
def test_a_non_computable_component_makes_ecs_not_computable(
    scores: tuple, missing: str
) -> None:
    result = _ecs(*scores)
    assert result.status == "not_computable"
    assert result.score is None  # never 0, never renormalized over the others
    assert result.missing_components == (missing,)
    assert missing in (result.reason or "")
    present = {"agreement": result.agreement_score, "stability": result.stability_score,
               "faithfulness": result.faithfulness_score}
    assert present.pop(missing) is None
    assert all(v is not None for v in present.values())  # computed components still visible


@pytest.mark.parametrize(
    "weights", [(-0.1, 0.6, 0.5), (0.33, 0.33, 0.33), (0.5, 0.5, 0.5), (float("nan"), 0.5, 0.5)]
)
def test_invalid_weights_fail(weights: tuple[float, float, float]) -> None:
    with pytest.raises(InvalidMetricConfigurationError):
        EcsWeights(*weights)


def test_custom_weights_are_applied() -> None:
    config = EcsConfig(weights=EcsWeights(0.5, 0.25, 0.25))
    assert _ecs(0.8, 0.4, 0.2, config).score == pytest.approx(0.5 * 0.8 + 0.25 * 0.4 + 0.25 * 0.2)


def test_components_for_different_predictions_are_rejected() -> None:
    with pytest.raises(ExplanationAlignmentError):
        EcsCalculator().compute(
            agreement_result(0.9),
            stability_result(0.6, model_version="other-model"),
            faithfulness_result(0.3),
        )


# --- sensitivity ---------------------------------------------------------------


def test_sensitivity_scenarios_match_hand_calculation() -> None:
    a, s, f = 0.9, 0.6, 0.3
    calculator = EcsCalculator()
    components = (agreement_result(a), stability_result(s), faithfulness_result(f))
    baseline = calculator.compute(*components)
    result = calculator.sensitivity(*components)

    expected = {
        "equal_weight": ((1 / 3, 1 / 3, 1 / 3), (a + s + f) / 3),
        "agreement_emphasis": ((0.50, 0.25, 0.25), 0.50 * a + 0.25 * s + 0.25 * f),
        "stability_emphasis": ((0.25, 0.50, 0.25), 0.25 * a + 0.50 * s + 0.25 * f),
        "faithfulness_emphasis": ((0.25, 0.25, 0.50), 0.25 * a + 0.25 * s + 0.50 * f),
    }
    assert [sc.name for sc in result.scenarios] == list(expected)
    for scenario in result.scenarios:
        weights, score = expected[scenario.name]
        assert tuple(scenario.weights.values()) == pytest.approx(weights)
        assert scenario.score == pytest.approx(score)
    assert result.baseline_version == "ecs_v1_equal_weight"
    assert result.prediction_id == baseline.prediction_id
    assert calculator.compute(*components) == baseline  # baseline untouched


def test_sensitivity_with_missing_component_is_not_computable() -> None:
    result = EcsCalculator().sensitivity(
        agreement_result(0.9), stability_result(None), faithfulness_result(0.3)
    )
    assert all(sc.status == "not_computable" and sc.score is None for sc in result.scenarios)
