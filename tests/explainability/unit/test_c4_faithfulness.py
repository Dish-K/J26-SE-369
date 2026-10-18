"""Perturbation Faithfulness (faithfulness_v1)."""

import copy
from typing import Any

import numpy as np
import pytest

from backend.app.explainability.config import FaithfulnessConfig
from backend.app.explainability.errors import (
    ExplanationAlignmentError,
    InvalidMetricConfigurationError,
)
from backend.app.explainability.evaluation.faithfulness import FaithfulnessEvaluator, ranking_of
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.explanations.shap_explainer import ShapExplainer
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import AnalysisContext

RANKING_FEATURES = ("a", "b", "c", "d")


def _fixed(probabilities: dict[tuple[str, ...], list[float]]):  # type: ignore[no-untyped-def]
    return lambda features: np.array(probabilities[tuple(features)])


# --- golden math (formula only) ----------------------------------------------


def test_golden_faithfulness_value() -> None:
    """p0 = 0.80, p_top = 0.50, p_base = 0.70 -> D_top 0.30, D_base 0.10."""
    evaluator = FaithfulnessEvaluator()
    perturbed = _fixed({("a",): [0.40, 0.60], ("d",): [0.65, 0.75]})
    result = evaluator._at_k(1, RANKING_FEATURES, 0.80, perturbed)
    eps = evaluator.config.epsilon
    assert result.p_top == pytest.approx(0.50)
    assert result.p_base == pytest.approx(0.70)
    assert result.raw_delta_top == pytest.approx(0.30)
    assert result.d_top == pytest.approx(0.30)
    assert result.d_base == pytest.approx(0.10)
    assert result.score == pytest.approx(0.30 / (0.30 + 0.10 + eps))
    assert result.top_probabilities == (0.40, 0.60)


def test_probability_increase_is_clipped_but_kept_signed() -> None:
    perturbed = _fixed({("a",): [0.84], ("d",): [0.50]})
    result = FaithfulnessEvaluator()._at_k(1, RANKING_FEATURES, 0.80, perturbed)
    assert result.raw_delta_top == pytest.approx(-0.04)
    assert result.d_top == 0.0
    assert result.d_base == pytest.approx(0.30)
    assert result.score == pytest.approx(0.0)


def test_negligible_effect_is_not_computable_not_zero() -> None:
    perturbed = _fixed({("a",): [0.7999], ("d",): [0.7998]})
    result = FaithfulnessEvaluator()._at_k(1, RANKING_FEATURES, 0.80, perturbed)
    assert result.status == "not_computable"
    assert result.score is None
    assert result.reason and result.reason.startswith("insufficient_probability_effect")


def test_baseline_uses_bottom_features_with_the_same_cardinality() -> None:
    perturbed = _fixed({("a", "b"): [0.2], ("c", "d"): [0.7]})
    result = FaithfulnessEvaluator()._at_k(2, RANKING_FEATURES, 0.80, perturbed)
    assert result.top_features == ("a", "b")
    assert result.baseline_features == ("c", "d")


# --- protocol on the real prototype model --------------------------------------


@pytest.fixture
def shap_ranking(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> tuple[str, ...]:
    return ranking_of(ShapExplainer().explain(external_context, adapter, background))


def test_p0_is_the_reproduced_target_probability(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...],
) -> None:
    result = FaithfulnessEvaluator().evaluate(
        external_context, adapter, background, shap_ranking, ranking_source="shap"
    )
    assert result.p0 == pytest.approx(
        external_context.reproduction.reproduced_probabilities["external_ai"], abs=1e-12
    )
    assert result.ranking_source == "shap"
    assert result.ranking == shap_ranking
    assert [r.k for r in result.at_k] == [1, 3, 5]
    for at_k in result.at_k:
        assert at_k.top_features == shap_ranking[: at_k.k]
        assert at_k.baseline_features == shap_ranking[-at_k.k :]
        assert len(at_k.top_probabilities) == len(at_k.baseline_probabilities) == 50


def test_same_background_rows_replace_top_and_baseline_through_the_adapter(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[np.ndarray] = []
    original = adapter.predict_proba_batch

    def spy(rows: Any) -> np.ndarray:
        seen.append(np.array(rows, dtype=float))
        return original(rows)

    monkeypatch.setattr(adapter, "predict_proba_batch", spy)
    config = FaithfulnessConfig(top_k_values=(3,), primary_top_k=3)
    result = FaithfulnessEvaluator(config).evaluate(
        external_context, adapter, background, shap_ranking, ranking_source="shap"
    )

    # p0 on the instance, then top and baseline perturbations: all via the adapter
    assert [m.shape[0] for m in seen] == [1, 50, 50]
    names = list(adapter.get_feature_names())
    rows = background.matrix()[list(result.replacement_row_indices)]
    instance = np.array([external_context.model_input[n] for n in names])
    np.testing.assert_array_equal(seen[0][0], instance)
    for matrix, features in ((seen[1], result.at_k[0].top_features),
                             (seen[2], result.at_k[0].baseline_features)):
        for j, name in enumerate(names):
            expected = rows[:, j] if name in features else np.full(len(rows), instance[j])
            np.testing.assert_array_equal(matrix[:, j], expected)
    probabilities = original(seen[1])[:, external_context.target_class_index]
    assert result.at_k[0].p_top == pytest.approx(probabilities.mean())


def test_shap_ranking_is_more_faithful_than_its_reverse(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...],
) -> None:
    evaluator = FaithfulnessEvaluator()
    forward = evaluator.evaluate(external_context, adapter, background, shap_ranking,
                                 ranking_source="shap")
    backward = evaluator.evaluate(external_context, adapter, background, shap_ranking[::-1],
                                  ranking_source="shap")
    forward3 = next(r for r in forward.at_k if r.k == 3)
    backward3 = next(r for r in backward.at_k if r.k == 3)
    assert forward3.d_top > forward3.d_base
    assert forward3.d_top == pytest.approx(backward3.d_base)  # same sets, swapped roles
    assert forward.score > backward.score


def test_result_is_deterministic(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...],
) -> None:
    run = lambda: FaithfulnessEvaluator().evaluate(  # noqa: E731
        external_context, adapter, background, shap_ranking, ranking_source="shap"
    )
    assert run() == run()


def test_degenerate_effect_on_real_model_is_not_computable(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...],
) -> None:
    config = FaithfulnessConfig(min_total_effect=2.0)  # no probability change can exceed 2
    result = FaithfulnessEvaluator(config).evaluate(
        external_context, adapter, background, shap_ranking, ranking_source="shap"
    )
    assert result.status == "not_computable" and result.score is None
    assert all(r.status == "not_computable" for r in result.at_k)


def test_nothing_is_mutated(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...], prototype_model: Any,
) -> None:
    model_input = dict(external_context.model_input)
    frame = background.frame.copy()
    coefficients = copy.deepcopy(prototype_model.named_steps["classify"].coef_)
    FaithfulnessEvaluator().evaluate(
        external_context, adapter, background, shap_ranking, ranking_source="shap"
    )
    assert external_context.model_input == model_input
    assert background.frame.equals(frame)
    np.testing.assert_array_equal(prototype_model.named_steps["classify"].coef_, coefficients)


@pytest.mark.parametrize(
    "bad_ranking",
    [("paste_ratio",) * 10, ("paste_ratio", "pause_ms"), ("not_a_feature",) * 10],
)
def test_ranking_must_be_a_permutation_of_model_features(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    bad_ranking: tuple[str, ...],
) -> None:
    with pytest.raises(ExplanationAlignmentError):
        FaithfulnessEvaluator().evaluate(
            external_context, adapter, background, bad_ranking, ranking_source="shap"
        )


@pytest.mark.parametrize(
    "config",
    [
        FaithfulnessConfig(top_k_values=(6,), primary_top_k=6),  # top-6 and bottom-6 overlap
        FaithfulnessConfig(perturbation_samples=1000),  # more rows than the background has
    ],
)
def test_invalid_configuration_fails(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    shap_ranking: tuple[str, ...], config: FaithfulnessConfig,
) -> None:
    with pytest.raises(InvalidMetricConfigurationError):
        FaithfulnessEvaluator(config).evaluate(
            external_context, adapter, background, shap_ranking, ranking_source="shap"
        )


def test_unknown_ranking_source_is_rejected_by_config() -> None:
    with pytest.raises(InvalidMetricConfigurationError):
        FaithfulnessConfig(ranking_source="gut_feeling")
