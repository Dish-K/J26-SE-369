"""Repeated-seed LIME Stability (stability_v1)."""

from collections.abc import Callable
from typing import Any

import pytest

from backend.app.explainability.config import LimeConfig, StabilityConfig
from backend.app.explainability.errors import (
    ExplanationAlignmentError,
    InvalidMetricConfigurationError,
    LimeExplanationError,
)
from backend.app.explainability.evaluation.stability import StabilityEvaluator
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.explanations.lime_explainer import LimeExplainer
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import AnalysisContext, LimeExplanationResult
from tests.explainability.fixtures.explanation_builders import lime_result
from tests.explainability.fixtures.prototype_c2 import FEATURE_NAMES

DESCENDING = {name: float(len(FEATURE_NAMES) - i) for i, name in enumerate(FEATURE_NAMES)}
ASCENDING = {name: float(i + 1) for i, name in enumerate(FEATURE_NAMES)}


class StubLime:
    """Returns hand-chosen LIME results per seed, with the context's provenance."""

    def __init__(
        self, weights_for: Callable[[int], dict[str, float]], score_for: Callable[[int], float]
    ) -> None:
        self.config = LimeConfig()
        self.calls: list[tuple[Any, Any, Any, int]] = []
        self._weights_for = weights_for
        self._score_for = score_for

    def explain(
        self, context: AnalysisContext, adapter: Any, background: Any, *, seed: int
    ) -> LimeExplanationResult:
        self.calls.append((context, adapter, background, seed))
        prediction = context.representative_prediction
        return lime_result(
            self._weights_for(seed),
            seed=seed,
            score=self._score_for(seed),
            prediction_id=prediction.prediction_id,
            feature_vector_id=prediction.feature_vector_id,
            model_version=context.model_version,
            target_class=context.target_class,
            target_class_index=context.target_class_index,
        )


def _evaluate(stub: Any, context, adapter, background, config=None):  # type: ignore[no-untyped-def]
    return StabilityEvaluator(stub, config).evaluate(context, adapter, background)


def test_twenty_repetitions_give_190_unique_pairs_and_deterministic_seeds(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    stub = StubLime(lambda s: DESCENDING, lambda s: 0.6)
    result = _evaluate(stub, external_context, adapter, background)
    assert result.repetitions_requested == result.successful_runs == len(result.runs) == 20
    assert result.pair_count == 190 == 20 * 19 // 2
    assert result.seeds == tuple(range(42, 62))
    assert len(set(result.seeds)) == 20
    assert [run.seed for run in result.runs] == list(result.seeds)


def test_every_run_uses_the_same_context_adapter_and_background(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    stub = StubLime(lambda s: DESCENDING, lambda s: 0.6)
    _evaluate(stub, external_context, adapter, background)
    assert all(c is external_context and a is adapter and b is background for c, a, b, _ in stub.calls)


def test_identical_runs_are_perfectly_stable(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    result = _evaluate(StubLime(lambda s: DESCENDING, lambda s: 0.6), external_context, adapter,
                       background)
    assert result.rank_stability == pytest.approx(1.0)
    for at_k in result.at_k:
        assert at_k.set_stability == 1.0
        assert at_k.stability_score == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)


def test_reversed_rankings_have_zero_stability(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    """Two runs: identical except fully reversed. Top-3/Top-5 sets of 10 are disjoint."""
    stub = StubLime(lambda s: DESCENDING if s % 2 == 0 else ASCENDING, lambda s: 0.6)
    result = _evaluate(stub, external_context, adapter, background,
                       StabilityConfig(repetitions=2))
    assert result.pair_count == 1
    assert result.mean_spearman_normalized == pytest.approx(0.0)
    assert result.mean_kendall_normalized == pytest.approx(0.0)
    assert result.rank_stability == pytest.approx(0.0)
    assert all(at_k.set_stability == 0.0 for at_k in result.at_k)
    assert result.score == pytest.approx(0.0)


def test_varied_runs_lower_stability_exactly(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    """Three runs, two identical, one with its 5th and 6th features swapped.

    Top-5 sets: identical pair -> Jaccard 1; each pair with the swapped run
    shares 4 of 6 -> 4/6. Mean over 3 pairs = (1 + 4/6 + 4/6) / 3.
    """
    swapped = dict(DESCENDING)
    fifth, sixth = FEATURE_NAMES[4], FEATURE_NAMES[5]
    swapped[fifth], swapped[sixth] = DESCENDING[sixth], DESCENDING[fifth]
    stub = StubLime(lambda s: swapped if s == 44 else DESCENDING, lambda s: 0.6)
    result = _evaluate(stub, external_context, adapter, background,
                       StabilityConfig(repetitions=3))
    top5 = next(r for r in result.at_k if r.k == 5)
    assert top5.set_stability == pytest.approx((1 + 4 / 6 + 4 / 6) / 3)
    assert next(r for r in result.at_k if r.k == 3).set_stability == 1.0
    assert result.rank_stability < 1.0


def test_local_fit_is_kept_for_every_run_including_low_fit(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    stub = StubLime(lambda s: DESCENDING, lambda s: 0.01 if s == 50 else 0.6)
    result = _evaluate(stub, external_context, adapter, background)
    assert [run.score for run in result.runs].count(0.01) == 1
    assert result.lime_fit.count == 20
    assert result.lime_fit.minimum == 0.01
    assert result.lime_fit.maximum == 0.6


def test_a_failed_run_makes_stability_not_computable(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    def weights(seed: int) -> dict[str, float]:
        if seed == 45:
            raise LimeExplanationError("simulated failure")
        return DESCENDING

    result = _evaluate(StubLime(weights, lambda s: 0.6), external_context, adapter, background)
    assert result.status == "not_computable" and result.score is None
    assert result.successful_runs == 19
    assert [f.seed for f in result.failed_runs] == [45]
    assert result.failed_runs[0].error_type == "LimeExplanationError"
    assert "1 of 20" in (result.reason or "")
    assert all(r.status == "not_computable" for r in result.at_k)


def test_a_run_for_another_prediction_is_rejected(
    external_context: AnalysisContext, ide_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground,
) -> None:
    class WrongPrediction(StubLime):
        def explain(self, context, adapter, background, *, seed):  # type: ignore[no-untyped-def]
            return super().explain(ide_context, adapter, background, seed=seed)

    with pytest.raises(ExplanationAlignmentError):
        _evaluate(WrongPrediction(lambda s: DESCENDING, lambda s: 0.6), external_context,
                  adapter, background)


def test_k_larger_than_features_fails(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    config = StabilityConfig(top_k_values=(11,), primary_top_k=11)
    with pytest.raises(InvalidMetricConfigurationError):
        _evaluate(StubLime(lambda s: DESCENDING, lambda s: 0.6), external_context, adapter,
                  background, config)


def test_single_repetition_is_rejected() -> None:
    with pytest.raises(InvalidMetricConfigurationError):
        StabilityConfig(repetitions=1)


# --- real LIME ---------------------------------------------------------------


def test_real_lime_runs_use_the_seed_schedule_and_reproduce(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    config = StabilityConfig(repetitions=4)
    first = StabilityEvaluator(LimeExplainer(), config).evaluate(external_context, adapter, background)
    second = StabilityEvaluator(LimeExplainer(), config).evaluate(external_context, adapter, background)
    assert [run.seed for run in first.runs] == [42, 43, 44, 45]
    assert first == second
    assert first.runs[0] == LimeExplainer().explain(external_context, adapter, background)
    assert first.status == "computed" and 0.0 <= first.score <= 1.0
