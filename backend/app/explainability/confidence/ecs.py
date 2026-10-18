"""Explanation Confidence Score (ecs_v1_equal_weight) — a proposed C4 composite.

    ECS = w_A * A + w_S * S + w_F * F,   w >= 0,  sum(w) = 1

Baseline weights are equal (1/3 each) because no evidence yet favours one
dimension; sensitivity scenarios re-weight the same components. ECS is a
composite reliability INDICATOR under this framework — not the probability
that an explanation is correct, and there is no "reliable" threshold.

If any component is not computable, ECS is not computable: the component is
never replaced by 0 or 0.5 and the remaining weights are never renormalized.
"""

import logging

from backend.app.explainability.config import EcsConfig, EcsWeights
from backend.app.explainability.errors import ExplanationAlignmentError
from backend.app.explainability.schemas import (
    AgreementResult,
    EcsResult,
    EcsSensitivityResult,
    EcsSensitivityScenario,
    FaithfulnessResult,
    StabilityResult,
)

logger = logging.getLogger(__name__)

_IDENTITY_FIELDS = (
    "prediction_id",
    "feature_vector_id",
    "model_version",
    "target_class",
    "target_class_index",
)


def _check_same_prediction(
    agreement: AgreementResult, stability: StabilityResult, faithfulness: FaithfulnessResult
) -> None:
    for name in _IDENTITY_FIELDS:
        values = {getattr(r, name) for r in (agreement, stability, faithfulness)}
        if len(values) != 1:
            raise ExplanationAlignmentError(
                f"ECS components describe different {name}: {sorted(map(str, values))}"
            )


def _weighted(scores: tuple[float, float, float], weights: EcsWeights) -> float:
    a, s, f = scores
    value = weights.agreement * a + weights.stability * s + weights.faithfulness * f
    return min(1.0, max(0.0, value))  # absorbs float noise from weights summing to 1


class EcsCalculator:
    def __init__(self, config: EcsConfig | None = None) -> None:
        self._config = config or EcsConfig()

    @property
    def config(self) -> EcsConfig:
        return self._config

    @staticmethod
    def _components(
        agreement: AgreementResult, stability: StabilityResult, faithfulness: FaithfulnessResult
    ) -> tuple[tuple[float, float, float] | None, tuple[str, ...]]:
        named = (("agreement", agreement), ("stability", stability), ("faithfulness", faithfulness))
        missing = tuple(name for name, r in named if r.status != "computed" or r.score is None)
        if missing:
            return None, missing
        return (agreement.score, stability.score, faithfulness.score), ()  # type: ignore[return-value]

    def compute(
        self,
        agreement: AgreementResult,
        stability: StabilityResult,
        faithfulness: FaithfulnessResult,
    ) -> EcsResult:
        _check_same_prediction(agreement, stability, faithfulness)
        scores, missing = self._components(agreement, stability, faithfulness)
        if scores is None:
            score, status = None, "not_computable"
            reason = f"ECS components not computable: {list(missing)}"
        else:
            score, status, reason = _weighted(scores, self._config.weights), "computed", None

        result = EcsResult(
            version=self._config.version,
            status=status,
            prediction_id=agreement.prediction_id,
            feature_vector_id=agreement.feature_vector_id,
            model_version=agreement.model_version,
            target_class=agreement.target_class,
            target_class_index=agreement.target_class_index,
            agreement_score=agreement.score,
            stability_score=stability.score,
            faithfulness_score=faithfulness.score,
            weights=self._config.weights.as_dict(),
            score=score,
            missing_components=missing,
            reason=reason,
            agreement_version=agreement.version,
            agreement_top_k=agreement.primary_top_k,
            stability_version=stability.version,
            stability_top_k=stability.primary_top_k,
            stability_repetitions=stability.repetitions_requested,
            faithfulness_version=faithfulness.version,
            faithfulness_top_k=faithfulness.primary_top_k,
            faithfulness_ranking_source=faithfulness.ranking_source,
        )
        logger.info(
            "ECS %s for prediction %s: status=%s score=%s",
            result.version,
            result.prediction_id,
            result.status,
            result.score,
        )
        return result

    def sensitivity(
        self,
        agreement: AgreementResult,
        stability: StabilityResult,
        faithfulness: FaithfulnessResult,
    ) -> EcsSensitivityResult:
        """Every configured weighting scenario; none is preferred or selected."""
        _check_same_prediction(agreement, stability, faithfulness)
        scores, _ = self._components(agreement, stability, faithfulness)
        scenarios = tuple(
            EcsSensitivityScenario(
                name=name,
                weights=weights.as_dict(),
                status="computed" if scores is not None else "not_computable",
                score=None if scores is None else _weighted(scores, weights),
            )
            for name, weights in self._config.sensitivity_scenarios
        )
        return EcsSensitivityResult(
            baseline_version=self._config.version,
            prediction_id=agreement.prediction_id,
            feature_vector_id=agreement.feature_vector_id,
            model_version=agreement.model_version,
            target_class=agreement.target_class,
            target_class_index=agreement.target_class_index,
            scenarios=scenarios,
        )
