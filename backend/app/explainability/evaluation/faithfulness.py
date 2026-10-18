"""Perturbation Faithfulness (faithfulness_v1) — a proposed C4 transformation.

Asks whether features an explanation ranks as important move the model's
TARGET-CLASS PROBABILITY more than equally many low-ranked features, under
``background_feature_replacement_v1``:

  for each of N background rows (seeded, recorded), copy the explained vector
  and replace only the chosen features with that row's values; score the
  copies through ClassifierAdapter.

    p0     = P(target | x)
    p_top  = mean P(target) after replacing the K top-ranked features
    p_base = mean P(target) after replacing the K bottom-ranked features
             (same K, same background rows)
    D      = max(0, p0 - p_perturbed)        signed deltas are kept too
    F      = D_top / (D_top + D_base + epsilon)

If D_top + D_base <= min_total_effect there is no perturbation evidence and F
is not computable (never reported as 0).

Known limitation: replacing columns independently can break relationships
between derived behavioural features (e.g. paste_ratio vs. edit counts).
Phase 4 adds consistency rules; any change here becomes a new version.
"""

import logging
from collections.abc import Callable, Sequence

import numpy as np

from backend.app.explainability.config import FaithfulnessConfig
from backend.app.explainability.errors import (
    ExplanationAlignmentError,
    InvalidMetricConfigurationError,
)
from backend.app.explainability.explanations.alignment import (
    check_alignment,
    checked_target_probability,
    instance_row,
)
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import (
    AnalysisContext,
    FaithfulnessAtKResult,
    FaithfulnessResult,
    LimeExplanationResult,
    ShapExplanationResult,
)

logger = logging.getLogger(__name__)


def ranking_of(explanation: ShapExplanationResult | LimeExplanationResult) -> tuple[str, ...]:
    """Feature names in the explanation's own rank order (|attribution| desc)."""
    return tuple(f.name for f in explanation.features)


class FaithfulnessEvaluator:
    def __init__(self, config: FaithfulnessConfig | None = None) -> None:
        self._config = config or FaithfulnessConfig()

    @property
    def config(self) -> FaithfulnessConfig:
        return self._config

    def evaluate(
        self,
        context: AnalysisContext,
        adapter: ClassifierAdapter,
        background: XaiBackground,
        ranking: Sequence[str],
        *,
        ranking_source: str,
    ) -> FaithfulnessResult:
        check_alignment(context, adapter, background)
        ranking = tuple(ranking)
        if sorted(ranking) != sorted(context.feature_names) or len(set(ranking)) != len(ranking):
            raise ExplanationAlignmentError(
                "faithfulness ranking must be a permutation of the model features"
            )
        n = len(ranking)
        for k in self._config.top_k_values:
            if k < 1 or 2 * k > n:
                raise InvalidMetricConfigurationError(
                    f"faithfulness K={k}: top-K and bottom-K must be disjoint, so 2K <= {n}"
                )

        data = background.matrix()
        samples = self._config.perturbation_samples
        if samples > data.shape[0]:
            raise InvalidMetricConfigurationError(
                f"perturbation_samples {samples} exceeds the {data.shape[0]} background rows"
            )
        rng = np.random.default_rng(self._config.perturbation_seed)
        rows = np.sort(rng.choice(data.shape[0], size=samples, replace=False))
        replacements = data[rows]

        target = context.target_class_index
        p0 = checked_target_probability(context, adapter, adapter.predict_proba(context.model_input))
        original = instance_row(context, adapter)[0]
        column = {name: i for i, name in enumerate(adapter.get_feature_names())}

        def perturbed_probabilities(features: tuple[str, ...]) -> np.ndarray:
            matrix = np.tile(original, (samples, 1))
            columns = [column[name] for name in features]
            matrix[:, columns] = replacements[:, columns]
            return adapter.predict_proba_batch(matrix)[:, target]

        at_k = tuple(
            self._at_k(k, ranking, p0, perturbed_probabilities) for k in self._config.top_k_values
        )
        primary = next(r for r in at_k if r.k == self._config.primary_top_k)
        prediction = context.representative_prediction
        result = FaithfulnessResult(
            version=self._config.version,
            prediction_id=prediction.prediction_id,
            feature_vector_id=prediction.feature_vector_id,
            model_version=context.model_version,
            target_class=context.target_class,
            target_class_index=target,
            ranking_source=ranking_source,
            ranking=ranking,
            perturbation_method=self._config.perturbation_method,
            perturbation_samples=samples,
            perturbation_seed=self._config.perturbation_seed,
            replacement_row_indices=tuple(int(i) for i in rows),
            background_id=background.background_id,
            background_version=background.background_version,
            epsilon=self._config.epsilon,
            min_total_effect=self._config.min_total_effect,
            p0=p0,
            at_k=at_k,
            primary_top_k=self._config.primary_top_k,
            status=primary.status,
            score=primary.score,
            reason=primary.reason,
        )
        logger.info(
            "Faithfulness %s for prediction %s (ranking %s, K=%d): status=%s score=%s",
            result.version,
            result.prediction_id,
            ranking_source,
            result.primary_top_k,
            result.status,
            result.score,
        )
        return result

    def _at_k(
        self,
        k: int,
        ranking: tuple[str, ...],
        p0: float,
        perturbed_probabilities: Callable[[tuple[str, ...]], np.ndarray],
    ) -> FaithfulnessAtKResult:
        top_features = ranking[:k]
        baseline_features = ranking[-k:]
        top = perturbed_probabilities(top_features)
        base = perturbed_probabilities(baseline_features)
        p_top, p_base = float(top.mean()), float(base.mean())
        raw_top, raw_base = p0 - p_top, p0 - p_base
        d_top, d_base = max(0.0, raw_top), max(0.0, raw_base)

        total = d_top + d_base
        if total <= self._config.min_total_effect:
            score, status = None, "not_computable"
            reason = (
                f"insufficient_probability_effect: D_top + D_base = {total:.3g} "
                f"<= min_total_effect {self._config.min_total_effect:g}"
            )
        else:
            score = d_top / (total + self._config.epsilon)
            status, reason = "computed", None

        return FaithfulnessAtKResult(
            k=k,
            top_features=top_features,
            baseline_features=baseline_features,
            p_top=p_top,
            p_base=p_base,
            raw_delta_top=raw_top,
            raw_delta_base=raw_base,
            d_top=d_top,
            d_base=d_base,
            top_probabilities=tuple(float(p) for p in top),
            baseline_probabilities=tuple(float(p) for p in base),
            status=status,
            score=score,
            reason=reason,
        )
