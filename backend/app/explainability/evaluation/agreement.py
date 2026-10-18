"""SHAP-LIME Agreement (agreement_v1_hierarchical) — a proposed C4 aggregation.

Compares feature SELECTION, RANKING and DIRECTION — never attribution
magnitudes, because SHAP (e.g. logit space) and LIME (local linear surrogate
of probability) are on different scales.

    A_set       = (Overlap@K + Jaccard) / 2           over Top-K sets
    A_rank      = (rho' + tau') / 2                   over all features, |attribution|
    A_direction = same-direction / comparable         over the Top-K intersection
    A           = (A_set + A_rank + A_direction) / 3

Defined v1 convention: if the Top-K sets share no feature, there is nothing
whose direction can agree, so A_direction = 0 with
direction_status = "no_common_top_k_features".
"""

import logging

from backend.app.explainability.config import AgreementConfig
from backend.app.explainability.errors import ExplanationAlignmentError
from backend.app.explainability.evaluation.ranking import (
    check_top_k,
    jaccard,
    overlap_at_k,
    rank_correlation,
    top_k,
)
from backend.app.explainability.schemas import (
    AgreementAtKResult,
    AgreementResult,
    LimeExplanationResult,
    ShapExplanationResult,
)

logger = logging.getLogger(__name__)

_IDENTITY_FIELDS = (
    "prediction_id",
    "feature_vector_id",
    "model_version",
    "target_class",
    "target_class_index",
)


def check_explanation_pair(shap: ShapExplanationResult, lime: LimeExplanationResult) -> None:
    """Both explanations must describe the same prediction and features."""
    for name in _IDENTITY_FIELDS:
        if getattr(shap, name) != getattr(lime, name):
            raise ExplanationAlignmentError(
                f"SHAP and LIME explain different {name}: "
                f"{getattr(shap, name)!r} vs {getattr(lime, name)!r}"
            )
    shap_names = {f.name for f in shap.features}
    lime_names = {f.name for f in lime.features}
    if shap_names != lime_names:
        raise ExplanationAlignmentError(
            f"SHAP and LIME cover different features: {sorted(shap_names ^ lime_names)}"
        )


class AgreementEvaluator:
    def __init__(self, config: AgreementConfig | None = None) -> None:
        self._config = config or AgreementConfig()

    @property
    def config(self) -> AgreementConfig:
        return self._config

    def evaluate(
        self, shap: ShapExplanationResult, lime: LimeExplanationResult
    ) -> AgreementResult:
        check_explanation_pair(shap, lime)
        feature_count = len(shap.features)
        for k in self._config.top_k_values:
            check_top_k(k, feature_count, "agreement")

        shap_ranked = [f.name for f in shap.features]  # already rank order
        lime_ranked = [f.name for f in lime.features]
        shap_direction = {f.name: f.direction for f in shap.features}
        lime_direction = {f.name: f.direction for f in lime.features}

        universe = sorted(shap_direction)  # any fixed alignment of the same names
        shap_abs = {f.name: f.absolute_shap_value for f in shap.features}
        lime_abs = {f.name: f.absolute_weight for f in lime.features}
        correlation = rank_correlation(
            [shap_abs[n] for n in universe], [lime_abs[n] for n in universe]
        )
        rank_agreement = correlation.rank_agreement

        at_k = tuple(
            self._at_k(k, shap_ranked, lime_ranked, shap_direction, lime_direction, rank_agreement,
                       correlation.reason)
            for k in self._config.top_k_values
        )
        primary = next(r for r in at_k if r.k == self._config.primary_top_k)
        result = AgreementResult(
            version=self._config.version,
            prediction_id=shap.prediction_id,
            feature_vector_id=shap.feature_vector_id,
            model_version=shap.model_version,
            target_class=shap.target_class,
            target_class_index=shap.target_class_index,
            feature_count=feature_count,
            spearman_raw=correlation.spearman,
            spearman_normalized=correlation.spearman_normalized,
            kendall_raw=correlation.kendall,
            kendall_normalized=correlation.kendall_normalized,
            rank_agreement=rank_agreement,
            rank_status="computed" if correlation.computable else "not_computable",
            rank_reason=correlation.reason,
            at_k=at_k,
            primary_top_k=self._config.primary_top_k,
            status=primary.status,
            score=primary.agreement_score,
            reason=primary.reason,
        )
        logger.info(
            "Agreement %s for prediction %s (model %s, target %s): K=%d status=%s score=%s",
            result.version,
            result.prediction_id,
            result.model_version,
            result.target_class,
            result.primary_top_k,
            result.status,
            result.score,
        )
        return result

    @staticmethod
    def _at_k(
        k: int,
        shap_ranked: list[str],
        lime_ranked: list[str],
        shap_direction: dict[str, str],
        lime_direction: dict[str, str],
        rank_agreement: float | None,
        rank_reason: str | None,
    ) -> AgreementAtKResult:
        shap_top = top_k(shap_ranked, k)
        lime_top = top_k(lime_ranked, k)
        common = tuple(name for name in shap_top if name in set(lime_top))
        overlap = overlap_at_k(shap_top, lime_top, k)
        jac = jaccard(shap_top, lime_top)
        feature_set = (overlap + jac) / 2

        same = tuple(n for n in common if shap_direction[n] == lime_direction[n])
        different = tuple(n for n in common if shap_direction[n] != lime_direction[n])
        if common:
            direction_status = "computed"
            direction = len(same) / len(common)
        else:
            direction_status = "no_common_top_k_features"
            direction = 0.0

        if rank_agreement is None:
            score, status, reason = None, "not_computable", rank_reason
        else:
            score = (feature_set + rank_agreement + direction) / 3
            status, reason = "computed", None

        return AgreementAtKResult(
            k=k,
            shap_top_features=shap_top,
            lime_top_features=lime_top,
            intersection=common,
            overlap_at_k=overlap,
            jaccard=jac,
            feature_set_agreement=feature_set,
            comparable_direction_features=common,
            same_direction_features=same,
            different_direction_features=different,
            direction_status=direction_status,
            direction_agreement=direction,
            status=status,
            agreement_score=score,
            reason=reason,
        )
