"""Build typed SHAP/LIME/metric results from hand-chosen numbers.

For golden-math tests: the attributions are chosen by the test, so expected
metric values can be calculated by hand rather than by production code.
Feature order of the dict passed in is the canonical (tie-break) order.
"""

from typing import Any
from uuid import UUID

from backend.app.explainability.explanations.attribution import rank_attributions
from backend.app.explainability.schemas import (
    AgreementAtKResult,
    AgreementResult,
    FaithfulnessAtKResult,
    FaithfulnessResult,
    LimeExplanationResult,
    LimeFeatureContribution,
    LimeFitSummary,
    ShapExplanationResult,
    ShapFeatureContribution,
    StabilityAtKResult,
    StabilityResult,
)

PREDICTION_ID = UUID("00000000-0000-0000-0000-0000000000a1")
FEATURE_VECTOR_ID = UUID("00000000-0000-0000-0000-0000000000f1")
MODEL = "golden-model-v0"


def provenance(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "prediction_id": PREDICTION_ID,
        "feature_vector_id": FEATURE_VECTOR_ID,
        "model_version": MODEL,
        "target_class": "external_ai",
        "target_class_index": 0,
    }
    values.update(overrides)
    return values


def shap_result(attributions: dict[str, float], **overrides: Any) -> ShapExplanationResult:
    ranked = rank_attributions(list(attributions), list(attributions.values()))
    total = sum(attributions.values())
    data: dict[str, Any] = {
        **provenance(),
        "model_family": "LogisticRegression",
        "target_probability": 0.9,
        "strategy": "linear",
        "strategy_reason": "golden fixture",
        "explainer_type": "LinearExplainer",
        "output_space": "raw",
        "output_description": "golden fixture",
        "background_id": "bg",
        "background_version": "bg-v0",
        "base_value": 0.0,
        "model_output": total,
        "reconstructed_output": total,
        "reconstruction_error": 0.0,
        "reconstruction_tolerance": 1e-6,
        "features": tuple(
            ShapFeatureContribution(
                name=r.name,
                value=0.0,
                shap_value=r.attribution,
                absolute_shap_value=r.magnitude,
                rank=r.rank,
                direction=r.direction,
            )
            for r in ranked
        ),
    }
    data.update(overrides)
    return ShapExplanationResult(**data)


def lime_result(
    weights: dict[str, float], *, seed: int = 42, score: float | None = 0.5, **overrides: Any
) -> LimeExplanationResult:
    names = list(weights)
    ranked = rank_attributions(names, list(weights.values()))
    data: dict[str, Any] = {
        **provenance(),
        "target_probability": 0.9,
        "background_id": "bg",
        "background_version": "bg-v0",
        "seed": seed,
        "num_samples": 5000,
        "discretize_continuous": True,
        "intercept": 0.1,
        "local_prediction": 0.8,
        "score": score,
        "features": tuple(
            LimeFeatureContribution(
                name=r.name,
                feature_index=r.index,
                value=0.0,
                condition=f"{r.name} > 0.0",
                weight=r.attribution,
                absolute_weight=r.magnitude,
                rank=r.rank,
                direction=r.direction,
            )
            for r in ranked
        ),
    }
    data.update(overrides)
    return LimeExplanationResult(**data)


def _status(score: float | None) -> dict[str, Any]:
    if score is None:
        return {"status": "not_computable", "score": None, "reason": "golden: not computable"}
    return {"status": "computed", "score": score, "reason": None}


def agreement_result(score: float | None, **overrides: Any) -> AgreementResult:
    status = _status(score)
    at_k = AgreementAtKResult(
        k=1,
        shap_top_features=("a",),
        lime_top_features=("a",),
        intersection=("a",),
        overlap_at_k=1.0,
        jaccard=1.0,
        feature_set_agreement=1.0,
        comparable_direction_features=("a",),
        same_direction_features=("a",),
        different_direction_features=(),
        direction_status="computed",
        direction_agreement=1.0,
        status=status["status"],
        agreement_score=status["score"],
        reason=status["reason"],
    )
    data: dict[str, Any] = {
        **provenance(),
        "version": "agreement_v1_hierarchical",
        "feature_count": 2,
        "rank_status": "computed",
        "rank_agreement": 1.0,
        "spearman_raw": 1.0,
        "spearman_normalized": 1.0,
        "kendall_raw": 1.0,
        "kendall_normalized": 1.0,
        "at_k": (at_k,),
        "primary_top_k": 1,
        **status,
    }
    data.update(overrides)
    return AgreementResult(**data)


def stability_result(score: float | None, **overrides: Any) -> StabilityResult:
    status = _status(score)
    runs = (lime_result({"a": 1.0, "b": 0.5}, seed=42), lime_result({"a": 1.0, "b": 0.5}, seed=43))
    data: dict[str, Any] = {
        **provenance(),
        "version": "stability_v1",
        "repetitions_requested": 2,
        "successful_runs": 2,
        "failed_runs": (),
        "seeds": (42, 43),
        "lime_num_samples": 5000,
        "lime_discretize_continuous": True,
        "background_id": "bg",
        "background_version": "bg-v0",
        "pair_count": 1,
        "rank_status": "computed",
        "rank_stability": 1.0,
        "at_k": (
            StabilityAtKResult(
                k=1,
                pair_count=1,
                set_stability=1.0,
                status=status["status"],
                stability_score=status["score"],
                reason=status["reason"],
            ),
        ),
        "primary_top_k": 1,
        "lime_fit": LimeFitSummary(count=2, mean=0.5, median=0.5, minimum=0.5, maximum=0.5, std=0.0),
        "runs": runs,
        **status,
    }
    data.update(overrides)
    return StabilityResult(**data)


def faithfulness_result(score: float | None, **overrides: Any) -> FaithfulnessResult:
    status = _status(score)
    at_k = FaithfulnessAtKResult(
        k=1,
        top_features=("a",),
        baseline_features=("b",),
        p_top=0.5,
        p_base=0.7,
        raw_delta_top=0.3,
        raw_delta_base=0.1,
        d_top=0.3,
        d_base=0.1,
        top_probabilities=(0.5,),
        baseline_probabilities=(0.7,),
        status=status["status"],
        score=status["score"],
        reason=status["reason"],
    )
    data: dict[str, Any] = {
        **provenance(),
        "version": "faithfulness_v1",
        "ranking_source": "shap",
        "ranking": ("a", "b"),
        "perturbation_method": "background_feature_replacement_v1",
        "perturbation_samples": 1,
        "perturbation_seed": 42,
        "replacement_row_indices": (0,),
        "background_id": "bg",
        "background_version": "bg-v0",
        "epsilon": 1e-12,
        "min_total_effect": 1e-3,
        "p0": 0.8,
        "at_k": (at_k,),
        "primary_top_k": 1,
        **status,
    }
    data.update(overrides)
    return FaithfulnessResult(**data)
