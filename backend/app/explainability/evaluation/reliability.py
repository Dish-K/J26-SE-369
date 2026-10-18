"""Run the Phase 3 reliability chain for one validated AnalysisContext.

    SHAP + LIME (Phase 2)  ->  Agreement
    repeated seeded LIME   ->  Stability
    ranked perturbation    ->  Faithfulness
    A + S + F              ->  ECS (+ sensitivity scenarios)

Internal orchestration only; the public API arrives in a later phase.
"""

from pydantic import BaseModel, ConfigDict

from backend.app.explainability.confidence.ecs import EcsCalculator
from backend.app.explainability.errors import InvalidMetricConfigurationError
from backend.app.explainability.evaluation.agreement import AgreementEvaluator
from backend.app.explainability.evaluation.faithfulness import FaithfulnessEvaluator, ranking_of
from backend.app.explainability.evaluation.stability import StabilityEvaluator
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.explanations.lime_explainer import LimeExplainer
from backend.app.explainability.explanations.shap_explainer import ShapExplainer
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import (
    AgreementResult,
    AnalysisContext,
    EcsResult,
    EcsSensitivityResult,
    FaithfulnessResult,
    LimeExplanationResult,
    ShapExplanationResult,
    StabilityResult,
)


class ReliabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    shap: ShapExplanationResult
    lime: LimeExplanationResult
    agreement: AgreementResult
    stability: StabilityResult
    faithfulness: FaithfulnessResult
    ecs: EcsResult
    ecs_sensitivity: EcsSensitivityResult


class ReliabilityEvaluator:
    def __init__(
        self,
        *,
        shap_explainer: ShapExplainer | None = None,
        lime_explainer: LimeExplainer | None = None,
        agreement: AgreementEvaluator | None = None,
        stability: StabilityEvaluator | None = None,
        faithfulness: FaithfulnessEvaluator | None = None,
        ecs: EcsCalculator | None = None,
    ) -> None:
        self._shap = shap_explainer or ShapExplainer()
        self._lime = lime_explainer or LimeExplainer()
        self._agreement = agreement or AgreementEvaluator()
        self._stability = stability or StabilityEvaluator(self._lime)
        self._faithfulness = faithfulness or FaithfulnessEvaluator()
        self._ecs = ecs or EcsCalculator()

    def evaluate(
        self, context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
    ) -> ReliabilityReport:
        shap = self._shap.explain(context, adapter, background)
        lime = self._lime.explain(context, adapter, background)
        agreement = self._agreement.evaluate(shap, lime)
        stability = self._stability.evaluate(context, adapter, background)

        source = self._faithfulness.config.ranking_source
        if source == "shap":
            ranking = ranking_of(shap)
        elif source == "lime":
            ranking = ranking_of(lime)
        else:
            raise InvalidMetricConfigurationError(
                f"faithfulness ranking_source {source!r} is not implemented in faithfulness_v1"
            )
        faithfulness = self._faithfulness.evaluate(
            context, adapter, background, ranking, ranking_source=source
        )
        return ReliabilityReport(
            shap=shap,
            lime=lime,
            agreement=agreement,
            stability=stability,
            faithfulness=faithfulness,
            ecs=self._ecs.compute(agreement, stability, faithfulness),
            ecs_sensitivity=self._ecs.sensitivity(agreement, stability, faithfulness),
        )
