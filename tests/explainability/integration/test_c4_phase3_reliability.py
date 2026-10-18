"""Phase 3 end to end on the prototype: no assertion forces a 'good' score.

flag -> AnalysisContext -> SHAP + LIME -> Agreement -> repeated LIME ->
Stability -> Faithfulness -> ECS -> ECS sensitivity.
"""

import copy
import json

import pytest

from backend.app.explainability.evaluation.faithfulness import ranking_of
from backend.app.explainability.evaluation.reliability import ReliabilityEvaluator
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import AnalysisContext
from tests.explainability.fixtures.prototype_c2 import C2Records

_IDENTITY = ("prediction_id", "feature_vector_id", "model_version", "target_class",
             "target_class_index")


@pytest.mark.parametrize("fixture", ["external_context", "ide_context"])
def test_full_reliability_chain(
    fixture: str, request: pytest.FixtureRequest, adapter: ClassifierAdapter,
    background: XaiBackground, records: C2Records,
) -> None:
    context: AnalysisContext = request.getfixturevalue(fixture)
    before = copy.deepcopy(records)

    report = ReliabilityEvaluator().evaluate(context, adapter, background)

    # one prediction, one provenance, across every result
    stages = (report.shap, report.lime, report.agreement, report.stability,
              report.faithfulness, report.ecs, report.ecs_sensitivity)
    for name in _IDENTITY:
        assert len({getattr(stage, name) for stage in stages}) == 1, name
    assert report.ecs.prediction_id == context.flag.representative_prediction_id
    assert report.ecs.target_class == context.flag.category

    # every stage computed and in range on this prototype
    for stage in (report.agreement, report.stability, report.faithfulness, report.ecs):
        assert stage.status == "computed"
        assert 0.0 <= stage.score <= 1.0
    assert report.stability.pair_count == 190
    assert report.stability.seeds == tuple(range(42, 62))
    assert report.stability.lime_fit.count == 20
    assert report.faithfulness.ranking == ranking_of(report.shap)
    assert report.faithfulness.ranking_source == "shap"

    # ECS is exactly the equal-weight mean of the primary components
    assert report.ecs.score == pytest.approx(
        (report.agreement.score + report.stability.score + report.faithfulness.score) / 3
    )
    assert [s.name for s in report.ecs_sensitivity.scenarios] == [
        "equal_weight", "agreement_emphasis", "stability_emphasis", "faithfulness_emphasis"
    ]
    assert report.ecs_sensitivity.scenarios[0].score == pytest.approx(report.ecs.score)

    # Phase 1 guarantee intact; C2 output untouched; results serialisable
    assert context.reproduction.max_abs_difference <= context.reproduction.tolerance
    assert records.flags == before.flags
    assert records.predictions == before.predictions
    assert records.feature_vectors == before.feature_vectors
    json.dumps(report.model_dump(mode="json"))


def test_reliability_is_reproducible(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    first = ReliabilityEvaluator().evaluate(external_context, adapter, background)
    second = ReliabilityEvaluator().evaluate(external_context, adapter, background)
    assert first == second
