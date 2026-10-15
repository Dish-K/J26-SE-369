import math
from typing import Any

import numpy as np
import pytest

from backend.app.explainability.config import LimeConfig
from backend.app.explainability.errors import (
    ExplanationBackgroundMismatchError,
    ModelVersionMismatchError,
)
from backend.app.explainability.explanations.background import XaiBackground
from backend.app.explainability.explanations.feature_normalizer import feature_name_from_condition
from backend.app.explainability.explanations.lime_explainer import LimeExplainer
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import AnalysisContext
from tests.explainability.fixtures.prototype_c2 import FEATURE_NAMES, background_reference


@pytest.fixture
def lime() -> LimeExplainer:
    return LimeExplainer()


def test_explains_the_representative_prediction(
    lime: LimeExplainer, external_context: AnalysisContext, adapter: ClassifierAdapter,
    background: XaiBackground,
) -> None:
    result = lime.explain(external_context, adapter, background)
    prediction = external_context.representative_prediction
    assert result.prediction_id == prediction.prediction_id
    assert result.feature_vector_id == prediction.feature_vector_id
    assert result.model_version == external_context.model_version
    assert result.background_version == background.background_version


@pytest.mark.parametrize("fixture", ["external_context", "ide_context"])
def test_target_class_and_index_come_from_the_model_class_order(
    lime: LimeExplainer, adapter: ClassifierAdapter, background: XaiBackground,
    fixture: str, request: pytest.FixtureRequest,
) -> None:
    context: AnalysisContext = request.getfixturevalue(fixture)
    result = lime.explain(context, adapter, background)
    assert result.target_class == context.representative_prediction.predicted_category
    assert adapter.get_class_names()[result.target_class_index] == result.target_class
    assert result.target_probability == pytest.approx(
        context.reproduction.reproduced_probabilities[context.target_class], abs=1e-12
    )


def test_prediction_callback_is_the_adapter_on_the_same_instance(
    lime: LimeExplainer, external_context: AnalysisContext, adapter: ClassifierAdapter,
    background: XaiBackground, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[np.ndarray] = []
    original = adapter.predict_proba_batch

    def spy(rows: Any) -> np.ndarray:
        seen.append(np.asarray(rows, dtype=float))
        return original(rows)

    monkeypatch.setattr(adapter, "predict_proba_batch", spy)
    lime.explain(external_context, adapter, background)

    assert seen, "LIME never called ClassifierAdapter.predict_proba_batch"
    instance = [external_context.model_input[name] for name in FEATURE_NAMES]
    np.testing.assert_array_equal(seen[0][0], instance)  # first LIME row = the instance
    assert seen[0].shape[1] == len(FEATURE_NAMES)


def test_features_are_canonical_complete_and_ranked(
    lime: LimeExplainer, external_context: AnalysisContext, adapter: ClassifierAdapter,
    background: XaiBackground,
) -> None:
    result = lime.explain(external_context, adapter, background)
    assert sorted(f.name for f in result.features) == sorted(FEATURE_NAMES)
    magnitudes = [f.absolute_weight for f in result.features]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for feature in result.features:
        assert FEATURE_NAMES[feature.feature_index] == feature.name
        assert feature_name_from_condition(feature.condition, FEATURE_NAMES) == feature.name
        assert feature.value == external_context.model_input[feature.name]  # original value
        assert math.isfinite(feature.weight)
        expected = (
            "toward_target" if feature.weight > 0 else "away_from_target" if feature.weight < 0
            else "neutral"
        )
        assert feature.direction == expected


def test_seed_is_stored_and_reproducible(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    first = LimeExplainer(LimeConfig(seed=7)).explain(external_context, adapter, background)
    second = LimeExplainer(LimeConfig(seed=7)).explain(external_context, adapter, background)
    other = LimeExplainer(LimeConfig(seed=8)).explain(external_context, adapter, background)
    assert first.seed == 7
    assert first == second
    assert [f.weight for f in first.features] != [f.weight for f in other.features]


def test_wrong_background_is_rejected(
    lime: LimeExplainer, external_context: AnalysisContext, adapter: ClassifierAdapter,
    background: XaiBackground,
) -> None:
    other = XaiBackground(
        reference=background_reference(dataset_version="some-other-background"),
        frame=background.frame,
    )
    with pytest.raises(ExplanationBackgroundMismatchError):
        lime.explain(external_context, adapter, other)


def test_context_without_background_is_rejected(
    lime: LimeExplainer, external_context: AnalysisContext, adapter: ClassifierAdapter,
    background: XaiBackground,
) -> None:
    context = external_context.model_copy(update={"background_ref": None})
    with pytest.raises(ExplanationBackgroundMismatchError, match="no XAI background"):
        lime.explain(context, adapter, background)


def test_model_version_mismatch_is_rejected(
    lime: LimeExplainer, external_context: AnalysisContext, adapter: ClassifierAdapter,
    background: XaiBackground,
) -> None:
    context = external_context.model_copy(update={"model_version": "synthetic-prototype-v1"})
    with pytest.raises(ModelVersionMismatchError):
        lime.explain(context, adapter, background)
