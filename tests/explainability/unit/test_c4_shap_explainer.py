"""Local and global SHAP. Requires the shap package (blocked on Windows hosts
with Smart App Control enforcing; run under Linux/WSL or CI)."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.app.explainability.config import ShapConfig
from backend.app.explainability.errors import (
    ExplanationBackgroundMismatchError,
    ShapReconstructionError,
    TargetClassNotFoundError,
)
from backend.app.explainability.explanations.background import XaiBackground, load_background
from backend.app.explainability.explanations.shap_explainer import (
    ShapExplainer,
    verify_reconstruction,
)
from backend.app.explainability.feature_registry import FeatureRegistry
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from backend.app.explainability.model.explainer_factory import ExplainerFactory
from backend.app.explainability.schemas import AnalysisContext, ShapExplanationResult
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import (
    FEATURE_NAMES,
    MODEL_VERSION,
    InMemoryC2Reader,
    _training_frame,
    analysis_frame,
    artifact_dict,
    background_reference,
    emulate_c2,
    load_scenarios,
    write_artifact,
    write_background,
)


@pytest.fixture
def shap_explainer() -> ShapExplainer:
    return ShapExplainer()


class _ForceModelAgnostic(ExplainerFactory):
    def select_kind(self, adapter: ClassifierAdapter) -> tuple[str, str]:
        return "model_agnostic", "forced by test"


# --- local SHAP: provenance, target, features ---------------------------------


def test_local_shap_explains_the_representative_prediction(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground,
) -> None:
    result = shap_explainer.explain(external_context, adapter, background)
    prediction = external_context.representative_prediction
    assert result.prediction_id == prediction.prediction_id
    assert result.feature_vector_id == prediction.feature_vector_id
    assert result.model_version == MODEL_VERSION
    assert result.target_class == prediction.predicted_category == "external_ai"
    assert result.target_class_index == adapter.get_class_names().index("external_ai")
    assert result.background_version == background.background_version
    assert result.strategy == "linear"


@pytest.mark.parametrize("fixture", ["external_context", "ide_context"])
def test_target_index_follows_model_class_order(
    shap_explainer: ShapExplainer, adapter: ClassifierAdapter, background: XaiBackground,
    fixture: str, request: pytest.FixtureRequest,
) -> None:
    context: AnalysisContext = request.getfixturevalue(fixture)
    result = shap_explainer.explain(context, adapter, background)
    assert adapter.get_class_names()[result.target_class_index] == result.target_class
    assert result.target_class == context.flag.category


def test_features_are_original_values_ranked_by_magnitude(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground,
) -> None:
    result = shap_explainer.explain(external_context, adapter, background)
    assert sorted(f.name for f in result.features) == sorted(FEATURE_NAMES)
    assert [f.rank for f in result.features] == list(range(1, len(FEATURE_NAMES) + 1))
    magnitudes = [f.absolute_shap_value for f in result.features]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for feature in result.features:
        assert feature.value == external_context.model_input[feature.name]  # not scaled
        assert feature.absolute_shap_value == abs(feature.shap_value)
        if feature.shap_value > 0:
            assert feature.direction == "toward_target"
        elif feature.shap_value < 0:
            assert feature.direction == "away_from_target"


def test_linear_attributions_match_the_closed_form_on_raw_features(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground, prototype_model: Any,
) -> None:
    """Interventional linear SHAP is w_k * (z - mean(z_background)) per raw feature."""
    result = shap_explainer.explain(external_context, adapter, background)
    scaler, linear = prototype_model.named_steps["scale"], prototype_model.named_steps["classify"]
    k = result.target_class_index
    x = adapter.prepare_features(external_context.model_input)
    z = scaler.transform(x)[0]
    z_bg = scaler.transform(background.frame).mean(axis=0)
    expected = dict(zip(FEATURE_NAMES, linear.coef_[k] * (z - z_bg), strict=True))
    for feature in result.features:
        assert feature.shap_value == pytest.approx(expected[feature.name], abs=1e-9)


# --- output space and reconstruction -----------------------------------------


def test_linear_output_space_is_the_decision_function_not_probability(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground, prototype_model: Any,
) -> None:
    result = shap_explainer.explain(external_context, adapter, background)
    x = adapter.prepare_features(external_context.model_input)
    logits = prototype_model.decision_function(x)[0]
    assert result.output_space == "raw"
    assert result.model_output == pytest.approx(logits[result.target_class_index], abs=1e-12)
    assert result.target_probability == pytest.approx(
        external_context.reproduction.reproduced_probabilities["external_ai"], abs=1e-12
    )
    assert result.model_output != pytest.approx(result.target_probability)


def test_reconstruction_passes_within_tolerance(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground,
) -> None:
    result = shap_explainer.explain(external_context, adapter, background)
    total = result.base_value + sum(f.shap_value for f in result.features)
    assert result.reconstructed_output == pytest.approx(total, abs=1e-12)
    assert result.reconstruction_error == abs(result.model_output - result.reconstructed_output)
    assert result.reconstruction_error <= result.reconstruction_tolerance == 1e-6


def test_corrupted_model_output_is_detected(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = ExplainerFactory()
    strategy = factory.create(adapter, background)
    corrupted = replace(strategy, _model_outputs=lambda rows: strategy.model_outputs(rows) + 0.01)
    monkeypatch.setattr(factory, "create", lambda *_: corrupted)
    with pytest.raises(ShapReconstructionError, match="output_space=raw"):
        ShapExplainer(factory).explain(external_context, adapter, background)


def test_verify_reconstruction_rejects_corrupted_attributions(
    adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    strategy = ExplainerFactory().create(adapter, background)
    verify_reconstruction(
        base_value=0.5, attributions=np.array([0.2, 0.3]), model_output=1.0,
        tolerance=1e-6, strategy=strategy,
    )
    with pytest.raises(ShapReconstructionError):
        verify_reconstruction(
            base_value=0.5, attributions=np.array([0.2, 0.31]), model_output=1.0,
            tolerance=1e-6, strategy=strategy,
        )


def test_result_schema_refuses_an_unreconstructed_explanation(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground,
) -> None:
    result = shap_explainer.explain(external_context, adapter, background)
    data = result.model_dump()
    data["reconstruction_error"] = 1e-3
    with pytest.raises(ValueError):
        ShapExplanationResult.model_validate(data)


# --- other strategies --------------------------------------------------------


def test_model_agnostic_exact_is_probability_space(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    result = ShapExplainer(_ForceModelAgnostic()).explain(external_context, adapter, background)
    assert result.strategy == "model_agnostic"
    assert result.output_space == "probability"
    assert result.model_output == pytest.approx(result.target_probability, abs=1e-12)
    assert result.reconstruction_error <= result.reconstruction_tolerance


def test_model_agnostic_permutation_above_exact_limit(
    external_context: AnalysisContext, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    factory = _ForceModelAgnostic(ShapConfig(max_exact_features=5))
    result = ShapExplainer(factory).explain(external_context, adapter, background)
    assert "Permutation" in result.explainer_type
    assert result.reconstruction_error <= result.reconstruction_tolerance


def test_random_forest_tree_strategy_reconstructs_probability(
    tmp_path: Path, registry: FeatureRegistry
) -> None:
    frame, labels = _training_frame()
    forest = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=0).fit(frame, labels)
    write_artifact(tmp_path, artifact_dict(forest, model_family=None))
    write_background(tmp_path)
    records = emulate_c2(forest, load_scenarios())
    service = AnalysisService(
        InMemoryC2Reader(records), ArtifactLoader(tmp_path), registry,
        dev_background_ref=background_reference(),
    )
    context = asyncio.run(service.build_context(records.flag_ids_by_name["external_ai_segment"]))
    adapter = service.adapter_for(MODEL_VERSION)
    background = load_background(background_reference(), context.artifact, base_dir=tmp_path)

    result = ShapExplainer().explain(context, adapter, background)
    assert result.strategy == "tree"
    assert result.output_space == "probability"
    assert result.model_output == pytest.approx(result.target_probability, abs=1e-12)
    assert result.reconstruction_error <= result.reconstruction_tolerance


def test_factory_caches_per_model_and_background(
    adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    factory = ExplainerFactory()
    assert factory.create(adapter, background) is factory.create(adapter, background)
    other = XaiBackground(
        reference=background_reference(dataset_version="other-background-v1"),
        frame=background.frame,
    )
    assert factory.create(adapter, other) is not factory.create(adapter, background)


def test_wrong_background_is_rejected(
    shap_explainer: ShapExplainer, external_context: AnalysisContext,
    adapter: ClassifierAdapter, background: XaiBackground,
) -> None:
    other = XaiBackground(
        reference=background_reference(dataset_version="some-other-background"),
        frame=background.frame,
    )
    with pytest.raises(ExplanationBackgroundMismatchError):
        shap_explainer.explain(external_context, adapter, other)


# --- global SHAP -------------------------------------------------------------


def test_global_shap_summarises_analysis_samples(
    shap_explainer: ShapExplainer, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    samples = analysis_frame()
    result = shap_explainer.explain_global(
        adapter, background, samples, target_class="external_ai",
        dataset_reference="prototype-analysis-slice",
    )
    assert result.sample_count == len(samples) > 1
    assert result.target_class == "external_ai"
    assert result.target_class_index == adapter.get_class_names().index("external_ai")
    assert result.dataset_reference == "prototype-analysis-slice"
    assert result.background_version == background.background_version
    assert result.feature_schema_version == background.reference.schema_version
    assert sorted(f.name for f in result.feature_importance) == sorted(FEATURE_NAMES)
    assert all(f.mean_abs_shap >= 0 for f in result.feature_importance)
    importances = [f.mean_abs_shap for f in result.feature_importance]
    assert importances == sorted(importances, reverse=True)
    assert result.max_reconstruction_error <= shap_explainer.tolerance
    assert result.beeswarm is not None
    assert len(result.beeswarm.shap_values) == len(samples)
    assert len(result.beeswarm.shap_values[0]) == len(FEATURE_NAMES)


def test_global_shap_mean_abs_matches_local_values(
    shap_explainer: ShapExplainer, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    result = shap_explainer.explain_global(
        adapter, background, analysis_frame(), target_class="ide_ai", dataset_reference="slice"
    )
    shap_matrix = np.array(result.beeswarm.shap_values)  # type: ignore[union-attr]
    by_name = {f.name: f.mean_abs_shap for f in result.feature_importance}
    for column, name in enumerate(result.beeswarm.feature_names):  # type: ignore[union-attr]
        assert by_name[name] == pytest.approx(np.abs(shap_matrix[:, column]).mean(), abs=1e-12)


def test_global_shap_is_deterministic_and_supports_no_ai(
    shap_explainer: ShapExplainer, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    kwargs = dict(target_class="no_ai", dataset_reference="slice", include_beeswarm=False)
    first = shap_explainer.explain_global(adapter, background, analysis_frame(), **kwargs)
    second = shap_explainer.explain_global(adapter, background, analysis_frame(), **kwargs)
    assert first == second
    assert first.target_class == "no_ai"
    assert first.beeswarm is None


def test_global_shap_rejects_unknown_class_and_incomplete_samples(
    shap_explainer: ShapExplainer, adapter: ClassifierAdapter, background: XaiBackground
) -> None:
    with pytest.raises(TargetClassNotFoundError):
        shap_explainer.explain_global(
            adapter, background, analysis_frame(), target_class="cheating", dataset_reference="s"
        )
    with pytest.raises(ExplanationBackgroundMismatchError, match="paste_ratio"):
        shap_explainer.explain_global(
            adapter, background, analysis_frame().drop(columns=["paste_ratio"]),
            target_class="external_ai", dataset_reference="s",
        )
