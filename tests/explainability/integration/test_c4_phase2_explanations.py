"""Phase 2 end to end: flag -> AnalysisContext -> background -> SHAP + LIME.

Requires the shap package (run under Linux/WSL or CI).
"""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from backend.app.explainability.config import ExplainabilitySettings
from backend.app.explainability.explanations.background import load_background
from backend.app.explainability.explanations.lime_explainer import LimeExplainer
from backend.app.explainability.explanations.shap_explainer import ShapExplainer
from backend.app.explainability.persistence.repository import PostgresC2Repository
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import (
    FEATURE_NAMES,
    REGISTRY_PATH,
    C2Records,
    FakeAsyncpgPool,
    background_reference,
    write_background,
)


@pytest.mark.parametrize("flag_name", ["external_ai_segment", "ide_ai_segment"])
def test_shap_and_lime_explain_the_same_prediction(
    records: C2Records, artifact_dir: Path, tmp_path: Path, flag_name: str
) -> None:
    # The background is configured exactly as a deployment would: a JSON
    # descriptor whose uri is resolved next to it.
    background_dir = tmp_path / "background"
    background_dir.mkdir()
    write_background(background_dir)
    descriptor = background_dir / "xai_background.json"
    descriptor.write_text(background_reference().model_dump_json(), encoding="utf-8")

    settings = ExplainabilitySettings(
        model_artifact_dir=artifact_dir,
        feature_registry_path=REGISTRY_PATH,
        xai_background_ref_path=descriptor,
    )
    service = AnalysisService.from_settings(
        PostgresC2Repository(FakeAsyncpgPool(records)), settings  # type: ignore[arg-type]
    )
    before = copy.deepcopy(records)

    context = asyncio.run(service.build_context(records.flag_ids_by_name[flag_name]))
    adapter = service.adapter_for(context.model_version)
    assert context.background_ref is not None
    background = load_background(context.background_ref, context.artifact, base_dir=descriptor.parent)

    shap_result = ShapExplainer().explain(context, adapter, background)
    lime_result = LimeExplainer().explain(context, adapter, background)

    # SHAP/LIME target alignment
    assert shap_result.prediction_id == lime_result.prediction_id == (
        context.flag.representative_prediction_id
    )
    assert shap_result.feature_vector_id == lime_result.feature_vector_id == (
        context.feature_vector.feature_vector_id
    )
    assert shap_result.model_version == lime_result.model_version == context.flag.model_version
    assert shap_result.target_class == lime_result.target_class == context.flag.category
    assert shap_result.target_class_index == lime_result.target_class_index
    assert shap_result.background_version == lime_result.background_version
    assert {f.name for f in shap_result.features} == {f.name for f in lime_result.features} == set(
        FEATURE_NAMES
    )
    assert shap_result.target_probability == lime_result.target_probability

    # SHAP is additive in its own output space
    assert shap_result.reconstruction_error <= shap_result.reconstruction_tolerance

    # Phase 1 guarantee still holds and nothing C2 produced was modified
    assert context.reproduction.max_abs_difference <= context.reproduction.tolerance
    assert records.flags == before.flags
    assert records.predictions == before.predictions
    assert records.feature_vectors == before.feature_vectors
    json.dumps(shap_result.model_dump(mode="json"))  # results are serialisable
    json.dumps(lime_result.model_dump(mode="json"))
