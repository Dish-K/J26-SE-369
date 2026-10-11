"""MAIN PHASE 1 ACCEPTANCE TEST: C4 reproduces the stored C2 prediction."""

import asyncio

import pytest

from backend.app.explainability.config import DEFAULT_REPRODUCTION_TOLERANCE
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import C2Records


@pytest.mark.parametrize("flag_name", ["external_ai_segment", "ide_ai_segment"])
def test_representative_prediction_is_reproduced(
    records: C2Records, service: AnalysisService, flag_name: str
) -> None:
    flag_id = records.flag_ids_by_name[flag_name]
    stored = records.predictions[records.flags[flag_id]["representative_prediction_id"]]

    context = asyncio.run(service.build_context(flag_id))
    reproduced = context.reproduction.reproduced_probabilities

    for name, probability in stored["class_probabilities"].items():
        assert abs(probability - reproduced[name]) <= DEFAULT_REPRODUCTION_TOLERANCE
    assert max(reproduced, key=reproduced.get) == stored["predicted_category"]
    assert context.reproduction.reproduced_category == stored["predicted_category"]
    assert context.reproduction.max_abs_difference <= DEFAULT_REPRODUCTION_TOLERANCE


def test_reproduction_uses_the_exact_stored_model_input(
    records: C2Records, service: AnalysisService
) -> None:
    flag_id = records.flag_ids_by_name["external_ai_segment"]
    context = asyncio.run(service.build_context(flag_id))
    adapter = service.adapter_for(context.model_version)

    rerun = adapter.predict_proba(context.model_input)
    assert rerun == context.reproduction.reproduced_probabilities
