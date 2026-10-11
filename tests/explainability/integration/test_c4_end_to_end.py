"""flag_id -> PostgresC2Repository -> artifact -> adapter -> AnalysisContext.

Runs the real repository SQL path against FakeAsyncpgPool (no database is
available in this environment) and the real joblib artifact path.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from backend.app.explainability.config import ExplainabilitySettings
from backend.app.explainability.persistence.repository import PostgresC2Repository
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import (
    FEATURE_NAMES,
    MODEL_VERSION,
    REGISTRY_PATH,
    SCHEMA_VERSION,
    C2Records,
    FakeAsyncpgPool,
)


def test_phase1_end_to_end_contract_valid(
    records: C2Records, artifact_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = ExplainabilitySettings(
        model_artifact_dir=artifact_dir, feature_registry_path=REGISTRY_PATH
    )
    reader = PostgresC2Repository(FakeAsyncpgPool(records))  # type: ignore[arg-type]
    service = AnalysisService.from_settings(reader, settings)

    flag_id = records.flag_ids_by_name["external_ai_segment"]
    stored_flag = records.flags[flag_id]

    with caplog.at_level(logging.INFO, logger="backend.app.explainability"):
        context = asyncio.run(service.build_context(flag_id))

    # flag -> representative prediction -> feature vector, exactly as C2 linked them
    assert context.flag.flag_id == flag_id
    assert context.representative_prediction.prediction_id == (
        stored_flag["representative_prediction_id"]
    )
    assert context.feature_vector.feature_vector_id == (
        context.representative_prediction.feature_vector_id
    )
    assert set(context.contributing_prediction_ids) == set(records.members[flag_id])
    assert len(context.contributing_prediction_ids) == stored_flag["window_count"] == 3

    # exact artifact and contract
    assert context.model_version == MODEL_VERSION
    assert context.feature_schema_version == SCHEMA_VERSION
    assert context.feature_names == FEATURE_NAMES
    assert tuple(context.model_input) == FEATURE_NAMES
    assert set(context.feature_metadata) == set(FEATURE_NAMES)

    # target alignment for later explanation phases
    assert context.target_class == "external_ai"
    assert context.class_names[context.target_class_index] == "external_ai"

    # reproduction
    assert context.reproduction.max_abs_difference <= context.reproduction.tolerance
    assert context.reproduction.reproduced_category == "external_ai"

    assert "C2 -> C4 contract valid" in caplog.text
