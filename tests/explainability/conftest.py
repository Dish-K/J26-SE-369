"""Shared fixtures for C4 tests. All data here is PROTOTYPE data."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from backend.app.explainability.explanations.background import XaiBackground, load_background
from backend.app.explainability.feature_registry import FeatureRegistry, load_feature_registry
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from backend.app.explainability.model.adapter import ClassifierAdapter
from backend.app.explainability.schemas import AnalysisContext
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import (
    MODEL_VERSION,
    REGISTRY_PATH,
    C2Records,
    InMemoryC2Reader,
    artifact_dict,
    background_reference,
    build_prototype_model,
    emulate_c2,
    load_scenarios,
    write_artifact,
    write_background,
)


@pytest.fixture(scope="session")
def prototype_model() -> Any:
    return build_prototype_model()


@pytest.fixture(scope="session")
def c2_records(prototype_model: Any) -> C2Records:
    return emulate_c2(prototype_model, load_scenarios())


@pytest.fixture
def records(c2_records: C2Records) -> C2Records:
    """A private copy each test may mutate to break the contract."""
    return c2_records.copy()


@pytest.fixture
def artifact_dir(tmp_path: Path, prototype_model: Any) -> Path:
    write_artifact(tmp_path, artifact_dict(prototype_model))
    return tmp_path


@pytest.fixture(scope="session")
def registry() -> FeatureRegistry:
    return load_feature_registry(REGISTRY_PATH)


@pytest.fixture
def service(records: C2Records, artifact_dir: Path, registry: FeatureRegistry) -> AnalysisService:
    return AnalysisService(InMemoryC2Reader(records), ArtifactLoader(artifact_dir), registry)


# --- Phase 2 fixtures ---------------------------------------------------------


@pytest.fixture
def background_dir(tmp_path: Path) -> Path:
    write_background(tmp_path)
    return tmp_path


@pytest.fixture
def explain_service(
    records: C2Records, artifact_dir: Path, registry: FeatureRegistry
) -> AnalysisService:
    """Phase 1 service whose contexts resolve the prototype XAI background."""
    return AnalysisService(
        InMemoryC2Reader(records),
        ArtifactLoader(artifact_dir),
        registry,
        dev_background_ref=background_reference(),
    )


def _context(service: AnalysisService, records: C2Records, flag_name: str) -> AnalysisContext:
    return asyncio.run(service.build_context(records.flag_ids_by_name[flag_name]))


@pytest.fixture
def external_context(explain_service: AnalysisService, records: C2Records) -> AnalysisContext:
    return _context(explain_service, records, "external_ai_segment")


@pytest.fixture
def ide_context(explain_service: AnalysisService, records: C2Records) -> AnalysisContext:
    return _context(explain_service, records, "ide_ai_segment")


@pytest.fixture
def adapter(explain_service: AnalysisService) -> ClassifierAdapter:
    return explain_service.adapter_for(MODEL_VERSION)


@pytest.fixture
def background(artifact_dir: Path, background_dir: Path) -> XaiBackground:
    metadata = ArtifactLoader(artifact_dir).load(MODEL_VERSION).metadata
    return load_background(background_reference(), metadata, base_dir=background_dir)
