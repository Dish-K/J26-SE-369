"""Shared fixtures for C4 Phase 1 tests. All data here is PROTOTYPE data."""

from pathlib import Path
from typing import Any

import pytest

from backend.app.explainability.feature_registry import FeatureRegistry, load_feature_registry
from backend.app.explainability.model.artifact_loader import ArtifactLoader
from backend.app.explainability.service import AnalysisService
from tests.explainability.fixtures.prototype_c2 import (
    REGISTRY_PATH,
    C2Records,
    InMemoryC2Reader,
    artifact_dict,
    build_prototype_model,
    emulate_c2,
    load_scenarios,
    write_artifact,
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
