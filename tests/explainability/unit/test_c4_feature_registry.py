from pathlib import Path

import pytest

from backend.app.explainability.errors import FeatureRegistryError, MissingFeatureError
from backend.app.explainability.feature_registry import (
    FeatureRegistry,
    load_feature_registry,
    parse_feature_registry,
)
from tests.explainability.fixtures.prototype_c2 import FEATURE_NAMES, SCHEMA_VERSION


def test_registry_loads(registry: FeatureRegistry) -> None:
    assert registry.schema_version == SCHEMA_VERSION
    assert "backspace_rate" in registry.features


def test_every_model_feature_resolves_in_model_order(registry: FeatureRegistry) -> None:
    resolved = registry.resolve(FEATURE_NAMES)
    assert tuple(resolved) == FEATURE_NAMES


def test_missing_registry_feature_is_detected(registry: FeatureRegistry) -> None:
    with pytest.raises(MissingFeatureError, match="not_a_feature"):
        registry.resolve([*FEATURE_NAMES, "not_a_feature"])


def test_schema_metadata_is_available_where_defined(registry: FeatureRegistry) -> None:
    entry = registry.resolve(["backspace_rate"])["backspace_rate"]
    assert entry.dtype == "float"
    assert entry.valid_range == (0.0, 1.0)
    assert entry.display_as == "percent"
    assert entry.actionable is True


def test_optional_and_unknown_metadata_do_not_fail() -> None:
    registry = parse_feature_registry(
        {
            "schema_version": "v",
            "features": {"bare": None, "extra": {"dtype": "int", "future_key": 1}},
        }
    )
    assert registry.features["bare"].dtype is None
    assert registry.features["extra"].model_extra == {"future_key": 1}


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"features": {"a": {}}},
        {"schema_version": "v", "features": {}},
        {"schema_version": "v", "features": {"a": {"valid_range": [1.0, 0.0]}}},
    ],
)
def test_malformed_registry_fails(document: object) -> None:
    with pytest.raises(FeatureRegistryError):
        parse_feature_registry(document)


def test_missing_registry_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FeatureRegistryError):
        load_feature_registry(tmp_path / "absent.yaml")
