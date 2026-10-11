"""Read-only loader for the shared C1/C2/C4 feature registry.

C4 does not own or define the feature registry. C1 specification §16 assigns
it to one shared YAML file; this module only consumes that format. Until the
file lands in the repository, C4 points at it through
``C4_FEATURE_REGISTRY_PATH``.

Expected file shape (per-feature entries exactly as in C1 spec §16; the
``schema_version`` envelope lets C4 check compatibility with the artifact and
the stored feature vectors):

    schema_version: <Contract-A version string>
    features:
      backspace_rate:
        dtype: float
        unit: fraction
        valid_range: [0.0, 1.0]
        label: "Correction rate while typing"
        description: "..."
        display_as: percent
        actionable: true

Every per-feature key is optional here. Phase 1 only needs to resolve names;
later phases decide which metadata they require.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from backend.app.explainability.errors import FeatureRegistryError, MissingFeatureError


class FeatureMetadata(BaseModel):
    """One registry entry. Unknown keys are preserved, not rejected."""

    model_config = ConfigDict(frozen=True, extra="allow")

    dtype: str | None = None
    unit: str | None = None
    valid_range: tuple[float, float] | None = None
    label: str | None = None
    description: str | None = None
    display_as: str | None = None
    actionable: bool | None = None

    @field_validator("valid_range")
    @classmethod
    def _ordered_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError(f"valid_range lower bound exceeds upper bound: {value}")
        return value


@dataclass(frozen=True)
class FeatureRegistry:
    schema_version: str
    features: Mapping[str, FeatureMetadata]

    def resolve(self, names: Iterable[str]) -> dict[str, FeatureMetadata]:
        """Return metadata for ``names`` in the given order, or fail on any gap."""
        names = list(names)
        missing = [name for name in names if name not in self.features]
        if missing:
            raise MissingFeatureError(
                f"model features not in feature registry {self.schema_version!r}: {missing}"
            )
        return {name: self.features[name] for name in names}


def parse_feature_registry(document: Any, *, source: str = "<memory>") -> FeatureRegistry:
    if not isinstance(document, Mapping):
        raise FeatureRegistryError(f"{source}: registry must be a mapping")

    schema_version = document.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise FeatureRegistryError(f"{source}: registry needs a non-empty 'schema_version'")

    entries = document.get("features")
    if not isinstance(entries, Mapping) or not entries:
        raise FeatureRegistryError(f"{source}: registry needs a non-empty 'features' mapping")

    features: dict[str, FeatureMetadata] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not name:
            raise FeatureRegistryError(f"{source}: feature names must be non-empty strings")
        try:
            features[name] = FeatureMetadata.model_validate(entry or {})
        except ValidationError as exc:
            raise FeatureRegistryError(f"{source}: invalid entry for {name!r}: {exc}") from exc

    return FeatureRegistry(schema_version=schema_version, features=features)


def load_feature_registry(path: Path) -> FeatureRegistry:
    if not path.is_file():
        raise FeatureRegistryError(f"feature registry not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FeatureRegistryError(f"{path}: invalid YAML: {exc}") from exc
    return parse_feature_registry(document, source=str(path))
