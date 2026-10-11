"""Load a C2 model artifact by its exact ``model_version``.

C2 has not yet shipped its own loader. This module reads the artifact shape
defined in the C2 implementation plan §13 — a joblib-serialised mapping:

    {
        "model": <fitted estimator or sklearn Pipeline>,
        "model_version": "synthetic-prototype-v0",
        "feature_schema_version": "...",
        "feature_names": [...],              # exact model input order
        "class_order": ["no_ai", "ide_ai", "external_ai"],
        "classification_threshold": <float>,
        "training_dataset_version": "...",   # optional
        "generator_version": "...",          # optional
        "training_timestamp": "...",         # optional
        "model_family": "...",               # optional, derived if absent
        "xai_background_ref": {...},         # optional
    }

stored at ``<artifact_dir>/<model_version>/artifact.joblib``. When C2 delivers
its runtime loader, replace ``_read_artifact`` with a call to it and keep the
validation here.

joblib files are pickles: only load artifacts produced by the project.
"""

import logging
import pickle
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from pydantic import ValidationError

from backend.app.explainability.errors import (
    ClassOrderMismatchError,
    ModelArtifactInvalidError,
    ModelArtifactNotFoundError,
    ModelVersionMismatchError,
)
from backend.app.explainability.schemas import ModelArtifactMetadata

logger = logging.getLogger(__name__)

ARTIFACT_FILENAME = "artifact.joblib"

# A model_version becomes one directory name; anything else would let a stored
# string escape the artifact directory.
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_METADATA_KEYS = frozenset(ModelArtifactMetadata.model_fields)


@dataclass(frozen=True)
class LoadedArtifact:
    metadata: ModelArtifactMetadata
    model: Any
    model_family: str
    path: Path
    # Class name of each predict_proba column, in the model's column order.
    output_classes: tuple[str, ...]


def derive_model_family(model: Any) -> str:
    """Name of the final estimator, looking through an sklearn Pipeline."""
    steps = getattr(model, "steps", None)
    if steps:
        model = steps[-1][1]
    return type(model).__name__


def resolve_output_classes(model: Any, class_order: tuple[str, ...]) -> tuple[str, ...]:
    """Class name of each ``predict_proba`` column, in column order.

    The fitted model is authoritative for column order:

    * string ``classes_`` (e.g. sklearn trained on class names) name the columns
      directly and must be exactly the classes in ``class_order``. They are
      often alphabetical, so ``class_order`` is not assumed to be column order;
    * integer ``classes_`` (e.g. a label-encoded XGBoost model) must be
      ``0..n-1`` and ``class_order`` supplies the name of each index.

    Never assumes index 0 is ``no_ai``.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ModelArtifactInvalidError(
            "model exposes no classes_; its probability column order cannot be verified"
        )
    labels = [label.item() if hasattr(label, "item") else label for label in classes]
    if all(isinstance(label, str) for label in labels):
        if len(labels) != len(class_order) or set(labels) != set(class_order):
            raise ClassOrderMismatchError(
                f"model classes_ {labels} are not the artifact classes {list(class_order)}"
            )
        return tuple(labels)
    if all(isinstance(label, int) and not isinstance(label, bool) for label in labels):
        if labels != list(range(len(class_order))):
            raise ClassOrderMismatchError(
                f"integer model classes_ {labels} are not 0..{len(class_order) - 1}"
            )
        return class_order
    raise ClassOrderMismatchError(f"unsupported model classes_ labels: {labels}")


def _check_feature_names(model: Any, feature_names: tuple[str, ...]) -> None:
    fitted = getattr(model, "feature_names_in_", None)
    if fitted is not None and tuple(str(name) for name in fitted) != feature_names:
        raise ModelArtifactInvalidError(
            "artifact feature_names do not match the column order the model was fitted on: "
            f"{list(feature_names)} vs {list(fitted)}"
        )
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and expected_count != len(feature_names):
        raise ModelArtifactInvalidError(
            f"model expects {expected_count} features but artifact lists {len(feature_names)}"
        )


def validate_artifact(raw: Any, *, requested_version: str, path: Path) -> LoadedArtifact:
    if not isinstance(raw, Mapping):
        raise ModelArtifactInvalidError(f"{path}: artifact must be a mapping, got {type(raw)}")
    if "model" not in raw:
        raise ModelArtifactInvalidError(f"{path}: artifact has no 'model' entry")

    model = raw["model"]
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelArtifactInvalidError(f"{path}: model has no predict_proba")

    try:
        metadata = ModelArtifactMetadata.model_validate(
            {key: value for key, value in raw.items() if key in _METADATA_KEYS}
        )
    except ValidationError as exc:
        raise ModelArtifactInvalidError(f"{path}: invalid artifact metadata: {exc}") from exc

    if metadata.model_version != requested_version:
        raise ModelVersionMismatchError(
            f"{path}: requested model_version {requested_version!r} "
            f"but artifact declares {metadata.model_version!r}"
        )

    output_classes = resolve_output_classes(model, metadata.class_order)
    _check_feature_names(model, metadata.feature_names)

    return LoadedArtifact(
        metadata=metadata,
        model=model,
        model_family=metadata.model_family or derive_model_family(model),
        path=path,
        output_classes=output_classes,
    )


class ArtifactLoader:
    """Resolves and caches C2 artifacts. Never substitutes another version."""

    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir
        self._cache: dict[str, LoadedArtifact] = {}

    def artifact_path(self, model_version: str) -> Path:
        if not _SAFE_VERSION.fullmatch(model_version):
            raise ModelArtifactNotFoundError(f"unsafe model_version for a path: {model_version!r}")
        return self._artifact_dir / model_version / ARTIFACT_FILENAME

    def load(self, model_version: str) -> LoadedArtifact:
        cached = self._cache.get(model_version)
        if cached is not None:
            return cached

        path = self.artifact_path(model_version)
        if not path.is_file():
            raise ModelArtifactNotFoundError(
                f"no artifact for model_version {model_version!r} at {path}"
            )

        try:
            raw = joblib.load(path)
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            ImportError,
            AttributeError,
            pickle.UnpicklingError,
        ) as exc:
            raise ModelArtifactInvalidError(f"{path}: artifact could not be read: {exc}") from exc

        artifact = validate_artifact(raw, requested_version=model_version, path=path)
        self._cache[model_version] = artifact
        logger.info(
            "Loaded C2 artifact model_version=%s family=%s feature_schema_version=%s",
            model_version,
            artifact.model_family,
            artifact.metadata.feature_schema_version,
        )
        return artifact
