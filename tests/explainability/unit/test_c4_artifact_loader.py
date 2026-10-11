from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from backend.app.explainability.errors import (
    ClassOrderMismatchError,
    ModelArtifactInvalidError,
    ModelArtifactNotFoundError,
    ModelVersionMismatchError,
)
from backend.app.explainability.model.artifact_loader import ARTIFACT_FILENAME, ArtifactLoader
from tests.explainability.fixtures.prototype_c2 import (
    CLASS_ORDER,
    FEATURE_NAMES,
    MODEL_VERSION,
    PROTOTYPE_THRESHOLD,
    SCHEMA_VERSION,
    artifact_dict,
    write_artifact,
)


def test_existing_artifact_loads_with_metadata_preserved(artifact_dir: Path) -> None:
    artifact = ArtifactLoader(artifact_dir).load(MODEL_VERSION)
    metadata = artifact.metadata
    assert metadata.model_version == MODEL_VERSION
    assert metadata.feature_schema_version == SCHEMA_VERSION
    assert metadata.feature_names == FEATURE_NAMES
    assert metadata.class_order == CLASS_ORDER
    assert metadata.classification_threshold == PROTOTYPE_THRESHOLD
    assert metadata.training_dataset_version == "synthetic-prototype-fixture"
    assert artifact.model_family == "LogisticRegression"


def test_output_classes_follow_the_fitted_model_not_class_order(artifact_dir: Path) -> None:
    # sklearn sorts string labels; class_order is the documented C2 order.
    artifact = ArtifactLoader(artifact_dir).load(MODEL_VERSION)
    assert artifact.output_classes == ("external_ai", "ide_ai", "no_ai")
    assert artifact.metadata.class_order == ("no_ai", "ide_ai", "external_ai")


def test_loader_caches_by_version(artifact_dir: Path) -> None:
    loader = ArtifactLoader(artifact_dir)
    assert loader.load(MODEL_VERSION) is loader.load(MODEL_VERSION)


def test_missing_artifact_raises(artifact_dir: Path) -> None:
    with pytest.raises(ModelArtifactNotFoundError, match="synthetic-prototype-v9"):
        ArtifactLoader(artifact_dir).load("synthetic-prototype-v9")


@pytest.mark.parametrize("version", ["../escape", "a/b", "", ".hidden"])
def test_unsafe_model_version_is_rejected(artifact_dir: Path, version: str) -> None:
    with pytest.raises(ModelArtifactNotFoundError):
        ArtifactLoader(artifact_dir).load(version)


def test_model_family_is_derived_when_absent(tmp_path: Path, prototype_model: Any) -> None:
    write_artifact(tmp_path, artifact_dict(prototype_model, model_family=None))
    assert ArtifactLoader(tmp_path).load(MODEL_VERSION).model_family == "LogisticRegression"


def test_declared_version_must_match_requested(tmp_path: Path, prototype_model: Any) -> None:
    raw = artifact_dict(prototype_model, model_version="other-version")
    path = tmp_path / MODEL_VERSION / ARTIFACT_FILENAME
    path.parent.mkdir(parents=True)
    joblib.dump(raw, path)
    with pytest.raises(ModelVersionMismatchError):
        ArtifactLoader(tmp_path).load(MODEL_VERSION)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.pop("model"),
        lambda a: a.pop("feature_schema_version"),
        lambda a: a.pop("classification_threshold"),
        lambda a: a.update(feature_names=list(reversed(FEATURE_NAMES))),  # fitted order differs
        lambda a: a.update(feature_names=list(FEATURE_NAMES[:-1])),
        lambda a: a.update(model=object()),  # no predict_proba
    ],
)
def test_malformed_artifact_fails_clearly(
    tmp_path: Path, prototype_model: Any, mutate: Any
) -> None:
    raw = artifact_dict(prototype_model)
    mutate(raw)
    path = tmp_path / MODEL_VERSION / ARTIFACT_FILENAME
    path.parent.mkdir(parents=True)
    joblib.dump(raw, path)
    with pytest.raises(ModelArtifactInvalidError):
        ArtifactLoader(tmp_path).load(MODEL_VERSION)


def test_unreadable_artifact_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / MODEL_VERSION / ARTIFACT_FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a joblib file")
    with pytest.raises(ModelArtifactInvalidError):
        ArtifactLoader(tmp_path).load(MODEL_VERSION)


def test_model_classes_must_be_the_artifact_classes(tmp_path: Path) -> None:
    model = LogisticRegression().fit(
        np.arange(30, dtype=float).reshape(10, 3), ["no_ai"] * 4 + ["ide_ai"] * 3 + ["other"] * 3
    )
    raw = artifact_dict(model, feature_names=["a", "b", "c"])
    write_artifact(tmp_path, raw)
    with pytest.raises(ClassOrderMismatchError):
        ArtifactLoader(tmp_path).load(MODEL_VERSION)


def test_integer_labels_are_named_by_class_order(tmp_path: Path) -> None:
    model = LogisticRegression().fit(
        np.arange(30, dtype=float).reshape(10, 3), [0] * 4 + [1] * 3 + [2] * 3
    )
    raw = artifact_dict(
        model, feature_names=["a", "b", "c"], class_order=["ide_ai", "external_ai", "no_ai"]
    )
    write_artifact(tmp_path, raw)
    artifact = ArtifactLoader(tmp_path).load(MODEL_VERSION)
    assert artifact.output_classes == ("ide_ai", "external_ai", "no_ai")


def test_integer_labels_must_be_contiguous(tmp_path: Path) -> None:
    model = LogisticRegression().fit(
        np.arange(30, dtype=float).reshape(10, 3), [0] * 4 + [1] * 3 + [5] * 3
    )
    write_artifact(tmp_path, artifact_dict(model, feature_names=["a", "b", "c"]))
    with pytest.raises(ClassOrderMismatchError):
        ArtifactLoader(tmp_path).load(MODEL_VERSION)
