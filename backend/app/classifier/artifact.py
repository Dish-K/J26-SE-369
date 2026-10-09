from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

CLASS_ORDER = ("no_ai", "ide_ai", "external_ai")
CLASS_SET = frozenset(CLASS_ORDER)


@dataclass
class ModelArtifact:
    """
    Complete package required to reproduce one classifier model version.

    The trained estimator alone is not enough. The artifact also records 
    the feature contract, class order, threshold, and training provenance.
    """

    model: Any

    model_version: str
    model_type: str

    schema_version: str
    feature_names: list[str]

    class_order: list[str]

    threshold: float

    training_metadata: dict[str, Any]

    def validate(self) -> None:
        """
        Fail early if an invalid artifact is created or loaded.
        """

        if not self.model_version.strip():
            raise ValueError("model_version must not be empty")

        if not self.model_type.strip():
            raise ValueError("model_type must not be empty")

        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")

        if not self.feature_names:
            raise ValueError("feature_names must not be empty")

        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must not contain duplicates")

        if len(self.class_order) != 3:
            raise ValueError("class_order must contain exactly three classes")

        if len(self.class_order) != len(set(self.class_order)):
            raise ValueError("class_order must not contain duplicate classes")

        if set(self.class_order) != CLASS_SET:
            raise ValueError("class_order must contain exactly:" "no_ai, ide_ai, external_ai")

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("classifier model must provide predict_proba()")

        if hasattr(self.model, "classes_"):
            estimator_classes = [str(value) for value in self.model.classes_]

            if estimator_classes != self.class_order:
                raise ValueError("artifact class_order does not match " f"model.classes_: {estimator_classes}")


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    """
    Validate and save a classifier artifact.
    """

    artifact.validate()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, output_path)


def load_artifact(path: str | Path) -> ModelArtifact:
    """
    Load and validate a classifier artifact.
    """

    artifact = joblib.load(path)

    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Loaded file is not a valid ModelArtifact")

    artifact.validate()

    return artifact