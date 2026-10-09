from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform

import sklearn
from sklearn.dummy import DummyClassifier

from backend.app.classifier.artifact import (ModelArtifact, save_artifact)

MODEL_VERSION = "dummy-contract-v1"
MODEL_TYPE = "dummy_classifier"

# define temporary integration-test features
FEATURE_NAMES = [
    "fixture_feature_1",
    "fixture_feature_2",
]

# Dummy integration threshold
DUMMY_THRESHOLD = 1.0

def train_dummy() -> ModelArtifact:
    """
    Train a tiny three-class DummyClassifier used only to prove
    the Classifier artifact/inference/integration pipeline.
    """

    # Tiny artificial fixture dataset.
    X_fixture = [
        [0.0, 0.0],
        [0.1, 0.2],
        [1.0, 1.0],
        [1.1, 1.2],
        [2.0, 2.0],
        [2.1, 2.2],
    ]

    # Two examples of each required classifier categories
    y_fixture = [
        "no_ai",
        "no_ai",
        "ide_ai",
        "ide_ai",
        "external_ai",
        "external_ai",
    ]

    # "prior" ignores the actual feature patterns and simply learns
    # the class proportions.
    model = DummyClassifier(
        strategy="prior",
        random_state=42,
    )

    model.fit(X_fixture, y_fixture)

    artifact = ModelArtifact(
        model=model,
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        schema_version="fixture-schema-v0",
        feature_names=FEATURE_NAMES,

        # Store the estimator's actual probability-column order.
        class_order=[
            str(class_name)
            for class_name in model.classes_
        ],

        threshold=DUMMY_THRESHOLD,

        training_metadata={
            "purpose": "integration_only",
            "research_valid": False,
            "dataset_id": "dummy-fixture-v1",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },
    )

    artifact.validate()

    return artifact

def main() -> None:
    """
    Train and save dummy-contract-v1.
    """

    artifact = train_dummy()

    project_root = Path(__file__).resolve().parents[2]

    output_path = (
        project_root
        / "ml"
        / "models"
        / MODEL_VERSION
        / "classifier.joblib"
    )

    save_artifact(artifact, output_path)

    print("Dummy classifier artifact created successfully.")
    print(f"Saved to: {output_path}")
    print(f"Model version: {artifact.model_version}")
    print(f"Model type: {artifact.model_type}")
    print(f"Model class order: {artifact.class_order}")
    print(f"Feature names: {artifact.feature_names}")
    print(f"Threshold: {artifact.threshold}")

if __name__ == "__main__":
    main()