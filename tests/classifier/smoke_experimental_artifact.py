from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

# Support the same direct-script invocation as the existing classifier smoke tools.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
from ml.training.package_experimental_artifact import (
    read,
    require,
    runtime_modules,
    sha,
    validate_contract,
)


def reference_windows(dataset, expected):
    with (dataset / "training_labels.csv").open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    with (dataset / "training_features.csv").open(newline="") as handle:
        features = list(csv.DictReader(handle))
    require(len(labels) == len(features), "Feature/label row count mismatch.")
    require(
        len({row["row_id"] for row in labels}) == len(labels),
        "Duplicate row identities.",
    )
    indexed = {
        row["row_id"]: values for row, values in zip(labels, features, strict=True)
    }
    return [
        {
            "row_id": row["row_id"],
            "features": {
                name: None if value == "" else float(value)
                for name, value in indexed[row["row_id"]].items()
            },
        }
        for row in expected
    ]


class FixedProbabilities:
    """In-memory decision-boundary probe, never stored in a model artifact."""

    classes_ = np.array(["external_ai", "ide_ai", "no_ai"])

    def __init__(self, row):
        self.row = row

    def predict_proba(self, matrix):
        return np.tile(self.row, (len(matrix), 1))


def verify(inputs, dataset, output, report_path, art, inf, official):
    artifact = art.load_artifact(output)
    manifest = read(inputs / "training_manifest.json")
    with (inputs / "synthetic_test_predictions.csv").open(newline="") as handle:
        expected = list(csv.DictReader(handle))
    windows = reference_windows(dataset, expected)
    contract = {
        "schema_version": manifest["schema_version"],
        "feature_names": manifest["feature_names"],
    }
    checks = []

    def checked(name, operation):
        operation()
        checks.append(name)

    checked(
        "actual_loader_returns_ModelArtifact",
        lambda: require(
            isinstance(artifact, art.ModelArtifact), "Wrong artifact type."
        ),
    )
    checked(
        "official_40_feature_schema_and_order",
        lambda: validate_contract(artifact, contract, official),
    )
    checked(
        "frozen_version_threshold_and_estimator_class_order",
        lambda: require(
            artifact.model_version == manifest["model_version"]
            and artifact.threshold == manifest["threshold"]
            and artifact.class_order
            == list(artifact.model.classes_)
            == manifest["class_order"],
            "Version, threshold, or estimator class order mismatch.",
        ),
    )

    original = joblib.load(inputs / "selected_experimental_pipeline.joblib")
    values = [
        [
            np.nan if row["features"][n] is None else row["features"][n]
            for n in artifact.feature_names
        ]
        for row in windows
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names.*"
        )
        before = original.predict_proba(values)
        after = artifact.model.predict_proba(values)
    checked(
        "serialized_pipeline_preserves_all_probabilities",
        lambda: np.testing.assert_allclose(before, after, rtol=0, atol=0),
    )
    require(len(windows) == len(expected), "Window/reference count mismatch.")
    predictions = []
    examples = []
    for window, reference in zip(windows, expected, strict=True):
        require(window["row_id"] == reference["row_id"], "Reference row misalignment.")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names.*"
            )
            p = inf.predict_window(
                artifact=artifact,
                feature_vector_id=uuid5(NAMESPACE_URL, window["row_id"]),
                task_id=uuid5(NAMESPACE_URL, reference["family_id"]),
                features=window["features"],
            )
        require(
            list(p.class_probabilities) == list(art.CLASS_ORDER),
            "Output class order mismatch.",
        )
        for name, value in p.class_probabilities.items():
            require(
                abs(value - float(reference["p_" + name])) <= 1e-6,
                "Runtime probabilities differ from saved test predictions.",
            )
        require(
            p.predicted_category == reference["predicted_category"], "Category changed."
        )
        require(
            abs(p.confidence - float(reference["confidence"])) <= 1e-6,
            "Confidence changed.",
        )
        require(
            p.ai_positive == (reference["ai_positive"].lower() == "true"),
            "Flag changed.",
        )
        predictions.append(p)
        if len(examples) < 6:
            examples.append(
                {
                    "row_id": window["row_id"],
                    "predicted_category": p.predicted_category,
                    "class_probabilities": p.class_probabilities,
                    "confidence": p.confidence,
                    "ai_positive": p.ai_positive,
                }
            )
    checks.append("actual_predict_window_matches_all_259_saved_predictions")

    def predict(features, target=artifact):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names.*"
            )
            return inf.predict_window(
                artifact=target,
                feature_vector_id=uuid5(NAMESPACE_URL, "probe"),
                task_id=uuid5(NAMESPACE_URL, "task"),
                features=features,
            )

    features = windows[0]["features"]
    checked(
        "mapping_key_order_does_not_change_model_feature_order",
        lambda: require(
            predict(dict(reversed(list(features.items())))).class_probabilities
            == predictions[0].class_probabilities,
            "Mapping order changed prediction.",
        ),
    )

    def rejects(operation):
        try:
            operation()
        except ValueError:
            return
        raise ValueError("Invalid input was accepted.")

    checked(
        "missing_feature_rejected",
        lambda: rejects(
            lambda: predict(
                {k: v for k, v in features.items() if k != artifact.feature_names[0]}
            )
        ),
    )
    checked(
        "unexpected_feature_rejected",
        lambda: rejects(lambda: predict(dict(features, not_a_feature=1))),
    )
    invalid = copy.copy(artifact)
    invalid.class_order = list(reversed(artifact.class_order))
    checked(
        "incorrect_estimator_probability_order_rejected",
        lambda: rejects(invalid.validate),
    )
    invalid_order = copy.copy(artifact)
    invalid_order.feature_names = list(reversed(artifact.feature_names))
    checked(
        "incorrect_artifact_feature_order_rejected_by_contract_check",
        lambda: rejects(lambda: validate_contract(invalid_order, contract, official)),
    )

    def boundary_checks():
        probe = copy.copy(artifact)
        probe.class_order = list(FixedProbabilities.classes_)
        probe.threshold = 0.5
        for probs, category, flag in [
            ([0.5, 0.3, 0.2], "external_ai", True),
            ([0.49, 0.3, 0.21], "external_ai", False),
            ([0.2, 0.3, 0.5], "no_ai", False),
            ([0.5, 0, 0.5], "no_ai", False),
        ]:
            probe.model = FixedProbabilities(probs)
            p = predict(features, probe)
            require(
                p.predicted_category == category and p.ai_positive == flag,
                "Existing decision boundary or canonical tie break changed.",
            )

    checked("actual_inference_threshold_boundary_no_ai_and_tie_rules", boundary_checks)
    checked(
        "null_features_accepted_by_fitted_imputer",
        lambda: require(
            any(v is None for row in windows for v in row["features"].values()),
            "No null-feature cases were exercised.",
        ),
    )

    report = {
        "status": "EXPERIMENTAL_C2_ARTIFACT_EXPORTED_AND_SUPPLIED_RUNTIME_VERIFIED",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks_passed": len(checks),
        "checks": checks,
        "windows_verified": len(windows),
        "artifact_sha256": sha(output),
        "model_version": artifact.model_version,
        "threshold": artifact.threshold,
        "estimator_class_order": artifact.class_order,
        "output_class_order": list(art.CLASS_ORDER),
        "feature_count": len(artifact.feature_names),
        "schema_version": artifact.schema_version,
        "artifact_module": art.__name__,
        "inference_module": inf.__name__,
        "artifact_source_sha256": sha(art.__file__),
        "inference_source_sha256": sha(inf.__file__),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "joblib",
                "pandera",
            ]
        },
        "retrained": False,
        "test_used_for_new_model_or_threshold_selection": False,
        "database_or_service_integration_verified": False,
        "TSTR_completed": False,
        "robustness_evaluation_completed": False,
        "real_participant_accuracy_established": False,
        "reliable_mixed_behavior_detection_established": False,
        "prediction_examples": examples,
    }
    require(not report_path.exists(), "Report exists; choose a new report path.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "checks_passed",
                    "windows_verified",
                    "model_version",
                    "threshold",
                ]
            },
            indent=2,
        )
    )
    print("Verification report:", report_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("backend"))
    parser.add_argument("--shared-root", type=Path, default=Path("."))
    parser.add_argument("--module-prefix", default="app.classifier")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    art, inf, official = runtime_modules(
        args.runtime_root, args.module_prefix, args.shared_root
    )
    verify(
        args.training_run, args.dataset, args.artifact, args.report, art, inf, official
    )


if __name__ == "__main__":
    main()
