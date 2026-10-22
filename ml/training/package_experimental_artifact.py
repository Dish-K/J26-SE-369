from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_modules(runtime_root, module_prefix, shared_root):
    # Import actual files, with no changes to their source or contract.
    require(
        not any(
            k == module_prefix or k.startswith(module_prefix + ".") for k in sys.modules
        ),
        "Use a fresh Python process.",
    )
    sys.path[:0] = [str(Path(runtime_root).resolve()), str(Path(shared_root).resolve())]
    art = importlib.import_module(module_prefix + ".artifact")
    inf = importlib.import_module(module_prefix + ".inference")
    official = importlib.import_module("shared_features")
    require(
        Path(art.__file__).resolve().is_relative_to(Path(runtime_root).resolve()),
        "Artifact module did not come from the specified runtime root.",
    )
    require(
        Path(inf.__file__).resolve().is_relative_to(Path(runtime_root).resolve()),
        "Inference module did not come from the specified runtime root.",
    )
    require(
        Path(official.__file__).resolve().is_relative_to(Path(shared_root).resolve()),
        "Official feature module did not come from the specified shared root.",
    )
    return art, inf, official


def validate_contract(artifact, contract, official):
    require(
        artifact.feature_names
        == contract["feature_names"]
        == list(official.FEATURE_NAMES),
        "Feature order differs from the unchanged official 40-feature contract.",
    )
    require(len(artifact.feature_names) == 40, "Expected exactly 40 features.")
    require(
        artifact.schema_version
        == contract["schema_version"]
        == official.SCHEMA_VERSION,
        "Feature schema version mismatch.",
    )


def export(inputs, dataset, output, art, official):
    manifest = read(inputs / "training_manifest.json")
    contract = {
        "schema_version": manifest["schema_version"],
        "feature_names": manifest["feature_names"],
    }
    threshold = read(inputs / "threshold_selection.json")
    freeze = read(inputs / "selection_frozen_before_test.json")
    model_path = inputs / "selected_experimental_pipeline.joblib"
    require(
        sha(model_path) == manifest["output_sha256"][model_path.name],
        "Selected pipeline hash mismatch.",
    )
    require(
        sha(dataset / "dataset_manifest.json") == manifest["dataset_manifest_sha256"],
        "Dataset identity mismatch.",
    )
    require(
        not manifest["test_used_for_selection"] and not threshold["test_data_used"],
        "Selection must precede test evaluation.",
    )
    model = joblib.load(model_path)
    classes = [str(v) for v in model.classes_]
    require(classes == manifest["class_order"], "Estimator class order mismatch.")
    require(
        manifest["threshold"] == threshold["selected"]["threshold"],
        "Threshold mismatch.",
    )
    metadata = {
        "experimental_only": True,
        "real_participant_validated": False,
        "reliable_mixed_behavior_detection": False,
        "training_manifest": manifest,
        "dataset_manifest": read(dataset / "dataset_manifest.json"),
        "threshold_selection": threshold,
        "selection_frozen_before_test": freeze,
        "pipeline_sha256": sha(model_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "runtime_artifact_module": art.__name__,
        "source_artifact_py_sha256": sha(art.__file__),
        "verification_status": "PENDING_SEPARATE_LOADER_INFERENCE_CHECK",
    }
    artifact = art.ModelArtifact(
        model=model,
        model_version=manifest["model_version"],
        model_type=manifest["model_type"],
        schema_version=manifest["schema_version"],
        feature_names=manifest["feature_names"],
        class_order=classes,
        threshold=manifest["threshold"],
        training_metadata=metadata,
    )
    validate_contract(artifact, contract, official)
    require(not output.exists(), "Output exists; choose a new versioned path.")
    art.save_artifact(artifact, output)
    print("Exported:", output.resolve())
    print(
        "No runtime configuration was changed. Run the classifier smoke check in a new process."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export the existing trained pipeline through the actual C2 artifact contract."
    )
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("backend"))
    parser.add_argument("--shared-root", type=Path, default=Path("."))
    parser.add_argument("--module-prefix", default="app.classifier")
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    art, _, official = runtime_modules(
        args.runtime_root, args.module_prefix, args.shared_root
    )
    export(args.training_run, args.dataset, args.artifact, art, official)


if __name__ == "__main__":
    main()
