import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from synthetic.build_dataset import CLASSES, sha
from synthetic import build_dataset
from ml.training.train_experimental import aligned_probabilities


def audit(dataset, models=None, require_generator_match=False):
    from shared_features import FEATURE_NAMES, SCHEMA_VERSION
    from shared_features.feature_schema import FEATURE_SCHEMA, IDENTITY_COLUMNS

    m = json.loads((dataset / "dataset_manifest.json").read_text())
    assert m["holdout_examined"] is False and m["experimental_training_ready"] is True
    assert m["research_validated"] is False
    # Saved-dataset integrity and current-source identity are separate checks.
    recorded_generator_sha256 = m["generator_sha256"]
    current_generator_sha256 = sha(Path(build_dataset.__file__))
    generator_source_matches = current_generator_sha256 == recorded_generator_sha256
    if require_generator_match and not generator_source_matches:
        raise ValueError(
            "Current generator source differs from the dataset provenance. "
            "Exact source identity was explicitly required. "
            f"Dataset generator: {recorded_generator_sha256}; "
            f"current generator: {current_generator_sha256}"
        )
    for name, h in m["output_sha256"].items():
        assert sha(dataset / name) == h, name
    all_X = pd.read_csv(dataset / "all_features.csv")
    w = pd.read_csv(dataset / "windows_and_labels.csv")
    X = pd.read_csv(dataset / "training_features.csv")
    y = pd.read_csv(dataset / "training_labels.csv")
    assert (
        list(all_X.columns)
        == list(X.columns)
        == list(FEATURE_NAMES)
        == m["feature_names"]
    )
    assert (
        len(w) == len(all_X) == m["all_windows"]
        and len(X) == len(y) == m["included_windows"]
    )
    assert set(y["label"]) == set(CLASSES) and w["row_id"].is_unique
    assert np.allclose(
        X.to_numpy(), all_X.loc[w["included"]].to_numpy(), equal_nan=True
    )
    assert y["row_id"].tolist() == w.loc[w["included"], "row_id"].tolist()
    assert y["label"].tolist() == w.loc[w["included"], "label"].tolist()
    assert w.loc[~w["included"], "label"].isna().all()
    assert w.loc[w["included"], "reason"].eq("included_single_known_category").all()
    # Validate the saved all-window values against official C1 schema, including identity columns.
    ids = w[["session_id", "task_id", "window_start", "window_end"]].copy()
    ids["window_start"] = pd.to_datetime(
        ids["window_start"], format="ISO8601", utc=True
    )
    ids["window_end"] = pd.to_datetime(ids["window_end"], format="ISO8601", utc=True)
    ids["run_id"] = "final"
    full = pd.concat([ids[list(IDENTITY_COLUMNS)], all_X], axis=1)
    FEATURE_SCHEMA.validate(full, lazy=False)
    assert (
        all_X["run_error_rate"].isna().all()
        and all_X["runs_before_first_success"].isna().all()
    )
    for raw in (dataset / "raw").glob("*/events.csv"):
        cols = pd.read_csv(raw, nrows=0).columns
        assert not set(
            ["label", "categories", "family_id", "planned_context", "mechanism"]
        ) & set(cols)
    result = {
        "audit": "PASSED",
        "feature_count": len(FEATURE_NAMES),
        "schema_version": SCHEMA_VERSION,
        "included_windows": len(X),
        "excluded_windows_retained": int((~w["included"]).sum()),
        "class_counts": m["class_counts"],
        "holdout_examined": False,
        "dataset_file_hashes_verified": True,
        "dataset_generator_sha256": recorded_generator_sha256,
        "current_generator_sha256": current_generator_sha256,
        "generator_source_matches": generator_source_matches,
        "exact_generator_identity_required": require_generator_match,
        "generator_provenance_note": (
            "Current generator bytes match the dataset provenance."
            if generator_source_matches
            else "Current generator bytes differ. Saved dataset file integrity was "
            "verified independently. Generator equivalence/reproduction was not verified."
        ),
    }
    if models:
        tm = json.loads((models / "training_manifest.json").read_text())
        for name, h in tm["output_sha256"].items():
            assert sha(models / name) == h, name
        membership = pd.read_csv(models / "split_membership.csv")
        assert membership["family_id"].groupby(
            membership["family_id"]
        ).size().sum() == len(y)
        assert membership.groupby("family_id")["split"].nunique().max() == 1
        pipe = joblib.load(models / "selected_experimental_pipeline.joblib")
        trainmask = membership["split"].eq("train").to_numpy()
        expected = X.loc[trainmask].median().fillna(0).to_numpy()
        assert np.allclose(pipe.named_steps["imputer"].statistics_, expected)
        held = membership["split"].eq("test").to_numpy()
        probs = aligned_probabilities(pipe, X.loc[held])
        saved = pd.read_csv(models / "synthetic_test_predictions.csv")
        assert np.allclose(
            probs, saved[["p_" + c for c in CLASSES]].to_numpy(), atol=1e-12
        )
        assert np.allclose(probs.sum(axis=1), 1)
        result.update(
            {
                "training_preprocessing_fit_only": True,
                "persona_split_leakage": False,
                "selected_model": tm["model_type"],
                "synthetic_test_macro_f1": tm["synthetic_test_macro_f1"],
                "estimator_roundtrip_passed": True,
                "runtime_verification_included_in_this_audit": False,
                "TSTR_completed": False,
            }
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--models", type=Path)
    p.add_argument(
        "--require-generator-match",
        action="store_true",
        help="Also require exact generator source bytes; optional for saved-dataset reuse.",
    )
    a = p.parse_args()
    print(json.dumps(audit(a.dataset, a.models, a.require_generator_match), indent=2))
