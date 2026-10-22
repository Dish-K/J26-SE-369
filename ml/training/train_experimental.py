from __future__ import annotations
import argparse, hashlib, json, time, platform
from datetime import datetime, timezone
import sklearn
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from synthetic.build_dataset import CLASSES, save_json, sha


def split_groups(labels, seed, fractions):
    groups = sorted(labels["family_id"].unique())
    if len(groups) < 10:
        raise ValueError("At least 10 persona families required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * fractions[0])
    n_val = max(1, int(n * fractions[1]))
    parts = {
        "train": set(groups[:n_train]),
        "validation": set(groups[n_train : n_train + n_val]),
        "test": set(groups[n_train + n_val :]),
    }
    if any(not x for x in parts.values()):
        raise ValueError("Empty split")
    if any(
        parts[a] & parts[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Group leakage")
    rows = {
        name: np.flatnonzero(labels["family_id"].isin(g).to_numpy())
        for name, g in parts.items()
    }
    for name, idx in rows.items():
        if set(labels.iloc[idx]["label"]) != set(CLASSES):
            raise ValueError(f"{name}: missing class; use more persona families")
    return rows, parts


def aligned_probabilities(model, X):
    p = model.predict_proba(X)
    order = list(model.classes_)
    return p[:, [order.index(c) for c in CLASSES]]


def probability_metrics(y, p):
    truth = np.array([[int(label == c) for c in CLASSES] for label in y])
    idx = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = np.asarray(CLASSES)[idx] == np.asarray(y)
    bins = []
    ece = 0.0
    for left in np.linspace(0, 0.9, 10):
        right = left + 0.1
        mask = np.minimum((conf * 10).astype(int), 9) == int(round(left * 10))
        if mask.any():
            acc = float(correct[mask].mean())
            mean = float(conf[mask].mean())
            n = int(mask.sum())
            ece += n / len(y) * abs(acc - mean)
            bins.append(
                {
                    "lower": float(left),
                    "upper": float(right),
                    "n": n,
                    "accuracy": acc,
                    "mean_confidence": mean,
                }
            )
    return {
        "log_loss": float(
            -np.mean(np.sum(truth * np.log(np.clip(p, 1e-15, 1.0)), axis=1))
        ),
        "multiclass_brier_sum": float(np.mean(np.sum((p - truth) ** 2, axis=1))),
        "confidence_ece_10_bins": float(ece),
        "confidence_bins": bins,
        "probability_calibration_applied": False,
        "interpretation": "Synthetic-data probability diagnostics only; deployment confidence not validated.",
    }


def classification_metrics(y, p):
    pred = np.asarray(CLASSES)[p.argmax(axis=1)]
    report = classification_report(
        y, pred, labels=list(CLASSES), output_dict=True, zero_division=0
    )
    truth = np.asarray(y)
    clean = truth == "no_ai"
    return {
        "macro_f1": float(
            f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(CLASSES)).tolist(),
        "class_order": list(CLASSES),
        "unaided_winning_class_false_positive_rate": (
            float(np.mean(pred[clean] != "no_ai")) if clean.any() else None
        ),
        "probability": probability_metrics(y, p),
    }


def threshold_search(y, p, grid):
    pred = np.asarray(CLASSES)[p.argmax(axis=1)]
    conf = p.max(axis=1)
    truth = np.asarray(y) != "no_ai"
    clean = ~truth
    rows = []
    for t in grid:
        flagged = (pred != "no_ai") & (conf >= t)
        rows.append(
            {
                "threshold": float(t),
                "ai_flag_f1": float(f1_score(truth, flagged, zero_division=0)),
                "unaided_flag_false_positive_rate": (
                    float(flagged[clean].mean()) if clean.any() else None
                ),
                "flag_count": int(flagged.sum()),
            }
        )
    # Validation-only engineering criterion, never a runtime rule change.
    best = max(
        rows,
        key=lambda r: (
            r["ai_flag_f1"],
            -(r["unaided_flag_false_positive_rate"] or 0),
            r["threshold"],
        ),
    )
    return best, rows


def candidates(seed):
    from xgboost import XGBClassifier

    imputer = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    rows = [
        (
            "majority_baseline",
            {},
            Pipeline(
                [
                    ("imputer", imputer()),
                    ("model", DummyClassifier(strategy="most_frequent")),
                ]
            ),
        )
    ]
    for c in [0.3, 1.0]:
        rows.append(
            (
                "logistic_regression",
                {"C": c},
                Pipeline(
                    [
                        ("imputer", imputer()),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c,
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    for depth in [6, None]:
        rows.append(
            (
                "random_forest",
                {"max_depth": depth},
                Pipeline(
                    [
                        ("imputer", imputer()),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=120,
                                max_depth=depth,
                                class_weight="balanced",
                                random_state=seed,
                                n_jobs=2,
                            ),
                        ),
                    ]
                ),
            )
        )
    for depth in [3, 5]:
        # Native sklearn VotingClassifier performs label encoding and exposes string classes.
        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=depth,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=2,
        )
        rows.append(
            (
                "xgboost",
                {"max_depth": depth},
                Pipeline(
                    [
                        ("imputer", imputer()),
                        (
                            "model",
                            VotingClassifier(
                                estimators=[("xgb", xgb)], voting="soft", n_jobs=1
                            ),
                        ),
                    ]
                ),
            )
        )
    return rows


def train(dataset, output, seed=160420):
    if output.exists():
        raise FileExistsError("Choose a fresh training output folder")
    m = json.loads((dataset / "dataset_manifest.json").read_text())
    if not m["experimental_training_ready"] or m["research_validated"]:
        raise ValueError("Unexpected experimental dataset status")
    for name, h in m["output_sha256"].items():
        if sha(dataset / name) != h:
            raise ValueError(f"Dataset changed: {name}")
    from shared_features import FEATURE_NAMES, SCHEMA_VERSION

    X = pd.read_csv(dataset / "training_features.csv")
    labels = pd.read_csv(dataset / "training_labels.csv")
    if (
        list(X.columns) != list(FEATURE_NAMES)
        or list(X.columns) != m["feature_names"]
        or SCHEMA_VERSION != "1.0.0"
    ):
        raise ValueError("Wrong input feature contract")
    if len(X) != len(labels) or labels["row_id"].duplicated().any():
        raise ValueError("Row alignment invalid")
    # Detect literal duplicated traces before allowing them to cross a split.
    # Repeated numeric vectors are legitimate windows and are not themselves source trace identities.
    config = json.loads((dataset / "generator_configuration.json").read_text())
    idx, groups = split_groups(labels, seed, config["training_split_fractions"])
    y = labels["label"]
    output.mkdir(parents=True)
    split_frame = labels.copy()
    split_frame["split"] = ""
    for name, rows in idx.items():
        split_frame.loc[rows, "split"] = name
    split_frame.to_csv(output / "split_membership.csv", index=False)
    save_json(
        output / "split_manifest.json",
        {
            "seed": seed,
            "groups": {k: sorted(v) for k, v in groups.items()},
            "rows": {k: len(v) for k, v in idx.items()},
            "same_split_for_all_models": True,
            "split_rule": "one entire simulated persona family, including category variants, per split",
        },
    )
    train_idx, val_idx, test_idx = idx["train"], idx["validation"], idx["test"]
    metrics = []
    best = None
    best_key = None
    all_rows = []
    for n, (kind, params, model) in enumerate(candidates(seed)):
        start = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = aligned_probabilities(model, X.iloc[val_idx])
        result = classification_metrics(y.iloc[val_idx], p)
        cid = f"{kind}-{n:02d}"
        record = {
            "candidate_id": cid,
            "model_type": kind,
            "parameters": params,
            "fit_seconds": time.perf_counter() - start,
            "validation": result,
        }
        metrics.append(record)
        all_rows.append(
            {
                "candidate_id": cid,
                "model_type": kind,
                "parameters": json.dumps(params),
                "validation_macro_f1": result["macro_f1"],
                "validation_log_loss": result["probability"]["log_loss"],
            }
        )
        joblib.dump(model, output / f"{cid}.joblib")
        print(f'{cid}: validation macro-F1={result["macro_f1"]:.4f}', flush=True)
        if kind != "majority_baseline":
            key = (result["macro_f1"], -result["probability"]["log_loss"])
            if best_key is None or key > best_key:
                best_key = key
                best = (cid, kind, params, model, p)
    save_json(output / "validation_comparison.json", metrics)
    pd.DataFrame(all_rows).to_csv(output / "validation_comparison.csv", index=False)
    cid, kind, params, model, p_val = best
    threshold, grid = threshold_search(y.iloc[val_idx], p_val, config["threshold_grid"])
    save_json(
        output / "threshold_selection.json",
        {
            "criterion": "maximize validation AI-flag F1; ties prefer lower unaided false positives, then higher threshold",
            "runtime_rule": "winning class in ide_ai/external_ai AND winning probability >= threshold",
            "selected": threshold,
            "validation_grid": grid,
            "test_data_used": False,
        },
    )
    # Freeze selection before accessing the held-out synthetic test labels/features.
    selection = {
        "candidate_id": cid,
        "model_type": kind,
        "parameters": params,
        "threshold": threshold["threshold"],
        "selection_basis": "validation macro-F1; ties prefer validation log loss",
        "test_used_for_selection": False,
        "class_order": list(model.classes_),
        "feature_names": list(FEATURE_NAMES),
        "schema_version": SCHEMA_VERSION,
    }
    save_json(output / "selection_frozen_before_test.json", selection)
    p_test = aligned_probabilities(model, X.iloc[test_idx])
    test = classification_metrics(y.iloc[test_idx], p_test)
    save_json(output / "synthetic_test_report.json", test)
    pred = np.asarray(CLASSES)[p_test.argmax(axis=1)]
    conf = p_test.max(axis=1)
    pred_rows = labels.iloc[test_idx].copy()
    pred_rows["predicted_category"] = pred
    pred_rows["confidence"] = conf
    pred_rows["ai_positive"] = (pred != "no_ai") & (conf >= threshold["threshold"])
    for j, c in enumerate(CLASSES):
        pred_rows["p_" + c] = p_test[:, j]
    pred_rows.to_csv(output / "synthetic_test_predictions.csv", index=False)
    selected_path = output / "selected_experimental_pipeline.joblib"
    joblib.dump(model, selected_path)
    reloaded = joblib.load(selected_path)
    if not np.allclose(
        aligned_probabilities(reloaded, X.iloc[test_idx]), p_test, atol=1e-12
    ):
        raise ValueError("Estimator serialization roundtrip failed")
    # Retain excluded test-family predictions without assigning invented truth labels.
    windows = pd.read_csv(dataset / "windows_and_labels.csv")
    all_X = pd.read_csv(dataset / "all_features.csv")
    excluded = (~windows["included"]) & windows["family_id"].isin(groups["test"])
    if excluded.any():
        p_ex = aligned_probabilities(model, all_X.loc[excluded])
        out = windows.loc[excluded].copy()
        out["predicted_category"] = np.asarray(CLASSES)[p_ex.argmax(axis=1)]
        out["confidence"] = p_ex.max(axis=1)
        for j, c in enumerate(CLASSES):
            out["p_" + c] = p_ex[:, j]
        out.to_csv(
            output / "excluded_test_windows_unscored_predictions.csv", index=False
        )
    manifest = {
        "status": "EXPERIMENTAL_SYNTHETIC_TRAINING_COMPLETE_RUNTIME_HANDOFF_PENDING",
        "model_version": "experimental-synthetic-v1-"
        + sha(dataset / "dataset_manifest.json")[:10],
        "dataset_manifest_sha256": sha(dataset / "dataset_manifest.json"),
        "dataset_version": m["dataset_version"],
        "label_policy": m["label_policy"],
        **selection,
        "synthetic_test_macro_f1": test["macro_f1"],
        "estimator_serialization_roundtrip": True,
        "current_runtime_loader_verified": False,
        "runtime_artifact_exported": False,
        "TSTR_completed": False,
        "robustness_evaluation_completed": False,
        "reliable_mixed_behavior_detection_claimed": False,
        "all_missing_training_features": list(
            X.iloc[train_idx].columns[X.iloc[train_idx].isna().all()]
        ),
        "constant_training_features": list(
            X.iloc[train_idx].columns[X.iloc[train_idx].nunique(dropna=False) <= 1]
        ),
        "probability_calibration_applied": False,
        "trainer_sha256": sha(Path(__file__)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_metadata": {
            "seed": seed,
            "train_only_preprocessing": True,
            "metadata_in_model_inputs": False,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
                "xgboost": __import__("xgboost").__version__,
            },
        },
        "runtime_handoff_note": "The experimental pipeline is not a verified C2 runtime artifact. Export through the current artifact contract and verify with the actual loader/inference source before integration.",
        "limitations": m["limitations"],
        "output_sha256": {
            p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
        },
    }
    save_json(output / "training_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=160420)
    a = p.parse_args()
    result = train(a.dataset, a.output, a.seed)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "model_type",
                    "synthetic_test_macro_f1",
                    "threshold",
                    "current_runtime_loader_verified",
                    "TSTR_completed",
                ]
            },
            indent=2,
        )
    )
