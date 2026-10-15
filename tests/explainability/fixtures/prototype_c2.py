"""PROTOTYPE C2 stand-in for C4 Phase 1 tests. NOT a research classifier.

C2 has not delivered its artifact or persisted outputs yet. This module:

1. packages a small synthetic-prototype Logistic Regression Pipeline in the
   C2 artifact format (C2 implementation plan §13);
2. emulates what C2 persists for the scenarios in ``c2_scenarios.json``:
   per-window predictions (argmax category, winning-class confidence,
   threshold-based ``ai_positive``), flags whose representative is the
   highest-confidence member and whose probabilities are copied unchanged,
   and membership rows;
3. serves those rows through an in-memory ``C2OutputReader``.

The training rows are random draws around hand-picked class centres. They
exist only to make the pipeline produce confident, distinct probabilities.
Nothing here is evidence about real coding behaviour.
"""

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.explainability.errors import (
    FeatureVectorNotFoundError,
    FlagNotFoundError,
    PredictionNotFoundError,
)
from backend.app.explainability.model.artifact_loader import ARTIFACT_FILENAME
from backend.app.explainability.persistence.repository import parse_c2_row
from backend.app.explainability.schemas import (
    BackgroundReference,
    ClassifierFlag,
    FeatureVector,
    WindowPrediction,
)

FIXTURE_DIR = Path(__file__).parent
SCENARIO_PATH = FIXTURE_DIR / "c2_scenarios.json"
REGISTRY_PATH = FIXTURE_DIR / "feature_registry.provisional.yaml"

MODEL_VERSION = "synthetic-prototype-v0"
SCHEMA_VERSION = "contract-a-provisional-v0"
# Temporary integration value only; the real threshold is C2's, chosen
# empirically and shipped inside its artifact.
PROTOTYPE_THRESHOLD = 0.7
CLASS_ORDER = ("no_ai", "ide_ai", "external_ai")

FEATURE_NAMES = (
    "keystroke_count",
    "mean_dwell_ms",
    "std_dwell_ms",
    "mean_flight_ms",
    "std_flight_ms",
    "backspace_rate",
    "paste_ratio",
    "inserted_len_max",
    "pause_ms",
    "focus_switch_count",
)

# (mean, sd) per feature per class, in FEATURE_NAMES order.
_CLASS_CENTRES: dict[str, list[tuple[float, float]]] = {
    "no_ai": [(90, 20), (110, 15), (35, 8), (260, 50), (140, 30),
              (0.12, 0.04), (0.01, 0.02), (4, 2), (4000, 1500), (0.3, 0.5)],
    "ide_ai": [(50, 15), (104, 15), (30, 8), (215, 45), (100, 25),
               (0.03, 0.02), (0.04, 0.03), (40, 12), (2800, 1000), (0.3, 0.5)],
    "external_ai": [(30, 12), (105, 15), (31, 8), (290, 60), (150, 35),
                    (0.02, 0.02), (0.5, 0.15), (300, 80), (17000, 4000), (3, 1.5)],
}
_FRACTIONS = {"backspace_rate", "paste_ratio"}


def _training_frame(seed: int = 20261009, per_class: int = 200) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    columns: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    labels: list[str] = []
    for label, centres in _CLASS_CENTRES.items():
        for name, (mean, sd) in zip(FEATURE_NAMES, centres, strict=True):
            values = rng.normal(mean, sd, per_class)
            values = np.clip(values, 0.0, 1.0 if name in _FRACTIONS else None)
            columns[name].extend(values.tolist())
        labels.extend([label] * per_class)
    return pd.DataFrame(columns, columns=list(FEATURE_NAMES)), labels


def build_prototype_model() -> Pipeline:
    frame, labels = _training_frame()
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("classify", LogisticRegression(max_iter=2000))]
    )
    return pipeline.fit(frame, labels)


def artifact_dict(model: Any, **overrides: Any) -> dict[str, Any]:
    artifact = {
        "model": model,
        "model_version": MODEL_VERSION,
        "model_family": "LogisticRegression",
        "feature_schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "class_order": list(CLASS_ORDER),
        "classification_threshold": PROTOTYPE_THRESHOLD,
        "training_dataset_version": "synthetic-prototype-fixture",
        "generator_version": None,
        "training_timestamp": "2026-10-09T00:00:00Z",
    }
    artifact.update(overrides)
    return artifact


def write_artifact(artifact_dir: Path, artifact: dict[str, Any]) -> Path:
    path = artifact_dir / artifact["model_version"] / ARTIFACT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_scenarios() -> dict[str, Any]:
    return json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))


@dataclass
class C2Records:
    """What C2 would have persisted, as plain JSON-like rows."""

    feature_vectors: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    predictions: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    flags: dict[UUID, dict[str, Any]] = field(default_factory=dict)
    members: dict[UUID, list[UUID]] = field(default_factory=dict)
    flag_ids_by_name: dict[str, UUID] = field(default_factory=dict)

    def copy(self) -> "C2Records":
        return copy.deepcopy(self)


def _score(model: Any, features: dict[str, Any]) -> dict[str, float]:
    row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=list(FEATURE_NAMES))
    probabilities = model.predict_proba(row)[0]
    scored = {str(label): float(p) for label, p in zip(model.classes_, probabilities, strict=True)}
    # Round-trip through JSON exactly as JSONB storage would.
    return json.loads(json.dumps({name: scored[name] for name in CLASS_ORDER}))


def emulate_c2(model: Any, scenarios: dict[str, Any]) -> C2Records:
    """Apply C2's frozen persistence rules to the scenario windows."""
    records = C2Records()
    for task in scenarios["tasks"]:
        task_id, session_id = UUID(task["task_id"]), UUID(task["session_id"])
        windows = {}
        for window in task["windows"]:
            fv_id, pred_id = UUID(window["feature_vector_id"]), UUID(window["prediction_id"])
            records.feature_vectors[fv_id] = {
                "feature_vector_id": fv_id,
                "session_id": session_id,
                "task_id": task_id,
                "window_start": datetime.fromisoformat(window["window_start"]),
                "window_end": datetime.fromisoformat(window["window_end"]),
                "run_id": "live",
                "schema_version": scenarios["schema_version"],
                "features": window["features"],
            }
            probabilities = _score(model, window["features"])
            category = max(probabilities, key=probabilities.__getitem__)
            confidence = probabilities[category]
            records.predictions[pred_id] = {
                "prediction_id": pred_id,
                "feature_vector_id": fv_id,
                "task_id": task_id,
                "model_version": scenarios["model_version"],
                "predicted_category": category,
                "class_probabilities": probabilities,
                "confidence": confidence,
                "ai_positive": category != "no_ai" and confidence >= PROTOTYPE_THRESHOLD,
            }
            windows[pred_id] = records.feature_vectors[fv_id]

        for spec in task["flags"]:
            member_ids = [UUID(pid) for pid in spec["member_prediction_ids"]]
            members = [records.predictions[pid] for pid in member_ids]
            categories = {m["predicted_category"] for m in members}
            if len(categories) != 1 or not all(m["ai_positive"] for m in members):
                raise AssertionError(
                    f"fixture {spec['name']}: members must be ai_positive and one category, "
                    f"got {[(m['predicted_category'], m['confidence']) for m in members]}"
                )
            representative = max(members, key=lambda m: m["confidence"])
            flag_id = UUID(spec["flag_id"])
            records.flags[flag_id] = {
                "flag_id": flag_id,
                "task_id": task_id,
                "model_version": scenarios["model_version"],
                "category": representative["predicted_category"],
                "representative_prediction_id": representative["prediction_id"],
                "class_probabilities": dict(representative["class_probabilities"]),
                "confidence": representative["confidence"],
                "detected_start_line": spec["detected_start_line"],
                "detected_end_line": spec["detected_end_line"],
                "evidence_kinds": spec["evidence_kinds"],
                "segment_basis": spec["segment_basis"],
                "window_count": len(members),
                "detected_at": min(windows[pid]["window_start"] for pid in member_ids),
                "detected_until": max(windows[pid]["window_end"] for pid in member_ids),
            }
            records.members[flag_id] = member_ids
            records.flag_ids_by_name[spec["name"]] = flag_id
    return records


class InMemoryC2Reader:
    """``C2OutputReader`` over emulated C2 rows. Parses rows like the DB reader."""

    def __init__(self, records: C2Records) -> None:
        self.records = records

    async def get_flag(self, flag_id: UUID) -> ClassifierFlag:
        if flag_id not in self.records.flags:
            raise FlagNotFoundError(f"classifier flag {flag_id} does not exist")
        return parse_c2_row(ClassifierFlag, self.records.flags[flag_id], "flag")

    async def get_prediction(self, prediction_id: UUID) -> WindowPrediction:
        if prediction_id not in self.records.predictions:
            raise PredictionNotFoundError(f"window prediction {prediction_id} does not exist")
        return parse_c2_row(WindowPrediction, self.records.predictions[prediction_id], "prediction")

    async def get_feature_vector(self, feature_vector_id: UUID) -> FeatureVector:
        if feature_vector_id not in self.records.feature_vectors:
            raise FeatureVectorNotFoundError(f"feature vector {feature_vector_id} does not exist")
        return parse_c2_row(
            FeatureVector, self.records.feature_vectors[feature_vector_id], "feature vector"
        )

    async def get_contributing_predictions(self, flag_id: UUID) -> list[WindowPrediction]:
        return [
            parse_c2_row(WindowPrediction, self.records.predictions[pid], "member")
            for pid in self.records.members.get(flag_id, [])
        ]

    async def get_task_window_predictions(
        self, task_id: UUID, model_version: str
    ) -> list[WindowPrediction]:
        return [
            parse_c2_row(WindowPrediction, row, "prediction")
            for row in _task_predictions(self.records, task_id, model_version)
        ]


def _task_predictions(records: C2Records, task_id: UUID, model_version: str) -> list[dict]:
    rows = [
        p
        for p in records.predictions.values()
        if p["task_id"] == task_id and p["model_version"] == model_version
    ]
    rows.sort(key=lambda p: records.feature_vectors[p["feature_vector_id"]]["window_start"])
    return rows


def as_db_row(row: dict[str, Any]) -> dict[str, Any]:
    """Shape a row like asyncpg returns it: JSONB as text, TEXT[] as list."""
    out = dict(row)
    for column in ("class_probabilities", "features"):
        if column in out:
            out[column] = json.dumps(out[column])
    if "evidence_kinds" in out:
        out["evidence_kinds"] = list(out["evidence_kinds"])
    return out


class FakeAsyncpgPool:
    """Answers PostgresC2Repository's queries from emulated C2 rows.

    Dispatches on the table each query reads, so the repository's real SQL
    strings and parameter binding are exercised without a database.
    """

    def __init__(self, records: C2Records) -> None:
        self.records = records
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        key = args[0]
        if "FROM classifier_flags" in query:
            table = self.records.flags
        elif "FROM feature_vectors" in query:
            table = self.records.feature_vectors
        elif "WHERE p.prediction_id = $1" in query:
            table = self.records.predictions
        else:
            raise AssertionError(f"unexpected fetchrow query: {query}")
        row = table.get(key)
        return None if row is None else as_db_row(row)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        if "FROM classifier_flag_predictions" in query:
            rows = [self.records.predictions[p] for p in self.records.members.get(args[0], [])]
        elif "WHERE p.task_id = $1" in query:
            rows = _task_predictions(self.records, args[0], args[1])
        else:
            raise AssertionError(f"unexpected fetch query: {query}")
        return [as_db_row(row) for row in rows]


# --- Phase 2: C4 XAI background ---------------------------------------------

BACKGROUND_FILENAME = "xai_background.csv"
BACKGROUND_VERSION = "synthetic-prototype-background-v0"


def background_frame() -> pd.DataFrame:
    """Every 6th row of the prototype training frame (100 rows).

    Training-reference data, as the plan requires: never the scenario windows
    being explained and never a held-out test set.
    """
    frame, _ = _training_frame()
    return frame.iloc[::6].reset_index(drop=True)


def analysis_frame() -> pd.DataFrame:
    """A different slice of the prototype distribution for global SHAP."""
    frame, _ = _training_frame()
    return frame.iloc[3::20].reset_index(drop=True)


def background_reference(**overrides: Any) -> BackgroundReference:
    values: dict[str, Any] = {
        "reference_id": "prototype-xai-background",
        "dataset_version": BACKGROUND_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": "synthetic prototype training frame (tests/explainability/fixtures)",
        "split": "training_reference",
        "sample_count": len(background_frame()),
        "uri": BACKGROUND_FILENAME,
        "compatible_model_version": MODEL_VERSION,
        "training_dataset_version": "synthetic-prototype-fixture",
    }
    values.update(overrides)
    return BackgroundReference(**values)


def write_background(directory: Path) -> Path:
    path = directory / BACKGROUND_FILENAME
    background_frame().to_csv(path, index=False)
    return path
