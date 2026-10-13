"""Rebuild shared_features/example_fixture.json from its raw events.
   python tools/regenerate_example_fixture.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_features import (
    FEATURE_NAMES,
    SCHEMA_VERSION,
    edit_evidence,
    extract_features,
    feature_payload,
)

FIXTURE = ROOT / "shared_features" / "example_fixture.json"


def _plain(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def build_fixture(fixture: dict) -> dict:
    events = pd.DataFrame(fixture["raw_events"])
    tasks = pd.DataFrame(fixture["tasks"])
    # The stored times mix whole seconds and microseconds, so the format must be explicit.
    events["time"] = pd.to_datetime(events["time"], utc=True, format="ISO8601")
    for column in ("started_at", "submitted_at"):
        tasks[column] = pd.to_datetime(tasks[column], utc=True, format="ISO8601")

    frame = extract_features(events, tasks, run_id="final")
    payloads = [feature_payload(row) for row in frame.to_dict("records")]

    first = payloads[0]
    evidence = edit_evidence(
        events,
        first["session_id"],
        first["task_id"],
        first["window_start"],
        first["window_end"],
    )

    return {
        **fixture,
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_payloads": payloads,
        "edit_evidence_window_0": [
            {column: _plain(value) for column, value in row.items()}
            for row in evidence.to_dict("records")
        ],
    }


def render(fixture: dict) -> str:
    return json.dumps(fixture, indent=2) + "\n"


def main() -> None:
    fixture = build_fixture(json.loads(FIXTURE.read_text(encoding="utf-8")))
    FIXTURE.write_text(render(fixture), encoding="utf-8")
    print(f"{FIXTURE.name}: {len(fixture['feature_payloads'])} windows, schema_version {SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
