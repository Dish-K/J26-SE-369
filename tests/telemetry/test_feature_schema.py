"""The registry and the validation contract."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from shared_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    IDENTITY_COLUMNS,
    REGISTRY,
    SCHEMA_VERSION,
    feature_payload,
    frame_from_payloads,
)
from shared_features.feature_schema import _load_registry  # noqa: PLC2701

from conftest import SESSION, T0, TASK


def _valid_row():
    row = {
        "session_id": str(SESSION),
        "task_id": str(TASK),
        "window_start": T0,
        "window_end": T0 + pd.Timedelta(seconds=30),
        "run_id": "live",
    }
    for name, spec in REGISTRY.items():
        row[name] = 0 if spec.dtype == "int" else 0.0
    row["min_line"] = 1
    row["max_line"] = 1
    row["runs_before_first_success"] = 1
    return row


def _valid_frame():
    return pd.DataFrame([_valid_row()], columns=list(IDENTITY_COLUMNS) + list(FEATURE_NAMES))


def test_registry_loads_and_every_entry_is_complete():
    assert len(REGISTRY) == 40
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.label and spec.description
        assert spec.dtype in {"int", "float"}
        assert spec.actionable is None or isinstance(spec.actionable, bool)


def test_feature_names_are_unique_and_ordered():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert FEATURE_NAMES == tuple(REGISTRY)


def test_schema_column_order_is_identity_then_registry_order():
    assert tuple(FEATURE_SCHEMA.columns) == IDENTITY_COLUMNS + FEATURE_NAMES


def test_nullable_integers_use_a_dtype_that_can_hold_na():
    for spec in REGISTRY.values():
        if spec.dtype == "int" and spec.nullable:
            assert spec.pandas_dtype == "Int64", f"{spec.name} cannot hold NA"


def test_counts_are_never_nullable_and_statistics_always_are():
    """Missing-value policy: a count is never null, a statistic always may be."""
    assert not REGISTRY["keystroke_count"].nullable
    assert not REGISTRY["edit_count"].nullable
    assert not REGISTRY["edited_line_count"].nullable
    for name in ("mean_dwell_ms", "std_flight_ms", "backspace_rate",
                 "paste_ratio", "pause_ms", "inserted_len_max",
                 "min_line", "max_line"):
        assert REGISTRY[name].nullable, f"{name} must be nullable"


def test_a_valid_frame_validates():
    assert len(FEATURE_SCHEMA.validate(_valid_frame())) == 1


def test_out_of_range_values_are_rejected():
    frame = _valid_frame()
    frame.loc[0, "backspace_rate"] = 1.5          # a fraction above 1
    with pytest.raises(Exception):
        FEATURE_SCHEMA.validate(frame)


def test_a_missing_feature_is_rejected():
    frame = _valid_frame().drop(columns=["burst_len"])
    with pytest.raises(Exception):
        FEATURE_SCHEMA.validate(frame)


def test_an_unexpected_column_is_rejected():
    frame = _valid_frame()
    frame["speculative_new_feature"] = 1
    with pytest.raises(Exception):
        FEATURE_SCHEMA.validate(frame)


def test_a_window_that_ends_before_it_starts_is_rejected():
    frame = _valid_frame()
    frame.loc[0, "window_end"] = frame.loc[0, "window_start"] - pd.Timedelta(seconds=1)
    with pytest.raises(Exception):
        FEATURE_SCHEMA.validate(frame)


def test_payload_round_trip_preserves_nulls():
    frame = _valid_frame()
    frame.loc[0, "mean_dwell_ms"] = None
    frame.loc[0, "paste_ratio"] = None
    frame.loc[0, "min_line"] = None
    frame = FEATURE_SCHEMA.validate(frame)

    payload = feature_payload(frame.iloc[0])

    assert payload["schema_version"] == SCHEMA_VERSION
    assert set(payload["features"]) == set(FEATURE_NAMES), "every key in every payload"
    assert payload["features"]["mean_dwell_ms"] is None
    assert payload["features"]["paste_ratio"] is None
    assert payload["features"]["min_line"] is None
    assert payload["features"]["keystroke_count"] == 0

    # A null must survive as null, never become 0.0 on the way back.
    back = frame_from_payloads([payload])
    assert pd.isna(back.loc[0, "mean_dwell_ms"])
    assert pd.isna(back.loc[0, "paste_ratio"])
    assert pd.isna(back.loc[0, "min_line"])
    assert back.loc[0, "keystroke_count"] == 0


def test_payload_values_are_json_native():
    frame = FEATURE_SCHEMA.validate(_valid_frame())
    payload = feature_payload(frame.iloc[0])
    import json
    assert json.loads(json.dumps(payload))["features"].keys() == payload["features"].keys()


def test_a_payload_from_another_schema_version_is_refused():
    frame = FEATURE_SCHEMA.validate(_valid_frame())
    payload = feature_payload(frame.iloc[0])
    payload["schema_version"] = "0.9.0"
    with pytest.raises(ValueError, match="schema_version"):
        frame_from_payloads([payload])


def test_a_payload_missing_a_feature_is_refused():
    frame = FEATURE_SCHEMA.validate(_valid_frame())
    payload = feature_payload(frame.iloc[0])
    payload["features"].pop("burst_len")
    with pytest.raises(ValueError, match="missing"):
        frame_from_payloads([payload])


def test_registry_rejects_an_unknown_vocabulary_value(tmp_path):
    import yaml
    from shared_features.feature_schema import _REGISTRY_PATH  # noqa: PLC2701

    raw = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    raw["features"]["burst_len"]["display_as"] = "sparkline"
    bad = tmp_path / "registry.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="display_as"):
        _load_registry(bad)
