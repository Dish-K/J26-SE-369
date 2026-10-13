"""Event codes, validation schemas and JSONB payload helpers.

Everything here is derived from ``registry.yaml``. That file is the contract;
this module is the typed view of it. Nothing hardcodes a feature name, a dtype
or an enum value -- if you find yourself about to, edit the registry instead.

Public names
------------
``REGISTRY`` ``FEATURE_NAMES`` ``SCHEMA_VERSION`` ``WINDOW_WIDTH`` ``WINDOW_STEP``
``PARAMETERS`` ``EVENT_SCHEMA`` ``TASK_SCHEMA`` ``FEATURE_SCHEMA`` ``IDENTITY_COLUMNS``
``Kind`` ``Origin`` ``KeyClass`` ``Modifier`` ``ExitStatus`` ``SourcePane``
``PasteOrigin`` ``Visibility`` ``Fullscreen`` ``Network`` ``Display``
``feature_payload`` ``frame_from_payloads``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pandera.pandas as pa
import yaml

__all__ = [
    "REGISTRY", "FEATURE_NAMES", "SCHEMA_VERSION", "WINDOW_WIDTH", "WINDOW_STEP",
    "PARAMETERS", "EVENT_SCHEMA", "TASK_SCHEMA", "FEATURE_SCHEMA", "IDENTITY_COLUMNS",
    "FeatureSpec", "ENUMS",
    "Kind", "Origin", "KeyClass", "Modifier", "ExitStatus", "SourcePane",
    "PasteOrigin", "Visibility", "Fullscreen", "Network", "Display",
    "feature_payload", "frame_from_payloads",
]

_REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

# Vocabularies. A typo in the registry should fail at import, not silently
# produce a feature nobody can render.
_FAMILIES = {"keystroke", "edit", "run", "environment", "window"}
_DTYPES = {"int", "float"}
_UNITS = {"count", "ms", "s", "fraction", "line"}
_DISPLAY_AS = {"count", "percent", "seconds", "milliseconds", "line", "number"}


# Registry
@dataclass(frozen=True)
class FeatureSpec:
    """Every fact about one feature. Immutable on purpose."""

    name: str
    family: str
    dtype: str
    unit: str
    valid_range: tuple[float | None, float | None]
    nullable: bool
    label: str
    description: str
    display_as: str
    actionable: bool | None

    @property
    def pandas_dtype(self) -> str:
        """``Int64`` for a nullable integer -- plain ``int64`` cannot hold NA."""
        if self.dtype == "float":
            return "float64"
        return "Int64" if self.nullable else "int64"


def _load_registry(path: Path = _REGISTRY_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    specs: dict[str, FeatureSpec] = {}
    for name, entry in raw["features"].items():
        missing = {
            "family", "dtype", "unit", "valid_range", "nullable",
            "label", "description", "display_as", "actionable",
        } - entry.keys()
        if missing:
            raise ValueError(f"{name}: registry entry missing {sorted(missing)}")

        for key, allowed in (
            ("family", _FAMILIES), ("dtype", _DTYPES),
            ("unit", _UNITS), ("display_as", _DISPLAY_AS),
        ):
            if entry[key] not in allowed:
                raise ValueError(
                    f"{name}: {key}={entry[key]!r} is not one of {sorted(allowed)}"
                )

        low, high = entry["valid_range"]
        if low is not None and high is not None and low > high:
            raise ValueError(f"{name}: valid_range {entry['valid_range']} is inverted")

        specs[name] = FeatureSpec(
            name=name,
            family=entry["family"],
            dtype=entry["dtype"],
            unit=entry["unit"],
            valid_range=(low, high),
            nullable=bool(entry["nullable"]),
            label=entry["label"],
            description=entry["description"],
            display_as=entry["display_as"],
            actionable=entry["actionable"],
        )

    raw["features"] = specs
    return raw


_RAW = _load_registry()

REGISTRY: Mapping[str, FeatureSpec] = _RAW["features"]
FEATURE_NAMES: tuple[str, ...] = tuple(REGISTRY)
SCHEMA_VERSION: str = _RAW["schema_version"]
WINDOW_WIDTH: int = _RAW["window"]["width_s"]
WINDOW_STEP: int = _RAW["window"]["step_s"]
PARAMETERS: Mapping[str, int] = dict(_RAW["parameters"])
ENUMS: Mapping[str, Any] = _RAW["enums"]


# Event codes, built from the registry so they cannot drift from it
def _make_enum(name: str, base=IntEnum):
    spec = ENUMS[name]
    members = spec.get("values") or spec["flags"]
    enum = base(name, members)
    enum.__doc__ = (
        f"{spec['column']} -- authority: {spec['authority']}. "
        + ("Pinned by a SQL CHECK constraint and asserted in CI."
           if spec["authority"] == "sql" else
           "The schema does not constrain this column; the registry is the only authority.")
    )
    return enum


Kind = _make_enum("Kind")
Origin = _make_enum("Origin")
ExitStatus = _make_enum("ExitStatus")
SourcePane = _make_enum("SourcePane")
KeyClass = _make_enum("KeyClass")
PasteOrigin = _make_enum("PasteOrigin")
CaptureReason = _make_enum("CaptureReason")
Modifier = _make_enum("Modifier", IntFlag)
Visibility = _make_enum("Visibility")
Fullscreen = _make_enum("Fullscreen")
Network = _make_enum("Network")
Display = _make_enum("Display")

#: Kinds the database requires a ``task_id`` on (``events_task_scoped_have_task_chk``).
TASK_SCOPED_KINDS = frozenset(
    {Kind.keydown, Kind.keyup, Kind.edit, Kind.run_started, Kind.run_finished}
)

#: Kinds whose ``start_line``/``end_line`` describe a *selection*, not an edit.
#: Specification 10.1. Nothing in SQL prevents confusing the two.
CLIPBOARD_KINDS = frozenset({Kind.copy, Kind.cut})


# Input schemas
def _frame_check(fn, name: str, description: str) -> pa.Check:
    return pa.Check(fn, name=name, error=description)


def _task_scoped_rows_have_task(df: pd.DataFrame) -> bool:
    scoped = df["kind"].isin([int(k) for k in TASK_SCOPED_KINDS])
    return bool(not (scoped & df["task_id"].isna()).any())


def _ordering_triple_unique(df: pd.DataFrame) -> bool:
    cols = ["session_id", "load_seq", "batch_seq", "seq_in_batch"]
    return bool(not df.duplicated(subset=cols).any())


#: Extractor input. Mirrors the ``events`` table. Extra columns are allowed and ignored -- including the stored ``gap_ms``, which the extractor never reads.
EVENT_SCHEMA = pa.DataFrameSchema(
    {
        "session_id":   pa.Column(nullable=False),
        "task_id":      pa.Column(nullable=True),
        "load_seq":     pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
        "batch_seq":    pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
        "seq_in_batch": pa.Column("int64", pa.Check.ge(0), nullable=False, coerce=True),
        "time":         pa.Column("datetime64[ns, UTC]", nullable=False, coerce=True),
        "perf_now":     pa.Column("float64", nullable=False, coerce=True),
        "kind":         pa.Column("int64", pa.Check.isin([int(k) for k in Kind]),
                                  nullable=False, coerce=True),
        # Kind-specific. Every one is optional, because a window of pure typing
        # legitimately carries no edit or clipboard columns at all.
        "key_class":    pa.Column("Int64", nullable=True, required=False, coerce=True),
        "key_code":     pa.Column(nullable=True, required=False),
        "is_repeat":    pa.Column("boolean", nullable=True, required=False, coerce=True),
        "is_trusted":   pa.Column("boolean", nullable=True, required=False, coerce=True),
        "origin":       pa.Column("Int64", nullable=True, required=False, coerce=True),
        "version_id":   pa.Column("Int64", nullable=True, required=False, coerce=True),
        "change_index": pa.Column("Int64", nullable=True, required=False, coerce=True),
        "start_line":   pa.Column("Int64", nullable=True, required=False, coerce=True),
        "end_line":     pa.Column("Int64", nullable=True, required=False, coerce=True),
        "inserted_len": pa.Column("Int64", nullable=True, required=False, coerce=True),
        "removed_len":  pa.Column("Int64", nullable=True, required=False, coerce=True),
        "inserted_lines": pa.Column("Int64", nullable=True, required=False, coerce=True),
        "paste_origin": pa.Column("Int64", nullable=True, required=False, coerce=True),
        "env_state":    pa.Column("Int64", nullable=True, required=False, coerce=True),
        "run_id":       pa.Column(nullable=True, required=False),
        "exit_status":  pa.Column("Int64", nullable=True, required=False, coerce=True),
        "copied_len":   pa.Column("Int64", nullable=True, required=False, coerce=True),
        "source_pane":  pa.Column("Int64", nullable=True, required=False, coerce=True),
    },
    checks=[
        _frame_check(_task_scoped_rows_have_task, "task_scoped_rows_have_task",
                     "kinds 0,1,2,8,9 require a task_id (events_task_scoped_have_task_chk)"),
        _frame_check(_ordering_triple_unique, "ordering_triple_unique",
                     "(session_id, load_seq, batch_seq, seq_in_batch) must be unique"),
    ],
    strict=False,
    name="events",
)

#: One row per task. ``submitted_at`` is null while the task is still open,
#: which is the normal case during live extraction.
TASK_SCHEMA = pa.DataFrameSchema(
    {
        "task_id":      pa.Column(nullable=False),
        "session_id":   pa.Column(nullable=False),
        "started_at":   pa.Column("datetime64[ns, UTC]", nullable=False, coerce=True),
        "submitted_at": pa.Column("datetime64[ns, UTC]", nullable=True, required=False,
                                  coerce=True),
    },
    strict=False,
    name="tasks",
)


# Output schema
IDENTITY_COLUMNS: tuple[str, ...] = (
    "session_id", "task_id", "window_start", "window_end", "run_id",
)


def _column_for(spec: FeatureSpec) -> pa.Column:
    low, high = spec.valid_range
    checks = []
    if low is not None:
        checks.append(pa.Check.ge(low))
    if high is not None:
        checks.append(pa.Check.le(high))
    return pa.Column(
        spec.pandas_dtype,
        checks=checks or None,
        nullable=spec.nullable,
        coerce=True,
    )


def _window_order(df: pd.DataFrame) -> bool:
    return bool((df["window_end"] > df["window_start"]).all())


#: Extractor output, and what C2 reads back out of ``feature_vectors``.
#: ``ordered=True`` makes column order part of the contract, so a reordered
#: registry is a visible change rather than a silent one.
FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "session_id":   pa.Column(nullable=False),
        "task_id":      pa.Column(nullable=False),
        "window_start": pa.Column("datetime64[ns, UTC]", nullable=False, coerce=True),
        "window_end":   pa.Column("datetime64[ns, UTC]", nullable=False, coerce=True),
        "run_id":       pa.Column(checks=pa.Check.str_length(min_value=1), nullable=False),
        **{name: _column_for(spec) for name, spec in REGISTRY.items()},
    },
    checks=[_frame_check(_window_order, "window_end_after_start",
                         "window_end must be strictly after window_start")],
    strict=True,
    ordered=True,
    name=f"feature_vectors@{SCHEMA_VERSION}",
)


# JSONB round-tripping
def _plain(value: Any) -> Any:
    """pandas NA/NaN -> ``None``; numpy scalar -> Python scalar."""
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def feature_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """One extractor row -> the record written to ``feature_vectors``.

    **Every registry key is present in every payload.** A null value and an
    absent key must never be confused -- one says "measured nothing", the other
    says "this version did not have the feature".
    """
    return {
        "session_id":     str(row["session_id"]),
        "task_id":        str(row["task_id"]),
        "window_start":   pd.Timestamp(row["window_start"]).isoformat(),
        "window_end":     pd.Timestamp(row["window_end"]).isoformat(),
        "run_id":         str(row["run_id"]),
        "schema_version": SCHEMA_VERSION,
        "features":       {name: _plain(row[name]) for name in FEATURE_NAMES},
    }


def frame_from_payloads(payloads: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Payloads read back from ``feature_vectors`` -> a validated frame.

    The inverse of :func:`feature_payload`. A payload whose ``schema_version``
    disagrees with this package is rejected rather than silently reinterpreted.
    """
    rows = []
    for payload in payloads:
        got = payload.get("schema_version")
        if got != SCHEMA_VERSION:
            raise ValueError(
                f"payload schema_version {got!r} != package {SCHEMA_VERSION!r}; "
                "refusing to reinterpret features across versions"
            )
        features = payload["features"]
        unknown = set(features) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"payload carries unknown features: {sorted(unknown)}")
        missing = set(FEATURE_NAMES) - set(features)
        if missing:
            raise ValueError(f"payload is missing features: {sorted(missing)}")

        rows.append({
            "session_id":   payload["session_id"],
            "task_id":      payload["task_id"],
            "window_start": pd.Timestamp(payload["window_start"]),
            "window_end":   pd.Timestamp(payload["window_end"]),
            "run_id":       payload["run_id"],
            **{name: features[name] for name in FEATURE_NAMES},
        })

    frame = pd.DataFrame(rows, columns=list(IDENTITY_COLUMNS) + list(FEATURE_NAMES))
    return FEATURE_SCHEMA.validate(frame)
