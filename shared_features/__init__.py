"""The C1 to C2 shared feature contract.

One registry, one validated schema, one extractor. Real telemetry, the
synthetic generator and every degraded copy go through the same code, which is
what makes their feature vectors comparable by construction.

    from shared_features import extract_features, FEATURE_SCHEMA, SCHEMA_VERSION

``registry.yaml`` is the contract; everything else is a typed view of it.
"""

from .feature_schema import (
    ENUMS,
    CaptureReason,
    EVENT_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    IDENTITY_COLUMNS,
    PARAMETERS,
    REGISTRY,
    SCHEMA_VERSION,
    TASK_SCHEMA,
    WINDOW_STEP,
    WINDOW_WIDTH,
    Display,
    ExitStatus,
    FeatureSpec,
    Fullscreen,
    Kind,
    KeyClass,
    Modifier,
    Network,
    Origin,
    PasteOrigin,
    SourcePane,
    Visibility,
    feature_payload,
    frame_from_payloads,
)
from .evidence import (
    EDIT_EVIDENCE_SQL,
    EDIT_EVIDENCE_SQL_ASYNCPG,
    EVIDENCE_COLUMNS,
    edit_evidence,
)
from .code_resolution import Resolution, resolve_flag_code
from .extractor import extract_features, extract_window, window_grid

__all__ = [
    "REGISTRY", "FEATURE_NAMES", "SCHEMA_VERSION", "WINDOW_WIDTH", "WINDOW_STEP",
    "PARAMETERS", "ENUMS", "FeatureSpec",
    "EVENT_SCHEMA", "TASK_SCHEMA", "FEATURE_SCHEMA", "IDENTITY_COLUMNS",
    "Kind", "Origin", "KeyClass", "Modifier", "ExitStatus", "SourcePane",
    "PasteOrigin", "Visibility", "Fullscreen", "Network", "Display",
    "CaptureReason",
    "feature_payload", "frame_from_payloads",
    "extract_features", "extract_window", "window_grid",
    "edit_evidence", "EDIT_EVIDENCE_SQL", "EDIT_EVIDENCE_SQL_ASYNCPG",
    "EVIDENCE_COLUMNS",
    "resolve_flag_code", "Resolution",
]
