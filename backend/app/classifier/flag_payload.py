from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Sequence
from uuid import UUID

from .flag_identity import flag_id_for_segment
from .segment_builder import SegmentDraft


_ALLOWED_EVIDENCE_KINDS = frozenset({
    "paste",
    "suggestion_accepted",
    "typing_burst",
    "uniform_intervals",
    "no_typing_errors",
})
_AI_CATEGORIES = frozenset({"ide_ai", "external_ai"})
_ALL_CATEGORIES = frozenset({"no_ai", "ide_ai", "external_ai"})
_MAX_SMALLINT = 32767
_MAX_BIGINT = (1 << 63) - 1
_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FlagInsertValues:
    """Fields for one ``classifier_flags`` row and its prediction links.

    ``prediction_ids`` belong in ``classifier_flag_predictions``. The original
    edits remain in Task 10/C1 evidence, retrievable through those predictions.
    No final submitted-code location or answer-level score is implied.
    """

    flag_id: UUID
    task_id: UUID
    model_version: str
    category: str
    representative_prediction_id: UUID
    class_probabilities: dict[str, float]
    confidence: float
    detected_start_line: int
    detected_end_line: int
    load_seq: int
    version_id: int
    evidence_kinds: tuple[str, ...]
    segment_basis: str
    window_count: int
    detected_at: datetime
    detected_until: datetime
    prediction_ids: tuple[UUID, ...]


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.utcoffset() is not None


def map_segment_to_flag(
    segment: SegmentDraft,
    *,
    verified_evidence_kinds: Sequence[str] = (),
) -> FlagInsertValues:
    """Map the approved window-driven C2 pipeline to database field values.

    The interval is the coverage of contributing observation windows, **not**
    the inferred period of AI use or the database write time. Named evidence
    must be verified externally; this mapper never derives it from a paste,
    Tab, or other event as a deterministic classification trigger.
    """
    # Also validates the immutable original source observation used for retries.
    flag_id = flag_id_for_segment(segment)

    if segment.category not in _AI_CATEGORIES:
        raise ValueError("Flag segments must have an AI-positive category")
    if not segment.model_version.strip():
        raise ValueError("model_version is required")
    if not (
        type(segment.load_seq) is int
        and 0 <= segment.load_seq <= _MAX_SMALLINT
        and type(segment.version_id) is int
        and 1 <= segment.version_id <= _MAX_BIGINT
    ):
        raise ValueError("Document identity is outside the database range")
    if not (
        type(segment.start_line) is int
        and type(segment.end_line) is int
        and 1 <= segment.start_line <= segment.end_line
    ):
        raise ValueError("Flag has an invalid line range")
    if not segment.members:
        raise ValueError("Flag must retain its original localization members")

    predictions = segment.prediction_ids
    if not (1 <= len(predictions) <= _MAX_SMALLINT):
        raise ValueError("Window count is outside the database range")
    if segment.representative_prediction_id not in predictions:
        raise ValueError("Representative prediction is not a segment member")

    for member in segment.members:
        if (
            member.task_id != segment.task_id
            or member.session_id != segment.session_id
            or member.model_version != segment.model_version
            or member.predicted_category != segment.category
            or member.load_seq != segment.load_seq
        ):
            raise ValueError("Segment member has mismatched provenance")
        if not (_aware(member.window_start) and _aware(member.window_end)):
            raise ValueError("Member window timestamps must be timezone-aware")
        if member.window_end <= member.window_start:
            raise ValueError("Member window has invalid duration")
        if not member.edit_events:
            raise ValueError("Segment member has no original edit-event evidence")

    if not (_aware(segment.first_window_start) and _aware(segment.last_window_end)):
        raise ValueError("Segment window timestamps must be timezone-aware")
    first = min(member.window_start for member in segment.members)
    last = max(member.window_end for member in segment.members)
    if segment.first_window_start != first or segment.last_window_end != last:
        raise ValueError("Segment observation timestamps do not match its members")
    if segment.last_window_end <= segment.first_window_start:
        raise ValueError("Segment observation period must be positive")

    probabilities = dict(segment.class_probabilities)
    if set(probabilities) != _ALL_CATEGORIES or any(
        type(value) not in (float, int) or not isfinite(value) or not 0 <= value <= 1
        for value in probabilities.values()
    ) or abs(sum(probabilities.values()) - 1) > _TOLERANCE:
        raise ValueError("Invalid three-class probabilities")
    if (
        type(segment.confidence) not in (float, int)
        or not isfinite(segment.confidence)
        or abs(segment.confidence - probabilities[segment.category]) > _TOLERANCE
    ):
        raise ValueError("Segment confidence must match its category probability")

    best_confidence = max(member.confidence for member in segment.members)
    if abs(best_confidence - segment.confidence) > _TOLERANCE:
        raise ValueError("Segment confidence is not its highest-confidence member")
    # Task 11 resolves equal-confidence ties to the earliest member.
    best_prediction = next(
        member.prediction_id
        for member in segment.members
        if abs(member.confidence - best_confidence) <= _TOLERANCE
    )
    if best_prediction != segment.representative_prediction_id:
        raise ValueError("Representative prediction does not match the best member")

    if isinstance(verified_evidence_kinds, (str, bytes)):
        raise ValueError("Verified evidence kinds must be a sequence of names")
    named_kinds = tuple(dict.fromkeys(verified_evidence_kinds))
    if any(type(name) is not str or name not in _ALLOWED_EVIDENCE_KINDS for name in named_kinds):
        raise ValueError("Unrecognized named evidence kind")

    return FlagInsertValues(
        flag_id=flag_id,
        task_id=segment.task_id,
        model_version=segment.model_version,
        category=segment.category,
        representative_prediction_id=segment.representative_prediction_id,
        class_probabilities=probabilities,
        confidence=float(segment.confidence),
        detected_start_line=segment.start_line,
        detected_end_line=segment.end_line,
        load_seq=segment.load_seq,
        version_id=segment.version_id,
        evidence_kinds=named_kinds,
        segment_basis="window_run",
        window_count=len(predictions),
        detected_at=segment.first_window_start,
        detected_until=segment.last_window_end,
        prediction_ids=predictions,
    )
