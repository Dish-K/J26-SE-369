from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from shared_features import EDIT_EVIDENCE_SQL_ASYNCPG

from .inference import AI_CATEGORIES


_PREDICTION_WINDOW_SQL = """
SELECT
    p.prediction_id, p.feature_vector_id, p.task_id,
    p.model_version, p.predicted_category, p.confidence, p.ai_positive,
    fv.session_id, fv.window_start, fv.window_end, fv.run_id
FROM classifier_window_predictions AS p
JOIN feature_vectors AS fv
  ON fv.feature_vector_id = p.feature_vector_id
 AND fv.task_id = p.task_id
WHERE p.prediction_id = $1
"""


@dataclass(frozen=True)
class PredictionEditEvidence:
    """Edit evidence for one AI-positive window."""

    prediction_id: UUID
    feature_vector_id: UUID
    session_id: UUID
    task_id: UUID
    model_version: str
    predicted_category: str
    confidence: float
    window_start: datetime
    window_end: datetime
    edit_events: tuple[dict[str, Any], ...]


async def fetch_prediction_edit_evidence(
    connection: Any,
    prediction_id: UUID,
) -> PredictionEditEvidence | None:
    """Fetch edit evidence for a stored AI-positive live prediction."""
    window = await connection.fetchrow(_PREDICTION_WINDOW_SQL, prediction_id)
    if window is None:
        raise LookupError(f"No classifier window prediction: {prediction_id}")

    if window["run_id"] != "live":
        raise ValueError(
            "Live localization requires a run_id='live' feature window; "
            f"received {window['run_id']!r}"
        )

    if not window["ai_positive"]:
        return None

    if window["predicted_category"] not in AI_CATEGORIES:
        raise ValueError("A positive window must have an AI-associated predicted category")

    rows = await connection.fetch(
        EDIT_EVIDENCE_SQL_ASYNCPG,
        window["session_id"],
        window["task_id"],
        window["window_start"],
        window["window_end"],
    )

    return PredictionEditEvidence(
        prediction_id=window["prediction_id"],
        feature_vector_id=window["feature_vector_id"],
        session_id=window["session_id"],
        task_id=window["task_id"],
        model_version=window["model_version"],
        predicted_category=window["predicted_category"],
        confidence=float(window["confidence"]),
        window_start=window["window_start"],
        window_end=window["window_end"],
        edit_events=tuple(dict(row) for row in rows),
    )


@dataclass(frozen=True)
class LocalizedCodeRegion:
    """Connected post-edit range in a single document version."""

    prediction_id: UUID
    feature_vector_id: UUID
    session_id: UUID
    task_id: UUID
    model_version: str
    predicted_category: str
    confidence: float
    window_start: datetime
    window_end: datetime
    load_seq: int
    version_id: int
    start_line: int
    end_line: int
    edit_events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LocalizationResult:
    """Localized regions and events without a safe code location."""

    prediction_id: UUID
    regions: tuple[LocalizedCodeRegion, ...]
    unlocalized_events: tuple[dict[str, Any], ...]
    # Reasons align with unlocalized_events.
    unlocalized_reasons: tuple[str, ...] = ()


def _valid_line(value: Any) -> bool:
    return type(value) is int and value >= 1


def _valid_load_seq(value: Any) -> bool:
    return type(value) is int and value >= 0


def _valid_version_id(value: Any) -> bool:
    return type(value) is int and value >= 1


def localize_prediction_edit_evidence(
    evidence: PredictionEditEvidence,
) -> LocalizationResult:
    """Group safe post-edit ranges and retain unlocalized evidence."""
    if evidence.predicted_category not in AI_CATEGORIES:
        raise ValueError("Localization requires an AI-associated predicted category")
    if not 0.0 <= evidence.confidence <= 1.0:
        raise ValueError("Prediction confidence must be in [0, 1]")

    # [start_line, end_line, first_event_index, indexed_events]
    per_version: dict[tuple[int, int], list[list[Any]]] = {}
    rejected: list[dict[str, Any]] = []
    rejection_reasons: list[str] = []

    for index, event in enumerate(evidence.edit_events):
        row = dict(event)
        if row.get("session_id") != evidence.session_id or row.get("task_id") != evidence.task_id:
            raise ValueError("C1 evidence session/task does not match the prediction")

        load_seq = row.get("load_seq")
        version_id = row.get("version_id")
        start = row.get("touched_start_line")
        end = row.get("touched_end_line")

        if (
            not _valid_load_seq(load_seq)
            or not _valid_version_id(version_id)
            or not _valid_line(start)
            or not _valid_line(end)
            or end < start
        ):
            rejected.append(row)
            rejection_reasons.append("invalid_location_or_version")
            continue

        inserted_len = row.get("inserted_len")
        removed_len = row.get("removed_len")
        if (
            type(inserted_len) is not int
            or type(removed_len) is not int
            or inserted_len < 0
            or removed_len < 0
        ):
            # Edit type cannot be established safely.
            rejected.append(row)
            rejection_reasons.append("missing_or_invalid_edit_lengths")
            continue
        if inserted_len == 0 and removed_len > 0:
            # Post-deletion positions may refer to unrelated code.
            rejected.append(row)
            rejection_reasons.append("deletion_only")
            continue
        if inserted_len == 0 and removed_len == 0:
            rejected.append(row)
            rejection_reasons.append("no_code_change")
            continue

        key = (load_seq, version_id)
        groups = per_version.setdefault(key, [])
        touching = [
            group for group in groups
            if start <= group[1] + 1 and group[0] <= end + 1
        ]
        if not touching:
            groups.append([start, end, index, [(index, row)]])
            continue

        new_start = min([start] + [group[0] for group in touching])
        new_end = max([end] + [group[1] for group in touching])
        new_first = min([index] + [group[2] for group in touching])
        members = [(index, row)]
        for group in touching:
            members.extend(group[3])
            groups.remove(group)
        groups.append([new_start, new_end, new_first, members])

    ordered_groups = [
        (key, group) for key, groups in per_version.items() for group in groups
    ]
    ordered_groups.sort(key=lambda item: item[1][2])

    # Keep contributing events in capture order.
    regions = tuple(
        LocalizedCodeRegion(
            prediction_id=evidence.prediction_id,
            feature_vector_id=evidence.feature_vector_id,
            session_id=evidence.session_id,
            task_id=evidence.task_id,
            model_version=evidence.model_version,
            predicted_category=evidence.predicted_category,
            confidence=evidence.confidence,
            window_start=evidence.window_start,
            window_end=evidence.window_end,
            load_seq=key[0],
            version_id=key[1],
            start_line=group[0],
            end_line=group[1],
            edit_events=tuple(row for _, row in sorted(group[3], key=lambda item: item[0])),
        )
        for key, group in ordered_groups
    )
    return LocalizationResult(
        prediction_id=evidence.prediction_id,
        regions=regions,
        unlocalized_events=tuple(rejected),
        unlocalized_reasons=tuple(rejection_reasons),
    )
