from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import isfinite
from typing import Any, Mapping, Sequence
from uuid import UUID

from .localization import LocalizedCodeRegion

_CLASSES = frozenset({"no_ai", "ide_ai", "external_ai"})
_AI_CLASSES = frozenset({"ide_ai", "external_ai"})
_WINDOW_STEP = timedelta(seconds=10)
_PROBABILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ClassifiedWindow:
    """Stored window prediction and its Task 10 regions."""

    prediction_id: UUID
    feature_vector_id: UUID
    session_id: UUID
    task_id: UUID
    model_version: str
    predicted_category: str
    class_probabilities: Mapping[str, float]
    confidence: float
    ai_positive: bool
    window_start: datetime
    window_end: datetime
    regions: tuple[LocalizedCodeRegion, ...] = ()


@dataclass(frozen=True)
class SegmentDraft:
    """One historical segment in a single document version."""

    session_id: UUID
    task_id: UUID
    model_version: str
    category: str
    load_seq: int
    version_id: int
    start_line: int
    end_line: int
    first_window_start: datetime
    last_window_start: datetime
    last_window_end: datetime
    representative_prediction_id: UUID
    class_probabilities: dict[str, float]
    confidence: float
    members: tuple[LocalizedCodeRegion, ...]

    @property
    def prediction_ids(self) -> tuple[UUID, ...]:
        return tuple(dict.fromkeys(region.prediction_id for region in self.members))

    @property
    def window_count(self) -> int:
        return len(self.prediction_ids)


def _validate_window(window: ClassifiedWindow) -> None:
    if window.predicted_category not in _CLASSES:
        raise ValueError("Invalid predicted category")
    if not window.model_version or not window.model_version.strip():
        raise ValueError("model_version is required")
    if (
        not isinstance(window.window_start, datetime)
        or not isinstance(window.window_end, datetime)
        or window.window_start.utcoffset() is None
        or window.window_end.utcoffset() is None
    ):
        raise ValueError("Window times must be timezone-aware")
    if window.window_end <= window.window_start:
        raise ValueError("Window must have positive duration")
    if type(window.ai_positive) is not bool:
        raise ValueError("ai_positive must be a boolean")
    if window.ai_positive and window.predicted_category not in _AI_CLASSES:
        raise ValueError("An AI-positive window requires an AI category")
    if set(window.class_probabilities) != _CLASSES:
        raise ValueError("Expected three canonical class probabilities")
    if any(
        type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1
        for value in window.class_probabilities.values()
    ):
        raise ValueError("Class probabilities must be finite and within [0, 1]")
    if abs(sum(window.class_probabilities.values()) - 1.0) > _PROBABILITY_TOLERANCE:
        raise ValueError("Class probabilities must sum to 1")
    if (
        type(window.confidence) not in (int, float)
        or not isfinite(window.confidence)
        or abs(window.confidence - window.class_probabilities[window.predicted_category])
        > _PROBABILITY_TOLERANCE
    ):
        raise ValueError("Confidence must match the predicted category probability")
    if not window.ai_positive and window.regions:
        raise ValueError("A non-positive window cannot contain flagged regions")
    for region in window.regions:
        if (
            region.prediction_id != window.prediction_id
            or region.feature_vector_id != window.feature_vector_id
            or region.session_id != window.session_id
            or region.task_id != window.task_id
            or region.model_version != window.model_version
            or region.predicted_category != window.predicted_category
            or region.confidence != window.confidence
            or region.window_start != window.window_start
            or region.window_end != window.window_end
        ):
            raise ValueError("Localized region does not match its prediction")
        if region.start_line < 1 or region.end_line < region.start_line:
            raise ValueError("Invalid localized line range")


def _valid_integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _version_edits(
    edit_history: Sequence[Mapping[str, Any]],
) -> dict[tuple[UUID, UUID, int, int], tuple[dict[str, Any], ...] | None]:
    groups: dict[tuple[UUID, UUID, int, int], list[dict[str, Any]]] = {}
    invalid: set[tuple[UUID, UUID, int, int]] = set()

    for source in edit_history:
        row = dict(source)
        session_id, task_id = row.get("session_id"), row.get("task_id")
        load_seq, version_id = row.get("load_seq"), row.get("version_id")
        if (
            not isinstance(session_id, UUID)
            or not isinstance(task_id, UUID)
            or not _valid_integer(load_seq)
            or not _valid_integer(version_id, minimum=1)
        ):
            raise ValueError("Edit history requires valid document identities")

        key = (session_id, task_id, load_seq, version_id)
        groups.setdefault(key, []).append(row)
        if any(
            not _valid_integer(row.get(field), minimum=minimum)
            for field, minimum in (
                ("batch_seq", 0), ("seq_in_batch", 0), ("change_index", 0),
                ("start_line", 1), ("end_line", 1), ("inserted_lines", 0),
            )
        ) or row["end_line"] < row["start_line"] or row.get("origin") == 4:
            invalid.add(key)
        elif row.get("net_line_delta") is not None and (
            type(row["net_line_delta"]) is not int
            or row["net_line_delta"] != row["inserted_lines"] - (row["end_line"] - row["start_line"])
        ):
            invalid.add(key)

    indexed: dict[tuple[UUID, UUID, int, int], tuple[dict[str, Any], ...] | None] = {}
    for key, rows in groups.items():
        event_order = [
            (row.get("batch_seq"), row.get("seq_in_batch"), row.get("change_index"))
            for row in rows
        ]
        indexed[key] = (
            None if key in invalid or len(set(event_order)) != len(event_order)
            else tuple(sorted(rows, key=lambda row: (
                row["batch_seq"], row["seq_in_batch"], row["change_index"]
            )))
        )
    return indexed


def _map_forward(
    start_line: int,
    end_line: int,
    *,
    session_id: UUID,
    task_id: UUID,
    load_seq: int,
    from_version: int,
    to_version: int,
    edits: Mapping[tuple[UUID, UUID, int, int], tuple[dict[str, Any], ...] | None],
) -> tuple[int, int] | None:
    if from_version > to_version:
        return None

    lo, hi = start_line, end_line
    for version_id in range(from_version + 1, to_version + 1):
        changes = edits.get((session_id, task_id, load_seq, version_id))
        if not changes:
            return None

        shift = 0
        for edit in changes:
            start, end = edit["start_line"], edit["end_line"]
            if end < lo:
                shift += edit["inserted_lines"] - (end - start)
            elif start <= hi:
                return None
        lo += shift
        hi += shift
        if lo < 1:
            return None
    return lo, hi


def _aligned_ranges(
    segment: SegmentDraft,
    region: LocalizedCodeRegion,
    edits: Mapping[tuple[UUID, UUID, int, int], tuple[dict[str, Any], ...] | None],
) -> tuple[int, int, int, int, int] | None:
    if segment.load_seq != region.load_seq:
        return None

    target_version = max(segment.version_id, region.version_id)
    if target_version == segment.version_id == region.version_id:
        old, new = (segment.start_line, segment.end_line), (region.start_line, region.end_line)
    else:
        old = _map_forward(
            segment.start_line, segment.end_line,
            session_id=segment.session_id, task_id=segment.task_id,
            load_seq=segment.load_seq, from_version=segment.version_id,
            to_version=target_version, edits=edits,
        ) if segment.version_id < target_version else (segment.start_line, segment.end_line)
        new = _map_forward(
            region.start_line, region.end_line,
            session_id=region.session_id, task_id=region.task_id,
            load_seq=region.load_seq, from_version=region.version_id,
            to_version=target_version, edits=edits,
        ) if region.version_id < target_version else (region.start_line, region.end_line)
        if old is None or new is None:
            return None

    if new[0] > old[1] + 1 or old[0] > new[1] + 1:
        return None
    return target_version, old[0], old[1], new[0], new[1]


def _new_segment(window: ClassifiedWindow, region: LocalizedCodeRegion) -> SegmentDraft:
    return SegmentDraft(
        session_id=window.session_id,
        task_id=window.task_id,
        model_version=window.model_version,
        category=window.predicted_category,
        load_seq=region.load_seq,
        version_id=region.version_id,
        start_line=region.start_line,
        end_line=region.end_line,
        first_window_start=window.window_start,
        last_window_start=window.window_start,
        last_window_end=window.window_end,
        representative_prediction_id=window.prediction_id,
        class_probabilities=dict(window.class_probabilities),
        confidence=window.confidence,
        members=(region,),
    )


def build_segment_drafts(
    windows: Sequence[ClassifiedWindow],
    *,
    edit_history: Sequence[Mapping[str, Any]] = (),
) -> tuple[SegmentDraft, ...]:
    """Form segment drafts using safely aligned document coordinates."""
    for window in windows:
        _validate_window(window)

    history = _version_edits(edit_history)

    streams: dict[tuple[UUID, str], list[ClassifiedWindow]] = {}
    for window in windows:
        streams.setdefault((window.task_id, window.model_version), []).append(window)

    drafts: list[SegmentDraft] = []
    for stream in streams.values():
        stream.sort(key=lambda window: window.window_start)
        active: list[int] = []
        episode: set[int] = set()
        previous: ClassifiedWindow | None = None
        for window in stream:
            if previous is not None:
                if window.window_start == previous.window_start:
                    raise ValueError("Duplicate window in one task/model stream")
                if window.session_id != previous.session_id:
                    raise ValueError("Task belongs to inconsistent sessions")
                if (
                    window.window_start - previous.window_start != _WINDOW_STEP
                    or not previous.ai_positive
                    or previous.predicted_category != window.predicted_category
                ):
                    active = []
                    episode = set()
            if not window.ai_positive or not window.regions:
                active = []
                episode = set()
                previous = window
                continue

            next_active: list[int] = []
            used: set[int] = set()
            for region in window.regions:
                matches = [
                    (index, aligned)
                    for index in sorted(episode)
                    if drafts[index].category == window.predicted_category
                    if (aligned := _aligned_ranges(drafts[index], region, history)) is not None
                ]
                if len(matches) > 1:
                    exact = [
                        match for match in matches
                        if match[1][1:3] == match[1][3:5]
                    ]
                    matches = exact if len(exact) == 1 else []
                if matches and (matches[0][0] not in active or matches[0][0] in used):
                    matches = []
                if matches:
                    index, aligned = matches[0]
                    target_version, old_start, old_end, new_start, new_end = aligned
                    old = drafts[index]
                    better = window.confidence > old.confidence
                    drafts[index] = replace(
                        old,
                        version_id=target_version,
                        start_line=min(old_start, new_start),
                        end_line=max(old_end, new_end),
                        last_window_start=window.window_start,
                        last_window_end=window.window_end,
                        representative_prediction_id=(
                            window.prediction_id if better else old.representative_prediction_id
                        ),
                        class_probabilities=(
                            dict(window.class_probabilities)
                            if better else old.class_probabilities
                        ),
                        confidence=window.confidence if better else old.confidence,
                        members=old.members + (region,),
                    )
                    used.add(index)
                else:
                    index = len(drafts)
                    drafts.append(_new_segment(window, region))
                next_active.append(index)
            active = next_active
            episode.update(next_active)
            previous = window

    return tuple(drafts)