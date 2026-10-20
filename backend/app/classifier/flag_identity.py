from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

if TYPE_CHECKING:
    from .segment_builder import SegmentDraft


# Versioned, fixed namespace. Do not change after flags have been persisted.
_FLAG_NAMESPACE = uuid5(NAMESPACE_URL, "codetrace/c2/flag/first-localized-region/v1")


def flag_id_for_segment(segment: SegmentDraft) -> UUID:
    """Derive a repeatable UUID from this draft's *first* localized observation.

    The first region remains in ``members[0]`` even if later windows merge or
    rebase the segment range. Different regions from a single prediction have
    distinct range/event anchors; repeated attempts to save the same draft
    retain the same identifier. Missing source evidence is rejected rather
    than replaced by a guessed identifier.
    """
    if not segment.members:
        raise ValueError("A flag segment must contain at least one localized region")

    first = segment.members[0]
    if (
        first.task_id != segment.task_id
        or first.session_id != segment.session_id
        or first.model_version != segment.model_version
        or first.predicted_category != segment.category
    ):
        raise ValueError("First localized region does not match its segment")

    if not first.edit_events:
        raise ValueError("First localized region has no original edit evidence")

    source = first.edit_events[0]
    event_coordinates = ("batch_seq", "seq_in_batch", "change_index")
    for name in event_coordinates:
        value = source.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Source edit event requires nonnegative integer {name}")

    if (
        source.get("load_seq") != first.load_seq
        or source.get("version_id") != first.version_id
    ):
        raise ValueError("Source edit event document identity does not match the region")

    if not (
        type(first.load_seq) is int
        and first.load_seq >= 0
        and type(first.version_id) is int
        and first.version_id >= 1
        and type(first.start_line) is int
        and first.start_line >= 1
        and type(first.end_line) is int
        and first.end_line >= first.start_line
    ):
        raise ValueError("First localized region has an invalid historical location")

    # JSON encodes field boundaries unambiguously and keeps the identity
    # independent of the segment's mutable merged end range.
    components = (
        "v1",
        str(segment.session_id),
        str(segment.task_id),
        segment.model_version,
        segment.category,
        str(first.prediction_id),
        str(first.feature_vector_id),
        str(first.load_seq),
        str(first.version_id),
        str(first.start_line),
        str(first.end_line),
        *(str(source[name]) for name in event_coordinates),
    )
    return uuid5(_FLAG_NAMESPACE, json.dumps(components, ensure_ascii=False, separators=(",", ":")))
