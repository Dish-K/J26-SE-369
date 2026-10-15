"""Merging overlapping flags into segments (brief section 6).

Line-range proximity is the only merge key: flags of different ``category``
or ``model_version`` still merge when their ranges overlap or touch. Whether
cross-model-version merging is right is a question for C2's owner; it is
recorded as a note here, not decided.
"""

import logging
from uuid import NAMESPACE_URL, UUID, uuid5

from backend.app.probing.errors import SegmentResolutionError
from backend.app.probing.languages import SegmentExtractor
from backend.app.probing.schemas import (
    ExtractionNote,
    FlagInput,
    RejectedFlag,
    ResolvedBoundary,
    Segment,
)

logger = logging.getLogger(__name__)

_SEGMENT_NAMESPACE = uuid5(NAMESPACE_URL, "codetrace/c3/segment")


def group_touching_flags(flags: list[FlagInput]) -> list[list[FlagInput]]:
    """Connected components of flags whose line ranges overlap or touch.

    Two inclusive ranges touch when the gap between them is zero lines:
    ``B.start <= A.end + 1``. Components are transitive: A-B and B-C put A and
    C in one group even when A and C do not touch. Because the ranges are
    intervals, one pass over them sorted by start finds exactly these
    components.

    Groups come back ordered by start line; members by (start, end, flag_id).
    """
    groups: list[list[FlagInput]] = []
    group_end = 0
    for flag in sorted(flags, key=lambda f: (f.start_line, f.end_line, f.flag_id)):
        if groups and flag.start_line <= group_end + 1:
            groups[-1].append(flag)
            group_end = max(group_end, flag.end_line)
        else:
            groups.append([flag])
            group_end = flag.end_line
    return groups


def choose_parent(group: list[FlagInput]) -> FlagInput:
    """Highest confidence; ties by earliest start_line, then lowest flag_id."""
    return min(group, key=lambda f: (-f.confidence, f.start_line, f.flag_id))


def build_segments(
    task_id: UUID,
    flags: list[FlagInput],
    code: str,
    extractor: SegmentExtractor,
) -> tuple[list[Segment], list[RejectedFlag], list[ExtractionNote]]:
    """Merge usable flags and resolve each group to a segment.

    ``flags`` must already have passed ``reject_unusable_flags``. If a group's
    union range cannot be resolved the group is dissolved: every member is
    resolved on its own range, becoming a standalone segment, and the
    ungrouping is recorded as a note. No partial regrouping is attempted.
    """
    segments: list[Segment] = []
    rejected: list[RejectedFlag] = []
    notes: list[ExtractionNote] = []

    for group in group_touching_flags(flags):
        flag_ids = tuple(sorted(f.flag_id for f in group))

        if len({f.model_version for f in group}) > 1:
            notes.append(ExtractionNote(kind="group_spans_model_versions", flag_ids=flag_ids))

        start = min(f.start_line for f in group)
        end = max(f.end_line for f in group)
        try:
            boundary = extractor.resolve_boundary(code, start, end)
        except SegmentResolutionError as exc:
            if len(group) == 1:
                rejected.append(_rejected(group[0], exc))
                continue
            logger.warning(
                "Union range %s-%s of %d flags is unresolvable (%s); ungrouping",
                start,
                end,
                len(group),
                exc,
            )
            notes.append(ExtractionNote(kind="group_ungrouped", flag_ids=flag_ids))
            for flag in group:
                try:
                    alone = extractor.resolve_boundary(code, flag.start_line, flag.end_line)
                except SegmentResolutionError as single_exc:
                    rejected.append(_rejected(flag, single_exc))
                else:
                    segments.append(_segment(task_id, [flag], flag, alone))
            continue

        segments.append(_segment(task_id, group, choose_parent(group), boundary))

    segments.sort(
        key=lambda s: (s.location.start_line, s.location.start_col, s.location.end_line, s.segment_id)
    )
    return segments, rejected, notes


def _segment(
    task_id: UUID, group: list[FlagInput], parent: FlagInput, boundary: ResolvedBoundary
) -> Segment:
    flag_ids = tuple(sorted(f.flag_id for f in group))
    # Deterministic: extracting the same task twice yields the same ids.
    segment_id = uuid5(_SEGMENT_NAMESPACE, f"{task_id}:" + ",".join(str(i) for i in flag_ids))
    return Segment(
        segment_id=segment_id,
        task_id=task_id,
        code=boundary.code,
        context=boundary.context,
        location=boundary.location,
        parent_flag_id=parent.flag_id,
        flag_ids=flag_ids,
    )


def _rejected(flag: FlagInput, exc: SegmentResolutionError) -> RejectedFlag:
    return RejectedFlag(
        flag_id=flag.flag_id, task_id=flag.task_id, reason=exc.reason, detail=str(exc)
    )
