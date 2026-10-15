"""Segment Extraction entry point (C3 part 1).

``extract_segments(task_id, reader)`` is the single function later C3 parts
call. It is a read path only: it reads the mapped flags and the submitted
file, and returns segments in memory. It writes nothing.

Order of work:
    flags -> reject unusable -> merge touching -> resolve boundaries
          -> attach entry function -> ExtractionResult
"""

import logging
from uuid import UUID

from backend.app.probing.languages import SegmentExtractor, get_extractor
from backend.app.probing.merge import build_segments
from backend.app.probing.rejection import reject_unusable_flags
from backend.app.probing.repository import ProbingInputReader
from backend.app.probing.schemas import EntryFunction, ExtractionResult

logger = logging.getLogger(__name__)

# Submissions are Python-only by design upstream; there is no detection step.
DEFAULT_LANGUAGE = "python"


async def extract_segments(
    task_id: UUID,
    reader: ProbingInputReader,
    *,
    extractor: SegmentExtractor | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> ExtractionResult:
    """Produce the segments for one task's mapped classifier flags.

    A task with no flags, or none that survive rejection, yields an empty
    ``segments`` tuple. That is a valid result, not an error. Only the
    individual unusable flag is ever dropped, and every drop is recorded in
    ``rejected_flags``.
    """
    extractor = extractor or get_extractor(language)

    flags = await reader.get_task_flags(task_id)
    if not flags:
        return _result(task_id, [], [], [])

    submission = await reader.get_submission(task_id)
    usable, rejected = reject_unusable_flags(flags, submission, extractor)
    if not usable:
        return _result(task_id, [], rejected, [])

    assert submission is not None and submission.code_text is not None  # usable => present
    code = submission.code_text

    segments, merge_rejected, notes = build_segments(task_id, usable, code, extractor)

    entry = _entry_function(extractor, code, task_id)
    if entry is not None:
        segments = [segment.model_copy(update={"entry_function": entry}) for segment in segments]

    logger.info(
        "Task %s: %d flag(s) -> %d segment(s), %d rejected, %d note(s)",
        task_id,
        len(flags),
        len(segments),
        len(rejected) + len(merge_rejected),
        len(notes),
    )
    return _result(task_id, segments, rejected + merge_rejected, notes)


def _entry_function(
    extractor: SegmentExtractor, code: str, task_id: UUID
) -> EntryFunction | None:
    """Best-effort and non-blocking: any failure just leaves it unset."""
    try:
        return extractor.find_entry_function(code)
    except Exception:  # noqa: BLE001 - the heuristic must never fail extraction
        logger.exception("Entry-function heuristic failed for task %s; continuing without it", task_id)
        return None


def _result(task_id, segments, rejected, notes) -> ExtractionResult:
    return ExtractionResult(
        task_id=task_id,
        segments=tuple(segments),
        rejected_flags=tuple(rejected),
        notes=tuple(notes),
    )
