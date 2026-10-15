"""Per-flag rejection of unusable flags (brief section 7).

Only the individual flag is rejected, never the task. This runs before
merging so a rejected flag cannot pull a healthy flag's range into a group.
"""

import logging

from backend.app.probing.errors import SegmentResolutionError
from backend.app.probing.languages import SegmentExtractor
from backend.app.probing.schemas import FlagInput, RejectedFlag, SubmissionInput

logger = logging.getLogger(__name__)


def reject_unusable_flags(
    flags: list[FlagInput],
    submission: SubmissionInput | None,
    extractor: SegmentExtractor,
) -> tuple[list[FlagInput], list[RejectedFlag]]:
    """Split one task's flags into (usable, rejected).

    A flag is rejected when:
      * the submission is redacted, or its code is missing or empty
        (``submission_redacted``);
      * its lines fall outside the submitted file (``out_of_range``);
      * its lines are in range but hold no code (``blank_range``);
      * they resolve to code the parser could not understand
        (``unparseable_range``).

    Input order is preserved. A task whose flags are all rejected simply has
    no usable flags; that is a valid outcome, not an error.
    """
    for flag in flags:
        if submission is not None and flag.task_id != submission.task_id:
            raise ValueError(
                f"flag {flag.flag_id} belongs to task {flag.task_id}, "
                f"not to submission task {submission.task_id}"
            )

    usable: list[FlagInput] = []
    rejected: list[RejectedFlag] = []

    redaction_detail = _unusable_submission_detail(submission)
    for flag in flags:
        if redaction_detail is not None:
            rejected.append(_reject(flag, "submission_redacted", redaction_detail))
            continue
        assert submission is not None and submission.code_text is not None
        try:
            extractor.resolve_boundary(submission.code_text, flag.start_line, flag.end_line)
        except SegmentResolutionError as exc:
            rejected.append(_reject(flag, exc.reason, str(exc)))
        else:
            usable.append(flag)
    return usable, rejected


def _unusable_submission_detail(submission: SubmissionInput | None) -> str | None:
    if submission is None:
        return "no task_submissions row"
    if submission.code_redacted_at is not None:
        return f"code redacted at {submission.code_redacted_at.isoformat()}"
    if submission.code_text is None or submission.code_text == "":
        return "submitted code is empty"
    return None


def _reject(flag: FlagInput, reason, detail: str) -> RejectedFlag:
    logger.warning("Rejecting flag %s (task %s): %s - %s", flag.flag_id, flag.task_id, reason, detail)
    return RejectedFlag(flag_id=flag.flag_id, task_id=flag.task_id, reason=reason, detail=detail)
