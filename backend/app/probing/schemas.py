"""Typed inputs and results of Segment Extraction (C3 part 1)."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

AICategory = Literal["ide_ai", "external_ai"]

# Reasons a single flag's line range cannot be turned into a segment.
RangeRejectionReason = Literal["out_of_range", "blank_range", "unparseable_range"]
# Every reason a flag can be rejected for (the above, plus a redacted submission).
RejectionReason = Literal[
    "submission_redacted", "out_of_range", "blank_range", "unparseable_range"
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FlagInput(_Frozen):
    """One row of ``classifier_flags_for_probing`` (mapped flags only).

    ``start_line``/``end_line`` are final-submission coordinates, already
    rebased upstream. The historical detected_* range is deliberately absent.
    """

    flag_id: UUID
    task_id: UUID
    model_version: str = Field(min_length=1)
    category: AICategory
    confidence: float = Field(ge=0.0, le=1.0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_range(self) -> "FlagInput":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class SubmissionInput(_Frozen):
    """The columns of ``task_submissions`` Segment Extraction reads."""

    task_id: UUID
    code_text: str | None
    code_redacted_at: datetime | None = None


class RejectedFlag(_Frozen):
    """A flag excluded from extraction, kept so the exclusion is auditable."""

    flag_id: UUID
    task_id: UUID
    reason: RejectionReason
    detail: str


class Location(_Frozen):
    """Where a segment sits in the submitted file.

    Lines and columns are 1-based, matching C1's ``events`` columns and the
    editor's own numbering. ``end_col`` is the column just after the last
    character (exclusive), the way Monaco reports a range end. Columns count
    characters, not bytes.
    """

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_col: int = Field(ge=1)
    end_col: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_order(self) -> "Location":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.end_line == self.start_line and self.end_col < self.start_col:
            raise ValueError("end_col must be >= start_col on a single-line location")
        return self


class ResolvedBoundary(_Frozen):
    """One resolved statement (or run of sibling statements) and its context.

    ``code`` is the exact source slice, original indentation on continuation
    lines included. ``context`` holds the enclosing compound-statement header
    texts, ordered outermost to innermost, and never contains a header that is
    part of ``code``. It stops below the enclosing function, class or module.
    """

    code: str = Field(min_length=1)
    context: tuple[str, ...]
    location: Location


class EntryFunction(_Frozen):
    """Best-effort entry-function annotation (filled in by a later step)."""

    name: str = Field(min_length=1)
    uncertain: bool
    # Why the guess is uncertain; None whenever it is confident.
    #   tied_candidates      several uncalled functions tied on size
    #   no_uncalled_function every function is called by another (a cycle)
    reason: Literal["tied_candidates", "no_uncalled_function"] | None = None

    @model_validator(mode="after")
    def _check_reason(self) -> "EntryFunction":
        if self.uncertain != (self.reason is not None):
            raise ValueError("reason must be set exactly when uncertain is true")
        return self


class Segment(_Frozen):
    """A complete, independently meaningful piece of the submitted code.

    ``flag_ids`` lists every flag that fed the segment, the parent included,
    sorted for determinism. ``extra`` is reserved for facts later C3 parts
    attach; Segment Extraction leaves it empty.
    """

    segment_id: UUID
    task_id: UUID
    code: str = Field(min_length=1)
    context: tuple[str, ...]
    location: Location
    parent_flag_id: UUID
    flag_ids: tuple[UUID, ...] = Field(min_length=1)
    entry_function: EntryFunction | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_flags(self) -> "Segment":
        if self.parent_flag_id not in self.flag_ids:
            raise ValueError("parent_flag_id must be one of flag_ids")
        if len(set(self.flag_ids)) != len(self.flag_ids):
            raise ValueError("flag_ids must not contain duplicates")
        return self


ExtractionNoteKind = Literal["group_ungrouped", "group_spans_model_versions"]


class ExtractionNote(_Frozen):
    """A diagnostic fact about the extraction run, not about one segment.

    * ``group_ungrouped``: the merged range could not be resolved, so these
      flags were resolved independently (their segments may overlap).
    * ``group_spans_model_versions``: these flags, merged into one group, come
      from more than one ``model_version``.
    """

    kind: ExtractionNoteKind
    flag_ids: tuple[UUID, ...] = Field(min_length=1)


class ExtractionResult(_Frozen):
    """Everything Segment Extraction produced for one task."""

    task_id: UUID
    segments: tuple[Segment, ...]
    rejected_flags: tuple[RejectedFlag, ...]
    notes: tuple[ExtractionNote, ...]
