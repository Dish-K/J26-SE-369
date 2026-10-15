"""Test doubles and builders for Segment Extraction inputs."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from backend.app.probing.schemas import FlagInput, SubmissionInput


def make_flag(
    start_line: int,
    end_line: int,
    *,
    task_id: UUID,
    confidence: float = 0.8,
    category: str = "external_ai",
    model_version: str = "model-v1",
    flag_id: UUID | None = None,
) -> FlagInput:
    return FlagInput(
        flag_id=flag_id or uuid4(),
        task_id=task_id,
        model_version=model_version,
        category=category,
        confidence=confidence,
        start_line=start_line,
        end_line=end_line,
    )


def make_submission(
    task_id: UUID, code: str | None, *, redacted_at: datetime | None = None
) -> SubmissionInput:
    return SubmissionInput(task_id=task_id, code_text=code, code_redacted_at=redacted_at)


class InMemoryProbingReader:
    """Same surface as ``PostgresProbingReader`` without a database."""

    def __init__(self) -> None:
        self._flags: dict[UUID, list[FlagInput]] = {}
        self._submissions: dict[UUID, SubmissionInput] = {}

    def add_flag(self, flag: FlagInput) -> None:
        self._flags.setdefault(flag.task_id, []).append(flag)

    def add_submission(self, submission: SubmissionInput) -> None:
        self._submissions[submission.task_id] = submission

    async def get_task_flags(self, task_id: UUID) -> list[FlagInput]:
        return sorted(
            self._flags.get(task_id, []),
            key=lambda f: (f.start_line, f.end_line, str(f.flag_id)),
        )

    async def get_submission(self, task_id: UUID) -> SubmissionInput | None:
        return self._submissions.get(task_id)


REDACTED_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)
