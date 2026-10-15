"""Read-only access to the two inputs of Segment Extraction.

Reads exactly ``classifier_flags_for_probing`` (view, migration 006) and
``task_submissions`` (C1). Nothing else, and never a write. Line numbers from
the view are final-submission coordinates and are passed through untouched.
"""

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

import asyncpg
from pydantic import ValidationError

from backend.app.probing.errors import ProbingInputError
from backend.app.probing.schemas import FlagInput, SubmissionInput


class ProbingInputReader(Protocol):
    """The data Segment Extraction depends on. Postgres in prod, fakes in tests."""

    async def get_task_flags(self, task_id: UUID) -> list[FlagInput]: ...

    async def get_submission(self, task_id: UUID) -> SubmissionInput | None: ...


_FLAGS_SQL = """
SELECT flag_id, task_id, model_version, category, confidence, start_line, end_line
  FROM classifier_flags_for_probing
 WHERE task_id = $1
 ORDER BY start_line, end_line, flag_id
"""

_SUBMISSION_SQL = """
SELECT task_id, code_text, code_redacted_at
  FROM task_submissions
 WHERE task_id = $1
"""


def _parse(schema, row: Mapping[str, Any], what: str):
    try:
        return schema.model_validate(dict(row))
    except ValidationError as exc:
        raise ProbingInputError(f"{what} violates the C1/C2 contract: {exc}") from exc


class PostgresProbingReader:
    """``ProbingInputReader`` over the shared asyncpg pool (``backend.app.db.pool``)."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_task_flags(self, task_id: UUID) -> list[FlagInput]:
        rows = await self._pool.fetch(_FLAGS_SQL, task_id)
        return [_parse(FlagInput, row, f"mapped flag of task {task_id}") for row in rows]

    async def get_submission(self, task_id: UUID) -> SubmissionInput | None:
        row = await self._pool.fetchrow(_SUBMISSION_SQL, task_id)
        if row is None:
            return None
        return _parse(SubmissionInput, row, f"submission of task {task_id}")
