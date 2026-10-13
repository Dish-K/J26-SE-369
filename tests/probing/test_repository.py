import asyncio
from uuid import uuid4

import pytest

from backend.app.probing.errors import ProbingInputError
from backend.app.probing.repository import PostgresProbingReader


class FakePool:
    def __init__(self, rows=None, row=None):
        self._rows, self._row = rows or [], row
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self._rows

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        return self._row


def flag_row(task_id, **overrides):
    row = {
        "flag_id": uuid4(),
        "task_id": task_id,
        "model_version": "model-v1",
        "category": "ide_ai",
        "confidence": 0.7,
        "start_line": 4,
        "end_line": 6,
    }
    row.update(overrides)
    return row


def test_flags_are_read_from_the_view_only_and_parsed():
    task = uuid4()
    pool = FakePool(rows=[flag_row(task)])
    flags = asyncio.run(PostgresProbingReader(pool).get_task_flags(task))
    assert [(f.start_line, f.end_line, f.category) for f in flags] == [(4, 6, "ide_ai")]
    sql, args = pool.calls[0]
    assert "classifier_flags_for_probing" in sql
    assert "detected_start_line" not in sql  # historical coordinates are never read
    assert args == (task,)


def test_submission_reads_only_the_three_columns():
    task = uuid4()
    pool = FakePool(row={"task_id": task, "code_text": "x = 1\n", "code_redacted_at": None})
    submission = asyncio.run(PostgresProbingReader(pool).get_submission(task))
    assert submission.code_text == "x = 1\n"
    assert "FROM task_submissions" in pool.calls[0][0]


def test_missing_submission_is_none():
    assert asyncio.run(PostgresProbingReader(FakePool(row=None)).get_submission(uuid4())) is None


def test_row_violating_the_contract_raises():
    task = uuid4()
    bad = FakePool(rows=[flag_row(task, start_line=9, end_line=2)])
    with pytest.raises(ProbingInputError):
        asyncio.run(PostgresProbingReader(bad).get_task_flags(task))
