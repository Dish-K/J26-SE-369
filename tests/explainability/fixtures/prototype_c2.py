"""Test access to the C4 PROTOTYPE fixtures.

The prototype model, C2 emulator, scenarios and XAI background live in
``ml.explainability.demo.prototype`` so the PP1 demo API can use them without
importing from tests. This module re-exports them and adds test-only helpers.
"""

import json
from typing import Any

from ml.explainability.demo.prototype import *  # noqa: F403
from ml.explainability.demo.prototype import (  # noqa: F401  (private helpers used by tests)
    C2Records,
    _task_predictions,
    _training_frame,
)


def as_db_row(row: dict[str, Any]) -> dict[str, Any]:
    """Shape a row like asyncpg returns it: JSONB as text, TEXT[] as list."""
    out = dict(row)
    for column in ("class_probabilities", "features"):
        if column in out:
            out[column] = json.dumps(out[column])
    if "evidence_kinds" in out:
        out["evidence_kinds"] = list(out["evidence_kinds"])
    return out


class FakeAsyncpgPool:
    """Answers PostgresC2Repository's queries from emulated C2 rows.

    Dispatches on the table each query reads, so the repository's real SQL
    strings and parameter binding are exercised without a database.
    """

    def __init__(self, records: C2Records) -> None:
        self.records = records
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        key = args[0]
        if "FROM classifier_flags" in query:
            table = self.records.flags
        elif "FROM feature_vectors" in query:
            table = self.records.feature_vectors
        elif "WHERE p.prediction_id = $1" in query:
            table = self.records.predictions
        else:
            raise AssertionError(f"unexpected fetchrow query: {query}")
        row = table.get(key)
        return None if row is None else as_db_row(row)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        if "FROM classifier_flag_predictions" in query:
            rows = [self.records.predictions[p] for p in self.records.members.get(args[0], [])]
        elif "WHERE p.task_id = $1" in query:
            rows = _task_predictions(self.records, args[0], args[1])
        else:
            raise AssertionError(f"unexpected fetch query: {query}")
        return [as_db_row(row) for row in rows]
