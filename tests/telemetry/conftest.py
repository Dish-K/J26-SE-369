"""Builder for raw telemetry events. ``at`` is always perf_now in milliseconds."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import pandas as pd
import pytest

for _folder in Path(__file__).resolve().parents:
    if (_folder / "shared_features").is_dir():
        sys.path.insert(0, str(_folder))
        break

from shared_features import ExitStatus, Kind, KeyClass, Origin, PasteOrigin, SourcePane  # noqa: E402
from backend.app.db.pool import close_db_pool, create_db_pool  # noqa: E402

T0 = pd.Timestamp("2026-10-09T10:00:00Z")
SESSION = UUID("11111111-1111-1111-1111-111111111111")
TASK = UUID("22222222-2222-2222-2222-222222222222")
TASK_B = UUID("33333333-3333-3333-3333-333333333333")


class EventStream:
    """Builds an ``events`` frame one event at a time."""

    def __init__(self, session_id=SESSION, task_id=TASK, t0=T0, load_seq=0):
        self.session_id = session_id
        self.task_id = task_id
        self.t0 = t0
        self.load_seq = load_seq
        self.rows: list[dict] = []

    def _add(self, kind, at, *, task_id=..., time=None, **fields):
        row = {
            "session_id": self.session_id,
            "task_id": self.task_id if task_id is ... else task_id,
            "load_seq": self.load_seq,
            "batch_seq": len(self.rows),
            "seq_in_batch": 0,
            "time": time if time is not None else self.t0 + pd.Timedelta(milliseconds=at),
            "perf_now": float(at),
            "kind": int(kind),
        }
        row.update(fields)
        self.rows.append(row)
        return self

    def reload(self, new_load_seq=None):
        """Simulate a page reload: a new load segment, a fresh clock."""
        self.load_seq = self.load_seq + 1 if new_load_seq is None else new_load_seq
        return self

    def keydown(self, code, at, *, key_class=KeyClass.letter, repeat=False,
                trusted=True, **kw):
        return self._add(Kind.keydown, at, key_code=code, key_class=int(key_class),
                         is_repeat=repeat, is_trusted=trusted, **kw)

    def keyup(self, code, at, **kw):
        return self._add(Kind.keyup, at, key_code=code, **kw)

    def types(self, code, down, up, **kw):
        """A complete keypress."""
        return self.keydown(code, down, **kw).keyup(code, up)

    def edit(self, at, *, origin=Origin.typed, start_line=1, end_line=None,
             inserted_lines=0, inserted_len=1, removed_len=0, paste_origin=None,
             version_id=None, start_col=1, end_col=1, range_offset=0, doc_len=100,
             change_index=0, **kw):
        return self._add(
            Kind.edit, at,
            origin=int(origin),
            start_line=start_line,
            end_line=start_line if end_line is None else end_line,
            start_col=start_col, end_col=end_col, range_offset=range_offset,
            inserted_lines=inserted_lines, inserted_len=inserted_len,
            removed_len=removed_len, doc_len=doc_len, change_index=change_index,
            paste_origin=None if paste_origin is None else int(paste_origin),
            version_id=version_id,
            **kw,
        )

    def paste(self, at, *, origin=PasteOrigin.external, **kw):
        kw.setdefault("inserted_len", 400)
        kw.setdefault("inserted_lines", 12)
        return self.edit(at, origin=Origin.pasted, paste_origin=origin, **kw)

    def blur(self, at, **kw):
        return self._add(Kind.blur, at, **kw)

    def focus(self, at, **kw):
        return self._add(Kind.focus, at, **kw)

    def visibility(self, at, *, hidden, **kw):
        return self._add(Kind.visibility_change, at, env_state=1 if hidden else 0, **kw)

    def copy(self, at, *, copied_len=50, pane=SourcePane.editor, **kw):
        return self._add(Kind.copy, at, copied_len=copied_len,
                         source_pane=int(pane), **kw)

    def cut(self, at, *, copied_len=50, pane=SourcePane.editor, **kw):
        return self._add(Kind.cut, at, copied_len=copied_len,
                         source_pane=int(pane), **kw)

    def run_started(self, at, **kw):
        return self._add(Kind.run_started, at, **kw)

    def run_finished(self, at, *, status=ExitStatus.ok, **kw):
        return self._add(Kind.run_finished, at, exit_status=int(status), **kw)

    def frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows)
        frame["seq_in_batch"] = 0
        return frame

    @property
    def last_at(self) -> float:
        return max((r["perf_now"] for r in self.rows), default=0.0)


def tasks_frame(*, task_id=TASK, session_id=SESSION, started_at=T0,
                submitted_at=None, extra=()):
    rows = [{
        "task_id": task_id,
        "session_id": session_id,
        "started_at": started_at,
        "submitted_at": submitted_at,
    }]
    rows.extend(extra)
    return pd.DataFrame(rows)


@pytest.fixture
def stream():
    return EventStream()


@pytest.fixture
def tasks():
    return tasks_frame()


def window_at(frame: pd.DataFrame, start):
    """The single row for the window beginning at ``start``."""
    start = pd.Timestamp(start)
    rows = frame[frame["window_start"] == start]
    assert len(rows) == 1, f"expected one window at {start}, found {len(rows)}"
    return rows.iloc[0]


# Skips rather than fails without a database, so C2 can still run tests/telemetry alone.
@pytest.fixture(scope="session")
def database():
    async def connect():
        await close_db_pool(await create_db_pool())

    try:
        asyncio.run(connect())
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"database not available: {exc}")


def on_connection(work):
    async def run():
        pool = await create_db_pool()
        try:
            async with pool.acquire() as connection:
                return await work(connection)
        finally:
            await close_db_pool(pool)

    return asyncio.run(run())


def run_sql(statement, *args):
    return on_connection(lambda connection: connection.fetch(statement, *args))
