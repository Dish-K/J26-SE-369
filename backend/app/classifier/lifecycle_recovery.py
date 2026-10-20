from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence
from uuid import UUID

import asyncpg

from .lifecycle_checkpoint import capture_checkpoint, restore_checkpoint
from .segment_builder import ClassifiedWindow
from .segment_lifecycle import LifecycleSnapshot, SegmentLifecycle, persist_closed_segments


class CheckpointConflict(RuntimeError):
    """Another writer advanced this stream; reload rather than overwrite it."""


def _document(value):
    return json.loads(value) if isinstance(value,str) else value


async def _validate_predictions(connection, flow: SegmentLifecycle) -> None:
    windows = [w for w,_ in flow.recovery_inputs().frames]
    if not windows:
        return
    rows = await connection.fetch(
        """SELECT p.*, fv.session_id, fv.window_start, fv.window_end, fv.run_id
        FROM classifier_window_predictions p JOIN feature_vectors fv
          ON fv.feature_vector_id=p.feature_vector_id AND fv.task_id=p.task_id
        WHERE p.prediction_id=ANY($1::uuid[])
        ORDER BY p.prediction_id FOR SHARE OF p, fv""",
        [w.prediction_id for w in windows],
    )
    indexed = {r["prediction_id"]:r for r in rows}
    if set(indexed) != {w.prediction_id for w in windows}:
        raise LookupError("Recovery requires every original stored prediction")
    for w in windows:
        row = indexed[w.prediction_id]
        fields = ("prediction_id","feature_vector_id","session_id","task_id","model_version",
                  "predicted_category","confidence","ai_positive","window_start","window_end")
        if (any(row[name] != getattr(w,name) for name in fields)
            or _document(row["class_probabilities"]) != dict(w.class_probabilities)
            or row["run_id"] != "live"):
            raise ValueError("Recovery inputs disagree with original stored predictions")


async def write_checkpoint(
    pool: asyncpg.Pool, flow: SegmentLifecycle, *, expected_revision: int,
) -> int:
    """Compare-and-swap one stream checkpoint; identical uncertain retries are safe."""
    if type(expected_revision) is not int or not 0 <= expected_revision < (1<<63)-1:
        raise ValueError("expected_revision must be a nonnegative database integer")
    document = capture_checkpoint(flow)
    identity = flow.recovery_inputs()
    # Validate the actual serialized representation before committing it.
    restore_checkpoint(document,session_id=identity.session_id,task_id=identity.task_id,
                       model_version=identity.model_version)
    async with pool.acquire() as connection:
        async with connection.transaction():
            await _validate_predictions(connection,flow)
            if expected_revision == 0:
                revision = await connection.fetchval(
                    """INSERT INTO classifier_lifecycle_checkpoints
                    (task_id,session_id,model_version,revision,checkpoint)
                    VALUES ($1,$2,$3,1,$4::jsonb)
                    ON CONFLICT (task_id,model_version) DO NOTHING RETURNING revision""",
                    identity.task_id,identity.session_id,identity.model_version,json.dumps(document),
                )
            else:
                revision = await connection.fetchval(
                    """UPDATE classifier_lifecycle_checkpoints
                    SET checkpoint=$4::jsonb,revision=revision+1,updated_at=now()
                    WHERE task_id=$1 AND session_id=$2 AND model_version=$3 AND revision=$5
                    RETURNING revision""",
                    identity.task_id,identity.session_id,identity.model_version,json.dumps(document),expected_revision,
                )
            if revision is not None:
                return revision
            row = await connection.fetchrow(
                """SELECT session_id,revision,checkpoint FROM classifier_lifecycle_checkpoints
                WHERE task_id=$1 AND model_version=$2 FOR UPDATE""",
                identity.task_id,identity.model_version,
            )
            if (row is not None and row["session_id"] == identity.session_id
                and row["revision"] == expected_revision+1
                and _document(row["checkpoint"]) == document):
                # The exact proposed save may already have committed before a
                # connection failure hid its acknowledgement from this caller.
                return row["revision"]
            raise CheckpointConflict("Checkpoint advanced or disappeared; reload the stream")


async def load_checkpoint(
    pool: asyncpg.Pool, *, session_id: UUID, task_id: UUID, model_version: str,
    create_if_missing: bool = False,
) -> tuple[SegmentLifecycle,int]:
    """Restore original C2 observations, never a fresh raw-telemetry localization."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            row = await connection.fetchrow(
                """SELECT session_id,revision,checkpoint FROM classifier_lifecycle_checkpoints
                WHERE task_id=$1 AND model_version=$2 FOR SHARE""",task_id,model_version,
            )
            if row is None:
                if create_if_missing is not True:
                    raise LookupError("No recovery checkpoint; explicit new-stream initialization is required")
                exists = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM tasks WHERE task_id=$1 AND session_id=$2)",
                    task_id,session_id,
                )
                if not exists:
                    raise LookupError("Recovery stream task/session does not exist")
                return SegmentLifecycle(session_id=session_id,task_id=task_id,model_version=model_version),0
            if row["session_id"] != session_id:
                raise ValueError("Recovery task belongs to another session")
            flow = restore_checkpoint(_document(row["checkpoint"]),session_id=session_id,
                                      task_id=task_id,model_version=model_version)
            await _validate_predictions(connection,flow)
            return flow,row["revision"]


class RecoverableSegmentLifecycle:
    """Internal runtime adapter: commit accepted state before returning it.

    No HTTP/WebSocket route, candidate access, auth check, queue drain, code
    mapping or UI publication is implemented here. Use snapshot() only after
    the operation succeeds. Cross-process stale writers get CheckpointConflict.
    """

    def __init__(self, pool, flow: SegmentLifecycle, revision: int):
        self._pool, self._flow, self._revision = pool,flow,revision
        self._lock = asyncio.Lock()

    @classmethod
    async def open(cls, pool, *, session_id: UUID, task_id: UUID, model_version: str,
                   create_if_missing: bool = False):
        flow,revision = await load_checkpoint(pool,session_id=session_id,task_id=task_id,
                                             model_version=model_version,create_if_missing=create_if_missing)
        return cls(pool,flow,revision)

    @property
    def revision(self) -> int:
        return self._revision

    def snapshot(self) -> LifecycleSnapshot:
        return self._flow.snapshot()

    def recovery_inputs(self):
        """Detached accepted inputs for internal submission completeness checks."""
        return self._flow.recovery_inputs()

    async def accept_window(
        self, window: ClassifiedWindow, *, edit_history: Sequence[Mapping[str,Any]] = (),
    ) -> LifecycleSnapshot:
        async with self._lock:
            candidate = deepcopy(self._flow)
            candidate.accept_window(window,edit_history=edit_history)
            revision = await write_checkpoint(self._pool,candidate,expected_revision=self._revision)
            self._flow,self._revision = candidate,revision
            return self.snapshot()

    async def finish(self, *, pending_live_processing_complete: bool) -> LifecycleSnapshot:
        async with self._lock:
            candidate = deepcopy(self._flow)
            candidate.finish(pending_live_processing_complete=pending_live_processing_complete)
            revision = await write_checkpoint(self._pool,candidate,expected_revision=self._revision)
            self._flow,self._revision = candidate,revision
            return self.snapshot()

    async def persist_closed(self) -> tuple[UUID,...]:
        async with self._lock:
            return await persist_closed_segments(self._flow,self._pool)
