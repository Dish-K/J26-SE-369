from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import UUID

import asyncpg

from .flag_identity import flag_id_for_segment
from .flag_payload import map_segment_to_flag
from .flag_persistence import save_segment_flag
from .segment_builder import ClassifiedWindow, SegmentDraft, build_segment_drafts


class LifecycleInputConflict(ValueError):
    """An accepted window/evidence stream would be revised or replayed differently."""


@dataclass(frozen=True)
class LifecycleSnapshot:
    """Detached internal state. Historical ranges are not live editor coordinates."""

    active_segments: tuple[SegmentDraft, ...]
    closed_segments: tuple[SegmentDraft, ...]
    persisted_flag_ids: tuple[UUID, ...]
    processed_window_count: int
    finished: bool

    @property
    def pending_segments(self) -> tuple[SegmentDraft, ...]:
        saved = set(self.persisted_flag_ids)
        return tuple(s for s in self.closed_segments if flag_id_for_segment(s) not in saved)


@dataclass(frozen=True)
class LifecycleRecoveryInputs:
    """Original accepted inputs, not a new C1 or dashboard contract."""

    session_id: UUID
    task_id: UUID
    model_version: str
    frames: tuple[tuple[ClassifiedWindow, tuple[dict[str, Any], ...]], ...]
    finished: bool


class SegmentLifecycle:
    """One session/task/model stream; callers serialize operations on an instance.

    Rebuilds the accepted prefix with the existing builder for behavioral parity.
    This first increment keeps that prefix in memory and makes no throughput or
    restart-recovery claim. Never rebuild original live evidence from a fresh
    unrestricted events query and assume that it is identical.
    """

    def __init__(self, *, session_id: UUID, task_id: UUID, model_version: str):
        if not isinstance(session_id, UUID) or not isinstance(task_id, UUID):
            raise ValueError("Lifecycle requires UUID session and task identities")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version is required")
        self._session_id = session_id
        self._task_id = task_id
        self._model_version = model_version
        self._windows: tuple[ClassifiedWindow, ...] = ()
        self._history: tuple[dict[str, Any], ...] = ()
        self._segments: tuple[SegmentDraft, ...] = ()
        self._closed_ids: set[UUID] = set()
        self._persisted: set[UUID] = set()
        self._finished = False
        self._recovery_frames: tuple[tuple[ClassifiedWindow, tuple[dict[str, Any], ...]], ...] = ()

    def _check_stream(self, window: ClassifiedWindow) -> None:
        if (
            window.session_id != self._session_id
            or window.task_id != self._task_id
            or window.model_version != self._model_version
        ):
            raise ValueError("Window belongs to another lifecycle stream")

    def _combine_history(
        self, incoming: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        # Exact repeated history rows across calls are harmless; conflicting rows
        # must not quietly change an already accepted segment's coordinates.
        rows = list(deepcopy(self._history))
        keys = {}
        names = ("load_seq", "version_id", "batch_seq", "seq_in_batch", "change_index")
        for index, row in enumerate(rows):
            keys[tuple(row[name] for name in names)] = index
        delivered_keys = set()
        for source in incoming:
            row = deepcopy(dict(source))
            if row.get("session_id") != self._session_id or row.get("task_id") != self._task_id:
                raise ValueError("Edit history belongs to another lifecycle stream")
            if any(type(row.get(name)) is not int or row[name] < (1 if name == "version_id" else 0)
                   for name in names):
                raise ValueError("Edit history requires valid event and document identities")
            key = tuple(row[name] for name in names)
            if key in delivered_keys:
                raise LifecycleInputConflict("Duplicate edit identity within one history delivery")
            delivered_keys.add(key)
            if key in keys:
                if rows[keys[key]] != row:
                    raise LifecycleInputConflict("Previously accepted edit evidence was changed")
            else:
                keys[key] = len(rows)
                rows.append(row)
        return tuple(rows)

    def accept_window(
        self, window: ClassifiedWindow, *, edit_history: Sequence[Mapping[str, Any]] = (),
    ) -> LifecycleSnapshot:
        """Accept the next completed live window, or an identical delivery retry.

        Readiness and ordering are caller responsibilities. A later window that
        establishes a genuine stream gap follows the builder's existing rule;
        network disconnection or elapsed wall time alone never closes a segment.
        Newly supplied history may help align the NEW window but may not revise
        results already accepted for an earlier prefix.
        """
        self._check_stream(window)
        candidate = deepcopy(window)
        history = self._combine_history(edit_history)
        matching = next((old for old in self._windows
                         if old.prediction_id == candidate.prediction_id), None)
        if matching is not None:
            if matching != candidate or history != self._history:
                raise LifecycleInputConflict("Repeated prediction has changed data or evidence")
            return self.snapshot()
        if self._finished:
            raise RuntimeError("Lifecycle is finished; no new windows may be appended")
        if any(old.feature_vector_id == candidate.feature_vector_id for old in self._windows):
            raise LifecycleInputConflict("Feature window already has a different prediction ID")

        # Validate with the existing builder before comparing timestamps, and
        # validate each localizable region through the approved storage mapper.
        for segment in build_segment_drafts([candidate], edit_history=history):
            map_segment_to_flag(segment)
        if self._windows and candidate.window_start <= self._windows[-1].window_start:
            raise LifecycleInputConflict("Completed windows must be delivered in stream order")
        if self._windows and history != self._history:
            before = build_segment_drafts(self._windows, edit_history=history)
            if before != self._segments:
                raise LifecycleInputConflict("New evidence would revise an accepted window prefix")

        windows = self._windows + (candidate,)
        segments = build_segment_drafts(windows, edit_history=history)
        indexed: dict[UUID, SegmentDraft] = {}
        for segment in segments:
            values = map_segment_to_flag(segment)
            if values.flag_id in indexed:
                raise LifecycleInputConflict("Different segment drafts share one stable identity")
            indexed[values.flag_id] = segment
        for old in self._segments:
            flag_id = flag_id_for_segment(old)
            if flag_id in self._closed_ids and indexed.get(flag_id) != old:
                raise LifecycleInputConflict("A closed historical segment would change")

        # In the existing builder every region in the latest window either
        # extends one draft or creates one. Only those drafts remain active.
        closed = {flag_id_for_segment(s) for s in segments
                  if s.last_window_start != candidate.window_start}
        if not self._closed_ids <= closed:
            raise LifecycleInputConflict("A closed segment would become active again")
        self._windows, self._history, self._segments = windows, history, segments
        self._closed_ids = closed
        self._recovery_frames += ((deepcopy(candidate), deepcopy(history)),)
        return self.snapshot()

    def recovery_inputs(self) -> LifecycleRecoveryInputs:
        """Return detached original observations and history as accepted per step.

        A recovery checkpoint must freeze these inputs rather than re-query
        current telemetry. Persistence acknowledgements are deliberately not
        included: recovered closed flags must be checked/retried through Task 12.
        """
        return deepcopy(LifecycleRecoveryInputs(
            session_id=self._session_id, task_id=self._task_id,
            model_version=self._model_version, frames=self._recovery_frames,
            finished=self._finished,
        ))

    def snapshot(self) -> LifecycleSnapshot:
        """Read state without closing segments; safe to inspect during an interruption."""
        active, closed = [], []
        for segment in self._segments:
            (closed if flag_id_for_segment(segment) in self._closed_ids else active).append(segment)
        persisted = tuple(flag_id_for_segment(s) for s in closed
                          if flag_id_for_segment(s) in self._persisted)
        return deepcopy(LifecycleSnapshot(
            active_segments=tuple(active), closed_segments=tuple(closed),
            persisted_flag_ids=persisted, processed_window_count=len(self._windows),
            finished=self._finished,
        ))

    def finish(self, *, pending_live_processing_complete: bool) -> LifecycleSnapshot:
        """Close remaining segments after the caller establishes the submission barrier.

        This does not establish the C1 cutoff, process partial windows, recompute
        run_id='final' vectors, save task scores, or mark submission ready for C3.
        Persistence must subsequently succeed for every pending segment.
        """
        if pending_live_processing_complete is not True:
            raise ValueError("Pending live processing must be complete before finishing")
        self._closed_ids = {flag_id_for_segment(s) for s in self._segments}
        self._finished = True
        return self.snapshot()

    def _acknowledge_persisted(self, flag_id: UUID) -> None:
        if flag_id not in self._closed_ids:
            raise ValueError("Only closed segments can be acknowledged as persisted")
        self._persisted.add(flag_id)


async def persist_closed_segments(
    lifecycle: SegmentLifecycle, pool: asyncpg.Pool,
) -> tuple[UUID, ...]:
    """Save pending closed flags through Task 12, acknowledging each success.

    Uses the already approved empty verified-evidence default. This coordinator
    does not invent named evidence. One save is atomic per flag and its links;
    a later failure does not undo earlier successful flags. On error/cancellation
    the failing and remaining flags stay pending for an identical retry.

    The caller must serialize this operation with other operations on the same
    coordinator. Distributed worker coordination is outside this increment.
    """
    saved = []
    for segment in lifecycle.snapshot().pending_segments:
        expected_id = flag_id_for_segment(segment)
        flag_id = await save_segment_flag(pool, segment)
        if flag_id != expected_id:
            raise RuntimeError("Persistence returned an unexpected segment identity")
        lifecycle._acknowledge_persisted(flag_id)
        saved.append(flag_id)
    return tuple(saved)
