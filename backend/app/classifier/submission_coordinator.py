from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID

import pandas as pd

from shared_features.extractor import window_grid
from shared_features.feature_schema import PARAMETERS, SCHEMA_VERSION, WINDOW_WIDTH

from .lifecycle_recovery import RecoverableSegmentLifecycle
from .segment_builder import ClassifiedWindow
from .segment_lifecycle import LifecycleSnapshot


class SubmissionInputConflict(ValueError):
    """Submission metadata and accepted observations cannot describe one stream."""


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Submission coordination requires timezone-aware datetimes")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class SubmissionWindow:
    window_start: datetime
    window_end: datetime
    ready_at: datetime


@dataclass(frozen=True)
class SubmissionPlan:
    """Trusted persisted metadata, not candidate-supplied authorization evidence.

    Required windows use the official grid and positive overlap with the task
    interval. A window starting exactly at submission has zero task coverage
    and is excluded. An instantaneous task therefore has no required windows.
    Existing live observations are never revised to match a submission plan.
    """

    session_id: UUID
    task_id: UUID
    model_version: str
    started_at: datetime
    submitted_at: datetime
    required_windows: tuple[SubmissionWindow, ...] = field(init=False)
    feature_schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self):
        if not all(isinstance(value, UUID) for value in
                   (self.session_id, self.task_id)):
            raise ValueError("Submission plan requires UUID identities")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("Submission plan requires a model version")
        start, end = _utc(self.started_at), _utc(self.submitted_at)
        if end < start:
            raise ValueError("Submission cannot precede task start")
        object.__setattr__(self, "started_at", start)
        object.__setattr__(self, "submitted_at", end)
        width = timedelta(seconds=WINDOW_WIDTH)
        grace = timedelta(milliseconds=PARAMETERS["window_grace_ms"])
        windows = tuple(
            SubmissionWindow(t.to_pydatetime(), t.to_pydatetime() + width,
                             t.to_pydatetime() + width + grace)
            for t in window_grid(pd.Timestamp(start), pd.Timestamp(end))
            if start < end and t < end and t + width > start
        )
        object.__setattr__(self, "required_windows", windows)


@dataclass(frozen=True)
class SubmissionProgress:
    """Internal progress only; complete means segment persistence, not C3 readiness."""

    phase: Literal["waiting_windows", "waiting_telemetry", "ready_to_finish",
                   "saving_flags", "complete"]
    pending_windows: tuple[SubmissionWindow, ...]
    next_ready_at: datetime | None
    lifecycle: LifecycleSnapshot


class SubmissionCoordinator:
    """Own one recovered lifecycle while draining a submitted task's windows.

    Do not mutate the wrapped lifecycle through a separate owner. Cross-worker
    races retain Increment 2's revision conflicts: reload rather than overwrite.
    The plan is reconstructed from authoritative persisted application metadata;
    it is not independently persisted by this classifier-side increment.
    """

    def __init__(self, plan: SubmissionPlan, lifecycle: RecoverableSegmentLifecycle):
        self._plan, self._lifecycle = plan, lifecycle
        self._lock = asyncio.Lock()
        self._accepted()

    @classmethod
    async def open(cls, pool, plan: SubmissionPlan, *, create_if_missing: bool = False):
        lifecycle = await RecoverableSegmentLifecycle.open(
            pool, session_id=plan.session_id, task_id=plan.task_id,
            model_version=plan.model_version, create_if_missing=create_if_missing,
        )
        return cls(plan, lifecycle)

    @property
    def plan(self) -> SubmissionPlan:
        return self._plan

    def _accepted(self) -> tuple[ClassifiedWindow, ...]:
        inputs = self._lifecycle.recovery_inputs()
        if (inputs.session_id, inputs.task_id, inputs.model_version) != (
            self.plan.session_id, self.plan.task_id, self.plan.model_version
        ):
            raise SubmissionInputConflict("Lifecycle belongs to another submission stream")
        windows = tuple(w for w, _ in inputs.frames)
        required = self.plan.required_windows
        if len(windows) > len(required):
            raise SubmissionInputConflict("Lifecycle contains windows outside the submission plan")
        for window, expected in zip(windows, required):
            if (window.window_start, window.window_end) != (
                expected.window_start, expected.window_end
            ):
                raise SubmissionInputConflict("Accepted windows must be a complete ordered plan prefix")
        if inputs.finished and len(windows) != len(required):
            raise SubmissionInputConflict("Finished lifecycle is missing required submission windows")
        return windows

    def progress(self, *, now: datetime, telemetry_complete: bool = False) -> SubmissionProgress:
        """Time alone never establishes telemetry or prediction completeness."""
        now = _utc(now)
        if type(telemetry_complete) is not bool:
            raise ValueError("telemetry_complete must be a trusted boolean signal")
        accepted = self._accepted()
        state = self._lifecycle.snapshot()
        pending = self.plan.required_windows[len(accepted):]
        future = tuple(w.ready_at for w in self.plan.required_windows if w.ready_at > now)
        if now < self.plan.submitted_at:
            future += (self.plan.submitted_at,)
        if state.finished:
            phase = "saving_flags" if state.pending_segments else "complete"
        elif pending or future:
            phase = "waiting_windows"
        elif not telemetry_complete:
            phase = "waiting_telemetry"
        else:
            phase = "ready_to_finish"
        return SubmissionProgress(phase, pending, min(future) if future else None, state)

    async def accept_window(
        self, window: ClassifiedWindow, *, now: datetime,
        edit_history: Sequence[Mapping[str, Any]] = (),
    ) -> SubmissionProgress:
        """Accept a ready stored prediction, including clean/unlocalized results.

        Changed duplicates and evidence are checked by the existing lifecycle.
        A missing earlier window is never silently skipped during submission.
        """
        now = _utc(now)
        async with self._lock:
            accepted = self._accepted()
            if (window.session_id, window.task_id, window.model_version) != (
                self.plan.session_id, self.plan.task_id, self.plan.model_version
            ):
                raise SubmissionInputConflict("Prediction belongs to another submission stream")
            duplicate = next((i for i, old in enumerate(accepted)
                              if old.prediction_id == window.prediction_id), None)
            index = len(accepted) if duplicate is None else duplicate
            if index >= len(self.plan.required_windows):
                raise SubmissionInputConflict("No further windows belong to this submission")
            expected = self.plan.required_windows[index]
            if (window.window_start, window.window_end) != (
                expected.window_start, expected.window_end
            ):
                raise SubmissionInputConflict("Deliver the next required window in order")
            if now < expected.ready_at:
                raise ValueError("Window has not reached its normal end plus grace")
            await self._lifecycle.accept_window(window, edit_history=edit_history)
            return self.progress(now=now)

    async def advance(self, *, now: datetime, telemetry_complete: bool) -> SubmissionProgress:
        """Finish only after all gates pass, then save/reconcile every closed flag.

        A failed save propagates its exception; the finished checkpoint remains
        recoverable with pending flags. Reopening forces Task 12 read-back/retry.
        A finished checkpoint already passed the barrier, so recovery can retry
        its saves without recreating a transient upstream acknowledgement.
        """
        async with self._lock:
            progress = self.progress(now=now, telemetry_complete=telemetry_complete)
            if progress.phase == "ready_to_finish":
                await self._lifecycle.finish(pending_live_processing_complete=True)
            elif not progress.lifecycle.finished:
                return progress
            await self._lifecycle.persist_closed()
            return self.progress(now=now, telemetry_complete=telemetry_complete)
