"""Raw telemetry events -> one feature vector per window.

Pure transformation: no database, no clock, no I/O. Real capture, the synthetic
generator and every degraded copy run through this same function, which is what
makes synthetic and real vectors comparable by construction. Never add a second
feature path -- change the registry and let the tests fail loudly.

Three rules here were expensive to learn and are easy to undo by accident:

1. **Order by ``(load_seq, batch_seq, seq_in_batch)``, never by time.** Browsers
   coarsen ``performance.now()``, so two events can share a timestamp, and a
   page reload produces a fresh clock offset, so ``time`` is not monotonic
   across a reload boundary. Specification 6.5.
2. **Recompute every duration from ``perf_now``.** The stored ``gap_ms`` column
   is never read. A degraded copy perturbs ``perf_now``; reading the stored
   column would hand back the clean session's timings and the whole degradation
   experiment would silently measure nothing. Specification 8.1.
3. **Flight excludes gaps over ``pause_threshold_ms``.** One 20-second think
   swamps ``std_flight_ms`` and the uniform-rhythm signal vanishes. Nothing is
   lost: the gap still lands in ``pause_ms`` and ends the burst.

``time`` *is* used, for one thing only: deciding which window an event falls in.
That is correct even across a reload, because the clock handshake re-runs and
the new wall clock is valid. Window membership and event ordering are different
questions and take different answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_schema import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    EVENT_SCHEMA,
    IDENTITY_COLUMNS,
    PARAMETERS,
    TASK_SCHEMA,
    WINDOW_STEP,
    WINDOW_WIDTH,
    ExitStatus,
    Kind,
    KeyClass,
    Origin,
    PasteOrigin,
    SourcePane,
    Visibility,
)

__all__ = ["extract_features", "extract_window", "window_grid"]

_ORDER = ["session_id", "load_seq", "batch_seq", "seq_in_batch"]
_SEGMENT = ["session_id", "task_id",       "load_seq"]
_STEP = pd.Timedelta(seconds=WINDOW_STEP)
_WIDTH = pd.Timedelta(seconds=WINDOW_WIDTH)
_TYPING_KINDS = [int(Kind.keydown), int(Kind.keyup), int(Kind.edit)]


# Window grid
def window_grid(started_at: pd.Timestamp, task_end: pd.Timestamp) -> pd.DatetimeIndex:
    """Window start times for one task.

    Epoch-aligned to the step, beginning ``WINDOW_WIDTH - WINDOW_STEP`` before
    the task's first bucket. That offset is what guarantees the specification's
    three-deep overlap holds *at the very start of a task* too, not only in the
    middle -- without it the opening seconds sit in fewer windows than the rest
    and their features are not comparable.
    """
    lead = _WIDTH - _STEP
    first = started_at.floor(_STEP) - lead
    last = max(task_end, started_at).floor(_STEP)
    return pd.date_range(first, last, freq=_STEP)


# Per-event derived timing, computed inside one (session, task, load_seq)
def _derive_timing(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["_non_reset_edit"] = (ev["kind"] != int(Kind.edit)) | (
        ev["origin"].fillna(-1) != int(Origin.reset)
    )
    ev["_is_typing"] = ev["kind"].isin(_TYPING_KINDS) & ev["_non_reset_edit"]
    ev["_is_keydown"] = ev["kind"] == int(Kind.keydown)
    ev["_repeat"] = ev["is_repeat"].fillna(False).astype(bool)
    ev["_fresh_keydown"] = ev["_is_keydown"] & ~ev["_repeat"]

    ev["gap_ms"] = np.nan
    ev["flight_ms"] = np.nan
    ev["dwell_ms"] = np.nan
    ev["burst_id"] = -1
    ev["unpaired"] = False

    burst_base = 0
    for _, idx in ev.groupby(_SEGMENT, sort=False, dropna=False).groups.items():
        seg = ev.loc[idx]

        # gap: time since the previous typing event, across kinds 0-2.
        typing = seg.index[seg["_is_typing"]]
        if len(typing):
            ev.loc[typing, "gap_ms"] = ev.loc[typing, "perf_now"].diff()

        # flight: between consecutive fresh keydowns. Long gaps are not flight.
        fresh = seg.index[seg["_fresh_keydown"]]
        if len(fresh):
            flight = ev.loc[fresh, "perf_now"].diff()
            flight[flight > PARAMETERS["pause_threshold_ms"]] = np.nan
            ev.loc[fresh, "flight_ms"] = flight
            # A new burst begins wherever flight is undefined -- the first
            # keystroke of the segment, or one that followed a long pause.
            ev.loc[fresh, "burst_id"] = burst_base + flight.isna().cumsum().to_numpy()
            burst_base = int(ev.loc[fresh, "burst_id"].max()) + 1

        _pair_dwell(ev, seg)

    return ev


def _pair_dwell(ev: pd.DataFrame, seg: pd.DataFrame) -> None:
    """Match each keyup to its keydown by key identity, not arrival order.

    Fast typists press the next key before releasing the last (``p down, r down,
    p up, r up``), so pairing adjacent events gives every fast typist wrong
    dwell values -- precisely the population the timing signal depends on.
    Auto-repeat rows take no part: they are not fresh presses.
    """
    open_at: dict[object, int] = {}
    for idx, kind, code, repeat, now in zip(
        seg.index, seg["kind"], seg["key_code"], seg["_repeat"], seg["perf_now"]
    ):
        if kind == int(Kind.keydown) and not repeat:
            if code in open_at:
                ev.at[open_at[code], "unpaired"] = True
            open_at[code] = idx
        elif kind == int(Kind.keyup):
            down = open_at.pop(code, None)
            if down is not None:
                ev.at[down, "dwell_ms"] = now - ev.at[down, "perf_now"]

    for idx in open_at.values():
        ev.at[idx, "unpaired"] = True


# Per-event environment state transitions
def _derive_environment(ev: pd.DataFrame) -> pd.DataFrame:
    """Mark the events that *change* focus state, and copies followed by a blur.

    ``focus_switch_count`` counts transitions into "away", not the events that
    report them -- a blur immediately followed by a hidden visibilitychange is
    one departure, not two.
    """
    ev["_away_transition"] = False
    ev["_copy_before_blur"] = False

    hidden = (ev["kind"] == int(Kind.visibility_change)) & (
        ev["env_state"].fillna(-1) == int(Visibility.hidden)
    )
    visible = (ev["kind"] == int(Kind.visibility_change)) & (
        ev["env_state"].fillna(-1) == int(Visibility.visible)
    )
    goes_away = (ev["kind"] == int(Kind.blur)) | (ev["kind"] == int(Kind.page_hide)) | hidden
    comes_back = (ev["kind"] == int(Kind.focus)) | visible

    for _, idx in ev.groupby(["session_id", "task_id"], sort=False, dropna=False).groups.items():
        seg = ev.loc[idx]
        away = False
        for i, is_away, is_back in zip(seg.index, goes_away.loc[idx], comes_back.loc[idx]):
            if is_away and not away:
                ev.at[i, "_away_transition"] = True
                away = True
            elif is_back:
                away = False

        # A copy or cut whose window into "focus left" closes within the cutoff.
        departures = seg.loc[goes_away.loc[idx], "perf_now"].to_numpy()
        if departures.size:
            copies = seg.index[seg["kind"].isin([int(Kind.copy), int(Kind.cut)])]
            for i in copies:
                now = ev.at[i, "perf_now"]
                after = departures[departures > now]
                if after.size and (after[0] - now) <= PARAMETERS["copy_before_blur_ms"]:
                    ev.at[i, "_copy_before_blur"] = True

    return ev


# Window-local aggregation
def _safe_div(num, den):
    """Ratio, or null where the denominator is zero.

    Never ``max(den, 1)``. A zero denominator means nothing was measured, and
    reporting 0.0 there is a confident claim about something never observed --
    "corrected nothing", "pasted none of their code". Specification 16.1.
    """
    num = pd.Series(num, dtype="float64")
    den = pd.Series(den, dtype="float64")
    return (num / den.where(den > 0)).astype("float64")


def _edit_lines(group: pd.DataFrame) -> tuple:
    """``min_line``, ``max_line`` and the count of distinct lines touched."""
    if group.empty:
        return (pd.NA, pd.NA, 0)
    start = group["start_line"].astype("Int64")
    span = (group["end_line"].astype("Int64") - start).fillna(0)
    added = group["inserted_lines"].astype("Int64").fillna(0)
    reach = start + span.where(span > added, added)

    touched: set[int] = set()
    for lo, hi in zip(start, reach):
        if pd.isna(lo):
            continue
        hi = lo if pd.isna(hi) else hi
        touched.update(range(int(lo), int(hi) + 1))
    if not touched:
        return (pd.NA, pd.NA, 0)
    return (min(touched), max(touched), len(touched))


def _aggregate_window(group: pd.DataFrame) -> dict:
    kind = group["kind"]
    keydowns = group[kind == int(Kind.keydown)]
    fresh = keydowns[~keydowns["_repeat"]]
    edits = group[(kind == int(Kind.edit)) & group["_non_reset_edit"]]

    dwell = group["dwell_ms"].dropna()
    flight = group["flight_ms"].dropna()

    keystrokes = len(keydowns)
    backspaces = int((keydowns["key_class"].fillna(-1) == int(KeyClass.backspace)).sum())

    # burst_len: the longest run *inside this window*. A burst crossing the
    # boundary is counted only as far as the window reaches.
    bursts = fresh[fresh["burst_id"] >= 0]
    burst_len = int(bursts.groupby("burst_id").size().max()) if len(bursts) else 0

    origin = edits["origin"].fillna(-1)
    pastes = edits[origin == int(Origin.pasted)]
    paste_origin = pastes["paste_origin"]
    min_line, max_line, edited_lines = _edit_lines(edits)

    runs_started = group[kind == int(Kind.run_started)]
    runs_done = group[kind == int(Kind.run_finished)]
    failed = runs_done["exit_status"].isin([int(ExitStatus.error), int(ExitStatus.timeout)])

    clip = group[kind.isin([int(Kind.copy), int(Kind.cut)])]
    gaps = group["gap_ms"].dropna()

    return {
        "keystroke_count": keystrokes,
        "mean_dwell_ms": dwell.mean() if len(dwell) else np.nan,
        "std_dwell_ms": dwell.std(ddof=0) if len(dwell) >= 2 else np.nan,
        "mean_flight_ms": flight.mean() if len(flight) else np.nan,
        "std_flight_ms": flight.std(ddof=0) if len(flight) >= 2 else np.nan,
        "backspace_count": backspaces,
        "_backspace_num": backspaces,
        "_keystroke_den": keystrokes,
        "_repeat_num": int(keydowns["_repeat"].sum()),
        "_untrusted_num": int((~keydowns["is_trusted"].fillna(True).astype(bool)).sum()),
        "_unpaired_num": int(fresh["unpaired"].sum()),
        "_fresh_den": len(fresh),
        "burst_len": burst_len,

        "edit_count": len(edits),
        "typing_edit_count": int((origin == int(Origin.typed)).sum()),
        "paste_edit_count": len(pastes),
        "external_paste_count": int((paste_origin == int(PasteOrigin.external)).sum()),
        "unknown_paste_count": int(
            (paste_origin.isna() | (paste_origin == int(PasteOrigin.unknown))).sum()
        ),
        "undo_count": int((origin == int(Origin.undo)).sum()),
        "redo_count": int((origin == int(Origin.redo)).sum()),
        "_paste_num": len(pastes),
        "_edit_den": len(edits),
        "inserted_len_sum": int(edits["inserted_len"].fillna(0).sum()),
        "inserted_len_max": (
            int(edits["inserted_len"].fillna(0).max()) if len(edits) else pd.NA
        ),
        "removed_len_sum": int(edits["removed_len"].fillna(0).sum()),
        "min_line": min_line,
        "max_line": max_line,
        "edited_line_count": edited_lines,
        "pause_ms": gaps.max() if len(gaps) else np.nan,

        "run_count": len(runs_started),
        "_run_fail_num": int(failed.sum()),
        "_run_done_den": len(runs_done),

        "blur_count": int((kind == int(Kind.blur)).sum()),
        "focus_count": int((kind == int(Kind.focus)).sum()),
        "visibility_change_count": int((kind == int(Kind.visibility_change)).sum()),
        "focus_switch_count": int(group["_away_transition"].sum()),
        "copy_count": len(clip),
        "copy_len_sum": int(clip["copied_len"].fillna(0).sum()),
        "problem_copy_count": int(
            (clip["source_pane"].fillna(-1) == int(SourcePane.problem_statement)).sum()
        ),
        "copy_before_blur": int(group["_copy_before_blur"].sum()),
    }


# Task-to-date run features
def _task_to_date(task_events: pd.DataFrame, started_at, window_ends) -> pd.DataFrame:
    """Run features that look back to the start of the task, not just the window.

    Each window sees only what had happened by its ``window_end``. A window must
    never be able to see the future -- a live prediction made at 10:00:30 that
    used a run at 10:02:00 would be untrainable and, during an interview, wrong.
    """
    starts = task_events[task_events["kind"] == int(Kind.run_started)].sort_values("time")
    done = task_events[task_events["kind"] == int(Kind.run_finished)].sort_values("time")
    edits = task_events[
        (task_events["kind"] == int(Kind.edit)) & task_events["_non_reset_edit"]
    ].sort_values("time")

    # searchsorted cannot mix tz-aware and tz-naive values, and a tz-aware
    # Series lands in numpy as an object array. Drop to naive UTC once, here,
    # rather than at every comparison below.
    def _naive(values) -> np.ndarray:
        return pd.DatetimeIndex(values).tz_convert(None).to_numpy()

    start_times = _naive(starts["time"])
    done_times = _naive(done["time"])
    done_ok = (done["exit_status"].fillna(-1) == int(ExitStatus.ok)).to_numpy()
    edit_times = _naive(edits["time"])
    task_start = _naive([started_at])[0]

    rows = []
    for end in _naive(window_ends):
        cutoff = end
        n_started = int(np.searchsorted(start_times, cutoff, side="left"))
        n_done = int(np.searchsorted(done_times, cutoff, side="left"))

        first_ok = pd.NA
        ok_so_far = np.flatnonzero(done_ok[:n_done])
        if ok_so_far.size:
            first_ok = int(ok_so_far[0]) + 1          # 1-based, includes the success

        first_run = np.nan
        if n_started:
            first_run = (start_times[0] - task_start) / np.timedelta64(1, "s")

        between = np.nan
        if n_started >= 2:
            counts = [
                int(np.searchsorted(edit_times, start_times[i + 1], side="left")
                    - np.searchsorted(edit_times, start_times[i], side="left"))
                for i in range(n_started - 1)
            ]
            between = float(np.mean(counts))

        rows.append({
            "runs_before_first_success": first_ok,
            "time_to_first_run": first_run,
            "edits_between_runs": between,
        })

    return pd.DataFrame(rows, index=pd.Index(range(len(rows))))


# Public API
def extract_features(
    events: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    run_id: str = "live",
    window_starts: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Feature vectors for every window of every task present in ``events``.

    Parameters
    ----------
    events
        Raw rows matching ``EVENT_SCHEMA``. Rows with a null ``task_id`` are
        dropped: they happened outside any task and belong to no window.
    tasks
        One row per task, matching ``TASK_SCHEMA``.
    run_id
        ``'live'`` during the task, ``'final'`` after submission, anything else
        for a degradation variant. Part of the vector's identity, so a degraded
        copy never overwrites the clean one.
    window_starts
        Restrict output to these window starts. Used by :func:`extract_window`.
    """
    events = EVENT_SCHEMA.validate(events.reset_index(drop=True), lazy=False)
    tasks = TASK_SCHEMA.validate(tasks.reset_index(drop=True), lazy=False)

    for column in ("origin", "is_repeat", "is_trusted", "key_class", "key_code",
                   "paste_origin", "env_state", "exit_status", "copied_len",
                   "source_pane", "start_line", "end_line", "inserted_lines",
                   "inserted_len", "removed_len"):
        if column not in events.columns:
            events[column] = pd.NA

    events = events[events["task_id"].notna()].copy()
    events = events.sort_values(_ORDER, kind="mergesort").reset_index(drop=True)

    if events.empty and tasks.empty:
        return _empty_frame()

    events = _derive_timing(events)
    events = _derive_environment(events)

    frames = [
        _extract_one_task(events, task, run_id, window_starts)
        for task in tasks.to_dict("records")
    ]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return _empty_frame()

    out = pd.concat(frames, ignore_index=True)
    return FEATURE_SCHEMA.validate(out, lazy=False)


def _extract_one_task(events, task, run_id, window_starts):
    task_id = task["task_id"]
    started_at = pd.Timestamp(task["started_at"])
    mine = events[events["task_id"] == task_id]

    submitted = task.get("submitted_at")
    if submitted is not None and not pd.isna(submitted):
        task_end = pd.Timestamp(submitted)
    elif not mine.empty:
        task_end = mine["time"].max()
    else:
        task_end = started_at

    grid = window_grid(started_at, task_end)
    if window_starts is not None:
        grid = grid.intersection(pd.DatetimeIndex(window_starts))
    if grid.empty:
        return None

    # Every event belongs to the three consecutive windows that contain it.
    if mine.empty:
        exploded = mine.assign(window_start=pd.Series(dtype="datetime64[ns, UTC]"))
    else:
        base = mine["time"].dt.floor(_STEP)
        exploded = pd.concat(
            [mine.assign(window_start=base - k * _STEP) for k in range(_WIDTH // _STEP)],
            ignore_index=True,
        )
        exploded = exploded[exploded["window_start"].isin(grid)]

    aggregated = {
        start: _aggregate_window(group)
        for start, group in exploded.groupby("window_start", sort=True)
    }
    empty = _aggregate_window(exploded.iloc[0:0])
    rows = [aggregated.get(start, empty) for start in grid]
    frame = pd.DataFrame(rows, index=pd.Index(range(len(grid))))

    window_end = grid + _WIDTH
    to_date = _task_to_date(mine, started_at, window_end)
    frame = pd.concat([frame, to_date], axis=1)

    # Ratios are formed here, from counts, so the numerator and denominator can
    # never drift apart the way two separately-defined features do.
    frame["backspace_rate"] = _safe_div(frame["_backspace_num"], frame["_keystroke_den"])
    frame["is_repeat_rate"] = _safe_div(frame["_repeat_num"], frame["_keystroke_den"])
    frame["untrusted_key_rate"] = _safe_div(frame["_untrusted_num"], frame["_keystroke_den"])
    frame["unpaired_keydown_rate"] = _safe_div(frame["_unpaired_num"], frame["_fresh_den"])
    frame["paste_ratio"] = _safe_div(frame["_paste_num"], frame["_edit_den"])
    frame["run_error_rate"] = _safe_div(frame["_run_fail_num"], frame["_run_done_den"])

    overlap_start = pd.Series(grid, index=frame.index).clip(lower=started_at)
    overlap_end = pd.Series(window_end, index=frame.index).clip(upper=task_end)
    frame["window_task_coverage_ms"] = (
        (overlap_end - overlap_start).dt.total_seconds() * 1000.0
    ).clip(lower=0.0)

    frame["session_id"] = task["session_id"]
    frame["task_id"] = task_id
    frame["window_start"] = grid
    frame["window_end"] = window_end
    frame["run_id"] = run_id

    return frame[list(IDENTITY_COLUMNS) + list(FEATURE_NAMES)]


def extract_window(
    events: pd.DataFrame,
    tasks: pd.DataFrame,
    window_start: pd.Timestamp,
    *,
    run_id: str = "live",
) -> pd.DataFrame:
    """One window, for the live scheduler.

    The caller must pass a slice covering
    ``[window_start - live_lookback_ms, window_start + WINDOW_WIDTH)`` **plus**
    every run event since the task started -- dwell and flight pairs straddling
    the window's start need the lead-in, and the task-to-date run features need
    the whole task. Anything outside the window is used for context only; it
    never lands in the output.
    """
    return extract_features(
        events, tasks, run_id=run_id,
        window_starts=pd.DatetimeIndex([pd.Timestamp(window_start)]),
    )


def _empty_frame() -> pd.DataFrame:
    return FEATURE_SCHEMA.validate(
        pd.DataFrame(columns=list(IDENTITY_COLUMNS) + list(FEATURE_NAMES))
    )
