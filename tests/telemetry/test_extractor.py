"""Extractor behaviour. Several tests guard decisions nothing else protects, so keep them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared_features import (
    FEATURE_SCHEMA,
    ExitStatus,
    KeyClass,
    Origin,
    PasteOrigin,
    SourcePane,
    extract_features,
    extract_window,
)

from conftest import SESSION, T0, TASK, TASK_B, EventStream, tasks_frame, window_at


def run(stream, tasks=None, **kw):
    return extract_features(stream.frame(), tasks if tasks is not None else tasks_frame(), **kw)


# Window geometry
def test_one_event_lands_in_exactly_three_windows():
    """The three-deep overlap C2's segment merging depends on."""
    out = run(EventStream().keydown("KeyA", 12_000))
    holding = out[out["keystroke_count"] > 0]
    assert len(holding) == 3


def test_windows_are_30s_wide_and_10s_apart():
    out = run(EventStream().keydown("KeyA", 12_000))
    assert ((out["window_end"] - out["window_start"]) == pd.Timedelta(seconds=30)).all()
    starts = out["window_start"].sort_values().diff().dropna().unique()
    assert list(starts) == [pd.Timedelta(seconds=10)]


def test_overlap_holds_at_the_very_start_of_a_task():
    """Without the lead-in, the opening seconds sit in fewer windows than the
    rest and their features are not comparable to mid-task windows."""
    out = run(EventStream().keydown("KeyA", 100))
    assert len(out[out["keystroke_count"] > 0]) == 3


def test_empty_windows_are_still_emitted():
    s = EventStream().keydown("KeyA", 1_000).keydown("KeyB", 61_000)
    out = run(s)
    assert (out["keystroke_count"] == 0).any(), "the quiet middle must be represented"


def test_window_task_coverage_marks_the_short_edge_windows():
    """The first windows of a task overlap it only partly, so their counts are
    naturally low. Coverage makes that visible instead of misleading."""
    s = EventStream()
    for i in range(12):
        s.types("KeyA", 1_000 + i * 1_000, 1_050 + i * 1_000)
    out = run(s, tasks_frame(submitted_at=T0 + pd.Timedelta(seconds=30)))

    first = out.iloc[0]
    assert first["window_task_coverage_ms"] < 30_000
    assert (out["window_task_coverage_ms"] <= 30_000).all()
    assert (out["window_task_coverage_ms"] >= 0).all()


# Keystroke timing
def test_rollover_dwell_pairs_by_key_identity():
    """Fast typists press the next key before releasing the last. Pairing by
    arrival order gives every fast typist wrong dwell values."""
    s = EventStream()
    s.keydown("KeyP", 1_000).keydown("KeyR", 1_060)
    s.keyup("KeyP", 1_100).keyup("KeyR", 1_200)

    row = window_at(run(s), T0)
    assert row["mean_dwell_ms"] == pytest.approx((100 + 140) / 2)


def test_repeats_are_excluded_from_timing_but_counted():
    s = EventStream()
    s.types("KeyA", 1_000, 1_080)
    s.keydown("KeyA", 1_200, repeat=True)
    s.keydown("KeyA", 1_300, repeat=True)
    s.types("KeyB", 2_000, 2_090)

    row = window_at(run(s), T0)
    assert row["keystroke_count"] == 4, "auto-repeat still counts as a keystroke"
    assert row["is_repeat_rate"] == pytest.approx(0.5)
    assert row["mean_dwell_ms"] == pytest.approx(85.0), "repeats contribute no dwell"
    assert row["mean_flight_ms"] == pytest.approx(1_000.0), "KeyA -> KeyB, not via repeats"


def test_a_lost_keyup_shows_up_as_an_unpaired_keydown():
    s = EventStream().types("KeyA", 1_000, 1_080).keydown("KeyB", 1_200)
    row = window_at(run(s), T0)
    assert row["unpaired_keydown_rate"] == pytest.approx(0.5)
    assert row["mean_dwell_ms"] == pytest.approx(80.0)


def test_flight_is_not_gap():
    """gap_ms on a keydown measures from the previous keyup or edit. Using it as
    flight underestimates by roughly one dwell, on every keystroke, forever."""
    s = EventStream()
    s.keydown("KeyA", 1_000).keyup("KeyA", 1_140).edit(1_150)
    s.keydown("KeyB", 1_300)

    row = window_at(run(s), T0)
    assert row["mean_flight_ms"] == pytest.approx(300.0), "keydown to keydown"
    assert row["pause_ms"] == pytest.approx(150.0), "gap is a different quantity"


def test_a_long_pause_leaves_flight_and_lands_in_pause_ms():
    s = EventStream()
    s.keydown("KeyF", 1_000).keydown("KeyO", 1_150).keydown("KeyR", 1_330)
    s.keydown("KeyI", 21_330).keydown("KeyN", 21_490)

    row = window_at(run(s), T0 + pd.Timedelta(seconds=10))
    assert row["mean_flight_ms"] < 500, "the 20s think must not enter flight"
    assert row["pause_ms"] == pytest.approx(20_000.0)


def test_the_flight_cutoff_is_strictly_greater_than_the_threshold():
    """C2 asked for the exact boundary. 2000.0 is kept; 2000.1 is not."""
    kept = EventStream().keydown("KeyA", 0).keydown("KeyB", 2_000)
    assert window_at(run(kept), T0)["mean_flight_ms"] == pytest.approx(2_000.0)

    dropped = EventStream().keydown("KeyA", 0).keydown("KeyB", 2_001)
    assert pd.isna(window_at(run(dropped), T0)["mean_flight_ms"])


def test_burst_len_is_the_longest_run_not_the_final_one():
    """A window ending mid-burst must not report a short burst as the answer."""
    s = EventStream()
    for i, at in enumerate([0, 150, 300, 450, 600]):      # a run of five
        s.keydown(f"Key{i}", at)
    s.keydown("KeyX", 9_000)                              # long gap, then two
    s.keydown("KeyY", 9_150)

    assert window_at(run(s), T0)["burst_len"] == 5


# Degradation and ordering
def test_a_deliberately_wrong_stored_gap_ms_is_ignored():
    """Guards the degradation path. The study perturbs perf_now; if the
    extractor read the stored gap_ms it would return the clean session's
    timings and the experiment would silently measure nothing."""
    s = EventStream().keydown("KeyA", 1_000).keydown("KeyB", 1_200).edit(1_400)
    frame = s.frame()
    frame["gap_ms"] = 999_999.0

    out = extract_features(frame, tasks_frame())
    row = window_at(out, T0)
    assert row["pause_ms"] == pytest.approx(200.0)
    assert row["mean_flight_ms"] == pytest.approx(200.0)


def test_shuffled_rows_and_equal_perf_now_give_the_same_output():
    """Ordering must come from the sequence triple. Browsers coarsen
    performance.now(), so two events can legitimately share a value."""
    s = EventStream()
    s.keydown("KeyA", 15_342).keyup("KeyA", 15_342).edit(15_342)
    s.keydown("KeyB", 15_500)

    ordered = extract_features(s.frame(), tasks_frame())
    shuffled = extract_features(
        s.frame().sample(frac=1.0, random_state=7).reset_index(drop=True),
        tasks_frame(),
    )
    pd.testing.assert_frame_equal(
        ordered.sort_values("window_start").reset_index(drop=True),
        shuffled.sort_values("window_start").reset_index(drop=True),
    )


def test_timing_never_crosses_a_reload_boundary():
    """A reload resets the clock. A flight computed across it is meaningless."""
    s = EventStream().keydown("KeyA", 20_000)
    s.reload().keydown("KeyB", 50) # perf_now restarted near zero
    row = window_at(run(s), T0)
    assert pd.isna(row["mean_flight_ms"]), "no flight spans a reload"


def test_timing_never_crosses_a_task_boundary():
    s = EventStream().keydown("KeyA", 1_000)
    s.task_id = TASK_B
    s.keydown("KeyB", 1_200)

    tasks = tasks_frame(extra=[{
        "task_id": TASK_B, "session_id": SESSION,
        "started_at": T0, "submitted_at": None,
    }])
    out = extract_features(s.frame(), tasks)
    assert out["mean_flight_ms"].isna().all()


# Edits
def test_a_reset_template_load_is_invisible_to_every_feature():
    """Otherwise C1's own starter template manufactures the external_ai
    signature: a large insertion with no keystrokes behind it."""
    s = EventStream().edit(100, origin=Origin.reset, inserted_len=900,
                           inserted_lines=40, start_line=1, end_line=1)
    s.types("KeyA", 5_000, 5_080)

    row = window_at(run(s), T0)
    assert row["edit_count"] == 0
    assert row["inserted_len_sum"] == 0
    assert pd.isna(row["min_line"])
    assert pd.isna(row["paste_ratio"])


def test_a_paste_produces_the_expected_edit_features():
    s = EventStream()
    s.edit(1_000, start_line=5, end_line=5, inserted_len=10, inserted_lines=0)
    s.paste(2_000, start_line=12, end_line=12, inserted_len=400, inserted_lines=29)

    row = window_at(run(s), T0)
    assert row["edit_count"] == 2
    assert row["paste_edit_count"] == 1
    assert row["paste_ratio"] == pytest.approx(0.5)
    assert row["inserted_len_max"] == 400
    assert row["min_line"] == 5
    assert row["max_line"] == 41
    assert row["edited_line_count"] == 31          # line 5, plus lines 12-41


def test_paste_origin_splits_confirmed_external_from_unknown():
    """An undetermined origin must not count as a confirmed one."""
    s = EventStream()
    s.paste(1_000, origin=PasteOrigin.external)
    s.paste(2_000, origin=PasteOrigin.internal)
    s.paste(3_000, origin=PasteOrigin.unknown)
    s.paste(4_000, origin=None)

    row = window_at(run(s), T0)
    assert row["paste_edit_count"] == 4
    assert row["external_paste_count"] == 1, "confirmed external only"
    assert row["unknown_paste_count"] == 2, "unknown and missing stay visible"


def test_paste_ratio_is_null_not_zero_when_nothing_was_edited():
    """A zero would claim the candidate pasted none of their code, in a window
    where nothing was measured at all."""
    row = window_at(run(EventStream().keydown("KeyA", 1_000)), T0)
    assert row["edit_count"] == 0
    assert pd.isna(row["paste_ratio"])


def test_an_empty_window_reports_null_rates_and_zero_counts():
    s = EventStream().keydown("KeyA", 1_000).keydown("KeyB", 61_000)
    quiet = run(s)
    quiet = quiet[quiet["keystroke_count"] == 0].iloc[0]

    assert quiet["keystroke_count"] == 0
    assert quiet["edited_line_count"] == 0
    assert pd.isna(quiet["backspace_rate"]), "0.0 would claim flawless typing"
    assert pd.isna(quiet["mean_dwell_ms"])
    assert pd.isna(quiet["pause_ms"])
    assert pd.isna(quiet["inserted_len_max"])


def test_backspaces_are_counted_once_not_twice():
    """keydown and keyup both carry key_code='Backspace'; counting rows rather
    than keydowns doubles the rate."""
    s = EventStream()
    s.types("Backspace", 1_000, 1_060, key_class=KeyClass.backspace)
    s.types("KeyA", 1_200, 1_260)

    row = window_at(run(s), T0)
    assert row["backspace_count"] == 1
    assert row["backspace_rate"] == pytest.approx(0.5)


# Runs
def test_a_failing_run_then_a_passing_one_gives_two():
    s = EventStream()
    s.run_started(1_000).run_finished(2_000, status=ExitStatus.error)
    s.run_started(3_000).run_finished(4_000, status=ExitStatus.ok)

    row = window_at(run(s), T0)
    assert row["runs_before_first_success"] == 2
    assert row["run_count"] == 2
    assert row["run_error_rate"] == pytest.approx(0.5)
    assert row["time_to_first_run"] == pytest.approx(1.0)


def test_runs_before_first_success_is_null_until_one_succeeds():
    s = EventStream().run_started(1_000).run_finished(2_000, status=ExitStatus.error)
    assert pd.isna(window_at(run(s), T0)["runs_before_first_success"])


def test_a_window_never_sees_a_run_from_its_future():
    """A live prediction made at 10:00:30 must not use a run at 10:02:00."""
    s = EventStream()
    s.run_started(5_000).run_finished(6_000, status=ExitStatus.ok)
    s.run_started(70_000).run_finished(71_000, status=ExitStatus.ok)

    early = window_at(run(s), T0)
    assert early["runs_before_first_success"] == 1
    assert pd.isna(early["edits_between_runs"]), "the second run has not happened yet"


# Environment and clipboard
def test_a_problem_statement_copy_followed_by_a_blur():
    s = EventStream()
    s.copy(1_000, copied_len=220, pane=SourcePane.problem_statement)
    s.blur(3_000)

    row = window_at(run(s), T0)
    assert row["copy_count"] == 1
    assert row["problem_copy_count"] == 1
    assert row["copy_len_sum"] == 220
    assert row["copy_before_blur"] == 1
    assert row["blur_count"] == 1
    assert row["focus_switch_count"] == 1


def test_copy_before_blur_boundary_is_inclusive_at_five_seconds():
    at_limit = EventStream().copy(1_000).blur(6_000)
    assert window_at(run(at_limit), T0)["copy_before_blur"] == 1

    past_limit = EventStream().copy(1_000).blur(6_001)
    assert window_at(run(past_limit), T0)["copy_before_blur"] == 0


def test_cut_counts_as_a_copy():
    s = EventStream().cut(1_000, copied_len=90)
    row = window_at(run(s), T0)
    assert row["copy_count"] == 1
    assert row["copy_len_sum"] == 90


def test_one_departure_reported_twice_is_one_focus_switch():
    """A blur immediately followed by a hidden visibilitychange is one
    departure, not two."""
    s = EventStream().blur(1_000).visibility(1_010, hidden=True)
    s.focus(5_000).visibility(5_010, hidden=False)
    s.blur(8_000)

    row = window_at(run(s), T0)
    assert row["blur_count"] == 2
    assert row["visibility_change_count"] == 2
    assert row["focus_switch_count"] == 2, "two departures, however they were reported"


def test_environment_events_do_not_fracture_a_pause():
    """A candidate away for two minutes must come back with the pause intact."""
    s = EventStream().keydown("KeyA", 1_000).blur(2_000).focus(20_000)
    s.keydown("KeyB", 21_000)
    row = window_at(run(s), T0 + pd.Timedelta(seconds=10))
    assert row["pause_ms"] == pytest.approx(20_000.0)


# Task scoping and output contract
def test_events_with_no_task_never_enter_a_window():
    s = EventStream()
    s.blur(1_000, task_id=None) # between tasks
    s.copy(1_500, task_id=None)
    s.keydown("KeyA", 2_000)

    row = window_at(run(s), T0)
    assert row["blur_count"] == 0
    assert row["copy_count"] == 0
    assert row["keystroke_count"] == 1


def test_environment_features_exist_when_capture_stamps_the_task():
    """The other side of the previous test: these eight features stay empty
    unless capture stamps task_id."""
    s = EventStream().blur(1_000).focus(2_000).copy(2_500)
    row = window_at(run(s), T0)
    assert row["blur_count"] == 1
    assert row["focus_count"] == 1
    assert row["copy_count"] == 1


def test_output_always_passes_the_feature_schema():
    s = EventStream()
    s.types("KeyA", 1_000, 1_080).paste(2_000).run_started(3_000)
    s.run_finished(4_000, status=ExitStatus.ok).blur(5_000).copy(5_500)
    out = run(s)
    FEATURE_SCHEMA.validate(out)


def test_run_id_labels_the_vector_and_keeps_variants_apart():
    s = EventStream().keydown("KeyA", 1_000)
    assert (run(s)["run_id"] == "live").all()
    assert (run(s, run_id="final")["run_id"] == "final").all()
    assert (run(s, run_id="degraded:jitter=50ms")["run_id"] != "live").all()


def test_extract_window_returns_exactly_one_window():
    s = EventStream().types("KeyA", 1_000, 1_080).types("KeyB", 1_200, 1_280)
    one = extract_window(s.frame(), tasks_frame(), T0)
    assert len(one) == 1
    assert one.iloc[0]["window_start"] == T0
    assert one.iloc[0]["keystroke_count"] == 2


def test_live_and_final_agree_when_nothing_was_lost():
    """The live pass is best-effort; when no event is missing it must match the
    authoritative final pass exactly, or the two are not comparable."""
    s = EventStream().types("KeyA", 1_000, 1_080).paste(2_000).blur(3_000)
    tasks = tasks_frame(submitted_at=T0 + pd.Timedelta(seconds=10))

    live = extract_features(s.frame(), tasks, run_id="live").drop(columns=["run_id"])
    final = extract_features(s.frame(), tasks, run_id="final").drop(columns=["run_id"])
    pd.testing.assert_frame_equal(live, final)
