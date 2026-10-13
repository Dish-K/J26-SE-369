"""edit_evidence and its SQL forms: the exact edit ranges C2 uses to localize a flag."""

from __future__ import annotations

import pandas as pd
import pytest

from shared_features import (
    EDIT_EVIDENCE_SQL,
    EDIT_EVIDENCE_SQL_ASYNCPG,
    EVIDENCE_COLUMNS,
    Origin,
    PasteOrigin,
    edit_evidence,
)

from conftest import SESSION, T0, TASK, EventStream

WINDOW_START = T0
WINDOW_END = T0 + pd.Timedelta(seconds=30)


def evidence(stream, start=WINDOW_START, end=WINDOW_END):
    return edit_evidence(stream.frame(), SESSION, TASK, start, end)


def test_two_disconnected_edit_regions_are_returned_separately():
    """Edits at line 5 and lines 12-41 must stay two regions, not one 5-41 span."""
    s = EventStream()
    s.edit(1_000, start_line=5, end_line=5, inserted_len=8, inserted_lines=0, version_id=100)
    s.paste(2_000, start_line=12, end_line=12, inserted_len=400,
            inserted_lines=29, version_id=101)

    rows = evidence(s)
    assert len(rows) == 2

    assert rows.loc[0, "touched_start_line"] == 5
    assert rows.loc[0, "touched_end_line"] == 5
    assert rows.loc[1, "touched_start_line"] == 12
    assert rows.loc[1, "touched_end_line"] == 41

    # The regions are disconnected; C2 decides that from exact ranges.
    assert rows.loc[0, "touched_end_line"] + 1 < rows.loc[1, "touched_start_line"]


def test_a_copy_selection_is_never_returned_as_an_edit():
    """Copy and cut reuse the line columns for the selected range, and no CHECK stops a mix-up."""
    s = EventStream()
    s.copy(500, copied_len=300)
    s.frame()
    s.rows[-1].update({"start_line": 60, "end_line": 80})
    s.edit(1_000, start_line=5, end_line=5, version_id=100)

    rows = evidence(s)
    assert len(rows) == 1
    assert rows.loc[0, "touched_start_line"] == 5
    assert 60 not in set(rows["start_line"].dropna())


def test_the_template_load_is_excluded():
    """An unfiltered query localizes flags onto the starter template."""
    s = EventStream()
    s.edit(100, origin=Origin.reset, start_line=1, end_line=1,
           inserted_len=900, inserted_lines=40, version_id=1)
    s.edit(1_000, start_line=7, end_line=7, version_id=2)

    rows = evidence(s)
    assert len(rows) == 1
    assert rows.loc[0, "version_id"] == 2


def test_rows_carry_the_document_version_and_change_index():
    s = EventStream().edit(1_000, version_id=137, change_index=2, start_line=9)
    rows = evidence(s)
    assert rows.loc[0, "version_id"] == 137
    assert rows.loc[0, "change_index"] == 2


def test_results_are_ordered_by_the_sequence_triple_not_by_time():
    """perf_now can tie and a reload can move time backwards, so order by the sequence triple."""
    s = EventStream()
    s.edit(1_000, start_line=1, version_id=1)
    s.edit(1_000, start_line=2, version_id=2) # identical perf_now
    s.edit(1_000, start_line=3, version_id=3)

    rows = evidence(s)
    assert list(rows["version_id"]) == [1, 2, 3]


def test_the_window_is_half_open_at_its_end():
    """An event on the boundary belongs to the next window, as in the extractor."""
    s = EventStream()
    s.edit(0, start_line=1, version_id=1)
    s.edit(30_000, start_line=2, version_id=2) # exactly window_end

    rows = evidence(s)
    assert list(rows["version_id"]) == [1]


def test_events_from_another_task_are_excluded():
    from conftest import TASK_B

    s = EventStream().edit(1_000, start_line=1, version_id=1)
    s.task_id = TASK_B
    s.edit(2_000, start_line=2, version_id=2)

    rows = evidence(s)
    assert list(rows["version_id"]) == [1]


def test_every_contract_column_is_present_and_ordered():
    s = EventStream().edit(1_000, start_line=3, version_id=5)
    assert tuple(evidence(s).columns) == EVIDENCE_COLUMNS


def test_an_empty_window_returns_no_rows_not_an_error():
    s = EventStream().keydown("KeyA", 1_000)
    assert evidence(s).empty


def test_the_sql_and_the_dataframe_paths_agree_on_their_filters():
    """Both forms must apply the same filters, or C2's answer depends on which it calls."""
    sql = EDIT_EVIDENCE_SQL
    assert "e.kind       = 2" in sql
    assert "e.origin IS DISTINCT FROM 4" in sql
    assert "ORDER BY e.load_seq, e.batch_seq, e.seq_in_batch, e.change_index" in sql
    assert "e.time >= %s" in sql and "e.time <  %s" in sql
    for column in EVIDENCE_COLUMNS:
        assert column in sql, f"{column} missing from the SQL contract"


def test_the_asyncpg_form_differs_from_psycopg_only_in_parameter_style():
    """C2's asyncpg pool cannot bind %s; any other difference would mislocalize flags silently."""
    ported = (
        EDIT_EVIDENCE_SQL_ASYNCPG
        .replace("$1", "%s").replace("$2", "%s")
        .replace("$3", "%s").replace("$4", "%s")
    )
    assert ported == EDIT_EVIDENCE_SQL


def test_the_asyncpg_form_numbers_exactly_four_parameters_in_order():
    sql = EDIT_EVIDENCE_SQL_ASYNCPG
    assert "%s" not in sql, "a psycopg placeholder survived into the asyncpg form"
    for n in (1, 2, 3, 4):
        assert sql.count(f"${n}") == 1
    assert "$5" not in sql
    # session_id, task_id, window_start, window_end, in that order.
    positions = [sql.index(f"${n}") for n in (1, 2, 3, 4)]
    assert positions == sorted(positions)
    assert "e.session_id = $1" in sql
    assert "e.task_id    = $2" in sql
    assert "e.time >= $3" in sql
    assert "e.time <  $4" in sql


def test_both_forms_return_every_contract_column():
    for sql in (EDIT_EVIDENCE_SQL, EDIT_EVIDENCE_SQL_ASYNCPG):
        for column in EVIDENCE_COLUMNS:
            assert column in sql, f"{column} missing from {sql[:40]!r}"


def test_a_deletion_reports_a_post_edit_range_of_one_line():
    """Monaco's range is pre-edit but version_id is post-edit, so a deletion collapses."""
    s = EventStream()
    s.edit(1_000, start_line=10, end_line=15, inserted_len=0, removed_len=180,
           inserted_lines=0, version_id=200)

    rows = evidence(s)
    assert rows.loc[0, "replaced_start_line"] == 10
    assert rows.loc[0, "replaced_end_line"] == 15 # pre-edit, as Monaco gave it
    assert rows.loc[0, "touched_start_line"] == 10
    assert rows.loc[0, "touched_end_line"] == 10 # post-edit, collapsed
    assert rows.loc[0, "net_line_delta"] == -5


def test_a_multi_line_paste_reports_the_lines_it_now_occupies():
    s = EventStream()
    s.paste(1_000, start_line=12, end_line=12, inserted_len=400,
            inserted_lines=29, version_id=101)

    rows = evidence(s)
    assert rows.loc[0, "replaced_start_line"] == 12
    assert rows.loc[0, "replaced_end_line"] == 12 # a point, nothing replaced
    assert rows.loc[0, "touched_end_line"] == 41
    assert rows.loc[0, "net_line_delta"] == 29


def test_net_line_delta_is_the_shift_c2_applies_to_regions_below():
    """net_line_delta is computed once here so C2 need not recompute it."""
    s = EventStream()
    # Replace 3 lines with 10, regions below move down by 7.
    s.edit(1_000, start_line=4, end_line=7, inserted_len=300,
           inserted_lines=10, version_id=300)

    assert evidence(s).loc[0, "net_line_delta"] == 7


def test_several_changes_in_one_editor_event_keep_their_array_order():
    """Replace-all writes one row per change sharing a version_id; only change_index orders them."""
    s = EventStream()
    s.edit(1_000, start_line=40, end_line=40, inserted_lines=0,
           version_id=500, change_index=2)
    s.edit(1_000, start_line=20, end_line=20, inserted_lines=0,
           version_id=500, change_index=0)
    s.edit(1_000, start_line=30, end_line=30, inserted_lines=0,
           version_id=500, change_index=1)
    # One editor event, so all three share a batch position.
    for row in s.rows:
        row["batch_seq"] = 4
        row["seq_in_batch"] = 0

    rows = evidence(s)
    assert list(rows["change_index"]) == [0, 1, 2]
    assert list(rows["start_line"]) == [20, 30, 40]


def test_identity_columns_are_returned_so_pooled_rows_stay_attributable():
    """C2 pools evidence across tasks, so every row must say whose it is."""
    s = EventStream().edit(1_000, start_line=3, version_id=5)
    rows = evidence(s)
    assert rows.loc[0, "session_id"] == SESSION
    assert rows.loc[0, "task_id"] == TASK
