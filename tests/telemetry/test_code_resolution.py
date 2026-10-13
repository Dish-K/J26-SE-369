"""Live code retrieval: never present code as the flagged original unless it is."""

from __future__ import annotations

import pandas as pd
import pytest

from shared_features import Origin, resolve_flag_code

from conftest import SESSION, T0, TASK, EventStream

SNAPSHOT_COLUMNS = [
    "snapshot_id", "session_id", "task_id", "version_id", "captured_at",
    "perf_now", "load_seq", "code_text", "doc_len", "code_redacted_at",
    "capture_reason",
]


def snapshots(*rows) -> pd.DataFrame:
    """Build a code_snapshots frame. Each row: (version_id, code_text, **kw)."""
    built = []
    for n, (version_id, code_text, kw) in enumerate(rows):
        built.append({
            "snapshot_id": f"snap-{n}",
            "session_id": kw.get("session_id", SESSION),
            "task_id": kw.get("task_id", TASK),
            "version_id": version_id,
            "captured_at": T0 + pd.Timedelta(seconds=10 * n),
            "perf_now": 10_000.0 * n,
            "load_seq": kw.get("load_seq", 0),
            "code_text": code_text,
            "doc_len": len(code_text),
            "code_redacted_at": kw.get("code_redacted_at", pd.NaT),
            "capture_reason": kw.get("capture_reason", 0),
        })
    return pd.DataFrame(built, columns=SNAPSHOT_COLUMNS)


def snap(version_id, code_text, **kw):
    return (version_id, code_text, kw)


def resolve(snaps, stream, *, version_id, start_line, end_line,
            load_seq=0, authorized=True):
    events = stream.frame() if stream is not None else pd.DataFrame()
    return resolve_flag_code(
        snaps, events,
        session_id=SESSION, task_id=TASK,
        load_seq=load_seq, version_id=version_id,
        start_line=start_line, end_line=end_line,
        authorized=authorized,
    )


DOC = "\n".join(f"line {n}" for n in range(1, 21))


# Unchanged code
def test_exact_returns_the_real_historical_code():
    s = snapshots(snap(100, DOC), snap(140, "something else entirely"))
    r = resolve(s, None, version_id=100, start_line=3, end_line=5)

    assert r.status == "exact"
    assert r.is_historical_original
    assert r.version_distance == 0
    assert r.direction == "exact"
    assert r.snapshot_version_id == 100
    assert r.snippet == "line 3\nline 4\nline 5"
    assert r.warning is None


def test_exact_prefers_the_matching_version_over_a_closer_one():
    """A snapshot at the flagged version is always the answer, whatever else
    is nearby."""
    s = snapshots(snap(99, "WRONG"), snap(100, DOC), snap(101, "WRONG"))
    r = resolve(s, None, version_id=100, start_line=1, end_line=1)
    assert r.status == "exact"
    assert r.snippet == "line 1"


# Code edited after the flag
def test_an_edit_after_the_snapshot_gives_a_labelled_nearest_result():
    """The flagged version was never captured. A later snapshot is returned,
    but it must announce itself."""
    s = snapshots(snap(100, DOC))
    # One edit far below the flagged region, so the mapping is safe.
    stream = EventStream().edit(1_000, start_line=18, end_line=18,
                                inserted_lines=0, version_id=101)

    r = resolve(s, stream, version_id=101, start_line=3, end_line=5)

    assert r.status == "nearest"
    assert not r.is_historical_original
    assert r.version_distance == 1
    assert r.direction == "before"
    assert r.warning is not None
    assert "NOT the flagged code" in r.warning
    assert "1 version(s) before" in r.warning


def test_a_nearest_result_states_distance_and_direction_in_both_directions():
    s = snapshots(snap(90, DOC), snap(130, DOC))

    stream = EventStream().edit(1_000, start_line=19, end_line=19,
                                inserted_lines=0, version_id=95)
    after = resolve(s, stream, version_id=95, start_line=2, end_line=3)
    assert after.direction == "before" and after.version_distance == 5

    stream2 = EventStream().edit(1_000, start_line=19, end_line=19,
                                 inserted_lines=0, version_id=128)
    before = resolve(s, stream2, version_id=128, start_line=2, end_line=3)
    assert before.direction == "after" and before.version_distance == 2


def test_the_later_snapshot_wins_a_tie():
    """Code flagged at version V existed at V, so a later document still holds
    it unless an edit removed it, which the mapping detects. An earlier
    snapshot may predate the code entirely, and nothing here can detect that."""
    s = snapshots(snap(98, DOC), snap(102, DOC))
    r = resolve(s, None, version_id=100, start_line=4, end_line=4)
    assert r.status == "nearest"
    assert r.snapshot_version_id == 102
    assert r.direction == "after"


# Code deleted after the flag
def test_a_deletion_of_the_flagged_region_is_unavailable_not_remapped():
    """The flagged lines were deleted between the flag and the snapshot. The
    lines now at 10-15 are different code; showing them is the failure this
    whole module exists to prevent."""
    s = snapshots(snap(200, DOC))
    stream = EventStream().edit(1_000, start_line=10, end_line=15,
                                inserted_len=0, removed_len=180,
                                inserted_lines=0, version_id=200)

    r = resolve(s, stream, version_id=199, start_line=10, end_line=15)

    assert r.status == "unavailable"
    assert r.reason == "range_not_safely_mappable"
    assert r.snippet is None
    assert r.code_text is None


def test_historical_text_survives_a_later_deletion_when_the_snapshot_exists():
    """C2's rule: a later deletion does not make an exact snapshot stale. The
    code was there at that version and the snapshot proves it."""
    s = snapshots(snap(199, DOC))
    stream = EventStream().edit(1_000, start_line=10, end_line=15,
                                inserted_len=0, inserted_lines=0,
                                version_id=200)

    r = resolve(s, stream, version_id=199, start_line=10, end_line=12)
    assert r.status == "exact"
    assert r.snippet == "line 10\nline 11\nline 12"


# Edits above or below the flag
def test_an_insert_above_the_region_shifts_the_mapped_range():
    """C2's worked example: flagged at lines 10-15, ten lines inserted above,
    so the same code sits at 20-25 in the later snapshot."""
    later = "\n".join(f"line {n}" for n in range(1, 31))
    s = snapshots(snap(300, later))
    stream = EventStream().paste(1_000, start_line=2, end_line=2,
                                 inserted_len=100, inserted_lines=10,
                                 version_id=300)

    r = resolve(s, stream, version_id=299, start_line=10, end_line=15)

    assert r.status == "nearest"
    assert (r.snapshot_start_line, r.snapshot_end_line) == (20, 25)
    # The range asked for is preserved untouched alongside the mapped one.
    assert (r.target_start_line, r.target_end_line) == (10, 15)


def test_an_edit_below_the_region_moves_nothing():
    s = snapshots(snap(300, DOC))
    stream = EventStream().edit(1_000, start_line=18, end_line=18,
                                inserted_lines=0, version_id=300)

    r = resolve(s, stream, version_id=299, start_line=3, end_line=5)
    assert (r.snapshot_start_line, r.snapshot_end_line) == (3, 5)


def test_mapping_runs_backwards_too():
    """Snapshot BEFORE the flag: the shift has to be inverted, or the range
    lands ten lines off in the opposite direction."""
    s = snapshots(snap(290, DOC))
    stream = EventStream().paste(1_000, start_line=2, end_line=2,
                                 inserted_len=100, inserted_lines=10,
                                 version_id=295)

    r = resolve(s, stream, version_id=299, start_line=20, end_line=25)
    assert r.status == "nearest"
    assert r.direction == "before"
    assert (r.snapshot_start_line, r.snapshot_end_line) == (10, 15)


# Several changes in one editor event
def test_multi_change_events_all_contribute_to_the_mapping():
    """Find-and-replace-all arrives as one editor event, one row per change
    sharing version_id. Every change above the region shifts it; missing one
    leaves the range silently wrong."""
    later = "\n".join(f"line {n}" for n in range(1, 27))
    s = snapshots(snap(400, later))

    stream = EventStream()
    stream.edit(1_000, start_line=1, end_line=1, inserted_lines=3,
                inserted_len=30, version_id=400, change_index=0)
    stream.edit(1_000, start_line=2, end_line=2, inserted_lines=3,
                inserted_len=30, version_id=400, change_index=1)
    for row in stream.rows:
        row["batch_seq"] = 9
        row["seq_in_batch"] = 0

    r = resolve(s, stream, version_id=399, start_line=10, end_line=12)
    assert r.status == "nearest"
    assert (r.snapshot_start_line, r.snapshot_end_line) == (16, 18)


def test_a_change_touching_the_region_fails_the_whole_mapping():
    """Fail closed. One ambiguous change in the array is enough, because a
    partial mapping would report a range built from only some of the edits."""
    s = snapshots(snap(400, DOC))
    stream = EventStream()
    stream.edit(1_000, start_line=1, end_line=1, inserted_lines=2,
                version_id=400, change_index=0)
    stream.edit(1_000, start_line=11, end_line=11, inserted_lines=1,
                version_id=400, change_index=1)       # inside 10-12
    for row in stream.rows:
        row["batch_seq"] = 9

    r = resolve(s, stream, version_id=399, start_line=10, end_line=12)
    assert r.status == "unavailable"
    assert r.reason == "range_not_safely_mappable"


# Page reloads
def test_a_flag_never_resolves_against_another_load_segments_snapshot():
    """Monaco's version_id restarts at 1 after a reload, so load_seq 0 version
    1 and load_seq 1 version 1 are different documents that share a number.
    Resolving across the boundary shows the interviewer the wrong document and
    labels it exact."""
    before_reload = "before = 1\nbefore = 2\nbefore = 3"
    after_reload = "after = 1\nafter = 2\nafter = 3"
    s = snapshots(
        snap(1, before_reload, load_seq=0),
        snap(1, after_reload, load_seq=1),
    )

    pre = resolve(s, None, version_id=1, start_line=1, end_line=1, load_seq=0)
    post = resolve(s, None, version_id=1, start_line=1, end_line=1, load_seq=1)

    assert pre.status == "exact" and pre.snippet == "before = 1"
    assert post.status == "exact" and post.snippet == "after = 1"
    assert pre.snapshot_load_seq == 0
    assert post.snapshot_load_seq == 1


def test_an_empty_load_segment_is_unavailable_not_borrowed_from_a_neighbour():
    s = snapshots(snap(1, DOC, load_seq=0))
    r = resolve(s, None, version_id=1, start_line=1, end_line=1, load_seq=1)

    assert r.status == "unavailable"
    assert r.reason == "no_snapshot_in_load_segment"
    assert r.snippet is None


def test_edits_from_another_load_segment_never_shift_the_mapping():
    s = snapshots(snap(100, DOC, load_seq=1))
    stream = EventStream(load_seq=0)
    stream.paste(1_000, start_line=1, end_line=1, inserted_lines=10,
                 version_id=100)

    r = resolve(s, stream, version_id=99, start_line=3, end_line=4, load_seq=1)
    assert r.status == "nearest"
    assert (r.snapshot_start_line, r.snapshot_end_line) == (3, 4)



# Redaction
def test_a_redacted_exact_snapshot_reports_redaction_not_absence():
    """The version was captured; its contents were removed by retention. That
    is a different fact from "never captured", and the interviewer should see
    which."""
    s = snapshots(snap(100, "", code_redacted_at=T0, capture_reason=0))
    r = resolve(s, None, version_id=100, start_line=1, end_line=1)

    assert r.status == "unavailable"
    assert r.reason == "redacted"
    assert r.snapshot_version_id == 100
    assert r.code_text is None


def test_a_redacted_snapshot_is_never_offered_as_nearest():
    s = snapshots(snap(100, "", code_redacted_at=T0))
    r = resolve(s, None, version_id=105, start_line=1, end_line=1)

    assert r.status == "unavailable"
    assert r.reason == "redacted"
    assert r.snippet is None


def test_an_unredacted_neighbour_is_still_usable():
    s = snapshots(snap(100, "", code_redacted_at=T0), snap(104, DOC))
    r = resolve(s, None, version_id=105, start_line=2, end_line=2)

    assert r.status == "nearest"
    assert r.snapshot_version_id == 104


# Authorization
def test_an_unauthorized_caller_gets_no_code_at_all():
    s = snapshots(snap(100, DOC))
    r = resolve(s, None, version_id=100, start_line=1, end_line=3,
                authorized=False)

    assert r.status == "unavailable"
    assert r.reason == "not_authorized"
    assert r.code_text is None
    assert r.snippet is None
    # Not even the existence of a snapshot leaks.
    assert r.snapshot_version_id is None


def test_authorization_cannot_be_omitted():
    """No default value, so a caller cannot forget it and silently get the
    permissive branch."""
    s = snapshots(snap(100, DOC))
    with pytest.raises(TypeError):
        resolve_flag_code(
            s, pd.DataFrame(),
            session_id=SESSION, task_id=TASK, load_seq=0,
            version_id=100, start_line=1, end_line=1,
        )


# Degenerate input
def test_no_snapshots_at_all_is_unavailable():
    r = resolve(pd.DataFrame(), None, version_id=100, start_line=1, end_line=1)
    assert r.status == "unavailable"
    assert r.reason == "no_snapshot_in_load_segment"


def test_a_range_past_the_end_of_the_document_is_unavailable():
    s = snapshots(snap(100, "one\ntwo"))
    r = resolve(s, None, version_id=100, start_line=40, end_line=41)
    assert r.status == "unavailable"
    assert r.reason == "range_outside_document"


def test_an_inverted_range_is_rejected():
    s = snapshots(snap(100, DOC))
    r = resolve(s, None, version_id=100, start_line=9, end_line=4)
    assert r.status == "unavailable"
    assert r.reason == "invalid_range"


def test_the_template_load_never_shifts_a_mapping():
    """origin=reset is the starter template, not the candidate's edit. It is
    excluded everywhere else and must be excluded here too."""
    s = snapshots(snap(100, DOC))
    stream = EventStream().edit(500, origin=Origin.reset, start_line=1,
                                end_line=1, inserted_lines=40, version_id=100)

    r = resolve(s, stream, version_id=99, start_line=3, end_line=4)
    assert r.status == "nearest"
    assert (r.snapshot_start_line, r.snapshot_end_line) == (3, 4)
