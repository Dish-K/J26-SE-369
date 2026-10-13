"""Exact edit ranges for one window, so C2 can localize a prediction.

C1 supplies evidence. C2 classifies, localizes and merges; C3 probes. Nothing
here decides anything -- it is a row filter, not a second feature extractor.

Why a shared helper exists at all, rather than C2 writing the query itself:
four rules are needed for a correct answer and **the database enforces none of
them.**

1. ``kind = 2``. Copy and cut events reuse ``start_line``/``start_col``/
   ``end_line``/``end_col`` to mean *the range the candidate selected and
   copied* -- not where anything was inserted. No CHECK prevents it, there is no
   index on ``events(kind)``, and an unfiltered range query will happily report
   a copy selection as an edit location. Specification 10.1.
2. ``origin != reset``. A reset edit is the starter template loading: a large
   insertion with no keystrokes behind it, which is the ``external_ai``
   signature manufactured by our own UI.
3. **Filter on ``time``, order on the sequence triple, then ``change_index``.**
   These are not in conflict. ``time`` is correct for deciding which window an
   event falls in -- after a reload the clock handshake re-runs and the wall
   clock is valid again. ``time`` is wrong for ordering *within* a window,
   because two events can share a coarsened ``perf_now`` and a reload produces a
   fresh offset. One editor event carrying several changes writes one row per
   change (specification 9.3), so ``change_index`` is the final tiebreak --
   without it, find-and-replace-all has no defined order and the forward
   mapping in 14.1 applies its changes in an arbitrary sequence.
   Specification 6.5.
4. **Two line ranges, not one.** See below -- this is the one that silently
   produces wrong coordinates.

The line ranges returned are **historical** -- the coordinates as they stood
during the detection episode, not in the submitted file. Migration 006 resolves
them at submission, and C3 reads ``classifier_flags_for_probing``.

Pre-edit range versus post-edit version
---------------------------------------
A Monaco change mixes two coordinate systems, and C2 asked for this to be
documented rather than assumed:

* ``start_line``/``start_col``/``end_line``/``end_col`` describe **what was
  replaced** -- positions in the document *before* the change (specification
  9.3). For a pure insertion they are a single point.
* ``version_id`` is the model revision *after* the change has been applied.

So a row's raw range and its ``version_id`` do not describe the same document.
Rather than leave that to be rediscovered, this contract returns both ranges
explicitly:

* ``replaced_start_line``/``replaced_end_line`` -- pre-edit, exactly as Monaco
  reported them.
* ``touched_start_line``/``touched_end_line`` -- **post-edit, aligned with
  ``version_id``**: the inserted text occupies ``inserted_lines`` newlines
  starting at ``start_line``, so it spans
  ``[start_line, start_line + inserted_lines]``.

The two coincide for same-line edits and diverge for every multi-line one. A
deletion of five lines has a pre-edit range five lines tall and a post-edit
range of one line, because after the change there is nothing left to span.
``net_line_delta`` (``inserted_lines - (end_line - start_line)``, specification
14.1) is returned alongside so C2's forward mapping does not have to recompute
it per row.
"""

from __future__ import annotations

import pandas as pd

from .feature_schema import Kind, Origin

__all__ = [
    "EDIT_EVIDENCE_SQL",
    "EDIT_EVIDENCE_SQL_ASYNCPG",
    "EVIDENCE_COLUMNS",
    "edit_evidence",
]

#: Columns every form of this contract returns, in this order.
EVIDENCE_COLUMNS: tuple[str, ...] = (
    "session_id", "task_id",
    "time", "perf_now", "load_seq", "batch_seq", "seq_in_batch",
    "origin", "paste_origin", "version_id", "change_index",
    "start_line", "start_col", "end_line", "end_col", "range_offset",
    "inserted_len", "removed_len", "inserted_lines", "doc_len",
    "replaced_start_line", "replaced_end_line",
    "touched_start_line", "touched_end_line",
    "net_line_delta",
)

# One body, two parameter styles. The placeholders are substituted at import
# time, so the psycopg and asyncpg forms cannot drift apart -- a column added to
# one is added to both or to neither. test_evidence.py asserts it.
_EDIT_EVIDENCE_TEMPLATE = """
SELECT
    e.session_id,
    e.task_id,
    e.time,
    e.perf_now,
    e.load_seq,
    e.batch_seq,
    e.seq_in_batch,
    e.origin,
    e.paste_origin,
    e.version_id,
    e.change_index,
    e.start_line,
    e.start_col,
    e.end_line,
    e.end_col,
    e.range_offset,
    e.inserted_len,
    e.removed_len,
    e.inserted_lines,
    e.doc_len,
    -- Pre-edit: the range Monaco reported as replaced.
    e.start_line AS replaced_start_line,
    e.end_line   AS replaced_end_line,
    -- Post-edit: aligned with version_id. The inserted text spans
    -- inserted_lines newlines from start_line, so a deletion collapses to one
    -- line and a 30-line paste spans 30.
    e.start_line AS touched_start_line,
    e.start_line + COALESCE(e.inserted_lines, 0) AS touched_end_line,
    COALESCE(e.inserted_lines, 0)
        - COALESCE(e.end_line - e.start_line, 0) AS net_line_delta
FROM events e
WHERE e.session_id = {p1}
  AND e.task_id    = {p2}
  AND e.kind       = 2            -- edits only; copy/cut reuse these columns
  AND e.origin IS DISTINCT FROM 4 -- exclude the programmatic template load
  AND e.time >= {p3}
  AND e.time <  {p4}              -- half-open, as the extractor windows are
ORDER BY e.load_seq, e.batch_seq, e.seq_in_batch, e.change_index NULLS FIRST
"""

#: psycopg / psycopg2. Parameters in order: session_id, task_id, window_start,
#: window_end.
#:
#: Half-open on the window end, matching the extractor exactly -- an event on
#: the boundary must not appear in both a window's features and the next
#: window's evidence.
EDIT_EVIDENCE_SQL = _EDIT_EVIDENCE_TEMPLATE.format(
    p1="%s", p2="%s", p3="%s", p4="%s"
)

#: asyncpg, which uses numbered placeholders rather than ``%s``. Identical
#: filters, identical order, identical columns -- only the parameter style
#: differs::
#:
#:     rows = await pool.fetch(
#:         EDIT_EVIDENCE_SQL_ASYNCPG,
#:         session_id, task_id, window_start, window_end,
#:     )
#:
#: asyncpg binds UUID and timestamptz natively, so ``session_id``/``task_id``
#: are passed as :class:`uuid.UUID` and the window bounds as **timezone-aware**
#: :class:`datetime.datetime`. A naive datetime is rejected by asyncpg rather
#: than silently reinterpreted, which is the behaviour we want.
EDIT_EVIDENCE_SQL_ASYNCPG = _EDIT_EVIDENCE_TEMPLATE.format(
    p1="$1", p2="$2", p3="$3", p4="$4"
)


def edit_evidence(
    events: pd.DataFrame,
    session_id,
    task_id,
    window_start,
    window_end,
) -> pd.DataFrame:
    """The DataFrame equivalent of :data:`EDIT_EVIDENCE_SQL`.

    Same filters, same order, same columns -- so a test can prove the contract
    without a database, and the synthetic generator can exercise it offline.
    """
    derived = {
        "replaced_start_line", "replaced_end_line",
        "touched_start_line", "touched_end_line", "net_line_delta",
    }

    frame = events.copy()
    for column in EVIDENCE_COLUMNS:
        if column not in frame.columns and column not in derived:
            frame[column] = pd.NA

    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)

    selected = frame[
        (frame["session_id"] == session_id)
        & (frame["task_id"] == task_id)
        & (frame["kind"] == int(Kind.edit))
        & (frame["origin"].fillna(-1) != int(Origin.reset))
        & (frame["time"] >= window_start)
        & (frame["time"] < window_end)
    ].sort_values(
        ["load_seq", "batch_seq", "seq_in_batch", "change_index"],
        kind="mergesort",
        na_position="first",
    )

    start = selected["start_line"].astype("Int64")
    end = selected["end_line"].astype("Int64")
    added = selected["inserted_lines"].astype("Int64").fillna(0)
    removed_span = (end - start).fillna(0)

    selected = selected.assign(
        replaced_start_line=start,
        replaced_end_line=end,
        touched_start_line=start,
        touched_end_line=start + added,
        net_line_delta=added - removed_span,
    )
    return selected[list(EVIDENCE_COLUMNS)].reset_index(drop=True)
