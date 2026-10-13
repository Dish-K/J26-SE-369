"""Resolve a flagged line range to the source code that was actually there.

C2 identifies *where* a flag is, in historical coordinates. This resolves that
range to text for the interviewer, and -- the entire point of the module --
**reports how good the answer is instead of assuming it is good.**

Why this is not simply a lookup,
Before submission there is no ``task_submissions`` row to rebase against, so a
live flag can name lines but not show them. ``code_snapshots`` (007) stores the
whole document periodically, which covers most of the gap but not all of it:
Monaco bumps ``version_id`` on every edit, so a candidate typing produces
roughly five versions a second, and a snapshot every ten seconds leaves many
versions uncaptured.

**Nothing recovers the text of an uncaptured version.** Edit rows carry
offsets and lengths, not characters, and ``inserted_text`` is pastes only. If
line 12 held ``x = 1`` at version 10 and holds ``x = 2`` at version 12, the
character ``1`` is not stored anywhere. Replaying the keystroke stream was
rejected for the same reason in 007: §3.3 assumes QWERTY rather than measuring
it, so a layout mismatch yields plausible but wrong source and nothing detects
that until submission.

So this returns one of three statuses, and never blurs them:

``exact``
    A snapshot exists at precisely ``(task_id, load_seq, version_id)`` and its
    body has not been redacted. This is the real historical code.

``nearest``
    The closest snapshot **in the same load segment**, with the flagged range
    mapped into its coordinates, plus the version distance and direction. It is
    labelled, and :attr:`Resolution.warning` says in words that it is not the
    flagged original. Presenting it as the original is the worst failure this
    system has.

``unavailable``
    No snapshot in that load segment, the only one is redacted, the range
    cannot be mapped safely, or the caller is not authorized. A short
    ``reason`` says which.

Three rules that are easy to get wrong:
1. **Never cross a ``load_seq`` boundary.** ``version_id`` restarts at 1 when a
   page reload builds a new Monaco model, so version 7 before a reload and
   version 7 after it are different documents. Migration 008 makes that a
   database constraint; this module makes it a lookup rule.
2. **Fail closed when mapping is ambiguous.** If an intervening edit touches,
   splits, replaces or deletes the flagged region, its identity in the other
   version cannot be established. Return ``unavailable`` rather than highlight
   a guessed span -- highlighting the wrong lines is indistinguishable, to the
   person reading it, from highlighting the right ones.
3. **Authorization is a required argument, not a default.** Full in-progress
   source is the most sensitive store in C1. ``authorized`` has no default
   value so it cannot be forgotten; a caller must state the decision. Actual
   authentication belongs in the service layer -- this only guarantees the
   decision was made.

Classification never reads any of this. Snapshot contents do not reach the
model, and a resolution result cannot change a category or a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .feature_schema import CaptureReason, Kind, Origin

__all__ = ["Resolution", "resolve_flag_code"]

Status = Literal["exact", "nearest", "unavailable"]
Direction = Literal["exact", "before", "after", "none"]


# Readable text for a ``code_snapshots.capture_reason`` value. The VALUES live in registry.yaml under ``enums.CaptureReason`` and are built into the class:`~shared_features.CaptureReason` IntEnum from there -- only the wording for the interviewer is here.
_CAPTURE_REASON_LABELS = {
    CaptureReason.tick: "periodic 10s capture",
    CaptureReason.after_paste: "captured immediately after a paste",
    CaptureReason.before_large_delete: "captured immediately before a large deletion",
}


def _capture_label(value) -> str:
    if value is None or pd.isna(value):
        return "capture reason not recorded"
    try:
        return _CAPTURE_REASON_LABELS[CaptureReason(int(value))]
    except ValueError:
        return f"unrecognised capture reason {int(value)}"


@dataclass(frozen=True)
class Resolution:
    """The answer, and how much weight it carries."""

    status: Status
    detail: str

    # What was asked for.
    target_load_seq: int | None = None
    target_version_id: int | None = None
    target_start_line: int | None = None
    target_end_line: int | None = None

    # What was found.
    snapshot_id: object = None
    snapshot_load_seq: int | None = None
    snapshot_version_id: int | None = None
    capture_reason: int | None = None

    # Absolute version gap, 0 when exact. Paired with ``direction`` rather than signed, because a signed number is read as a magnitude and the sign is missed.
    version_distance: int | None = None
    direction: Direction = "none"

    # The flagged range mapped into the snapshot's coordinates. None whenever the mapping could not be established safely.
    snapshot_start_line: int | None = None
    snapshot_end_line: int | None = None

    # Populated only when status is exact or nearest AND the caller was authorized.
    code_text: str | None = None
    snippet: str | None = None

    # Set for every nearest result. Shown to the interviewer verbatim.
    warning: str | None = None

    # Set for every unavailable result.
    reason: str | None = None

    @property
    def is_historical_original(self) -> bool:
        """True only for ``exact``.

        Any UI deciding whether to present text as "the flagged code" reads
        this rather than testing ``code_text is not None``, which is true for
        ``nearest`` as well.
        """
        return self.status == "exact"


_NEAREST_WARNING = (
    "This is NOT the flagged code. It is the closest stored version of the "
    "document, {distance} version(s) {direction} the one the flag was raised "
    "against. Lines may have been edited in between."
)


def _edits_between(
    events: pd.DataFrame,
    *,
    session_id,
    task_id,
    load_seq,
    low: int,
    high: int,
) -> pd.DataFrame:
    """Ordered edits whose resulting version lies in ``(low, high]``.

    Same two filters as :mod:`shared_features.evidence` -- ``kind = 2`` because
    copy and cut reuse the range columns, and ``origin != reset`` because the
    starter template is not the candidate's edit. Ordered by the sequence
    triple then ``change_index``, never by time: a reload produces a fresh
    clock offset, and one editor event can carry several changes.
    """
    if events is None or len(events) == 0:
        return events if events is not None else pd.DataFrame()

    frame = events
    for column in ("change_index", "inserted_lines", "end_line", "start_line"):
        if column not in frame.columns:
            frame = frame.assign(**{column: pd.NA})

    selected = frame[
        (frame["session_id"] == session_id)
        & (frame["task_id"] == task_id)
        & (frame["load_seq"] == load_seq)
        & (frame["kind"] == int(Kind.edit))
        & (frame["origin"].fillna(-1) != int(Origin.reset))
        & (frame["version_id"].astype("Int64") > low)
        & (frame["version_id"].astype("Int64") <= high)
    ]
    return selected.sort_values(
        ["load_seq", "batch_seq", "seq_in_batch", "change_index"],
        kind="mergesort",
        na_position="first",
    )


def _map_range(
    edits: pd.DataFrame,
    start_line: int,
    end_line: int,
    *,
    backwards: bool,
) -> tuple[int, int] | None:
    """Move a line range across a set of edits, or refuse.

    Two things make this less obvious than adding up line deltas.

    **The footprint depends on the direction.** A Monaco change reports the
    range it REPLACED, in pre-edit coordinates, while ``version_id`` names the
    document AFTER the change (see :mod:`shared_features.evidence`). Mapping a
    range forward means the range is in pre-edit coordinates, so the edit's
    pre-edit footprint ``[start_line, end_line]`` decides whether they
    collide. Mapping backward means the range is in post-edit coordinates, so
    the post-edit footprint ``[start_line, start_line + inserted_lines]``
    decides. Comparing against the wrong one rejects safe mappings and, worse,
    accepts unsafe ones -- a point insertion of thirty lines has a pre-edit
    footprint of one line and a post-edit footprint of thirty.

    **Changes sharing a version_id are one editor event.** They are all
    expressed against the same pre-edit document (specification 9.3), so they
    are evaluated against the same incoming range and their deltas summed,
    rather than applied one after another as if each saw the previous one's
    result. Specification 9.3 also records that Monaco's changes-array
    ordering is *expected* but unverified; grouping this way means the result
    does not depend on that ordering.

    Returns ``None`` -- deliberately, rather than a best guess -- as soon as
    any edit touches the region. C2's fail-closed rule: an edit that replaces,
    deletes, intersects or splits the region destroys the evidence that the
    region in the other version is the same region.
    """
    lo, hi = int(start_line), int(end_line)
    if edits is None or len(edits) == 0:
        return (lo, hi)

    groups = list(edits.groupby("version_id", sort=False, dropna=False))
    if backwards:
        groups = list(reversed(groups))

    for _version, group in groups:
        delta = 0
        for edit in group.itertuples(index=False):
            edit_start = edit.start_line
            if edit_start is None or pd.isna(edit_start):
                return None        # an edit with no location: cannot reason
            edit_start = int(edit_start)

            added = edit.inserted_lines
            added = 0 if added is None or pd.isna(added) else int(added)
            edit_end = edit.end_line
            edit_end = (
                edit_start if edit_end is None or pd.isna(edit_end)
                else int(edit_end)
            )

            if backwards:
                # The range is in post-edit coordinates: the inserted text
                # occupies inserted_lines newlines from start_line.
                foot_lo, foot_hi = edit_start, edit_start + added
            else:
                # The range is in pre-edit coordinates: what was replaced.
                foot_lo, foot_hi = edit_start, edit_end

            net = added - (edit_end - edit_start)

            if foot_hi < lo:
                delta += -net if backwards else net
            elif foot_lo > hi:
                continue           # strictly below: the region does not move
            else:
                return None        # overlaps or touches: fail closed

        lo += delta
        hi += delta
        if lo < 1:
            return None            # the region was pushed off the document

    return (lo, hi)


def _snippet(code_text: str, start_line: int, end_line: int) -> str | None:
    """Lines ``start_line..end_line`` inclusive, 1-based as Monaco counts."""
    if code_text is None:
        return None
    lines = code_text.split("\n")
    if start_line < 1 or end_line < start_line or start_line > len(lines):
        return None
    return "\n".join(lines[start_line - 1:min(end_line, len(lines))])


def _unavailable(reason: str, detail: str, **fields) -> Resolution:
    return Resolution(status="unavailable", reason=reason, detail=detail, **fields)


def resolve_flag_code(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    *,
    session_id,
    task_id,
    load_seq: int,
    version_id: int,
    start_line: int,
    end_line: int,
    authorized: bool,
) -> Resolution:
    """Resolve one flagged range to source code, with an honest status.

    ``snapshots`` carries the ``code_snapshots`` columns; ``events`` carries
    raw ``events`` rows. Both are plain DataFrames, so this is testable from
    fixtures and the synthetic generator exercises the same path as production.

    ``authorized`` is positional-by-keyword and has no default. Full
    in-progress source is the most sensitive store in C1; a caller states the
    decision or does not call.
    """
    asked = {
        "target_load_seq": int(load_seq),
        "target_version_id": int(version_id),
        "target_start_line": int(start_line),
        "target_end_line": int(end_line),
    }

    if not authorized:
        return _unavailable(
            "not_authorized",
            "Source retrieval requires an authorized interviewer session.",
            **asked,
        )

    if end_line < start_line or start_line < 1:
        return _unavailable(
            "invalid_range",
            f"Line range {start_line}-{end_line} is not a valid range.",
            **asked,
        )

    if snapshots is None or len(snapshots) == 0:
        return _unavailable(
            "no_snapshot_in_load_segment",
            "No document snapshot was stored for this task.",
            **asked,
        )

    # NEVER across load_seq. A reload restarts version_id, so a snapshot from
    # another load segment is a different document that happens to share a
    # number.
    candidates = snapshots[
        (snapshots["session_id"] == session_id)
        & (snapshots["task_id"] == task_id)
        & (snapshots["load_seq"] == load_seq)
    ]
    if candidates.empty:
        return _unavailable(
            "no_snapshot_in_load_segment",
            f"No snapshot was stored in load segment {load_seq}. Snapshots "
            f"from other load segments are different documents and are not "
            f"used.",
            **asked,
        )

    redacted = candidates["code_redacted_at"].notna() if "code_redacted_at" in candidates else False
    live = candidates[~redacted] if redacted is not False else candidates

    # exact
    at_version = candidates[candidates["version_id"].astype("Int64") == int(version_id)]
    if not at_version.empty:
        row = at_version.iloc[0]
        if "code_redacted_at" in row.index and pd.notna(row["code_redacted_at"]):
            return _unavailable(
                "redacted",
                "The snapshot for this exact document version exists but its "
                "contents were removed by the retention policy.",
                snapshot_id=row.get("snapshot_id"),
                snapshot_load_seq=int(row["load_seq"]),
                snapshot_version_id=int(row["version_id"]),
                version_distance=0,
                direction="exact",
                **asked,
            )
        code = row["code_text"]
        snippet = _snippet(code, start_line, end_line)
        if snippet is None:
            return _unavailable(
                "range_outside_document",
                f"Lines {start_line}-{end_line} are not present in the stored "
                f"document for version {version_id}.",
                snapshot_id=row.get("snapshot_id"),
                snapshot_load_seq=int(row["load_seq"]),
                snapshot_version_id=int(row["version_id"]),
                version_distance=0,
                direction="exact",
                **asked,
            )
        return Resolution(
            status="exact",
            detail=(
                f"Exact historical code: the document was stored at version "
                f"{version_id} ({_capture_label(row.get('capture_reason'))})."
            ),
            snapshot_id=row.get("snapshot_id"),
            snapshot_load_seq=int(row["load_seq"]),
            snapshot_version_id=int(row["version_id"]),
            capture_reason=None if pd.isna(row.get("capture_reason")) else int(row["capture_reason"]),
            version_distance=0,
            direction="exact",
            snapshot_start_line=int(start_line),
            snapshot_end_line=int(end_line),
            code_text=code,
            snippet=snippet,
            **asked,
        )

    # nearest
    if live.empty:
        return _unavailable(
            "redacted",
            "Every snapshot in this load segment has been redacted by the "
            "retention policy.",
            **asked,
        )

    ordered = live.assign(
        _distance=(live["version_id"].astype("Int64") - int(version_id)).abs(),
        # Tie-break towards the LATER snapshot. Code flagged at version V
        # existed at V, so a later document still contains it unless an edit
        # removed it -- which the mapping detects. An earlier snapshot may
        # predate the code entirely, and that is undetectable here.
        _prefer_after=(live["version_id"].astype("Int64") < int(version_id)).astype(int),
    ).sort_values(["_distance", "_prefer_after", "version_id"], kind="mergesort")

    first_failure: str | None = None

    for row in ordered.itertuples(index=False):
        snap_version = int(row.version_id)
        backwards = snap_version < int(version_id)
        low, high = sorted((snap_version, int(version_id)))
        edits = _edits_between(
            events,
            session_id=session_id,
            task_id=task_id,
            load_seq=load_seq,
            low=low,
            high=high,
        )
        mapped = _map_range(edits, start_line, end_line, backwards=backwards)
        if mapped is None:
            first_failure = first_failure or "range_not_safely_mappable"
            continue

        snippet = _snippet(row.code_text, mapped[0], mapped[1])
        if snippet is None:
            first_failure = first_failure or "range_outside_document"
            continue

        distance = abs(snap_version - int(version_id))
        direction: Direction = "before" if backwards else "after"
        return Resolution(
            status="nearest",
            detail=(
                f"Nearest stored version: {snap_version}, {distance} version(s) "
                f"{direction} the flagged version {version_id} "
                f"({_capture_label(getattr(row, 'capture_reason', None))}). "
                f"The flagged range maps to lines {mapped[0]}-{mapped[1]} here."
            ),
            snapshot_id=getattr(row, "snapshot_id", None),
            snapshot_load_seq=int(row.load_seq),
            snapshot_version_id=snap_version,
            capture_reason=(
                None if getattr(row, "capture_reason", None) is None
                or pd.isna(getattr(row, "capture_reason", None))
                else int(row.capture_reason)
            ),
            version_distance=distance,
            direction=direction,
            snapshot_start_line=mapped[0],
            snapshot_end_line=mapped[1],
            code_text=row.code_text,
            snippet=snippet,
            warning=_NEAREST_WARNING.format(distance=distance, direction=direction),
            **asked,
        )

    return _unavailable(
        first_failure or "range_not_safely_mappable",
        "A snapshot exists in this load segment, but an edit between it and "
        "the flagged version changed the flagged region itself, so its "
        "location there cannot be established. No code is shown rather than a "
        "guessed range.",
        **asked,
    )
