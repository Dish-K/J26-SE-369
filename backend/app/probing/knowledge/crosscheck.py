"""Compare the model's tags with the rule cues.

The check shows consistency, not correctness: a cue can be incidental and a
model can be right without one. Only strong cues can make a result doubtful.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.app.probing.knowledge.cues import CueHit, Strength

TagStatus = Literal["supported", "unchecked", "doubtful"]
OverallStatus = Literal["supported", "unchecked", "doubtful", "rule_only"]

STRENGTH_RANK: dict[Strength, int] = {"strong": 0, "medium": 1, "weak": 2}


class IgnoredCue(BaseModel):
    """A strong cue pointing at a tag the model did not choose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cue: str
    tag_id: str


class CrossCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tag_status: dict[str, TagStatus]
    ignored_strong_cues: tuple[IgnoredCue, ...]
    overall_status: OverallStatus


def cross_check(chosen_tag_ids: Sequence[str], hits: Sequence[CueHit]) -> CrossCheck:
    """``chosen_tag_ids`` are in rank order, best first."""
    supported = {s.tag_id for hit in hits for s in hit.supports}
    tag_status: dict[str, TagStatus] = {
        tag_id: "supported" if tag_id in supported else "unchecked" for tag_id in chosen_tag_ids
    }

    chosen = set(chosen_tag_ids)
    ignored: list[IgnoredCue] = []
    for hit in hits:
        for support in hit.supports:
            if support.strength == "strong" and support.tag_id not in chosen:
                item = IgnoredCue(cue=hit.text, tag_id=support.tag_id)
                if item not in ignored:
                    ignored.append(item)

    if ignored:
        overall: OverallStatus = "doubtful"
    elif chosen_tag_ids and tag_status[chosen_tag_ids[0]] == "supported":
        overall = "supported"
    else:
        overall = "unchecked"
    return CrossCheck(
        tag_status=tag_status,
        ignored_strong_cues=tuple(ignored),
        overall_status=overall,
    )


def rule_only_ranking(hits: Sequence[CueHit], catalogue_position) -> list[str]:
    """Tags from cues alone: strongest cue first, then most cues, then catalogue order.

    ``catalogue_position`` maps a tag id to its place in the catalogue so the
    result is deterministic. At most three tags.
    """
    best: dict[str, int] = {}
    count: dict[str, int] = {}
    for hit in hits:
        for support in hit.supports:
            rank = STRENGTH_RANK[support.strength]
            best[support.tag_id] = min(best.get(support.tag_id, 9), rank)
            count[support.tag_id] = count.get(support.tag_id, 0) + 1
    ordered = sorted(best, key=lambda t: (best[t], -count[t], catalogue_position(t)))
    return ordered[:3]
