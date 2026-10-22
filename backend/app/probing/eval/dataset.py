"""Records, splits and scoring for the Mapper accuracy test.

Follows ``label_map.json``. The records are whole solutions labelled by their
authors, so this measures tagging in general, not how the Mapper does on a
flagged segment. Pure functions only; the runner does the model calls.
"""

import gzip
import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.probing.knowledge.tags import Taxonomy

LABEL_MAP_PATH = Path(__file__).with_name("label_map.json")
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "dev_assets" / "km_dataset.jsonl.gz"


@dataclass(frozen=True)
class Record:
    record_id: str
    problem_id: str
    code: str
    labels: tuple[str, ...]
    license: str


@dataclass(frozen=True)
class Truth:
    """Ground-truth tag ids of one record."""

    strict: frozenset[str]  # primary tags only
    lenient: frozenset[str]  # primary plus the acceptable extras


def load_label_map(path: Path = LABEL_MAP_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> Iterator[Record]:
    """Python records labelled by their authors, with at least one label."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if (
                raw.get("language") == "python"
                and raw.get("label_quality") == "author-approach"
                and raw.get("labels")
            ):
                yield Record(
                    record_id=raw["id"],
                    problem_id=str(raw["problem_id"]),
                    code=raw["code"],
                    labels=tuple(raw["labels"]),
                    license=raw.get("license", ""),
                )


def truth_for(record: Record, label_map: dict[str, Any], taxonomy: Taxonomy) -> Truth | None:
    """The tags a record should get, or ``None`` if none of its labels is scorable."""
    strict: set[str] = set()
    lenient: set[str] = set()
    for label in record.labels:
        rule = label_map["labels"].get(label)
        if rule is None or rule["scoring"] != "include":
            continue
        if rule["primary"]:
            strict.add(taxonomy.by_name(rule["primary"]).id)
        for name in ([rule["primary"]] if rule["primary"] else []) + rule["acceptable"]:
            lenient.add(taxonomy.by_name(name).id)
    if not lenient:
        return None
    return Truth(strict=frozenset(strict), lenient=frozenset(lenient))


def taxonomy_gaps(records: Iterable[Record], label_map: dict[str, Any]) -> Counter:
    """Labels excluded because the taxonomy has no tag for them, with counts."""
    gaps: Counter = Counter()
    for record in records:
        for label in record.labels:
            rule = label_map["labels"].get(label)
            if rule is not None and rule["scoring"] != "include":
                gaps[label] += 1
    return gaps


def split_by_problem(
    scorable: Sequence[Record], *, dev: int, test: int, seed: int
) -> tuple[list[Record], list[Record]]:
    """One solution per problem; no problem in both sets; deterministic for a seed."""
    by_problem: dict[str, list[Record]] = {}
    for record in sorted(scorable, key=lambda r: r.record_id):
        by_problem.setdefault(record.problem_id, []).append(record)
    problems = sorted(by_problem)
    rng = random.Random(seed)
    rng.shuffle(problems)
    if dev + test > len(problems):
        raise ValueError(f"asked for {dev + test} problems but only {len(problems)} are scorable")
    picked = [rng.choice(by_problem[p]) for p in problems[: dev + test]]
    return picked[:dev], picked[dev : dev + test]


def hit_at_k(ranked: Sequence[str], truth: frozenset[str], k: int) -> bool:
    return any(tag_id in truth for tag_id in ranked[:k])
