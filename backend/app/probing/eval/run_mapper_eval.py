"""Mapper accuracy test on the public approach-labelled dataset.

Run from the repository root:

    python -m backend.app.probing.eval.run_mapper_eval --split dev --provider fake

Iterate on prompts with ``--split dev`` only. ``--split test`` is refused unless
you also pass ``--confirm-prompt-frozen``: touch the test set once, after the
prompt is frozen. Never tune on it and never claim a pass bar; the numbers are
reported and the owner decides what is good enough.

The run is resumable: every finished record is appended to a results file, and
a rerun skips them. Model responses are also cached on disk (public-dataset
code is allowed to be cached), so a rerun after a rate limit costs nothing for
finished calls.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from backend.app.probing.ai.cache import ResponseCache
from backend.app.probing.ai.config import GEMINI_PRIMARY_MODEL, AISettings
from backend.app.probing.ai.fake import FakeClient
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.gemini import GeminiClient
from backend.app.probing.ai.quota import DailyCounter
from backend.app.probing.eval.dataset import (
    DEFAULT_DATASET_PATH,
    Record,
    Truth,
    hit_at_k,
    load_label_map,
    read_records,
    split_by_problem,
    taxonomy_gaps,
    truth_for,
)
from backend.app.probing.knowledge.crosscheck import rule_only_ranking
from backend.app.probing.knowledge.cues import extract_cues
from backend.app.probing.knowledge.mapper import PROMPT_VERSION, KnowledgeMapper
from backend.app.probing.knowledge.tags import default_taxonomy
from backend.app.probing.python_extractor import PythonSegmentExtractor
from backend.app.probing.schemas import Location, Segment

RESULTS_DIR = Path(__file__).with_name("results")
KS = (1, 2, 3)


def record_to_segment(record: Record, extractor: PythonSegmentExtractor) -> Segment:
    """The whole solution as one segment (this test is not about flagged segments)."""
    lines = record.code.splitlines() or [record.code]
    entry = extractor.find_entry_function(record.code)
    flag_id = uuid5(NAMESPACE_URL, f"flag:{record.record_id}")
    return Segment(
        segment_id=uuid5(NAMESPACE_URL, f"segment:{record.record_id}"),
        task_id=uuid5(NAMESPACE_URL, f"task:{record.record_id}"),
        code=record.code,
        context=(),
        location=Location(start_line=1, end_line=len(lines), start_col=1, end_col=len(lines[-1]) + 1),
        parent_flag_id=flag_id,
        flag_ids=(flag_id,),
        entry_function=entry,
    )


class TimingClient:
    """Wraps a client and records each call's latency, for the report."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.latencies_ms: list[int] = []
        self.provider, self.family, self.hosted = inner.provider, inner.family, inner.hosted

    async def complete(self, *args, **kwargs):
        result = await self._inner.complete(*args, **kwargs)
        self.latencies_ms.append(result.latency_ms)
        return result


def build_client(provider: str):
    if provider == "fake":
        return FakeClient()
    if provider == "gemini":
        return GeminiClient()
    raise SystemExit(f"unknown provider {provider!r}: use 'fake' or 'gemini'")


def results_path(args: argparse.Namespace) -> Path:
    name = f"{args.provider}-{args.model}-{args.split}-seed{args.seed}-d{args.dev}t{args.test}.jsonl"
    return RESULTS_DIR / name


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["record_id"]] = row
    return done


async def evaluate(records: list[Record], args: argparse.Namespace, truths: dict[str, Truth]) -> list[dict]:
    taxonomy = default_taxonomy()
    extractor = PythonSegmentExtractor()
    settings = AISettings.from_env()
    if args.provider == "fake":
        # Nothing leaves the machine, so there is no free tier to protect.
        settings = replace(settings, max_rpm=100_000, max_tpm=100_000_000)
    daily = None
    if args.provider == "gemini":
        settings = settings.with_gemini_free_tier()
        daily = DailyCounter(settings.max_rpd, settings.cache_dir / "daily_requests.json")
    client = TimingClient(build_client(args.provider))
    gateway = LLMGateway(client, settings, cache=ResponseCache(settings.cache_dir), daily=daily)
    mapper = KnowledgeMapper(gateway, model=args.model, taxonomy=taxonomy)

    path = results_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path)
    rows: list[dict] = []
    for index, record in enumerate(records, start=1):
        if record.record_id in done:
            rows.append(done[record.record_id])
            continue
        segment = record_to_segment(record, extractor)
        result = await mapper.map_segment(segment, data_origin="public_dataset")
        hits = extract_cues(record.code, entry_function=segment.entry_function.name if segment.entry_function else None)
        row = {
            "record_id": record.record_id,
            "problem_id": record.problem_id,
            "truth_strict": sorted(truths[record.record_id].strict),
            "truth_lenient": sorted(truths[record.record_id].lenient),
            "model_ranked": [t.tag_id for t in result.tags],
            "rule_ranked": rule_only_ranking(hits, taxonomy.position),
            "overall_status": result.overall_status,
            "tag_statuses": [t.status for t in result.tags],
            "tokens_in": result.provenance.tokens_in,
            "tokens_out": result.provenance.tokens_out,
            "calls": result.provenance.calls,
            "cache_hit": result.provenance.cache_hit,
            "notes": list(result.notes),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"[{index}/{len(records)}] {record.record_id}: {row['model_ranked']}", file=sys.stderr)
    LATENCIES.extend(client.latencies_ms)
    return rows


LATENCIES: list[int] = []


def summarise(rows: list[dict], gaps: Counter, args: argparse.Namespace, pool: int, no_label: int) -> dict:
    def rates(key: str) -> dict:
        out = {}
        for k in KS:
            out[f"hit@{k}"] = {
                "lenient": _rate(sum(hit_at_k(r[key], frozenset(r["truth_lenient"]), k) for r in rows), len(rows)),
                "strict": _rate(sum(hit_at_k(r[key], frozenset(r["truth_strict"]), k) for r in rows), len(rows)),
            }
        return out

    per_tag: dict[str, dict] = {}
    for row in rows:
        for tag_id in row["truth_strict"]:
            entry = per_tag.setdefault(tag_id, {"n": 0, "hit@1_lenient": 0})
            entry["n"] += 1
            entry["hit@1_lenient"] += hit_at_k(row["model_ranked"], frozenset(row["truth_lenient"]), 1)

    return {
        "run": {
            "provider": args.provider,
            "model": args.model,
            "split": args.split,
            "seed": args.seed,
            "dev": args.dev,
            "test": args.test,
            "prompt_version": PROMPT_VERSION,
            "taxonomy_version": default_taxonomy().taxonomy_version,
            "date": date.today().isoformat(),
        },
        "records": len(rows),
        "scorable_pool": pool,
        "records_skipped_no_scorable_label": no_label,
        "model": rates("model_ranked"),
        "rule_only_baseline": rates("rule_ranked"),
        "records_with_at_least_one_rule_cue": sum(1 for r in rows if r["rule_ranked"]),
        "overall_status": dict(Counter(r["overall_status"] for r in rows)),
        "tag_status": dict(Counter(s for r in rows for s in r["tag_statuses"])),
        "model_calls_made_this_run": len(LATENCIES),
        "latency_ms_per_call": (
            {"mean": round(sum(LATENCIES) / len(LATENCIES)), "max": max(LATENCIES)} if LATENCIES else None
        ),
        "records_with_invalid_output_retry": sum(
            1 for r in rows if r["calls"] > 2
        ),
        "tokens_in": sum(r["tokens_in"] for r in rows),
        "tokens_out": sum(r["tokens_out"] for r in rows),
        "taxonomy_gaps_in_pool": dict(gaps),
        "per_tag_strict_truth": per_tag,
        "caveat": "Whole solutions, not flagged segments. Per-tag counts are small: treat as anecdotal.",
    }


def _rate(hits: int, total: int) -> dict:
    return {"hits": hits, "of": total, "rate": round(hits / total, 3) if total else None}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("KM_DATASET_PATH", DEFAULT_DATASET_PATH)))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--dev", type=int, default=20)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", default="fake")
    parser.add_argument("--model", default=None, help="default: dummy-v0 (fake) or the Gemini primary model")
    parser.add_argument("--confirm-prompt-frozen", action="store_true")
    parser.add_argument("--resume", action="store_true", help="accepted for clarity; runs always resume")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.model is None:
        args.model = GEMINI_PRIMARY_MODEL if args.provider == "gemini" else "dummy-v0"
    if args.split == "test" and not args.confirm_prompt_frozen:
        print("refusing to touch the test set: pass --confirm-prompt-frozen once the prompt is final", file=sys.stderr)
        return 2

    taxonomy, label_map = default_taxonomy(), load_label_map()
    all_records = list(read_records(args.dataset))
    truths = {r.record_id: t for r in all_records if (t := truth_for(r, label_map, taxonomy)) is not None}
    scorable = [r for r in all_records if r.record_id in truths]
    dev, test = split_by_problem(scorable, dev=args.dev, test=args.test, seed=args.seed)
    chosen = dev if args.split == "dev" else test

    rows = asyncio.run(evaluate(chosen, args, truths))
    summary = summarise(
        rows, taxonomy_gaps(all_records, label_map), args, len(scorable), len(all_records) - len(scorable)
    )
    out = RESULTS_DIR / f"{date.today().isoformat()}-{args.provider}-{args.split}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwritten to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
