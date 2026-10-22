from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib, json, random, uuid
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ("no_ai", "ide_ai", "external_ai")
POLICY = "known_single_contribution_category_v1"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_cache(path):
    obj = json.loads(Path(path).read_text())
    b = obj["payload"]
    if canonical(b) != obj["payload_sha256"]:
        raise ValueError("Calibration cache hash mismatch")
    if (
        b["format"] != "codetrace_run_start_sampling_cache_v1"
        or b["holdout_examined"] is not False
        or b["not_training_ready"] is not True
    ):
        raise ValueError("Unexpected cache provenance")
    if b["label_status"] != "provisional_human_reference_NOT_verified_no_ai":
        raise ValueError("CS1 source label status changed")
    return b


def choose(rng, weights):
    keys, values = zip(*weights)
    return rng.choices(keys, weights=values, k=1)[0]


class Document:
    """Artificial append/tail-edit document with exact contribution lineage."""

    def __init__(self, length):
        self.text = "x" * length
        self.spans = [[length, None]] if length else []

    def apply(self, inserted, removed, category, token):
        removed = min(int(removed), len(self.text))
        offset = len(self.text) - removed
        affected = set()
        left = removed
        while left:
            n, label = self.spans[-1]
            take = min(n, left)
            affected.add(label)
            self.spans[-1][0] -= take
            left -= take
            if self.spans[-1][0] == 0:
                self.spans.pop()
        self.text = self.text[:offset] + token * inserted
        if inserted:
            affected.add(category)
            if self.spans and self.spans[-1][1] == category:
                self.spans[-1][0] += inserted
            else:
                self.spans.append([inserted, category])
        return offset, removed, affected


def simulate_persona(pools, config, index):
    seed = config["seed"] + index * 1_000_003
    rng = random.Random(seed)
    family = f"synthetic-persona-{index:05d}"
    session = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f'codetrace/{config["version"]}/{family}/session'
        )
    )
    task = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f'codetrace/{config["version"]}/{family}/task')
    )
    base = BASE + timedelta(days=index)
    doc = Document(config["initial_document_chars"])
    events = []
    provenance = []
    episodes = []
    version = 0
    run_count = 0
    clipped_deletes = 0
    timing_scale = rng.uniform(*config["persona_timing_scale"])
    ide_p = rng.uniform(*config["ide_assistant_probability"])
    ext_p = rng.uniform(*config["external_assistant_probability"])
    chunks = {
        "ide_ai": [
            int(a)
            for a, b in pools["lengths"]["Paste"]
            if config["ide_chunk_empirical_length_bounds"][0]
            <= a
            <= config["ide_chunk_empirical_length_bounds"][1]
        ],
        "external_ai": [
            int(a)
            for a, b in pools["lengths"]["Paste"]
            if config["external_chunk_empirical_length_bounds"][0]
            <= a
            <= config["external_chunk_empirical_length_bounds"][1]
        ],
    }
    if not all(chunks.values()):
        raise ValueError("AI length filters have no empirical support")

    def identity(t):
        return {
            "session_id": session,
            "task_id": task,
            "load_seq": 0,
            "batch_seq": len(events),
            "seq_in_batch": 0,
            "time": (base + timedelta(milliseconds=int(t))).isoformat(
                timespec="microseconds"
            ),
            "perf_now": float(t),
        }

    def add_edit(t, inserted, removed, origin, category, mechanism, episode):
        nonlocal version, clipped_deletes
        before = len(doc.text)
        offset, actual_rm, categories = doc.apply(
            int(inserted), int(removed), category, "p" if origin == 1 else "t"
        )
        clipped_deletes += actual_rm != removed
        version += 1
        event = {
            **identity(t),
            "kind": 2,
            "origin": origin,
            "version_id": version,
            "change_index": 0,
            "start_line": 1,
            "end_line": 1,
            "start_col": offset + 1,
            "end_col": before + 1,
            "range_offset": offset,
            "inserted_len": int(inserted),
            "removed_len": actual_rm,
            "inserted_lines": 0,
            "doc_len": len(doc.text),
            "paste_origin": 2 if origin == 1 else None,
            "inserted_text": "p" * int(inserted) if origin == 1 else None,
        }
        events.append(event)
        provenance.append(
            {
                "session_id": session,
                "task_id": task,
                "batch_seq": event["batch_seq"],
                "time": event["time"],
                "episode_id": episode,
                "mechanism": mechanism,
                "categories": sorted(x for x in categories if x is not None),
                "uncertain": None in categories,
                "code_activity": bool(inserted or actual_rm),
                "inserted_category": category if inserted else None,
                "provenance_basis": "known_simulated_content_source_and_exact_affected_span_lineage",
            }
        )

    # Artificial baseline/reset contributes no training provenance.
    version += 1
    events.append(
        {
            **identity(0),
            "kind": 2,
            "origin": 4,
            "version_id": version,
            "change_index": 0,
            "start_line": 1,
            "end_line": 1,
            "start_col": 1,
            "end_col": 1,
            "range_offset": 0,
            "inserted_len": len(doc.text),
            "removed_len": 0,
            "inserted_lines": 0,
            "doc_len": len(doc.text),
            "paste_origin": None,
            "inserted_text": None,
        }
    )
    cursor = 1000
    for cycle in range(config["episode_cycles"]):
        order = list(CLASSES)
        rng.shuffle(order)
        for target in order:
            eid = f"{family}-episode-{len(episodes):02d}"
            duration = rng.randint(*config["episode_seconds"]) * 1000
            end = cursor + duration
            background = (
                0.0
                if target == "no_ai"
                else rng.choice(config["background_unaided_probability_choices"])
            )
            previous = None
            t = cursor
            actions = 0
            while t < end and actions < config["max_actions_per_episode"]:
                if previous is None:
                    action = choose(rng, list(pools["starts"].items()))
                else:
                    choices = [
                        (b, n)
                        for key, n in pools["transitions"].items()
                        for a, b in [key.split("->")]
                        if a == previous
                    ]
                    action = choose(rng, choices)
                    gap = rng.choice(pools["gaps"][previous + "->" + action])
                    t += max(1, int(gap * timing_scale))
                    if t >= end:
                        break
                if action == "RUN":
                    run_count += 1
                    events.append(
                        {
                            **identity(t),
                            "kind": 8,
                            "run_id": str(
                                uuid.uuid5(uuid.UUID(task), f"run/{run_count}")
                            ),
                            "exit_status": None,
                            "browser_duration_ms": None,
                            "sandbox_time_ms": None,
                            "sandbox_memory_kb": None,
                        }
                    )
                else:
                    inserted, removed = map(int, rng.choice(pools["lengths"][action]))
                    category = (
                        "no_ai"
                        if target == "no_ai" or rng.random() < background
                        else target
                    )
                    origin = 1 if action == "Paste" else 0
                    mechanism = (
                        "simulated_unaided_edit"
                        if category == "no_ai"
                        else "simulated_known_ai_content_retyping_or_revision"
                    )
                    if category != "no_ai" and action in ("Insert", "Paste"):
                        assist = rng.random() < (
                            ide_p if category == "ide_ai" else ext_p
                        )
                        if assist:
                            inserted = rng.choice(chunks[category])
                            removed = 0
                            if category == "ide_ai":
                                # Some known IDE-supplied code is explicitly retyped/pasted; acceptance is not a label rule.
                                origin = rng.choices([5, 0, 1], [0.7, 0.15, 0.15], k=1)[
                                    0
                                ]
                                mechanism = {
                                    5: "simulated_ide_acceptance",
                                    0: "simulated_ide_content_retyping",
                                    1: "simulated_ide_content_transfer",
                                }[origin]
                            else:
                                origin = rng.choices([1, 0], [0.8, 0.2], k=1)[0]
                                mechanism = (
                                    "simulated_external_content_paste"
                                    if origin == 1
                                    else "simulated_external_content_retyping"
                                )
                                t += rng.randint(
                                    *config["external_consultation_gap_ms"]
                                )
                                if t >= end:
                                    break
                    if category == "no_ai" and origin == 1:
                        mechanism = "simulated_unaided_copy_of_own_content"
                    add_edit(t, inserted, removed, origin, category, mechanism, eid)
                previous = action
                actions += 1
            episodes.append(
                {
                    "episode_id": eid,
                    "planned_context": target,
                    "start_ms": cursor,
                    "end_ms": end,
                    "background_unaided_probability": background,
                    "actions": actions,
                    "note": "Context is not a window label; actual contribution sidecar controls labeling.",
                }
            )
            cursor = end
    tasks = [
        {
            "session_id": session,
            "task_id": task,
            "started_at": base.isoformat(timespec="microseconds"),
            "submitted_at": (base + timedelta(milliseconds=cursor)).isoformat(
                timespec="microseconds"
            ),
        }
    ]
    audit = {
        "family_id": family,
        "seed": seed,
        "timing_scale": timing_scale,
        "ide_assistant_probability": ide_p,
        "external_assistant_probability": ext_p,
        "episodes": episodes,
        "run_starts": run_count,
        "delete_lengths_clipped_to_document": clipped_deletes,
        "final_document_sha256": hashlib.sha256(doc.text.encode()).hexdigest(),
    }
    replay(events, audit)
    return events, tasks, provenance, audit


def replay(events, audit):
    doc = ""
    version = 0
    ids = set()
    last = -1
    for i, e in enumerate(events):
        if e["batch_seq"] != i or e["perf_now"] < last:
            raise ValueError("Invalid event order")
        last = e["perf_now"]
        if e["kind"] == 8:
            if e["run_id"] in ids or "version_id" in e:
                raise ValueError("Invalid run identity/version")
            ids.add(e["run_id"])
            continue
        version += 1
        if e["version_id"] != version:
            raise ValueError("Invalid edit version")
        if e["origin"] == 4:
            doc = "x" * e["inserted_len"]
        else:
            offset = e["range_offset"]
            if offset != len(doc) - e["removed_len"]:
                raise ValueError("Invalid pre-edit offset")
            doc = doc[:offset] + ("p" if e["origin"] == 1 else "t") * e["inserted_len"]
        if len(doc) != e["doc_len"]:
            raise ValueError("Replay length mismatch")
    if hashlib.sha256(doc.encode()).hexdigest() != audit["final_document_sha256"]:
        raise ValueError("Replay hash mismatch")


def label_window(contributions, start, end):
    relevant = [
        e
        for e in contributions
        if start <= pd.Timestamp(e["time"]) < end and e["code_activity"]
    ]
    categories = sorted({c for e in relevant for c in e["categories"]})
    if not relevant:
        reason = "insufficient_activity"
    elif any(e["uncertain"] for e in relevant) or not categories:
        reason = "uncertain_provenance"
    elif len(categories) > 1:
        reason = "mixed_contribution_categories"
    else:
        reason = "included_single_known_category"
    return (
        (categories[0] if reason == "included_single_known_category" else None),
        reason,
        len(relevant),
        categories,
    )


def build(cache_path, config_path, output):
    from shared_features import (
        extract_features,
        FEATURE_NAMES,
        SCHEMA_VERSION,
        feature_payload,
    )
    from shared_features.feature_schema import FEATURE_SCHEMA

    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    pools = load_cache(cache_path)
    config = json.loads(config_path.read_text())
    if SCHEMA_VERSION != "1.0.0" or len(FEATURE_NAMES) != 40:
        raise ValueError("Wrong official feature contract")
    if config["personas"] < 10:
        raise ValueError("Use at least 10 persona families for grouped experiments")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    all_feature_rows = []
    window_rows = []
    family_audits = []
    for i in range(config["personas"]):
        events, tasks, prov, audit = simulate_persona(pools, config, i)
        person = raw / audit["family_id"]
        person.mkdir()
        pd.DataFrame(events).to_csv(person / "events.csv", index=False)
        pd.DataFrame(tasks).to_csv(person / "tasks.csv", index=False)
        with (person / "contributions.jsonl").open("w") as f:
            for e in prov:
                f.write(json.dumps(e, sort_keys=True) + "\n")
        save_json(person / "scenario.json", audit)
        ev, task = pd.DataFrame(events), pd.DataFrame(tasks)
        ev["time"] = pd.to_datetime(ev["time"], format="ISO8601", utc=True)
        for k in ["started_at", "submitted_at"]:
            task[k] = pd.to_datetime(task[k], format="ISO8601", utc=True)
        vectors = extract_features(ev, task, run_id="final")
        FEATURE_SCHEMA.validate(vectors)
        if (
            not vectors["run_error_rate"].isna().all()
            or not vectors["runs_before_first_success"].isna().all()
        ):
            raise ValueError("Starts-only generator invented outcomes")
        for _, row in vectors.iterrows():
            payload = feature_payload(row)
            label, reason, n, cats = label_window(
                prov, row["window_start"], row["window_end"]
            )
            rid = f"window-{len(window_rows):07d}"
            window_rows.append(
                {
                    "row_id": rid,
                    "family_id": audit["family_id"],
                    "session_id": row["session_id"],
                    "task_id": row["task_id"],
                    "window_start": row["window_start"].isoformat(),
                    "window_end": row["window_end"].isoformat(),
                    "label": label,
                    "included": label is not None,
                    "reason": reason,
                    "contribution_events": n,
                    "categories": json.dumps(cats),
                    "label_policy": POLICY,
                }
            )
            all_feature_rows.append(payload["features"])
        family_audits.append({k: v for k, v in audit.items() if k != "episodes"})
        print(
            f'Persona {i+1}/{config["personas"]}: raw replay + official C1 passed',
            flush=True,
        )
    features = pd.DataFrame(all_feature_rows, columns=FEATURE_NAMES)
    windows = pd.DataFrame(window_rows)
    features.to_csv(output / "all_features.csv", index=False)
    windows.to_csv(output / "windows_and_labels.csv", index=False)
    mask = windows["included"]
    features[mask].to_csv(output / "training_features.csv", index=False)
    windows.loc[mask, ["row_id", "family_id", "label"]].to_csv(
        output / "training_labels.csv", index=False
    )
    save_json(output / "generator_configuration.json", config)
    save_json(output / "persona_manifest.json", family_audits)
    counts = windows.loc[mask, "label"].value_counts().to_dict()
    if set(counts) != set(CLASSES):
        raise ValueError(
            "Missing training class; inspect exclusions without relabeling them"
        )
    report = {
        "dataset_version": config["version"],
        "status": "EXPERIMENTAL_SYNTHETIC_DATASET",
        "experimental_training_ready": True,
        "research_validated": False,
        "verified_real_no_ai_labels": False,
        "source_label_status": pools["label_status"],
        "holdout_examined": False,
        "label_policy": POLICY,
        "feature_schema_version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "all_windows": len(windows),
        "included_windows": int(mask.sum()),
        "exclusion_counts": {
            str(k): int(v)
            for k, v in windows.loc[~mask, "reason"].value_counts().items()
        },
        "personas": config["personas"],
        "generator_sha256": sha(Path(__file__)),
        "calibration_cache_sha256": sha(cache_path),
        "source_manifest_sha256": pools["source_manifest_sha256"],
        "configuration_sha256": sha(config_path),
        "simulation_assumptions": config["simulation_assumptions"],
        "limitations": [
            "No real TSTR or reliable mixed-behavior detection established.",
            "All excluded windows and raw sessions retained; no mixed-window ground truth inferred.",
            "One known affected-code edit is the minimum activity unit; this is not a confidence threshold.",
            "Physical keys/focus/completion outcomes absent consistently across classes.",
            "Task-to-date run features retain prior context; labels describe current-window contributions only.",
            "Single-line artificial code and provisional AI simulation settings limit deployment transfer.",
        ],
        "output_sha256": {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    save_json(output / "dataset_manifest.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = build(a.cache, a.config, a.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "class_counts",
                    "all_windows",
                    "included_windows",
                    "exclusion_counts",
                    "holdout_examined",
                ]
            },
            indent=2,
        )
    )
