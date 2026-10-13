# `shared_features/` — the C1 → C2 feature contract

One registry, one validated schema, one extractor. Real telemetry, the
synthetic generator and every degraded copy run through the **same** code, which
is what makes their feature vectors comparable by construction.

**Status: `1.0.0` — frozen 2026-10-09.** C2 approved the three remaining items
(`unknown_paste_count`, `window_task_coverage_ms`, the `unpaired_keydown_rate`
rename), so the 40-feature list, its order and its nullable-value semantics are
a contract. Any change to a name, dtype, unit or position is a new schema
version — never an edit to this one.

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install --requirement backend/requirements.txt
python -m pytest -q tests/telemetry    # 107 tests, no database needed
```

The versions pinned in `backend/requirements.txt` matter: training and inference
must run the same pandas, numpy and pandera, or the same events can produce
different features.

## Using it

```python
from shared_features import extract_features, feature_payload, SCHEMA_VERSION

vectors = extract_features(events, tasks, run_id="final") # one row per window
payload = feature_payload(vectors.iloc[0]) # -> feature_vectors.features
```

Reading them back out of the database:

```python
from shared_features import frame_from_payloads
frame = frame_from_payloads(rows) # validates, and refuses another version
```

`shared_features/example_fixture.json` holds a complete worked example: raw
events, the tasks frame, the resulting payloads with their nulls, and the edit
evidence for the first window.

| Name | What it is |
|---|---|
| `registry.yaml` | **The contract.** 40 features, the event code maps, the parameters. Data, not code |
| `feature_schema.py` | Typed view of the registry: `IntEnum`s, pandera schemas, payload helpers |
| `extractor.py` | `extract_features()` and `extract_window()` |
| `evidence.py` | `edit_evidence()` and `EDIT_EVIDENCE_SQL` for C2's localization |

**Adding a feature means editing `registry.yaml` and the extractor.** If it ever
means editing three files, it will soon mean editing two and forgetting the
third.

## Windows

30 seconds wide, every 10 seconds, so **every instant falls in exactly three
windows**. That is C1's geometry and it constrains C2: one flag per flagged
window reports a single paste three times.

The grid starts 20 s before the task's first bucket, which is what makes the
overlap three-deep *at the start of a task* as well as in the middle.

## Two passes

| | `run_id='live'` | `run_id='final'` |
|---|---|---|
| When | every 10 s during the task | once, at submission |
| Over | whatever has arrived | the complete event stream |
| For | C2 flagging during the interview | training, research, degradation |
| Revised? | **never** | n/a |

A window `[T, T+30s)` is scorable at `T + 30s + window_grace_ms`. **Detection
latency is 2–12 s**, floored by the window width — not by implementation
quality. Late events are dropped from `live` and present in `final`; revising a
live vector would mean retracting a flag an interviewer has already seen.

Comparing `live` with `final` measures what the live path loses, which the
reliability experiment wants anyway.

## Three rules that are easy to undo by accident

**Order by `(load_seq, batch_seq, seq_in_batch)`, never by time.** Browsers
coarsen `performance.now()`, so two events can share a timestamp, and a reload
gives a later event an earlier wall clock.

`time` *is* used for one thing: deciding which window an event falls in. That is
correct even across a reload, because the clock handshake re-runs. **Window
membership and event ordering are different questions.**

**Recompute every duration from `perf_now`.** The stored `gap_ms` column is
never read. A degraded copy perturbs `perf_now`; reading the stored column would
return the clean session's timings and the degradation study would silently
measure nothing. `test_a_deliberately_wrong_stored_gap_ms_is_ignored` guards it.

**Flight excludes gaps over 2 s.** One 20-second think swamps `std_flight_ms`
and the uniform-rhythm signal disappears. Nothing is lost — the gap still lands
in `pause_ms` and ends the burst.

## Missing values

**A count is 0; a statistic over nothing is null.** Never 0 for a timing or a
ratio. A `backspace_rate` of 0 claims flawless typing; in a window where the
candidate was reading the problem, that claim is fabricated.

Every registry key is present in every payload, so a null value and an absent
key never get confused. Imputation is C2's, inside its versioned model pipeline.

## Event codes

`registry.yaml` carries the code map under `enums:`, each marked with its
authority:

- **`sql`** — `Kind`, `Origin`, `ExitStatus`, `SourcePane`, `Modifier`. A CHECK
  constraint in `002_tables.sql` pins the range, and `test_enum_parity.py`
  parses that file and asserts the Python values match. Drift cannot survive CI.
- **`spec`** — `KeyClass`, from specification §4.
- **`c1`** — `PasteOrigin` and the four `env_state` enums. The migration defines
  these columns but constrains nothing and there is no `COMMENT ON` anywhere, so
  the registry is the only authority. Nothing can check them automatically; that
  is recorded rather than quietly assumed.

## Localization, for C2

```python
from shared_features import edit_evidence, EDIT_EVIDENCE_SQL
rows = edit_evidence(events, session_id, task_id, window_start, window_end)
```

Three rules the database will **not** enforce:

1. **`kind = 2`.** Copy and cut reuse `start_line`/`end_line` to mean the
   *selected* range. No constraint prevents the confusion and there is no index
   on `events(kind)`.
2. **`origin != reset`.** A reset edit is the starter template loading — a large
   insertion with no keystrokes, the `external_ai` signature made by our own UI.
3. **Filter on `time`, order on the sequence triple.**

Line ranges are **historical** — the coordinates during the detection episode.
Migration `006` resolves them at submission; C3 reads
`classifier_flags_for_probing`, never the raw historical range.

### asyncpg

Identical filters, identical order, identical columns — only the parameter
style differs. The two forms are generated from one template, and
`test_evidence.py` asserts that porting `$n` back to `%s` reproduces the other
exactly, so they cannot drift.

```python
from shared_features import EDIT_EVIDENCE_SQL_ASYNCPG, EVIDENCE_COLUMNS

async def window_evidence(pool, session_id, task_id, window_start, window_end):
    rows = await pool.fetch(
        EDIT_EVIDENCE_SQL_ASYNCPG,
        session_id, # uuid.UUID
        task_id, # uuid.UUID
        window_start, # timezone-AWARE datetime
        window_end,
    )
    return [dict(r) for r in rows]
```

asyncpg binds `uuid.UUID` and `timestamptz` natively. Pass **timezone-aware**
datetimes — asyncpg rejects naive ones rather than guessing a zone, which is
the behaviour to want here.

### Two line ranges, because a change mixes two coordinate systems

This one is easy to get wrong and fails silently. A Monaco change reports the
range it **replaced**, in *pre-edit* coordinates, while `version_id` names the
document *after* the change. So a row's raw range and its `version_id` do not
describe the same document.

| Column pair | Coordinate system |
|---|---|
| `replaced_start_line` / `replaced_end_line` | **Pre-edit** — what the change replaced, exactly as Monaco reported it |
| `touched_start_line` / `touched_end_line` | **Post-edit** — aligned with `version_id`: `[start_line, start_line + inserted_lines]` |

They coincide for same-line edits and diverge for every multi-line one.
Deleting lines 10–15 has a pre-edit range six lines tall and a post-edit range
of **one** line, because after the change there is nothing left to span.
`net_line_delta` (`inserted_lines − (end_line − start_line)`) comes computed,
so the forward mapping in §14.1 does not recompute it per row.

## Resolving a flag to source code

```python
from shared_features import resolve_flag_code

r = resolve_flag_code(
    snapshots, events,
    session_id=..., task_id=..., load_seq=0, version_id=137,
    start_line=12, end_line=41,
    authorized=True, # no default: state the decision or do not call
)
```

Three statuses, never blurred:

| `r.status` | Meaning |
|---|---|
| `exact` | A snapshot exists at precisely `(load_seq, version_id)` and is unredacted. **The real historical code.** |
| `nearest` | Closest snapshot **in the same load segment**, flagged range mapped into its coordinates, with `version_distance` and `direction`. `r.warning` says in words that it is not the original |
| `unavailable` | No snapshot in that load segment, the only one is redacted, the range cannot be mapped safely, or the caller is not authorized. `r.reason` says which |

Read `r.is_historical_original` — not `r.code_text is not None` — before
presenting anything as the flagged code. Text is populated for `nearest` too.

**What this does not do.** Nothing recovers the text of a version no snapshot
captured. Edit rows carry offsets and lengths, not characters, and
`inserted_text` is pastes only; if a line held `x = 1` at version 10 and holds
`x = 2` at version 12, the character `1` is stored nowhere. Three capture
triggers — the 10 s tick, immediately after every paste, and immediately before
any edit removing more than 200 characters — make `exact` the common case for
the evidence that carries most weight, because pasted code is both the dominant
`external_ai` signal and rare. They raise coverage; they are not a guarantee.

**Two rules it will not bend.** It never crosses a `load_seq` boundary —
`version_id` restarts at 1 when a reload builds a new Monaco model, so version 7
before and after a reload are different documents. And it fails closed: if an
intervening edit touches, splits or deletes the flagged region, it returns
`unavailable` rather than highlight a guessed span. Highlighting the wrong lines
is, to the person reading them, indistinguishable from highlighting the right
ones.
