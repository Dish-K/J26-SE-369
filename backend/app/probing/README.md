# C3 Adaptive Probing: decision log

Short on purpose. Each phase appends a section; earlier sections are not rewritten.

Note: the demo page (`backend/app/probing/demo/`) is kept out of this PR. Sections below that describe the demo page, its recorded answers or its start-up are background and refer to files that are not in this PR.

## Phase 1: Knowledge Mapper (2026-10-21)

**Decisions**
- Rules (tree-sitter cues C01–C25) gather evidence only. The model assigns up to 3 tags; the cross-check marks each `supported`, `unchecked`, `doubtful` or `rule_only`. Only a *strong* cue the model ignored makes a result `doubtful`. Tags are evidence about code, never proof about a person.
- Any model-path failure (consent refused, rate limited, bad output) falls back to a rule-only result with a fixed-text note. The Mapper does not raise to its caller.
- Candidate code goes to a hosted model only if `PROBING_ALLOW_CANDIDATE_CODE_TO_HOSTED_LLM=true`. No reasoning is stored for candidate origin and nothing from candidate origin is cached. Allowed origins: `fixture`, `public_dataset`, `owner_sample`.
- Cues differ slightly from the brief: overflow-safe midpoint in C05, extra literal moduli in C09, C07 needs exactly one other same-array cell, C17 skips C07 matches.
- Accuracy test: one solution per problem, no problem in both sets (dev 20 / test 50, seed 42). The test split is refused without `--confirm-prompt-frozen` and is run once, after the prompts are settled on dev.
- Models (Gemini free tier, `httpx` only): primary `gemini-3.5-flash-lite`, automatic fallback `gemini-3.1-flash-lite`, development reviewer `gemma-4-31b-it` (same provider, so a weaker check). Pacing 12 RPM / 200K TPM / 450 requests per model per day; the day counter is persisted in the cache folder and resets at midnight Pacific. A 429 with limit 0 is not retried and is reported as "quota is 0".
- Anything fake is labelled (`provider="fake"`, `model_version="dummy-v0"`). Numbers from the fake client are never quoted as accuracy.

**Working defaults (W1–W8)**: top-3 tags per segment; ladder for the top 2; three levels (5 / 10 / 20 min); middle level is the default; ladders generated on demand and cached; probe package carries about 6 model-written test cases plus an empty `oracle` slot; verification and approval are separate fields; reviewer behind `PROBING_REVIEWER_ENABLED`; reviewer output is advisory only.

**Other choices**: runner = contract and `NullRunner` only (nothing generated is executed this week). Storage = in memory plus a JSON-lines log. Demo page = a separate small FastAPI app on port 8100 with one static HTML file. Next free migration number is 012; migrations 009–011 (C2's) are not applied by this component.

**Deferred**: Priority Scorer, Response Evaluator, code runner and verification, database storage, hand-off to `tasks`, languages other than Python.

**Open questions for other owners** (not for this component to fix)
- C1/C2: when will anything write `task_submissions` and `classifier_flag_final_locations`? Until then C3 runs on fixtures.
- C1: where are question or problem statements stored? `tasks` has none.
- C2: flags from different `model_version`s are not merged; `touched_line_count` is not available.
- Team: `numpy==2.4.6` is pinned in `requirements.txt` but the working venv has 2.5.3.
- Migrations 009–011 contain their own `BEGIN`/`COMMIT` inside the runner's transaction (unverified concern).

## Phase 2 and 3: Probe Generator, repository, demo page (2026-10-21)

**Decisions**
- A probe package holds the question, signature, about 6 model-written cases (inputs and expected values as JSON text), a checker name, a reference solution, an empty `oracle` slot, static-check results, provenance, `observed_minutes` (empty), and separate `verification` (always `unverified` this week) and `approval` fields. It never holds candidate code: the generator receives only a tag id and a level, so its requests are cacheable.
- One model call per level (no think-first step), temperature 0.4. Output is parsed as data, then checked statically with `ast` and `json` only: valid signature, reference defines the function with matching parameters, imports limited to a small standard-library list, no `eval`/`exec`/`open`/dunder access, no unbounded `while True`, 4 to 10 well-formed distinct cases. A failed check gets one retry with the problems named; a level that still fails is left out of the ladder with a note. Generated code is never executed (`NullRunner`; `verify_probe` raises `RunnerUnavailable`).
- Approval needs a named person and a probe that passed its static checks. It never changes `verification`.
- Storage is in memory with an optional JSON-lines event log that is replayed on start, so approvals survive a restart. Mapper results and probes are logged; segment code is not.
- The demo is a separate FastAPI app (port 8100) with one static page. Demo data is one constructed solution with three invented flags; real Segment Extraction runs on it through a fixture reader. Banners: "Demo data", "Fake model", "Cached result", and "Unverified: tests were not executed" beside every probe. The page inserts all text with `textContent`.
- `PROBING_DEMO_PROVIDER=fake|gemini` (default `fake`, works offline). With `gemini`, `PROBING_DEMO_MODE=cache` (default) serves only saved answers from `demo/cache/` and cannot reach the network; `live` calls the provider through the same pacing and daily counter.
- The canned fake probes were checked by hand once; they are as unverified as any other probe.

**Not built**: the reviewer (`PROBING_REVIEWER_ENABLED`), ladders generated for more than the tags the user clicks, any runner or verification.

## Reviewer, stricter probe rules, Docker spike (2026-10-21)

**Decisions**
- Mapper prompt and cue strengths are frozen for the report; nothing was tuned on the test split (50 records, hit@1 0.78 lenient / 0.74 strict, rule-only 0.52 / 0.40). Of 6 doubtful test records, 3 ignored the strong recursion cue C19 where recursion was incidental; noted, not changed.
- The 10 and 20 minute probe prompts gained a rule (unique, exactly computable answer; no approximation problems; every case obeys every stated constraint), after a recorded 10-minute probe passed the static checks but had an expected value its own reference contradicted. It is kept in `demo/examples/flawed_10min_wrong_expected_value.json` as a documented example. The 5 minute prompt is unchanged. Provenance `prompt_version` is `probe_generate_v1` (5 min) and `probe_generate_v1.1` (10 and 20 min).
- Difficulty reviewer (`probes/reviewer.py`): `gemma-4-31b-it`, advisory only, behind `PROBING_REVIEWER_ENABLED`. A reviewer from the same provider as the generator needs `PROBING_ALLOW_SAME_FAMILY_REVIEWER=true` and is labelled "weaker check". It reads the probe, reasons, and returns estimated minutes, a difficulty verdict, concerns and suspect cases; it never changes approval, verification or static checks. It reads plain-text JSON (Gemma's JSON mode errored once) with a 150 s timeout. On the known-flawed probe it named the NP-hard problem and the wrong case; it raised nothing on the three clean probes.
- Docker spike (nothing built): Docker 29.8.2 works; container start plus a trivial run takes about 0.4 s. With `--network none --memory 128m --cpus 1 --pids-limit 64 --read-only --cap-drop ALL` a probe's reference ran against its six cases in under 1 ms (about 14 MB), outbound network was blocked, and a 300 MB allocation was killed (exit 137). Finding: killing the `docker run` client does not stop the container; a runner must name its containers and `docker kill` them on timeout.

## Groq client and reviewer chain (2026-10-21)

**Decisions**
- `ai/groq.py`: Groq over its OpenAI-compatible chat endpoint, `httpx` only, key from `GROQ_API_KEY` in an `Authorization` header (never logged). It shares the gateway, per-minute limiter, retry and daily counter with Gemini. Model ids were listed through the API first: `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` both exist.
- Limits: the console shows 30 RPM, 1K RPD, 8K TPM, 200K TPD per model. The code paces at 20 RPM, 6K TPM, 900 RPD and 180K TPD. The daily counter now also counts tokens per model per day (reasoning tokens included, since `completion_tokens` contains them) and refuses a request whose estimate would pass the daily limit. A 429 that names a per-day limit, or a Retry-After over 120 s, is treated as quota gone and is not retried.
- Groq's day is counted from midnight UTC. This is an assumption: the console does not state the reset time. Gemini's counter still uses midnight Pacific.
- Reasoning is `low` for both Groq models and the reviewer's output is capped at 1500 tokens, because reasoning tokens count toward the limits.
- Reviewer order: `openai/gpt-oss-120b`, then `qwen/qwen3.8-27b`, then `gemma-4-31b-it`. A model is passed over when its call fails or its output is invalid after one retry; the reason is stored in the review's provenance (`skipped`). Gemma is used only when `PROBING_ALLOW_SAME_FAMILY_REVIEWER=true`; it is the same provider as the Gemini generator, so its reviews are labelled "weaker check: same provider as the generator". If no model answers there is no review, never an error on the page.
- The demo page shows which model answered, with its label, and any models passed over.
- The reviewer sees only generated probes (`origin="fixture"`), never candidate code. Its reviews are advisory and change nothing else on a probe.
- The three demo reviews were re-recorded with `openai/gpt-oss-120b` (estimates 7, 10 and 15 minutes, all "about right", no wrong cases flagged), and the old Gemma recordings were deleted. The flawed example was also reviewed once and **was not caught**: the model gave only generic concerns (k = 0, negative durations) and called it about right. This is one example, not a measured result; it is the case a reference check is for.

**Environment variables** (names only; none is read from `.env.example`, which is unchanged)
- `GROQ_API_KEY`: Groq key, required only for live reviewer calls.
- `PROBING_REVIEWER_ENABLED`: `true` to switch the reviewer on (default off).
- `PROBING_ALLOW_SAME_FAMILY_REVIEWER`: `true` to allow Gemma as the last resort (default off).
- `PROBING_REVIEWER_MODEL`: replaces the first Groq model in the chain.
- `PROBING_DEMO_PROVIDER` (`fake` or `gemini`) and `PROBING_DEMO_MODE` (`cache` or `live`): as before. In `cache` mode the recorded Groq reviews are served with no network and no key.

## DockerRunner and the reference check (2026-10-21)

**Decisions**
- `probes/docker_runner.py`: `DockerRunner` runs one throwaway container per check: `python:3.12-slim`, `--network none`, `--memory 128m` (no swap), `--cpus 1`, `--pids-limit 64`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, an 8 MB `/tmp`. The code and cases go in on stdin to a small harness that runs each case under a per-case alarm (default 2 s) and ignores anything the code prints. At most two containers run at once.
- Every container is named (`codetrace-probe-<random>`). If a run does not end cleanly (wall-clock timeout, an exception, Ctrl+C) the runner calls `docker kill` and `docker rm -f` on that name, because killing the `docker run` client does not stop the container. A hang that swallows the alarm (`except BaseException`) is covered by this path; 7 live tests confirm no container is left behind.
- It is off by default. `PROBING_RUNNER=docker` turns it on. If Docker is not installed, not running, or the image is not already pulled (it never pulls), `make_runner` returns `NullRunner`, logs a warning, and probes stay unverified with no error on the page. One-time setup: `docker pull python:3.12-slim`.
- Only generated reference solutions are run. `verify.reference_check` passes `probe.reference_solution` and nothing else to the runner; a probe whose static checks failed is not run.
- New statuses: `reference_checked` (the reference reproduces every stated expected value) and `reference_failed` (a mismatch, an error, or a limit hit). Not `cross_checked`, which needs an independent second solution; both the reference and the expected values come from the same model. The badge reads "Reference-checked: not independently verified." and the old `verified` / `failed` values are unused. The `oracle` slot stays empty. A reference check never changes approval, and approval never changes verification.
- Comparison follows the probe's checker: `exact` (a bool is not a number; 3 equals 3.0; tuples equal lists), `unordered` (multiset), `float_tolerance` (relative 1e-6).
- Results are recorded in `demo/cache/reference_checks/`, keyed by the reference code, function, cases, checker, image and harness. The demo reads them with no Docker (shown as "recorded result"); a live run in `gemini` mode records new ones. The 3 demo-segment probes were checked: all `reference_checked`, 6 of 6 cases each, which matches the hand check from earlier.
- The flawed 10-minute example is shown on the demo page. Its reference returns 5 for case c2 where the model expected 6, so it is `reference_failed`. Run live in Docker in under a second. The reviewer model (gpt-oss-120b) did not catch the same flaw.
- Approval of a `reference_failed` probe is **blocked** (owner decision, 2026-10-21): the repository refuses it, the API answers 400 with the reason, and the page disables Approve and says why. Rejecting stays allowed. If a failed check arrives for a probe that was already approved, the approval is withdrawn and the probe goes back to pending.
- Tests: the normal suite is offline (a scripted runner and a patched `subprocess.run`). `tests/probing/test_docker_live.py` really starts containers and runs only with `PROBING_DOCKER_TESTS=1`.

**Environment variables**
- `PROBING_RUNNER`: `docker` to turn the runner on (default off).
- `PROBING_DOCKER_TESTS`: `1` to run the container tests.
