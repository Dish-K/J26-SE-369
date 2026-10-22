"""What the demo page does: map a segment, generate a ladder, approve a probe.

Model choice comes from the environment:

- ``PROBING_DEMO_PROVIDER`` = ``fake`` (default, works offline) or ``gemini``.
- ``PROBING_DEMO_MODE`` = ``cache`` (default) serves only answers already saved
  in ``demo/cache/``; ``live`` may call the provider (the gateway still caches,
  paces and counts every call). The mode only matters for ``gemini``.

Everything the page shows is data to be displayed: the browser code puts text in
with ``textContent`` and never as HTML.
"""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.app.probing.ai.cache import ResponseCache
from backend.app.probing.ai.config import (
    GEMINI_PRIMARY_MODEL,
    GEMINI_REVIEWER_MODEL,
    GROQ_REASONING,
    GROQ_REVIEWER_MODELS,
    AISettings,
)
from backend.app.probing.ai.errors import LLMError
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.gemini import GeminiClient, MissingApiKey
from backend.app.probing.ai.groq import GroqClient
from backend.app.probing.ai.quota import DailyCounter, utc_date
from backend.app.probing.demo.data import DEMO_TASK_ID, FixtureReader, flag_confidences
from backend.app.probing.knowledge.mapper import KnowledgeMapper
from backend.app.probing.knowledge.schemas import MapperResult
from backend.app.probing.probes.fake_probes import FakeDemoClient, FakeReviewerClient
from backend.app.probing.probes.docker_runner import RunnerChoice, make_runner
from backend.app.probing.probes.generator import ProbeGenerator
from backend.app.probing.probes.reference_cache import ReferenceCheckCache
from backend.app.probing.probes.reviewer import ProbeReviewer
from backend.app.probing.probes.schema import ProbePackage
from backend.app.probing.probes.verify import reference_check
from backend.app.probing.schemas import Segment
from backend.app.probing.service import extract_segments
from backend.app.probing.store.repository import InMemoryProbeRepository

DEMO_DIR = Path(__file__).resolve().parent
ORIGIN = "fixture"
REVIEWER_TIMEOUT_SECONDS = 150.0
FLAWED_EXAMPLE = DEMO_DIR / "examples" / "flawed_10min_wrong_expected_value.json"


class CacheOnlyClient:
    """Wraps a real client so cache mode can never make a network call."""

    hosted = True

    def __init__(self, provider: str = "gemini", family: str = "google") -> None:
        self.provider, self.family = provider, family

    async def complete(self, *args, **kwargs):
        raise LLMError("cache mode: there is no saved answer for this request")


class ReviewUnavailable(RuntimeError):
    """The advisory review was not produced; the message says why (fixed text)."""


class DemoService:
    def __init__(self, *, provider: str = "fake", mode: str = "cache", state_dir: Path | None = None,
                 cache_dir: Path | None = None, settings: AISettings | None = None,
                 runner: RunnerChoice | None = None) -> None:
        self.provider, self.mode = provider, mode
        self.runner_choice = runner or make_runner()  # PROBING_RUNNER=docker, else no runner
        self._reference_cache = ReferenceCheckCache((cache_dir or DEMO_DIR / "cache") / "reference_checks")
        settings = settings or AISettings.from_env()
        reviewer_enabled, allow_same = settings.reviewer_enabled, settings.allow_same_family_reviewer
        daily = None
        cache = None
        reviewer_settings = replace(settings, max_rpm=25, max_tpm=14_000)  # Gemma 4 free tier: 30 RPM, 16K TPM
        chain: list[tuple[LLMGateway, str]] = []
        if provider == "fake":
            client, reviewer_client = FakeDemoClient(), FakeReviewerClient()
            settings = AISettings(max_rpm=100_000, max_tpm=10**9, concurrency=settings.concurrency)
            model = "dummy-v0"
            chain.append((LLMGateway(reviewer_client, settings), "dummy-v0"))
        elif provider == "gemini":
            settings = settings.with_gemini_free_tier()
            model = settings.mapper_model or GEMINI_PRIMARY_MODEL
            cache = ResponseCache(cache_dir or DEMO_DIR / "cache")
            # Reviewer order: two Groq models, then Gemma (same provider as the generator, weaker check).
            groq_settings = settings.with_groq_free_tier()
            groq_models = (settings.reviewer_model, *GROQ_REVIEWER_MODELS[1:]) if settings.reviewer_model else GROQ_REVIEWER_MODELS
            groq_daily = None
            if mode == "live":
                client = GeminiClient()
                reviewer_client = GeminiClient(timeout=REVIEWER_TIMEOUT_SECONDS)  # the 31B model can take a while
                daily = DailyCounter(settings.max_rpd, settings.cache_dir / "daily_requests.json")
                try:
                    groq_client = GroqClient(reasoning=GROQ_REASONING)
                    groq_daily = DailyCounter(
                        groq_settings.max_rpd,
                        settings.cache_dir / "groq_daily.json",
                        token_limit=groq_settings.max_tpd,
                        day_of=utc_date,
                    )
                except MissingApiKey:
                    groq_client = None  # no Groq key: the chain starts at Gemma
            else:
                client = reviewer_client = CacheOnlyClient()
                groq_client = CacheOnlyClient("groq", "groq")
            if groq_client is not None:
                groq_gateway = LLMGateway(groq_client, groq_settings, cache=cache, daily=groq_daily)
                chain.extend((groq_gateway, name) for name in groq_models)
            gemma = settings.reviewer_model if settings.reviewer_model and not groq_client else GEMINI_REVIEWER_MODEL
            chain.append(
                (LLMGateway(reviewer_client, replace(reviewer_settings, fallback_model=None), cache=cache, daily=daily), gemma)
            )
        else:
            raise ValueError(f"PROBING_DEMO_PROVIDER must be fake or gemini, got {provider!r}")
        self.model = model
        self.generator_model = settings.generator_model or model
        self._gateway = LLMGateway(client, settings, cache=cache, daily=daily)
        self._mapper = KnowledgeMapper(self._gateway, model=model)
        self._generator = ProbeGenerator(self._gateway, model=self.generator_model)
        self.reviewer_models = [name for _, name in chain]
        self._reviewer = ProbeReviewer(chain=chain, enabled=reviewer_enabled, allow_same_family=allow_same)
        self.repo = InMemoryProbeRepository(state_dir / "events.jsonl" if state_dir else None)
        self._segments: dict[UUID, Segment] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "DemoService":
        AISettings.from_env()  # loads .env into the environment
        return cls(
            provider=os.environ.get("PROBING_DEMO_PROVIDER", "fake").strip().lower(),
            mode=os.environ.get("PROBING_DEMO_MODE", "cache").strip().lower(),
            state_dir=DEMO_DIR / "state",
        )

    async def load(self) -> None:
        result = await extract_segments(DEMO_TASK_ID, FixtureReader())
        self._segments = {s.segment_id: s for s in result.segments}

    # --- what the page reads -------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        confidences = flag_confidences()
        return {
            "banners": self.banners(),
            "provider": self.provider,
            "mode": self.mode,
            "model": self.model,
            "reviewer": {
                "enabled": self._reviewer.enabled,
                "models": self.reviewer_models,
                "note": "advisory only; each review names the model that answered; a reviewer from "
                "the same provider as the generator is a weaker check",
            },
            "runner": {"mode": self.runner_choice.mode, "note": self.runner_choice.note},
            "segments": [
                {
                    "segment_id": str(s.segment_id),
                    "start_line": s.location.start_line,
                    "end_line": s.location.end_line,
                    "flag_confidence": confidences.get(s.parent_flag_id),
                    "code": s.code,
                    "entry_function": s.entry_function.name if s.entry_function else None,
                    "mapped": self.repo.get_mapper_result(s.segment_id) is not None,
                }
                for s in sorted(self._segments.values(), key=lambda s: s.location.start_line)
            ],
        }

    def banners(self) -> list[str]:
        banners = ["Demo data"]
        if self.provider == "fake":
            banners.append("Fake model")
        if self.provider == "gemini" and self.mode == "cache":
            banners.append("Cached result")
        return banners

    def detail(self, segment_id: UUID) -> dict[str, Any]:
        self._segment(segment_id)
        mapped = self.repo.get_mapper_result(segment_id)
        ladders = {}
        if mapped:
            for tag in mapped.tags:
                ladder = self.repo.get_ladder(segment_id, tag.tag_id)
                if ladder:
                    ladders[tag.tag_id] = ladder.model_dump(mode="json")
        return {
            "mapper": mapped.model_dump(mode="json") if mapped else None,
            "ladders": ladders,
            "unverified_note": "Unverified: tests were not executed",
        }

    # --- what the page does ----------------------------------------------------------------

    async def map_segment(self, segment_id: UUID) -> MapperResult:
        segment = self._segment(segment_id)
        async with self._lock:
            existing = self.repo.get_mapper_result(segment_id)
            if existing is not None:
                return existing
            result = await self._mapper.map_segment(segment, data_origin=ORIGIN)
            self.repo.save_mapper_result(result)
            return result

    async def generate_ladder(self, segment_id: UUID, tag_id: str):
        mapped = self.repo.get_mapper_result(segment_id)
        if mapped is None:
            raise LookupError("map the segment first")
        if tag_id not in {t.tag_id for t in mapped.tags}:
            raise LookupError("that tag was not chosen for this segment")
        async with self._lock:
            existing = self.repo.get_ladder(segment_id, tag_id)
            if existing is not None:
                return existing
            ladder = await self._generator.generate_ladder(tag_id)
            self.repo.save_ladder(segment_id, ladder)
            return ladder

    async def review_probe(self, probe_id: UUID) -> ProbePackage:
        probe = self.repo.get_probe(probe_id)  # ProbeNotFound if unknown
        async with self._lock:
            if probe.review is not None:
                return probe
            review, reason = await self._reviewer.review(probe)
            if review is None:
                raise ReviewUnavailable(reason)
            return self.repo.attach_review(probe_id, review)

    async def check_reference(self, probe_id: UUID) -> ProbePackage:
        """Run (or read the recorded result of) the reference check. Unchanged if neither is possible."""
        probe = self.repo.get_probe(probe_id)  # ProbeNotFound if unknown
        async with self._lock:
            if probe.reference_check is not None or not probe.static_checks.passed:
                return probe
            check = await self._reference_result(probe)
            return probe if check is None else self.repo.attach_reference_check(probe_id, check)

    async def _reference_result(self, probe: ProbePackage, *, record: bool | None = None):
        """A recorded result, else a live run if a runner exists, else ``None``. Fake probes are not recorded."""
        recorded = self._reference_cache.get(probe)
        if recorded is not None:
            return recorded.model_copy(update={"served": "recorded"})
        if self.runner_choice.mode != "docker":
            return None  # no runner and no record: the probe stays unverified, quietly
        check = await reference_check(probe, self.runner_choice.runner)
        if record if record is not None else self.provider == "gemini":
            self._reference_cache.put(probe, check)
        return check

    async def flawed_example(self) -> dict[str, Any]:
        """The documented flawed probe with its reference-check result (not stored with the other probes)."""
        raw = json.loads(FLAWED_EXAMPLE.read_text(encoding="utf-8"))
        probe = ProbePackage.model_validate(raw["probe"])
        check = await self._reference_result(probe, record=True) if probe.static_checks.passed else None
        if check is not None:
            probe = probe.model_copy(update={"reference_check": check, "verification": check.status})
        return {"note": raw.get("_note", ""), "probe": probe.model_dump(mode="json")}

    def _segment(self, segment_id: UUID) -> Segment:
        try:
            return self._segments[segment_id]
        except KeyError:
            raise LookupError("unknown segment") from None
