"""Recorded Groq reviews served through the demo service. Offline: nothing is sent to a network.

Kept apart from test_groq_client.py because it needs backend/app/probing/demo/ and its recorded answers.
"""

from backend.app.probing.ai.config import AISettings
from tests.probing.test_groq_client import run


# --- the recorded demo reviews ----------------------------------------------------------------------


def test_recorded_reviews_are_served_from_the_cache_with_no_network(tmp_path, monkeypatch):
    from dataclasses import replace

    from backend.app.probing.demo.service import DemoService

    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = replace(AISettings(), reviewer_enabled=True)
    service = DemoService(provider="gemini", mode="cache", state_dir=tmp_path, settings=settings)

    async def go():
        await service.load()
        segment = next(s for s in service._segments.values() if s.location.start_line == 9)
        await service.map_segment(segment.segment_id)
        ladder = await service.generate_ladder(segment.segment_id, "heap_priority_queue")
        return [await service.review_probe(p.probe_id) for p in ladder.probes]

    reviewed = run(go())
    assert service.reviewer_models == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "gemma-4-31b-it"]
    for probe in reviewed:
        review = probe.review
        assert review.provenance.model == "openai/gpt-oss-120b" and review.provenance.provider == "groq"
        assert review.weaker_check is False and review.provenance.skipped == () and review.advisory_only
        assert probe.approval == "pending"
