"""The demo's live mode: the banner, and plain errors instead of a silent fake. Offline: no client is ever called."""

import pytest
from fastapi.testclient import TestClient

from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.gemini import MissingApiKey
from backend.app.probing.demo import service as demo_service
from backend.app.probing.demo.app import create_app
from backend.app.probing.demo.service import DemoService


class StandIn:
    """Takes the place of GeminiClient or GroqClient; building it never needs a key and calling it is a bug."""

    hosted = True
    provider, family = "stand-in", "stand-in"

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def complete(self, *args, **kwargs):
        raise AssertionError("a test must not call a provider")


def no_key(message):
    def build(*args, **kwargs):
        raise MissingApiKey(message)

    return build


def live(tmp_path, **kwargs):
    return DemoService(provider="live", state_dir=tmp_path, cache_dir=tmp_path / "cache", settings=AISettings(reviewer_enabled=True), **kwargs)


def test_live_is_a_value_of_the_provider_switch_and_the_banner_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_service, "GeminiClient", StandIn)
    monkeypatch.setattr(demo_service, "GroqClient", StandIn)
    service = live(tmp_path)
    state = service.state()
    assert (state["provider"], state["mode"]) == ("gemini", "live")
    assert state["banners"] == ["Demo data", "Live model"] and state["problems"] == []


def test_the_other_banners_are_unchanged(tmp_path):
    assert DemoService(provider="fake", state_dir=tmp_path, settings=AISettings()).banners() == ["Demo data", "Fake model"]
    cached = DemoService(provider="gemini", mode="cache", state_dir=tmp_path, settings=AISettings())
    assert cached.banners() == ["Demo data", "Cached result"]


def test_a_missing_gemini_key_starts_the_page_and_is_reported_on_it(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_service, "GeminiClient", no_key("GEMINI_API_KEY is not set; add it to .env"))
    monkeypatch.setattr(demo_service, "GroqClient", StandIn)
    with TestClient(create_app(live(tmp_path))) as client:
        problems = client.get("/api/state").json()["problems"]
        assert len(problems) == 1 and "GEMINI_API_KEY is not set" in problems[0]
        segment = client.get("/api/state").json()["segments"][0]["segment_id"]
        mapped = client.post(f"/api/segments/{segment}/map").json()
        assert mapped["overall_status"] and any(note.startswith("rule_only") for note in mapped["notes"])
        assert mapped["provenance"]["provider"] != "fake"
        ladder = client.post(f"/api/segments/{segment}/ladder/{mapped['tags'][0]['tag_id']}").json()
        assert ladder["probes"] == [] and len(ladder["notes"]) == 3
        assert all("not generated: MissingApiKey" in note for note in ladder["notes"])


def test_a_missing_groq_key_is_reported_and_the_reviewer_starts_at_gemma(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_service, "GeminiClient", StandIn)
    monkeypatch.setattr(demo_service, "GroqClient", no_key("GROQ_API_KEY is not set; add it to .env"))
    service = live(tmp_path)
    assert any("GROQ_API_KEY is not set" in p for p in service.state()["problems"])
    assert service.reviewer_models == ["gemma-4-31b-it"]


def test_the_fake_provider_has_no_problems(tmp_path):
    assert DemoService(provider="fake", state_dir=tmp_path, settings=AISettings()).state()["problems"] == []


def test_a_wrong_provider_value_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="PROBING_DEMO_PROVIDER"):
        DemoService(provider="nonsense", state_dir=tmp_path, settings=AISettings())
