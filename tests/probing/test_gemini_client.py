"""Gemini client, daily quota and fallback. Offline: HTTP is mocked.

The one real-API test runs only when PROBING_LIVE_TESTS=1 is set.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

import httpx
import pytest

from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.errors import LLMError, ProviderHTTPError, QuotaUnavailable, RateLimited
from backend.app.probing.ai.fake import FakeClient
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.gemini import GeminiClient, MissingApiKey, read_key
from backend.app.probing.ai.quota import DailyCounter, DailyQuotaExhausted, pacific_date

SECRET = "AIza-test-secret-key-0123456789"


def run(coro):
    return asyncio.run(coro)


async def no_sleep(_seconds):
    return None


def ok_body(text="hi", prompt=10, out=5, thoughts=0):
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": out}
    if thoughts:
        usage["thoughtsTokenCount"] = thoughts
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}], "usageMetadata": usage}


def client_with(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiClient(api_key=SECRET, http=http)


def ask(client, **kw):
    args = dict(model="m", temperature=0.0, max_tokens=50)
    args.update(kw)
    return run(client.complete("sys", "user", **args))


# --- the client ----------------------------------------------------------------------


def test_request_shape_and_key_only_in_header():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    result = ask(client_with(handler), json_schema={"type": "object"})
    assert seen["key"] == SECRET and SECRET not in seen["url"]
    assert seen["url"].endswith("/models/m:generateContent")
    config = seen["body"]["generationConfig"]
    assert config["responseMimeType"] == "application/json" and config["responseJsonSchema"] == {"type": "object"}
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert (result.text, result.tokens_in, result.tokens_out, result.provider) == ("hi", 10, 5, "gemini")


def test_thinking_tokens_are_counted_as_output():
    result = ask(client_with(lambda r: httpx.Response(200, json=ok_body(out=5, thoughts=40))))
    assert result.tokens_out == 45


def test_thought_parts_are_not_returned_as_text():
    body = ok_body()
    body["candidates"][0]["content"]["parts"] = [{"text": "private", "thought": True}, {"text": "answer"}]
    assert ask(client_with(lambda r: httpx.Response(200, json=body))).text == "answer"


def test_blocked_or_empty_answer_raises():
    with pytest.raises(LLMError):
        ask(client_with(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})))


def test_429_carries_retry_after_and_never_the_key():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "7"})

    with pytest.raises(ProviderHTTPError) as caught:
        ask(client_with(handler))
    assert caught.value.status == 429 and caught.value.retry_after == 7.0 and not caught.value.quota_zero
    assert SECRET not in str(caught.value)


def test_limit_zero_is_reported_clearly():
    message = "Quota exceeded for metric ... limit: 0, model: gemini-3.1-pro"

    def handler(request):
        return httpx.Response(429, json={"error": {"message": message}})

    with pytest.raises(ProviderHTTPError) as caught:
        ask(client_with(handler))
    assert caught.value.quota_zero and "quota for this model is 0" in str(caught.value)


def test_network_errors_become_retryable_503():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(ProviderHTTPError) as caught:
        ask(client_with(handler))
    assert caught.value.status == 503


def test_repr_hides_the_key():
    assert SECRET not in repr(client_with(lambda r: httpx.Response(200, json=ok_body())))


def test_missing_key_is_a_clear_error_that_names_only_the_variable():
    with pytest.raises(MissingApiKey, match="GEMINI_API_KEY"):
        read_key({})
    assert read_key({"GEMINI_API_KEY": f"  {SECRET} "}) == SECRET


# --- gateway: limit zero, fallback, daily counter ----------------------------------------


def gateway_for(handler, *, settings=None, daily=None):
    return LLMGateway(client_with(handler), settings or AISettings(max_rpm=1000, max_tpm=10**9), daily=daily, sleep=no_sleep)


def complete(gateway, model="primary"):
    return run(
        gateway.complete(system="s", user="u", model=model, temperature=0.0, max_tokens=10, origin="fixture")
    )


def test_limit_zero_is_not_retried_and_falls_back_to_the_second_model():
    calls = []

    def handler(request):
        model = request.url.path.split("/models/")[1].split(":")[0]
        calls.append(model)
        if model == "primary":
            return httpx.Response(429, json={"error": {"message": "limit: 0"}})
        return httpx.Response(200, json=ok_body("from fallback"))

    gateway = gateway_for(handler, settings=AISettings(max_rpm=1000, max_tpm=10**9, fallback_model="second"))
    result = complete(gateway)
    assert calls == ["primary", "second"]  # no retries on a hopeless 429
    assert (result.text, result.model) == ("from fallback", "second")


def test_without_a_fallback_limit_zero_surfaces_as_quota_unavailable():
    gateway = gateway_for(lambda r: httpx.Response(429, json={"error": {"message": "limit: 0"}}))
    with pytest.raises(QuotaUnavailable, match="quota for this model is 0"):
        complete(gateway)


def test_daily_limit_switches_to_the_fallback_then_raises():
    calls = []

    def handler(request):
        calls.append(request.url.path.split("/models/")[1].split(":")[0])
        return httpx.Response(200, json=ok_body())

    settings = AISettings(max_rpm=1000, max_tpm=10**9, fallback_model="second")
    gateway = gateway_for(handler, settings=settings, daily=DailyCounter(1))
    complete(gateway)
    complete(gateway)
    assert calls == ["primary", "second"]
    with pytest.raises(DailyQuotaExhausted):
        complete(gateway)
    assert len(calls) == 2  # nothing was sent once both were used up


def test_each_retry_counts_against_the_day():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(500, json={"error": {"message": "oops"}})

    counter = DailyCounter(100)
    with pytest.raises(RateLimited):
        complete(gateway_for(handler, daily=counter))
    assert counter.used("primary") == len(attempts) == 5


# --- the daily counter ------------------------------------------------------------------------


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_pacific_date_uses_daylight_saving_offsets():
    # PDT (UTC-7): 06:59 UTC is still the previous day, 07:00 UTC is midnight Pacific.
    assert pacific_date(utc(2026, 10, 21, 6, 59)).isoformat() == "2026-10-20"
    assert pacific_date(utc(2026, 10, 21, 7, 0)).isoformat() == "2026-10-21"
    # PST (UTC-8) after clocks change on 1 November 2026.
    assert pacific_date(utc(2026, 11, 2, 7, 59)).isoformat() == "2026-11-01"
    assert pacific_date(utc(2026, 11, 2, 8, 0)).isoformat() == "2026-11-02"
    # Before the spring change (8 March 2026, 10:00 UTC) the offset is still -8.
    assert pacific_date(utc(2026, 3, 8, 7, 59)).isoformat() == "2026-03-07"
    assert pacific_date(utc(2026, 3, 8, 8, 0)).isoformat() == "2026-03-08"
    assert pacific_date(utc(2026, 3, 9, 6, 59)).isoformat() == "2026-03-08"  # now -7
    assert pacific_date(utc(2026, 3, 9, 7, 0)).isoformat() == "2026-03-09"


def test_counter_survives_restart_and_resets_at_midnight_pacific(tmp_path):
    path = tmp_path / "daily.json"
    now = [utc(2026, 10, 21, 12, 0)]
    first = DailyCounter(2, path, clock=lambda: now[0])
    first.take("m")
    first.take("m")
    restarted = DailyCounter(2, path, clock=lambda: now[0])
    assert restarted.used("m") == 2
    with pytest.raises(DailyQuotaExhausted):
        restarted.take("m")
    now[0] = utc(2026, 10, 22, 7, 0)  # midnight Pacific
    restarted.take("m")
    assert restarted.used("m") == 1


def test_counter_file_holds_only_a_date_and_counts(tmp_path):
    path = tmp_path / "daily.json"
    DailyCounter(5, path).take("gemini-x")
    assert set(json.loads(path.read_text())) == {"day", "counts"}


def test_damaged_counter_file_starts_fresh(tmp_path):
    path = tmp_path / "daily.json"
    path.write_text("{not json", encoding="utf-8")
    assert DailyCounter(5, path).used("m") == 0


def test_free_tier_helper_respects_explicit_env_values():
    base = AISettings()
    tuned = base.with_gemini_free_tier({})
    assert (tuned.max_rpm, tuned.max_tpm, tuned.max_rpd) == (12, 200_000, 450)
    assert tuned.fallback_model == "gemini-3.1-flash-lite"
    kept = base.with_gemini_free_tier({"PROBING_LLM_MAX_RPM": "3"})
    assert kept.max_rpm == base.max_rpm  # left for from_env to have set it


def test_fallback_is_not_used_for_a_fake_client_without_one():
    gateway = LLMGateway(FakeClient(), AISettings(), sleep=no_sleep)
    assert gateway.settings.fallback_model is None


# --- opt-in real API ------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("PROBING_LIVE_TESTS") != "1", reason="set PROBING_LIVE_TESTS=1 to call the real API")
def test_live_smoke_one_tiny_call():
    client = GeminiClient()
    result = run(
        client.complete("Reply with one word.", "Say ok.", model="gemini-3.5-flash-lite", temperature=0.0, max_tokens=200)
    )
    assert result.text.strip() and result.tokens_in > 0
