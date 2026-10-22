"""Groq client, token-per-day counting and the reviewer chain. Offline: HTTP is mocked.

The one real-API test runs only when PROBING_LIVE_TESTS=1 is set.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

import httpx
import pytest

from backend.app.probing.ai.config import GROQ_FREE_TIER, GROQ_REASONING, GROQ_REVIEWER_MODELS, AISettings
from backend.app.probing.ai.errors import LLMError, ProviderHTTPError, QuotaUnavailable, RateLimited
from backend.app.probing.ai.fake import FakeClient
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.gemini import MissingApiKey
from backend.app.probing.ai.groq import GroqClient, read_key
from backend.app.probing.ai.quota import DailyCounter, DailyQuotaExhausted, utc_date
from backend.app.probing.probes.fake_probes import FakeDemoClient
from backend.app.probing.probes.generator import ProbeGenerator
from backend.app.probing.probes.reviewer import ProbeReviewer

SECRET = "gsk_test-secret-key-0123456789"
BIG = AISettings(max_rpm=100_000, max_tpm=10**9)


def run(coro):
    return asyncio.run(coro)


async def no_sleep(_seconds):
    return None


def ok_body(text="hi", prompt=10, out=5):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out},
    }


def client_with(handler, **kw):
    return GroqClient(api_key=SECRET, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


def ask(client, **kw):
    args = dict(model="m", temperature=0.0, max_tokens=50)
    args.update(kw)
    return run(client.complete("sys", "user", **args))


# --- the client -------------------------------------------------------------------------------------


def test_request_shape_key_only_in_header_and_reasoning_only_for_its_model():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    client = client_with(handler, reasoning={"m": {"reasoning_effort": "low"}})
    result = ask(client)
    assert seen["auth"] == f"Bearer {SECRET}" and SECRET not in seen["url"]
    assert seen["url"].endswith("/openai/v1/chat/completions")
    body = seen["body"]
    assert body["model"] == "m" and body["max_completion_tokens"] == 50 and body["reasoning_effort"] == "low"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert (result.text, result.tokens_in, result.tokens_out, result.provider, result.family) == (
        "hi", 10, 5, "groq", "groq")

    ask(client, model="other")
    assert "reasoning_effort" not in seen["body"]
    assert SECRET not in repr(client)


def test_configured_reasoning_is_low_for_both_groq_models():
    assert GROQ_REVIEWER_MODELS == ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")
    assert all(GROQ_REASONING[m] == {"reasoning_effort": "low"} for m in GROQ_REVIEWER_MODELS)


def test_think_blocks_are_removed_from_the_answer():
    result = ask(client_with(lambda r: httpx.Response(200, json=ok_body("<think>hmm</think>\n{\"a\": 1}"))))
    assert result.text == '{"a": 1}'


@pytest.mark.parametrize("body", [{"choices": []}, {}, ok_body(""), ok_body("<think>only thoughts</think>")])
def test_empty_or_missing_answer_is_an_llm_error(body):
    with pytest.raises(LLMError):
        ask(client_with(lambda r: httpx.Response(200, json=body)))


def test_non_json_reply_is_an_llm_error():
    with pytest.raises(LLMError):
        ask(client_with(lambda r: httpx.Response(200, content=b"<html>")))


def test_rate_limit_carries_retry_after_and_never_the_key():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "7"})

    with pytest.raises(ProviderHTTPError) as caught:
        ask(client_with(handler))
    assert caught.value.status == 429 and caught.value.retry_after == 7.0 and not caught.value.quota_zero
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize(
    "message, headers",
    [("Rate limit reached ... on tokens per day (TPD): Limit 200000", {"retry-after": "30"}),
     ("Rate limit reached", {"retry-after": "5000"})],
)
def test_a_daily_limit_is_not_retried(message, headers):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": message}}, headers=headers)

    gateway = LLMGateway(client_with(handler), BIG, sleep=no_sleep)
    with pytest.raises(QuotaUnavailable):
        run(gateway.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=10, origin="fixture"))
    assert len(calls) == 1


def test_429_is_retried_then_succeeds():
    replies = iter([httpx.Response(429, json={}, headers={"retry-after": "1"}), httpx.Response(200, json=ok_body())])
    gateway = LLMGateway(client_with(lambda r: next(replies)), BIG, sleep=no_sleep)
    result = run(gateway.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=10, origin="fixture"))
    assert result.text == "hi"


def test_server_error_is_retried_then_rate_limited():
    gateway = LLMGateway(client_with(lambda r: httpx.Response(503, json={})), BIG, sleep=no_sleep)
    with pytest.raises(RateLimited):
        run(gateway.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=10, origin="fixture"))


def test_auth_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    with pytest.raises(ProviderHTTPError):
        ask(client_with(handler))
    assert len(calls) == 1


def test_network_error_becomes_a_retryable_503():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(ProviderHTTPError) as caught:
        ask(client_with(handler))
    assert caught.value.status == 503


def test_missing_key_is_reported_without_a_value():
    with pytest.raises(MissingApiKey) as caught:
        read_key({})
    assert "GROQ_API_KEY" in str(caught.value)
    assert read_key({"GROQ_API_KEY": f" {SECRET} "}) == SECRET


def test_groq_settings_are_the_safe_margins():
    settings = AISettings().with_groq_free_tier()
    assert (settings.max_rpm, settings.max_tpm, settings.max_rpd, settings.max_tpd) == (20, 6000, 900, 180_000)
    assert settings.fallback_model is None and GROQ_FREE_TIER["max_tpd"] == 180_000


# --- tokens per day ---------------------------------------------------------------------------------


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_daily_tokens_are_counted_blocked_and_reset_at_utc_midnight(tmp_path):
    path = tmp_path / "groq.json"
    now = [utc(2026, 10, 21, 23, 0)]
    counter = DailyCounter(10, path, clock=lambda: now[0], token_limit=1000, day_of=utc_date)
    counter.take("m", 400)
    counter.add_tokens("m", 900)
    assert counter.tokens_used("m") == 900
    with pytest.raises(DailyQuotaExhausted):
        counter.take("m", 200)  # 900 + 200 would pass 1000
    assert counter.used("m") == 1  # the refused request was not counted
    assert DailyCounter(10, path, clock=lambda: now[0], token_limit=1000, day_of=utc_date).tokens_used("m") == 900
    now[0] = utc(2026, 10, 22, 0, 0)
    counter.take("m", 200)
    assert counter.tokens_used("m") == 0


def test_tokens_are_not_tracked_without_a_limit(tmp_path):
    path = tmp_path / "d.json"
    counter = DailyCounter(10, path)
    counter.take("m", 10**9)
    counter.add_tokens("m", 5)
    assert counter.tokens_used("m") == 0 and set(json.loads(path.read_text())) == {"day", "counts"}


def test_gateway_records_actual_tokens_including_reasoning(tmp_path):
    counter = DailyCounter(10, token_limit=10_000, day_of=utc_date)
    client = client_with(lambda r: httpx.Response(200, json=ok_body(prompt=100, out=700)))
    gateway = LLMGateway(client, BIG, daily=counter, sleep=no_sleep)
    run(gateway.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=900, origin="fixture"))
    assert counter.tokens_used("m") == 800


def test_gateway_refuses_before_sending_when_the_daily_tokens_are_gone():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=ok_body())

    counter = DailyCounter(10, token_limit=500, day_of=utc_date)
    gateway = LLMGateway(client_with(handler), BIG, daily=counter, sleep=no_sleep)
    with pytest.raises(DailyQuotaExhausted):
        run(gateway.complete(system="s", user="u", model="m", temperature=0.0, max_tokens=900, origin="fixture"))
    assert calls == []


# --- the reviewer chain -------------------------------------------------------------------------------


def a_probe(level=10):
    ladder = run(ProbeGenerator(LLMGateway(FakeDemoClient(), BIG, sleep=no_sleep), model="dummy-v0").generate_ladder("hash_table"))
    return next(p for p in ladder.probes if p.target_minutes == level)


def good(minutes=9):
    return json.dumps({"estimated_minutes": minutes, "difficulty": "about_right", "concerns": [], "case_problems": []})


def step(script, family):
    client = FakeClient(script=script)
    client.family = family
    return LLMGateway(client, BIG, sleep=no_sleep), client


def reviewer(*steps, allow_same_family=False):
    chain = [(gateway, name) for (gateway, _client), name in steps]
    return ProbeReviewer(chain=chain, enabled=True, allow_same_family=allow_same_family)


def test_first_model_answers_and_is_not_a_weaker_check():
    first, c1 = step([good(9)], "groq")
    second, c2 = step([good(20)], "groq")
    result, _ = run(reviewer(((first, c1), "gpt"), ((second, c2), "qwen")).review(a_probe()))
    assert result.estimated_minutes == 9 and result.weaker_check is False
    assert result.provenance.skipped == () and c2.calls == []


def test_a_failed_call_passes_on_to_the_next_model_and_records_why():
    first, c1 = step([ProviderHTTPError(503, "down")] * 10, "groq")
    second, c2 = step([good(12)], "groq")
    result, _ = run(reviewer(((first, c1), "gpt"), ((second, c2), "qwen")).review(a_probe()))
    assert result.estimated_minutes == 12 and result.provenance.skipped == ("gpt: RateLimited",)


def test_invalid_output_after_one_retry_passes_on_to_the_next_model():
    first, c1 = step(["nope", "still nope"], "groq")
    second, c2 = step([good(11)], "groq")
    result, _ = run(reviewer(((first, c1), "gpt"), ((second, c2), "qwen")).review(a_probe()))
    assert len(c1.calls) == 2 and result.estimated_minutes == 11
    assert result.provenance.skipped == ("gpt: output invalid after one retry",)


def test_same_provider_model_is_last_resort_and_needs_the_flag():
    generator_family = a_probe().provenance.family
    groq1, c1 = step([ProviderHTTPError(503, "down")] * 10, "groq")
    groq2, c2 = step(["nope", "nope"], "groq")
    gemma, c3 = step([good(14)], generator_family)

    off = reviewer(((groq1, c1), "gpt"), ((groq2, c2), "qwen"), ((gemma, c3), "gemma"))
    result, reason = run(off.review(a_probe()))
    assert result is None and reason.startswith("not reviewed:") and "same provider" in reason and c3.calls == []

    groq1, c1 = step([ProviderHTTPError(503, "down")] * 10, "groq")
    groq2, c2 = step(["nope", "nope"], "groq")
    gemma, c3 = step([good(14)], generator_family)
    on = reviewer(((groq1, c1), "gpt"), ((groq2, c2), "qwen"), ((gemma, c3), "gemma"), allow_same_family=True)
    result, _ = run(on.review(a_probe()))
    assert result.weaker_check is True and "weaker check: same provider" in result.label
    assert [s.split(":")[0] for s in result.provenance.skipped] == ["gpt", "qwen"]


def test_groq_answer_is_never_labelled_weaker_even_after_a_skip():
    first, c1 = step(["nope", "nope"], "groq")
    second, c2 = step([good()], "groq")
    result, _ = run(reviewer(((first, c1), "gpt"), ((second, c2), "qwen")).review(a_probe()))
    assert result.weaker_check is False and "weaker" not in result.label


def test_every_model_failing_gives_no_review_and_each_reason():
    first, c1 = step([ProviderHTTPError(503, "down")] * 10, "groq")
    second, c2 = step(["nope", "nope"], "groq")
    result, reason = run(reviewer(((first, c1), "gpt"), ((second, c2), "qwen")).review(a_probe()))
    assert result is None and "gpt: RateLimited" in reason and "qwen: output invalid after one retry" in reason


def test_the_reviewer_prompt_holds_the_probe_and_no_candidate_code():
    first, c1 = step([good()], "groq")
    probe = a_probe()
    run(reviewer(((first, c1), "gpt")).review(probe))
    user = c1.calls[0]["user"]
    assert probe.question in user and probe.signature in user
    assert first.client.calls[0]["json_schema"] is None
    assert "GROQ_API_KEY" not in user and "segment" not in user.lower()
    assert first.settings.allow_candidate_code_to_hosted_llm is False


def test_a_chain_or_gateway_is_required():
    with pytest.raises(ValueError):
        ProbeReviewer(enabled=True)


# --- opt-in real API ----------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("PROBING_LIVE_TESTS") != "1", reason="set PROBING_LIVE_TESTS=1 to call the real API")
def test_live_smoke_one_tiny_call():
    client = GroqClient(reasoning=GROQ_REASONING)
    result = run(client.complete("Reply with JSON only.", 'Return {"ok": true}.', model=GROQ_REVIEWER_MODELS[0],
                                 temperature=0.0, max_tokens=200))
    assert "ok" in result.text and result.provider == "groq"

