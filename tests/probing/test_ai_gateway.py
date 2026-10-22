import asyncio

import pytest

from backend.app.probing.ai.cache import ResponseCache
from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.consent import check_consent, is_cacheable
from backend.app.probing.ai.errors import (
    ConsentNotConfirmed,
    ProviderHTTPError,
    RateLimited,
    RequestTooLarge,
)
from backend.app.probing.ai.fake import FakeClient
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.provenance import ProvenanceBuilder
from backend.app.probing.ai.ratelimit import RateLimiter, retry_provider_call


def run(coro):
    return asyncio.run(coro)


class HostedFake(FakeClient):
    """A fake that pretends to be a hosted provider, to exercise the guard."""

    provider = "hosted-fake"
    hosted = True


async def no_sleep(_seconds):
    return None


def ask(gateway, origin="fixture", user="hello", **kwargs):
    return run(
        gateway.complete(
            system="sys",
            user=user,
            model="dummy-v0",
            temperature=0.0,
            max_tokens=100,
            origin=origin,
            **kwargs,
        )
    )


# --- settings ----------------------------------------------------------------


def test_settings_defaults_are_the_worst_case_and_closed():
    settings = AISettings.from_env({})
    assert (settings.max_rpm, settings.max_tpm, settings.concurrency) == (10, 6000, 1)
    assert settings.allow_candidate_code_to_hosted_llm is False
    assert settings.allow_same_family_reviewer is False
    assert settings.reviewer_enabled is False
    assert settings.demo_mode == "cache"


def test_settings_read_the_environment():
    settings = AISettings.from_env(
        {
            "PROBING_LLM_MAX_RPM": "30",
            "PROBING_ALLOW_CANDIDATE_CODE_TO_HOSTED_LLM": "TRUE",
            "PROBING_DEMO_MODE": "live",
            "PROBING_MAPPER_MODEL": "some-model",
        }
    )
    assert settings.max_rpm == 30
    assert settings.allow_candidate_code_to_hosted_llm is True
    assert settings.demo_mode == "live"
    assert settings.mapper_model == "some-model"


@pytest.mark.parametrize(
    "env",
    [
        {"PROBING_LLM_MAX_RPM": "zero"},
        {"PROBING_LLM_MAX_TPM": "0"},
        {"PROBING_ALLOW_CANDIDATE_CODE_TO_HOSTED_LLM": "maybe"},
        {"PROBING_DEMO_MODE": "sometimes"},
    ],
)
def test_bad_settings_are_rejected_loudly(env):
    with pytest.raises(ValueError):
        AISettings.from_env(env)


# --- consent guard -------------------------------------------------------------


def test_candidate_code_is_refused_for_a_hosted_client_by_default():
    with pytest.raises(ConsentNotConfirmed):
        check_consent("candidate", HostedFake(), AISettings())


@pytest.mark.parametrize("origin", ["fixture", "public_dataset", "owner_sample"])
def test_non_candidate_origins_may_go_to_a_hosted_client(origin):
    check_consent(origin, HostedFake(), AISettings())


def test_candidate_code_is_allowed_once_the_owner_confirms_consent():
    settings = AISettings(allow_candidate_code_to_hosted_llm=True)
    check_consent("candidate", HostedFake(), settings)


def test_candidate_code_may_go_to_a_client_that_sends_nothing_off_machine():
    check_consent("candidate", FakeClient(), AISettings())


def test_unknown_origin_is_an_error():
    with pytest.raises(ValueError):
        check_consent("somebody", FakeClient(), AISettings())  # type: ignore[arg-type]


def test_gateway_refuses_before_any_call_reaches_the_client():
    client = HostedFake()
    gateway = LLMGateway(client, AISettings(), sleep=no_sleep)
    with pytest.raises(ConsentNotConfirmed):
        ask(gateway, origin="candidate")
    assert client.calls == []


# --- cache ---------------------------------------------------------------------


def test_second_identical_request_is_served_from_cache():
    client = FakeClient()
    gateway = LLMGateway(client, AISettings(), cache=ResponseCache(), sleep=no_sleep)
    first = ask(gateway)
    second = ask(gateway)
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert second.text == first.text
    assert len(client.calls) == 1


def test_candidate_responses_are_never_cached():
    client = FakeClient()
    gateway = LLMGateway(client, AISettings(), cache=ResponseCache(), sleep=no_sleep)
    ask(gateway, origin="candidate")
    ask(gateway, origin="candidate")
    assert len(client.calls) == 2
    assert is_cacheable("candidate") is False


def test_a_different_request_is_a_cache_miss():
    client = FakeClient()
    gateway = LLMGateway(client, AISettings(), cache=ResponseCache(), sleep=no_sleep)
    ask(gateway, user="one")
    ask(gateway, user="two")
    assert len(client.calls) == 2


def test_disk_cache_survives_a_new_cache_object_and_stores_no_input(tmp_path):
    secret_input = "def very_private(): pass  # unique-marker-12345"
    client = FakeClient()
    ask(LLMGateway(client, AISettings(), cache=ResponseCache(tmp_path), sleep=no_sleep), user=secret_input)

    again = FakeClient()
    result = ask(
        LLMGateway(again, AISettings(), cache=ResponseCache(tmp_path), sleep=no_sleep),
        user=secret_input,
    )
    assert result.cache_hit is True and again.calls == []
    stored = "".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir())
    assert "unique-marker-12345" not in stored
    assert all(len(path.stem) == 64 for path in tmp_path.iterdir())  # hash-named


def test_damaged_cache_file_is_just_a_miss(tmp_path):
    client = FakeClient()
    gateway = LLMGateway(client, AISettings(), cache=ResponseCache(tmp_path), sleep=no_sleep)
    ask(gateway)
    for path in tmp_path.iterdir():
        path.write_text("not json", encoding="utf-8")
    fresh = LLMGateway(client, AISettings(), cache=ResponseCache(tmp_path), sleep=no_sleep)
    assert ask(fresh).cache_hit is False


# --- rate limit and retry ----------------------------------------------------------


def test_request_larger_than_the_token_cap_fails_before_sending():
    client = FakeClient()
    gateway = LLMGateway(client, AISettings(max_tpm=50), sleep=no_sleep)
    with pytest.raises(RequestTooLarge):
        ask(gateway, user="x" * 4000)
    assert client.calls == []


def test_limiter_waits_when_the_request_window_is_full():
    now = [0.0]
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, 10_000, clock=lambda: now[0], sleep=fake_sleep)

    async def three():
        for _ in range(3):
            await limiter.acquire(10)

    run(three())
    assert len(sleeps) >= 1 and sum(sleeps) == pytest.approx(60.0, abs=1.0)


def test_429_is_retried_with_backoff_then_succeeds():
    delays = []

    async def record(seconds):
        delays.append(seconds)

    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderHTTPError(429, "slow down")
        return "ok"

    assert run(retry_provider_call(flaky, sleep=record)) == "ok"
    assert delays == [1.0, 2.0]


def test_retry_after_header_is_honoured():
    delays = []

    async def record(seconds):
        delays.append(seconds)

    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ProviderHTTPError(503, "busy", retry_after=7.0)
        return "ok"

    run(retry_provider_call(flaky, sleep=record))
    assert delays == [7.0]


def test_exhausted_retries_raise_rate_limited():
    async def always():
        raise ProviderHTTPError(429, "no")

    with pytest.raises(RateLimited):
        run(retry_provider_call(always, sleep=no_sleep))


def test_client_errors_are_not_retried():
    attempts = {"n": 0}

    async def bad_request():
        attempts["n"] += 1
        raise ProviderHTTPError(400, "bad")

    with pytest.raises(ProviderHTTPError):
        run(retry_provider_call(bad_request, sleep=no_sleep))
    assert attempts["n"] == 1


# --- fake client and provenance -----------------------------------------------------------


def test_fake_client_is_labelled_and_scriptable():
    client = FakeClient(script=["first", ValueError("boom")])
    gateway = LLMGateway(client, AISettings(), sleep=no_sleep)
    result = ask(gateway)
    assert (result.text, result.provider, result.model) == ("first", "fake", "dummy-v0")
    with pytest.raises(ValueError):
        ask(gateway, user="again")


def test_provenance_sums_calls_and_flags_all_cached():
    builder = ProvenanceBuilder("mapper_v1")
    for cache_hit in (True, True):
        builder.add(
            provider="fake", model="dummy-v0", family="fake", temperature=0.2,
            tokens_in=10, tokens_out=5, cache_hit=cache_hit,
        )
    provenance = builder.build()
    assert (provenance.calls, provenance.tokens_in, provenance.tokens_out) == (2, 20, 10)
    assert provenance.cache_hit is True and provenance.temperatures == (0.2, 0.2)

    builder.add(
        provider="fake", model="dummy-v0", family="fake", temperature=0.0,
        tokens_in=1, tokens_out=1, cache_hit=False,
    )
    assert builder.build().cache_hit is False
