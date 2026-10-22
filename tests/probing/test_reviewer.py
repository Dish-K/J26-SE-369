import asyncio
import json

import pytest

from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.errors import ProviderHTTPError
from backend.app.probing.ai.fake import FakeClient
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.probes.fake_probes import FakeDemoClient, FakeReviewerClient
from backend.app.probing.probes.generator import CONSTRAINT_RULES, GUIDANCE, PROMPT_VERSIONS, ProbeGenerator
from backend.app.probing.probes.reviewer import ProbeReviewer
from backend.app.probing.store.repository import InMemoryProbeRepository

BIG = AISettings(max_rpm=100_000, max_tpm=10**9)


async def no_sleep(_seconds):
    return None


def a_probe(level=10):
    gateway = LLMGateway(FakeDemoClient(), BIG, sleep=no_sleep)
    ladder = asyncio.run(ProbeGenerator(gateway, model="dummy-v0").generate_ladder("hash_table"))
    return next(p for p in ladder.probes if p.target_minutes == level)


def reviewer_with(client, **kwargs):
    return ProbeReviewer(LLMGateway(client, BIG, sleep=no_sleep), model="dummy-v0", **kwargs)


def good_reply(**changes):
    body = {"estimated_minutes": 12, "difficulty": "harder", "concerns": ["too long"], "case_problems": ["c2: wrong"]}
    body.update(changes)
    return json.dumps(body)


def review(reviewer, probe):
    return asyncio.run(reviewer.review(probe))


# --- the reviewer ------------------------------------------------------------------------------


def test_disabled_reviewer_does_nothing_and_says_so():
    client = FakeReviewerClient()
    result, reason = review(reviewer_with(client, enabled=False), a_probe())
    assert result is None and "switched off" in reason and client.calls == []


def test_same_provider_is_refused_unless_allowed():
    client = FakeDemoClient()  # same 'family' as the generator that made the probe
    result, reason = review(reviewer_with(client, enabled=True), a_probe())
    assert result is None and "same provider" in reason and client.calls == []


def test_same_provider_review_is_labelled_a_weaker_check():
    client = FakeClient(script=[good_reply()])
    client.family = "fake"  # same family as the fake generator
    result, _ = review(reviewer_with(client, enabled=True, allow_same_family=True), a_probe())
    assert result.weaker_check is True and "weaker check" in result.label and result.advisory_only is True


def test_different_provider_review_is_not_labelled_weaker():
    result, _ = review(reviewer_with(FakeReviewerClient(), enabled=True), a_probe())
    assert result.weaker_check is False and "weaker check" not in result.label
    assert result.difficulty == "about_right" and "FAKE REVIEW" in result.concerns[0]
    assert (result.provenance.provider, result.provenance.model) == ("fake", "dummy-v0")


def test_prompt_shows_the_probe_and_no_one_elses_code():
    client = FakeReviewerClient()
    probe = a_probe()
    review(reviewer_with(client, enabled=True), probe)
    user = client.calls[0]["user"]
    assert probe.question in user and probe.signature in user and "c1:" in user
    assert probe.reference_solution.splitlines()[0] in user and "TARGET_MINUTES: 10" in user


def test_output_is_validated_extra_keys_dropped_and_lists_capped():
    messy = json.dumps(
        {"estimated_minutes": 8, "difficulty": "easier", "concerns": ["a"] * 9 + [5, ""], "case_problems": "no",
         "approval": "approved", "verification": "verified"}
    )
    result, _ = review(reviewer_with(FakeClient(script=[messy]), enabled=True, allow_same_family=True), a_probe())
    assert len(result.concerns) == 5 and result.case_problems == ()
    assert not hasattr(result, "approval")


@pytest.mark.parametrize("bad", ["nope", good_reply(difficulty="medium"), good_reply(estimated_minutes="ten"),
                                 good_reply(estimated_minutes=0), good_reply(estimated_minutes=True), "[1, 2]"])
def test_invalid_output_is_retried_once_then_given_up(bad):
    client = FakeClient(script=[bad, bad])
    result, reason = review(reviewer_with(client, enabled=True, allow_same_family=True), a_probe())
    assert result is None and "invalid after one retry" in reason and len(client.calls) == 2


def test_invalid_then_valid_works():
    client = FakeClient(script=["nope", good_reply()])
    result, _ = review(reviewer_with(client, enabled=True, allow_same_family=True), a_probe())
    assert result.estimated_minutes == 12 and result.provenance.calls == 2


def test_provider_failure_gives_a_reason_not_an_exception():
    client = FakeClient(script=[ProviderHTTPError(503, "down")] * 10)
    result, reason = review(reviewer_with(client, enabled=True, allow_same_family=True), a_probe())
    assert result is None and "RateLimited" in reason


def test_review_changes_nothing_else_on_the_probe():
    repo = InMemoryProbeRepository()
    probe = a_probe()
    from backend.app.probing.probes.schema import Ladder
    from uuid import uuid4

    segment = uuid4()
    repo.save_ladder(segment, Ladder(tag_id="hash_table", tag="Hash Table", taxonomy_version="x", probes=(probe,)))
    result, _ = review(reviewer_with(FakeReviewerClient(), enabled=True), probe)
    updated = repo.attach_review(probe.probe_id, result)
    assert updated.review == result
    assert (updated.approval, updated.verification, updated.static_checks, updated.oracle) == (
        probe.approval, probe.verification, probe.static_checks, probe.oracle)


def test_review_survives_a_restart(tmp_path):
    from backend.app.probing.probes.schema import Ladder
    from uuid import uuid4

    log = tmp_path / "events.jsonl"
    probe, segment = a_probe(), uuid4()
    repo = InMemoryProbeRepository(log)
    repo.save_ladder(segment, Ladder(tag_id="hash_table", tag="Hash Table", taxonomy_version="x", probes=(probe,)))
    result, _ = review(reviewer_with(FakeReviewerClient(), enabled=True), probe)
    repo.attach_review(probe.probe_id, result)
    assert InMemoryProbeRepository(log).get_probe(probe.probe_id).review == result


# --- the stricter generation rules --------------------------------------------------------------


def test_constraint_rules_apply_to_10_and_20_minutes_only():
    assert CONSTRAINT_RULES.strip() in GUIDANCE[10] and CONSTRAINT_RULES.strip() in GUIDANCE[20]
    assert CONSTRAINT_RULES.strip() not in GUIDANCE[5]
    assert "only gives an approximation" in CONSTRAINT_RULES and "obey every constraint" in CONSTRAINT_RULES
    assert PROMPT_VERSIONS[5] != PROMPT_VERSIONS[10] == PROMPT_VERSIONS[20]


def test_rules_reach_the_prompt_for_10_minutes_and_provenance_names_the_version():
    client = FakeDemoClient()
    gateway = LLMGateway(client, BIG, sleep=no_sleep)
    ladder = asyncio.run(ProbeGenerator(gateway, model="dummy-v0").generate_ladder("hash_table"))
    prompts = {c["user"].split("TARGET_MINUTES: ")[1].split("\n")[0]: c["user"] for c in client.calls}
    assert "exactly computable" in prompts["10"] and "exactly computable" not in prompts["5"]
    assert [p.provenance.prompt_version for p in ladder.probes] == [
        "probe_generate_v1", "probe_generate_v1.1", "probe_generate_v1.1"]

