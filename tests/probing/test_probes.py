import asyncio
import copy
import json

import pytest

from backend.app.probing.ai.cache import ResponseCache
from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.errors import ProviderHTTPError
from backend.app.probing.ai.fake import FakeClient
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.probes.fake_probes import CANNED, FakeDemoClient
from backend.app.probing.probes.generator import ProbeGenerator
from backend.app.probing.probes.runner import NullRunner, RunLimits, RunnerUnavailable
from backend.app.probing.probes.schema import ProbePackage
from backend.app.probing.probes.validation import check_draft
from backend.app.probing.probes.verify import verify_probe


async def no_sleep(_seconds):
    return None


def generator_with(client, *, cache=None):
    # Generous limits: with sleeping stubbed out, a tight limiter would spin on the real clock.
    gateway = LLMGateway(client, AISettings(max_rpm=100_000, max_tpm=10**9), cache=cache, sleep=no_sleep)
    return ProbeGenerator(gateway, model="dummy-v0")


def ladder_of(generator, tag_id="hash_table", **kwargs):
    return asyncio.run(generator.generate_ladder(tag_id, **kwargs))


def good():
    return copy.deepcopy(CANNED[("hash_table", 5)])


def failed(draft) -> set[str]:
    return {c.name for c in check_draft(draft).checks.checks if not c.passed}


# --- the generator on the fake client -------------------------------------------------------


def test_fake_ladder_has_three_levels_and_the_final_package_shape():
    ladder = ladder_of(generator_with(FakeDemoClient()))
    assert [p.target_minutes for p in ladder.probes] == [5, 10, 20]
    assert ladder.default_level == 10 and ladder.notes == ()
    probe = ladder.probes[1]
    assert isinstance(probe, ProbePackage)
    assert probe.verification == "unverified" and probe.approval == "pending" and probe.approved_by is None
    assert probe.oracle == {} and probe.observed_minutes is None
    assert probe.static_checks.passed and len(probe.cases) == 6
    assert probe.reference_solution and probe.signature and probe.function_name
    assert (probe.provenance.provider, probe.provenance.model) == ("fake", "dummy-v0")
    assert probe.notes == ("generated; tests were not executed",)


def test_every_canned_probe_passes_the_static_checks():
    for key, draft in CANNED.items():
        assert check_draft(draft).checks.passed, key


def test_unknown_tag_gets_a_labelled_placeholder_that_still_passes():
    ladder = ladder_of(generator_with(FakeDemoClient()), "sorting")
    assert len(ladder.probes) == 3 and "PLACEHOLDER from the fake model" in ladder.probes[0].question


def test_unknown_or_disabled_tags_are_refused():
    generator = generator_with(FakeDemoClient())
    for tag_id in ("nonsense", "design"):
        with pytest.raises(ValueError):
            ladder_of(generator, tag_id)


def test_prompt_carries_the_tag_and_level_and_no_code():
    client = FakeDemoClient()
    ladder_of(generator_with(client), "heap_priority_queue", levels=(10,))
    user = client.calls[0]["user"]
    assert "TAG_ID: heap_priority_queue" in user and "TARGET_MINUTES: 10" in user
    assert "```" not in user  # no fenced code of anyone's


def test_second_identical_ladder_is_served_from_cache():
    client = FakeDemoClient()
    generator = generator_with(client, cache=ResponseCache())
    ladder_of(generator)
    first_calls = len(client.calls)
    again = ladder_of(generator)
    assert len(client.calls) == first_calls and again.probes[0].provenance.cache_hit is True


def test_a_failed_static_check_is_retried_once_with_the_problem_named():
    bad = good()
    bad["reference_solution"] = "import os\n" + bad["reference_solution"]
    client = FakeClient(script=[json.dumps(bad), json.dumps(good())])
    ladder = ladder_of(generator_with(client), levels=(5,))
    assert len(ladder.probes) == 1 and ladder.probes[0].provenance.calls == 2
    assert "reference_imports_allowed" in client.calls[1]["user"]


def test_a_level_that_keeps_failing_is_left_out_with_a_note():
    bad = good()
    bad["signature"] = "not a signature"
    client = FakeClient(script=[json.dumps(bad), json.dumps(bad)])
    ladder = ladder_of(generator_with(client), levels=(5,))
    assert ladder.probes == () and "dropped after one retry" in ladder.notes[0]


def test_non_json_output_is_retried_and_then_dropped():
    client = FakeClient(script=["sorry, no", "still no"])
    ladder = ladder_of(generator_with(client), levels=(5,))
    assert ladder.probes == () and "not a JSON object" in ladder.notes[0]


def test_json_in_a_code_fence_is_accepted():
    fenced = "`" * 3 + "json\n" + json.dumps(good()) + "\n" + "`" * 3
    ladder = ladder_of(generator_with(FakeClient(script=[fenced])), levels=(5,))
    assert len(ladder.probes) == 1


def test_provider_outage_does_not_raise_and_leaves_a_note():
    client = FakeClient(script=[ProviderHTTPError(503, "down")] * 10)
    ladder = ladder_of(generator_with(client), levels=(5,))
    assert ladder.probes == () and "RateLimited" in ladder.notes[0]


# --- static checks ---------------------------------------------------------------------------------


def test_good_draft_has_no_failed_checks():
    assert failed(good()) == set()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda d: d.update(reference_solution="import os\n" + d["reference_solution"]), "reference_imports_allowed"),
        (lambda d: d.update(reference_solution="from socket import socket\n" + d["reference_solution"]), "reference_imports_allowed"),
        (lambda d: d.update(reference_solution=d["reference_solution"] + "\nopen('x')\n"), "reference_no_dangerous_names"),
        (lambda d: d.update(reference_solution="def has_duplicate(nums):\n    return eval('1')\n"), "reference_no_dangerous_names"),
        (lambda d: d.update(reference_solution="def has_duplicate(nums):\n    return nums.__class__\n"), "reference_no_dangerous_names"),
        (lambda d: d.update(reference_solution="def has_duplicate(nums:\n"), "reference_parses"),
        (lambda d: d.update(reference_solution="def other(nums):\n    return True\n"), "reference_defines_function"),
        (lambda d: d.update(reference_solution="def has_duplicate(a, b):\n    return True\n"), "reference_params_match"),
        (lambda d: d.update(reference_solution="def has_duplicate(nums):\n    while True:\n        pass\n"), "reference_no_unbounded_loop"),
        (lambda d: d.update(signature="has_duplicate(nums)"), "signature_parses"),
        (lambda d: d.update(function_name="renamed"), "signature_names_function"),
        (lambda d: d.update(question="short"), "question_present"),
        (lambda d: d.update(checker="fuzzy"), "checker_known"),
        (lambda d: d.update(cases=d["cases"][:2]), "cases_count"),
        (lambda d: d["cases"][0].update(args_json="[1, 2"), "cases_well_formed"),
        (lambda d: d["cases"][0].update(args_json="[[1], [2]]"), "cases_well_formed"),
        (lambda d: d["cases"][0].update(args_json="5"), "cases_well_formed"),
        (lambda d: d["cases"][1].update(args_json=d["cases"][0]["args_json"]), "cases_well_formed"),
    ],
)
def test_each_static_check_catches_its_problem(change, expected):
    draft = good()
    change(draft)
    assert expected in failed(draft)


def test_a_draft_that_is_not_a_dict_of_the_right_shape_fails_without_raising():
    assert not check_draft({}).checks.passed
    assert not check_draft({"question": 5, "cases": "x", "reference_solution": 3, "signature": [], "function_name": 1}).checks.passed


def test_checking_a_dangerous_reference_does_not_run_it(tmp_path):
    marker = tmp_path / "ran.txt"
    draft = good()
    draft["reference_solution"] = (
        f"import os\nopen({str(marker)!r}, 'w').write('x')\n" + draft["reference_solution"]
    )
    assert not check_draft(draft).checks.passed
    assert not marker.exists()


# --- the runner contract --------------------------------------------------------------------------


def test_null_runner_refuses_and_verification_is_not_available():
    runner = NullRunner()
    with pytest.raises(RunnerUnavailable):
        asyncio.run(runner.run_cases("x = 1", "f", [], RunLimits()))
    probe = ladder_of(generator_with(FakeDemoClient()), levels=(5,)).probes[0]
    with pytest.raises(RunnerUnavailable):
        asyncio.run(verify_probe(probe, runner))
