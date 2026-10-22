"""DockerRunner, the reference check and its statuses. Offline: nothing here starts Docker.

The runner is replaced by a scripted one, and the Docker command line is checked by
patching ``subprocess.run``. The tests that really start containers are in
``test_docker_live.py`` and run only with PROBING_DOCKER_TESTS=1.
"""

import asyncio
import json
import subprocess
import typing
from uuid import uuid4

import pytest

from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.probes import docker_runner
from backend.app.probing.probes.docker_runner import DockerRunner, RunnerChoice, make_runner
from backend.app.probing.probes.fake_probes import FakeDemoClient
from backend.app.probing.probes.generator import ProbeGenerator
from backend.app.probing.probes.reference_cache import ReferenceCheckCache, reference_key
from backend.app.probing.probes.runner import CaseOutcome, NullRunner, RunLimits, RunnerUnavailable
from backend.app.probing.probes.schema import (
    REFERENCE_CHECKED_LABEL,
    Ladder,
    ProbePackage,
    ReferenceCheck,
    Verification,
)
from backend.app.probing.probes.verify import matches, verify_probe
from backend.app.probing.store.repository import ApprovalError, InMemoryProbeRepository

BIG = AISettings(max_rpm=100_000, max_tpm=10**9)


def run(coro):
    return asyncio.run(coro)


async def no_sleep(_seconds):
    return None


def a_probe(level=10) -> ProbePackage:
    gateway = LLMGateway(FakeDemoClient(), BIG, sleep=no_sleep)
    ladder = run(ProbeGenerator(gateway, model="dummy-v0").generate_ladder("hash_table"))
    return next(p for p in ladder.probes if p.target_minutes == level)


class ScriptedRunner:
    """Returns prepared outcomes and never runs anything; remembers what it was asked to run."""

    description = "scripted runner"

    def __init__(self, probe: ProbePackage | None = None, *, wrong: dict | None = None, errors: dict | None = None):
        self.values = {c.case_id: c.expected for c in probe.cases} if probe else {}
        self.values.update(wrong or {})
        self.errors = errors or {}
        self.calls: list[tuple] = []

    async def run_cases(self, code, function_name, cases, limits):
        self.calls.append((code, function_name, cases, limits))
        return [
            CaseOutcome(cid, error=self.errors[cid], hit_limit="limit" in self.errors[cid])
            if cid in self.errors
            else CaseOutcome(cid, returned=self.values.get(cid))
            for cid, _ in cases
        ]


# --- comparing results --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "checker, expected, got, ok",
    [
        ("exact", 5, 5, True), ("exact", 5, 6, False), ("exact", 3, 3.0, True),
        ("exact", True, 1, False), ("exact", 1, True, False),
        ("exact", [1, 2], [1, 2], True), ("exact", [1, 2], [2, 1], False), ("exact", [1, 2], (1, 2), True),
        ("exact", {"a": [1]}, {"a": [1]}, True), ("exact", {"a": 1}, {"b": 1}, False),
        ("exact", None, None, True), ("exact", None, 0, False), ("exact", "ab", "ab", True),
        ("unordered", [3, 1, 2], [1, 2, 3], True), ("unordered", [1, 1, 2], [1, 2, 2], False),
        ("unordered", [[1, 2], [3]], [[3], [2, 1]], True), ("unordered", [1], [1, 1], False),
        ("float_tolerance", 0.1 + 0.2, 0.3, True), ("float_tolerance", 1.0, 1.1, False),
        ("float_tolerance", [1.0, 2.0], [1.0000001, 2.0], True), ("float_tolerance", "a", "b", False),
    ],
)
def test_matching_rules(checker, expected, got, ok):
    assert matches(checker, expected, got) is ok


# --- the reference check ----------------------------------------------------------------------------------


def test_status_names_are_new_and_never_cross_checked():
    statuses = set(typing.get_args(Verification))
    assert {"reference_checked", "reference_failed"} <= statuses
    assert "cross_checked" not in statuses
    assert REFERENCE_CHECKED_LABEL == "Reference-checked: not independently verified."


def test_matching_reference_is_reference_checked_and_nothing_else_changes():
    probe = a_probe()
    done = run(verify_probe(probe, ScriptedRunner(probe)))
    assert done.verification == "reference_checked"
    assert done.reference_check.label == "Reference-checked: not independently verified."
    assert all(c.passed for c in done.reference_check.cases) and len(done.reference_check.cases) == len(probe.cases)
    assert (done.approval, done.oracle, done.static_checks, done.cases) == (
        probe.approval, {}, probe.static_checks, probe.cases)


def test_only_the_generated_reference_solution_is_run():
    probe = a_probe()
    runner = ScriptedRunner(probe)
    run(verify_probe(probe, runner))
    ((code, function_name, cases, limits),) = runner.calls
    assert code == probe.reference_solution and function_name == probe.function_name
    assert [c[0] for c in cases] == [c.case_id for c in probe.cases]
    assert limits.network is False


def test_a_wrong_expected_value_is_reference_failed_with_the_case_shown():
    probe = a_probe()
    first = probe.cases[0]
    done = run(verify_probe(probe, ScriptedRunner(probe, wrong={first.case_id: "something else"})))
    assert done.verification == "reference_failed" and "does not reproduce" in done.reference_check.label
    bad = [c for c in done.reference_check.cases if not c.passed]
    assert [c.case_id for c in bad] == [first.case_id]
    assert bad[0].expected == first.expected and bad[0].got == "something else"


@pytest.mark.parametrize("error", ["ZeroDivisionError: division by zero", "time limit"])
def test_an_error_or_limit_fails_the_check(error):
    probe = a_probe()
    done = run(verify_probe(probe, ScriptedRunner(probe, errors={probe.cases[1].case_id: error})))
    failed = [c for c in done.reference_check.cases if not c.passed]
    assert done.verification == "reference_failed" and failed[0].error == error
    assert failed[0].hit_limit is (error == "time limit")


def test_a_probe_that_failed_static_checks_is_not_run():
    probe = a_probe()
    bad = probe.model_copy(update={"static_checks": probe.static_checks.model_copy(update={"passed": False})})
    runner = ScriptedRunner(probe)
    done = run(verify_probe(bad, runner))
    assert runner.calls == [] and done.verification == "unverified" and done.reference_check is None
    assert "static checks failed" in done.notes[-1]


def test_with_no_runner_nothing_is_verified():
    with pytest.raises(RunnerUnavailable):
        run(verify_probe(a_probe(), NullRunner()))


# --- choosing a runner ------------------------------------------------------------------------------------


def test_runner_is_off_by_default():
    choice = make_runner({})
    assert isinstance(choice.runner, NullRunner) and choice.mode == "none"
    assert make_runner({"PROBING_RUNNER": "nothing"}).mode == "none"


def test_docker_not_ready_falls_back_to_null_runner_quietly(monkeypatch):
    monkeypatch.setattr(docker_runner, "docker_ready", lambda: (False, "docker is not running"))
    choice = make_runner({"PROBING_RUNNER": "docker"})
    assert isinstance(choice.runner, NullRunner) and choice.mode == "none" and "not running" in choice.note


def test_docker_ready_gives_the_docker_runner(monkeypatch):
    monkeypatch.setattr(docker_runner, "docker_ready", lambda: (True, ""))
    choice = make_runner({"PROBING_RUNNER": " Docker "})
    assert isinstance(choice.runner, DockerRunner) and choice.mode == "docker"


# --- the Docker command (subprocess patched) ------------------------------------------------------------


class Recorder:
    def __init__(self, on_run):
        self.commands: list[list[str]] = []
        self.on_run = on_run

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        if "run" in command[:2]:
            return self.on_run(command, kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")


def docker_run(cases=(("a", (1,)),)):
    return run(DockerRunner().run_cases("def f(x):\n    return x\n", "f", list(cases), RunLimits()))


def result_line(items):
    return "\n@@RESULT@@" + json.dumps(items) + "\n"


def name_of(command):
    return command[command.index("--name") + 1]


def test_command_is_locked_down_named_and_not_killed_after_a_clean_run(monkeypatch):
    reply = result_line([{"id": "a", "ok": True, "value": 1}])
    rec = Recorder(lambda c, k: subprocess.CompletedProcess(c, 0, reply, ""))
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    outcomes = docker_run()
    command = rec.commands[0]
    assert outcomes == [CaseOutcome("a", returned=1)]
    for flag, value in [("--network", "none"), ("--memory", "128m"), ("--memory-swap", "128m"), ("--cpus", "1"),
                        ("--pids-limit", "64"), ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges")]:
        assert command[command.index(flag) + 1] == value
    assert "--rm" in command and "--read-only" in command and docker_runner.IMAGE in command
    assert name_of(command).startswith(docker_runner.NAME_PREFIX)
    assert len(rec.commands) == 1  # a clean run needs no kill


def test_every_container_gets_its_own_name(monkeypatch):
    reply = result_line([{"id": "a", "ok": True, "value": 1}])
    rec = Recorder(lambda c, k: subprocess.CompletedProcess(c, 0, reply, ""))
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    docker_run()
    docker_run()
    assert len({name_of(c) for c in rec.commands}) == 2


def test_a_timeout_kills_and_removes_the_named_container(monkeypatch):
    def timeout(command, kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    rec = Recorder(timeout)
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    outcomes = docker_run(cases=(("a", (1,)), ("b", (2,))))
    name = name_of(rec.commands[0])
    assert any(c[1:] == ["kill", name] for c in rec.commands[1:])
    assert any(c[1:] == ["rm", "-f", name] for c in rec.commands[1:])
    assert len(outcomes) == 2 and all(o.hit_limit and "container killed" in o.error for o in outcomes)


def test_an_unexpected_error_still_kills_the_container(monkeypatch):
    def boom(command, kwargs):
        raise KeyboardInterrupt

    rec = Recorder(boom)
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    with pytest.raises(KeyboardInterrupt):
        docker_run()
    assert any("kill" in c for c in rec.commands[1:])


def test_exit_137_means_the_memory_limit(monkeypatch):
    rec = Recorder(lambda c, k: subprocess.CompletedProcess(c, 137, "", ""))
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    (outcome,) = docker_run()
    assert outcome.hit_limit and "memory limit" in outcome.error


def test_garbage_output_gives_errors_not_a_crash(monkeypatch):
    rec = Recorder(lambda c, k: subprocess.CompletedProcess(c, 1, "traceback...\n@@RESULT@@{not json", ""))
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    (outcome,) = docker_run()
    assert outcome.error and outcome.returned is None and not outcome.hit_limit


def test_per_case_errors_and_limits_come_through(monkeypatch):
    reply = result_line([{"id": "a", "error": "ValueError: bad"}, {"id": "b", "limit": True, "error": "time limit"}])
    rec = Recorder(lambda c, k: subprocess.CompletedProcess(c, 0, "noise\n" + reply, ""))
    monkeypatch.setattr(docker_runner.subprocess, "run", rec)
    a, b, c = docker_run(cases=(("a", (1,)), ("b", (2,)), ("c", (3,))))
    assert (a.error, b.hit_limit, c.error) == ("ValueError: bad", True, "no result for this case")


def test_network_can_never_be_switched_on():
    with pytest.raises(ValueError):
        run(DockerRunner().run_cases("x", "f", [], RunLimits(network=True)))


def test_the_harness_is_valid_python():
    compile(docker_runner.HARNESS, "<harness>", "exec")


# --- storing and recording -----------------------------------------------------------------------------------


def checked(probe) -> ReferenceCheck:
    return run(verify_probe(probe, ScriptedRunner(probe))).reference_check


def test_repository_keeps_the_check_across_a_restart_and_leaves_approval_alone(tmp_path):
    log, probe, segment = tmp_path / "e.jsonl", a_probe(), uuid4()
    repo = InMemoryProbeRepository(log)
    repo.save_ladder(segment, Ladder(tag_id="hash_table", tag="Hash Table", taxonomy_version="x", probes=(probe,)))
    updated = repo.attach_reference_check(probe.probe_id, checked(probe))
    assert updated.verification == "reference_checked" and updated.approval == "pending"
    again = InMemoryProbeRepository(log).get_probe(probe.probe_id)
    assert again.verification == "reference_checked" and again.reference_check == updated.reference_check
    assert repo.approve(probe.probe_id, "Dana").verification == "reference_checked"  # approving never changes it


def test_recorded_results_are_keyed_by_what_decides_them(tmp_path):
    probe = a_probe()
    cache = ReferenceCheckCache(tmp_path)
    assert cache.get(probe) is None
    cache.put(probe, checked(probe))
    assert cache.get(probe).status == "reference_checked"
    edited_case = probe.cases[0].model_copy(update={"expected": "changed"})
    other = probe.model_copy(update={"cases": (edited_case, *probe.cases[1:])})
    other_code = probe.model_copy(update={"reference_solution": probe.reference_solution + "\n# x"})
    assert len({reference_key(probe), reference_key(other), reference_key(other_code)}) == 3
    assert cache.get(other) is None and cache.get(other_code) is None


def test_a_damaged_record_is_ignored(tmp_path):
    probe = a_probe()
    (tmp_path / f"{reference_key(probe)}.json").write_text("{nope", encoding="utf-8")
    assert ReferenceCheckCache(tmp_path).get(probe) is None


# --- approval is blocked after a failed check ------------------------------------------------------------


def failed_check(probe) -> ReferenceCheck:
    good = checked(probe)
    first = good.cases[0].model_copy(update={"passed": False, "got": "wrong"})
    return good.model_copy(update={"status": "reference_failed", "cases": (first, *good.cases[1:])})


def test_a_probe_whose_reference_check_failed_cannot_be_approved_but_can_be_rejected(tmp_path):
    log, probe, segment = tmp_path / "e.jsonl", a_probe(), uuid4()
    repo = InMemoryProbeRepository(log)
    repo.save_ladder(segment, Ladder(tag_id="hash_table", tag="Hash Table", taxonomy_version="x", probes=(probe,)))
    repo.attach_reference_check(probe.probe_id, failed_check(probe))
    with pytest.raises(ApprovalError, match="approval is disabled"):
        repo.approve(probe.probe_id, "Dana")
    assert repo.get_probe(probe.probe_id).approval == "pending"
    assert repo.reject(probe.probe_id, "Dana").approval == "rejected"
    assert InMemoryProbeRepository(log).get_probe(probe.probe_id).approval == "rejected"


def test_a_failed_check_after_approval_takes_the_approval_back(tmp_path):
    log, probe, segment = tmp_path / "e.jsonl", a_probe(), uuid4()
    repo = InMemoryProbeRepository(log)
    repo.save_ladder(segment, Ladder(tag_id="hash_table", tag="Hash Table", taxonomy_version="x", probes=(probe,)))
    repo.approve(probe.probe_id, "Dana")
    updated = repo.attach_reference_check(probe.probe_id, failed_check(probe))
    assert updated.approval == "pending" and updated.approved_by is None
    assert InMemoryProbeRepository(log).get_probe(probe.probe_id).approval == "pending"

