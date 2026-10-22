"""Tests that really start Docker containers. Opt-in: set PROBING_DOCKER_TESTS=1.

They need Docker running and the python:3.12-slim image already pulled. They run
only hand-written code, never model output. The normal suite stays offline.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from backend.app.probing.probes.docker_runner import NAME_PREFIX, DockerRunner, docker_ready
from backend.app.probing.probes.runner import RunLimits
from backend.app.probing.probes.schema import ProbePackage
from backend.app.probing.probes.verify import verify_probe

pytestmark = pytest.mark.skipif(
    os.environ.get("PROBING_DOCKER_TESTS") != "1", reason="set PROBING_DOCKER_TESTS=1 to use Docker"
)

FLAWED = Path(__file__).resolve().parent / "fixtures" / "flawed_10min_wrong_expected_value.json"


def run(coro):
    return asyncio.run(coro)


def containers() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={NAME_PREFIX}", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    ).stdout
    return out.split()


@pytest.fixture(autouse=True)
def docker_and_no_leftovers():
    ready, reason = docker_ready()
    if not ready:
        pytest.skip(reason)
    yield
    assert containers() == [], "a probe container was left behind"


def go(code, cases, **limits):
    return run(DockerRunner().run_cases(code, "f", cases, RunLimits(**limits)))


def test_a_correct_function_returns_its_values_and_printing_is_ignored():
    out = go("def f(a, b):\n    print('noise')\n    return {a + b, 1}\n", [("x", (2, 3)), ("y", (0, 1))])
    assert [(o.returned, o.error) for o in out] == [([1, 5], None), ([1], None)]


def test_errors_are_reported_per_case():
    out = go("def f(x):\n    return 1 // x\n", [("a", (1,)), ("b", (0,))])
    assert out[0].returned == 1 and "ZeroDivisionError" in out[1].error


def test_a_busy_loop_hits_the_per_case_limit():
    out = go("def f(x):\n    while True:\n        pass\n", [("a", (1,))], seconds=1.0)
    assert out[0].hit_limit


def test_code_that_swallows_the_alarm_is_killed_with_its_container():
    code = (
        "def f(x):\n    while True:\n        try:\n            while True:\n                pass\n"
        "        except BaseException:\n            pass\n"
    )
    out = go(code, [("a", (1,))], seconds=1.0)
    assert out[0].hit_limit and "container killed" in out[0].error


def test_memory_is_capped():
    out = go("def f(x):\n    return len(bytearray(300 * 1024 * 1024))\n", [("a", (1,))])
    assert out[0].hit_limit and "memory" in out[0].error


def test_the_network_is_blocked():
    code = "import socket\ndef f(x):\n    socket.create_connection(('1.1.1.1', 80), 2)\n"
    assert go(code, [("a", (1,))])[0].error.startswith("OSError")


def test_the_flawed_example_is_caught():
    if not FLAWED.exists():
        pytest.skip("the flawed example fixture is missing")
    probe = ProbePackage.model_validate(json.loads(FLAWED.read_text(encoding="utf-8"))["probe"])
    done = run(verify_probe(probe, DockerRunner()))
    bad = [c for c in done.reference_check.cases if not c.passed]
    assert done.verification == "reference_failed"
    assert [(c.case_id, c.expected, c.got) for c in bad] == [("c2", 6, 5)]
