"""A sandboxed code runner on Docker. Off unless ``PROBING_RUNNER=docker``.

Only generated reference solutions are run here (``verify.verify_probe`` is the one
caller); candidate code is never run. Each run is one throwaway container with no
network, a memory cap, one CPU, a process cap, a read-only filesystem and no
capabilities. Every container is named and, unless it ended on its own, is
``docker kill``-ed: killing the ``docker run`` client does not stop a container.

``subprocess`` in a worker thread is used instead of asyncio subprocesses, which
need a particular event loop on Windows.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any

from backend.app.probing.probes.runner import CaseOutcome, CodeRunner, NullRunner, RunLimits

logger = logging.getLogger(__name__)

RUNNER_VARIABLE = "PROBING_RUNNER"
IMAGE = "python:3.12-slim"
NAME_PREFIX = "codetrace-probe-"
MARKER = "@@RESULT@@"
MAX_CONCURRENT = 2
STARTUP_ALLOWANCE_SECONDS = 8.0
MAX_TOTAL_SECONDS = 60.0

# Runs inside the container. It reads one JSON request on stdin, runs each case under a
# per-case alarm, and prints one marked JSON line. Output the code prints is thrown away.
HARNESS = r'''
import sys, json, io, signal, copy
req = json.loads(sys.stdin.read())
real = sys.stdout
sys.stdout = io.StringIO()
class Limit(BaseException):
    pass
def on_alarm(*_):
    raise Limit()
signal.signal(signal.SIGALRM, on_alarm)
def conv(o):
    if isinstance(o, (set, frozenset)):
        try:
            return sorted(o)
        except TypeError:
            return list(o)
    raise TypeError("not JSON serialisable")
out = []
fn = None
setup_error = None
try:
    g = {"__name__": "reference"}
    exec(compile(req["code"], "<reference>", "exec"), g)
    fn = g[req["function"]]
except BaseException as e:
    setup_error = (type(e).__name__ + ": " + str(e))[:200]
for case in req["cases"]:
    if fn is None:
        out.append({"id": case["id"], "error": setup_error})
        continue
    try:
        signal.setitimer(signal.ITIMER_REAL, req["seconds"])
        value = fn(*copy.deepcopy(case["args"]))
        signal.setitimer(signal.ITIMER_REAL, 0)
        out.append({"id": case["id"], "ok": True, "value": json.loads(json.dumps(value, default=conv))})
    except Limit:
        out.append({"id": case["id"], "limit": True, "error": "time limit"})
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        out.append({"id": case["id"], "error": (type(e).__name__ + ": " + str(e))[:200]})
signal.setitimer(signal.ITIMER_REAL, 0)
real.write("\n" + "@@RESULT@@" + json.dumps(out) + "\n")
real.flush()
'''


@dataclass(frozen=True)
class RunnerChoice:
    runner: CodeRunner
    mode: str  # "docker" or "none"
    note: str


def _docker(*args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shutil.which("docker") or "docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def docker_ready() -> tuple[bool, str]:
    """Whether Docker is running and the image is already here (never pulls)."""
    if shutil.which("docker") is None:
        return False, "docker is not installed"
    try:
        if _docker("version", "--format", "{{.Server.Version}}", timeout=8).returncode != 0:
            return False, "docker is not running"
        if _docker("image", "inspect", IMAGE, timeout=8).returncode != 0:
            return False, f"image {IMAGE} is not pulled (docker pull {IMAGE})"
    except (OSError, subprocess.SubprocessError):
        return False, "docker did not answer"
    return True, ""


def make_runner(env: dict[str, str] | None = None) -> RunnerChoice:
    """``DockerRunner`` only when asked for and ready; otherwise ``NullRunner`` and no error."""
    source = os.environ if env is None else env
    if (source.get(RUNNER_VARIABLE) or "").strip().lower() != "docker":
        return RunnerChoice(NullRunner(), "none", "no code runner is configured (PROBING_RUNNER)")
    ready, reason = docker_ready()
    if not ready:
        logger.warning("PROBING_RUNNER=docker but %s; probes stay unverified", reason)
        return RunnerChoice(NullRunner(), "none", f"docker runner unavailable: {reason}")
    return RunnerChoice(DockerRunner(), "docker", f"docker, image {IMAGE}, no network")


class DockerRunner:
    description = f"docker {IMAGE}, --network none"

    def __init__(self, *, max_concurrent: int = MAX_CONCURRENT) -> None:
        self._slots = asyncio.Semaphore(max_concurrent)

    async def run_cases(
        self, code: str, function_name: str, cases: list[tuple[str, tuple[Any, ...]]], limits: RunLimits
    ) -> list[CaseOutcome]:
        if limits.network:
            raise ValueError("the runner never allows network access")
        request = json.dumps(
            {
                "code": code,
                "function": function_name,
                "cases": [{"id": case_id, "args": list(args)} for case_id, args in cases],
                "seconds": limits.seconds,
            }
        )
        total = min(limits.seconds * len(cases) + STARTUP_ALLOWANCE_SECONDS, MAX_TOTAL_SECONDS)
        async with self._slots:
            code_, stdout, timed_out = await asyncio.to_thread(self._run, request, limits.memory_mb, total)
        return _outcomes(cases, code_, stdout, timed_out)

    def _run(self, request: str, memory_mb: int, timeout: float) -> tuple[int | None, str, bool]:
        name = f"{NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        command = [
            shutil.which("docker") or "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
            "--cpus", "1", "--pids-limit", "64",
            "--read-only", "--tmpfs", "/tmp:size=8m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            IMAGE, "python", "-I", "-c", HARNESS,
        ]
        clean = False
        try:
            done = subprocess.run(
                command, input=request, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout,
            )
            clean = True
            return done.returncode, done.stdout, False
        except subprocess.TimeoutExpired:
            return None, "", True
        finally:
            if not clean:
                _kill(name)


def _kill(name: str) -> None:
    """Stop and remove a container by name; safe to call when it is already gone."""
    for args in (("kill", name), ("rm", "-f", name)):
        try:
            _docker(*args, timeout=15)
        except (OSError, subprocess.SubprocessError):
            logger.warning("could not run docker %s for a probe container", args[0])


def _outcomes(
    cases: list[tuple[str, tuple[Any, ...]]], returncode: int | None, stdout: str, timed_out: bool
) -> list[CaseOutcome]:
    if timed_out:
        return [CaseOutcome(case_id, error="time limit (container killed)", hit_limit=True) for case_id, _ in cases]
    parsed = _parse(stdout)
    if parsed is None:
        killed = returncode == 137
        reason = "memory limit (container killed)" if killed else f"no result from the container (exit {returncode})"
        return [CaseOutcome(case_id, error=reason, hit_limit=killed) for case_id, _ in cases]
    by_id = {item.get("id"): item for item in parsed if isinstance(item, dict)}
    outcomes = []
    for case_id, _ in cases:
        item = by_id.get(case_id)
        if item is None:
            outcomes.append(CaseOutcome(case_id, error="no result for this case"))
        elif item.get("ok"):
            outcomes.append(CaseOutcome(case_id, returned=item.get("value")))
        else:
            outcomes.append(
                CaseOutcome(case_id, error=str(item.get("error") or "error"), hit_limit=bool(item.get("limit")))
            )
    return outcomes


def _parse(stdout: str) -> list | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                data = json.loads(line[len(MARKER):])
            except ValueError:
                return None
            return data if isinstance(data, list) else None
    return None
