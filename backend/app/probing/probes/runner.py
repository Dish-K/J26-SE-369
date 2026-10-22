"""The contract for running code, and the runner used when none is configured.

Nothing the model writes is executed unless ``PROBING_RUNNER=docker`` is set and
Docker is ready (see ``docker_runner``); then only a probe's generated reference
solution is run, never candidate code. Without that, ``NullRunner`` refuses, so a
probe can never be marked checked by accident.

Run contract: input is the code, the function name, the arguments of each case and
limits (seconds per case, memory, no network); output is, for each case, the
returned value or an error, plus whether a limit was hit.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class RunnerUnavailable(RuntimeError):
    """No code runner is configured, so nothing was executed."""


@dataclass(frozen=True)
class RunLimits:
    seconds: float = 2.0
    memory_mb: int = 128
    network: bool = False


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    returned: Any = None
    error: str | None = None
    hit_limit: bool = False


class CodeRunner(Protocol):
    async def run_cases(
        self, code: str, function_name: str, cases: list[tuple[str, tuple[Any, ...]]], limits: RunLimits
    ) -> list[CaseOutcome]: ...


class NullRunner:
    async def run_cases(self, code, function_name, cases, limits) -> list[CaseOutcome]:
        raise RunnerUnavailable("no code runner is configured; generated code is never executed")
