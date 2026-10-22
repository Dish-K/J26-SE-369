"""The reference check: run a probe's generated reference against the model's own cases.

``reference_checked`` means the reference reproduces every stated expected value;
``reference_failed`` means at least one case did not (a wrong expected value, an
error, or a limit hit). Both the reference and the expected values came from the
same model, so a pass is **not** an independent verification and is never called
``cross_checked`` or ``verified``. Only ``probe.reference_solution`` is ever run:
this module has no way to run anything else, and candidate code never reaches it.

The probe is returned unchanged (apart from a note) if its static checks failed,
because code that failed them is not run.
"""

import math
from datetime import date
from typing import Any

from backend.app.probing.probes.runner import CaseOutcome, CodeRunner, RunLimits
from backend.app.probing.probes.schema import (
    REFERENCE_CHECKED_LABEL,
    REFERENCE_FAILED_LABEL,
    CaseCheck,
    ProbePackage,
    ReferenceCheck,
)

REL_TOLERANCE, ABS_TOLERANCE = 1e-6, 1e-9


async def reference_check(probe: ProbePackage, runner: CodeRunner, *, today: date | None = None) -> ReferenceCheck:
    """Run the reference on the probe's cases. Raises ``RunnerUnavailable`` with a ``NullRunner``."""
    cases = [(case.case_id, tuple(case.args)) for case in probe.cases]
    outcomes = await runner.run_cases(probe.reference_solution, probe.function_name, cases, RunLimits())
    by_id = {o.case_id: o for o in outcomes}
    checks = tuple(_check_case(case.case_id, case.expected, by_id.get(case.case_id), probe.checker) for case in probe.cases)
    passed = bool(checks) and all(c.passed for c in checks)
    return ReferenceCheck(
        status="reference_checked" if passed else "reference_failed",
        label=REFERENCE_CHECKED_LABEL if passed else REFERENCE_FAILED_LABEL,
        runner=getattr(runner, "description", type(runner).__name__),
        checker=probe.checker,
        date=(today or date.today()).isoformat(),
        cases=checks,
    )


async def verify_probe(probe: ProbePackage, runner: CodeRunner) -> ProbePackage:
    """The probe with ``reference_check`` and ``verification`` set; unchanged if its static checks failed."""
    if not probe.static_checks.passed:
        return probe.model_copy(update={"notes": (*probe.notes, "not reference-checked: static checks failed")})
    result = await reference_check(probe, runner)
    return probe.model_copy(update={"reference_check": result, "verification": result.status})


def _check_case(case_id: str, expected: Any, outcome: CaseOutcome | None, checker: str) -> CaseCheck:
    if outcome is None:
        return CaseCheck(case_id=case_id, passed=False, expected=expected, error="no result for this case")
    if outcome.error is not None:
        return CaseCheck(case_id=case_id, passed=False, expected=expected, error=outcome.error, hit_limit=outcome.hit_limit)
    return CaseCheck(case_id=case_id, passed=matches(checker, expected, outcome.returned), expected=expected, got=outcome.returned)


def matches(checker: str, expected: Any, got: Any) -> bool:
    if checker == "unordered":
        return _unordered(expected, got)
    if checker == "float_tolerance":
        return _close(expected, got)
    return _same(expected, got)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _same(a: Any, b: Any) -> bool:
    """Exact equality on JSON values: a bool is not a number, 3 equals 3.0, tuples are lists."""
    if _is_number(a) and _is_number(b):
        return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _unordered(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        remaining = list(b)
        for item in a:
            for index, other in enumerate(remaining):
                if _unordered(item, other):
                    del remaining[index]
                    break
            else:
                return False
        return True
    return _same(a, b)


def _close(a: Any, b: Any) -> bool:
    if _is_number(a) and _is_number(b):
        return math.isclose(a, b, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return _same(a, b)
