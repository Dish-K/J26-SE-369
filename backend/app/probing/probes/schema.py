"""The probe package: one question for one knowledge area at one level.

Everything the model wrote is data to be reviewed. The package never holds
candidate code (the generator is given only a tag id and a level). Test cases
and the reference solution come from the model and are ``unverified``: nothing
here has been run, except by the optional reference check (see ``ReferenceCheck``).
The ``oracle`` slot stays empty: it is for expected outputs checked independently. Verification and approval are separate fields: a
person approves a probe by reading it, which says nothing about whether it runs.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.probing.ai.provenance import Provenance

PROBE_SCHEMA_VERSION = 1

Level = Literal[5, 10, 20]
LEVELS: tuple[Level, ...] = (5, 10, 20)
DEFAULT_LEVEL: Level = 10

Checker = Literal["exact", "unordered", "float_tolerance"]
# reference_checked / reference_failed: the generated reference was run against the model's own
# expected values. That is weaker than an independent second solution, so it is not "cross_checked".
Verification = Literal["unverified", "reference_checked", "reference_failed", "verified", "failed"]
Approval = Literal["pending", "approved", "rejected"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProbeCase(_Frozen):
    """One test case written by the model. Expected values are the model's claim, not a fact."""

    case_id: str
    args: tuple[Any, ...]
    expected: Any
    note: str = ""
    source: Literal["llm"] = "llm"


class CheckResult(_Frozen):
    name: str
    passed: bool
    detail: str = ""


class StaticChecks(_Frozen):
    """Results of checks that read the probe without running it."""

    passed: bool
    checks: tuple[CheckResult, ...]


REFERENCE_CHECKED_LABEL = "Reference-checked: not independently verified."
REFERENCE_FAILED_LABEL = "Reference check failed: the reference does not reproduce the stated expected values."


class CaseCheck(_Frozen):
    case_id: str
    passed: bool
    expected: Any = None
    got: Any = None
    error: str | None = None
    hit_limit: bool = False


class ReferenceCheck(_Frozen):
    """The generated reference solution, run against the model's own cases in a sandbox.

    ``reference_checked`` means the reference reproduces every stated expected value. It does not
    mean the answer is right: both came from the same model. Only generated reference solutions
    are ever run, never candidate code.
    """

    status: Literal["reference_checked", "reference_failed"]
    label: str
    runner: str
    checker: Checker
    date: str
    cases: tuple[CaseCheck, ...]
    served: Literal["live", "recorded"] = "live"  # "recorded" when read from the demo cache, not run now


class ProbeReview(_Frozen):
    """A second model's opinion on difficulty. Advisory only: it changes nothing else on the probe.

    ``weaker_check`` is true when the reviewer comes from the same provider as the generator,
    which makes it a weaker check than a model from another provider.
    """

    advisory_only: Literal[True] = True
    weaker_check: bool
    label: str
    estimated_minutes: int = Field(ge=1, le=240)
    difficulty: Literal["easier", "about_right", "harder"]
    concerns: tuple[str, ...] = Field(default=(), max_length=5)
    case_problems: tuple[str, ...] = Field(default=(), max_length=6)
    provenance: Provenance


class ProbePackage(_Frozen):
    schema_version: Literal[1] = PROBE_SCHEMA_VERSION
    probe_id: UUID
    tag_id: str
    tag: str
    taxonomy_version: str
    target_minutes: Level
    question: str
    signature: str
    function_name: str
    cases: tuple[ProbeCase, ...]
    checker: Checker = "exact"
    reference_solution: str
    oracle: dict[str, Any] = Field(default_factory=dict)  # empty until a runner fills it
    static_checks: StaticChecks
    provenance: Provenance
    observed_minutes: float | None = None  # filled in only if someone times a real attempt
    verification: Verification = "unverified"
    approval: Approval = "pending"
    approved_by: str | None = None
    review: ProbeReview | None = None  # advisory difficulty review, if one was run
    reference_check: ReferenceCheck | None = None  # set only by a reference check; see ``verify``
    notes: tuple[str, ...] = ()


class Ladder(_Frozen):
    """The 5 / 10 / 20 minute probes for one tag. A level that failed its checks is absent."""

    tag_id: str
    tag: str
    taxonomy_version: str
    default_level: Level = DEFAULT_LEVEL
    probes: tuple[ProbePackage, ...]
    notes: tuple[str, ...] = ()
