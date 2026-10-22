"""Where Mapper results and probe ladders are kept this week.

Everything is held in memory and, if a path is given, also appended to a
JSON-lines event log that is replayed at start-up, so a restart keeps the
interviewer's approvals. There is no database: migrations are frozen.

Nothing stored here contains candidate code. Mapper results carry cue names and
tag ids, never the segment text, and no reasoning for candidate origin. The
only way a probe changes after generation is approval or rejection by a named
person; that never touches its verification status. Only a reference check changes
``verification``, and only to ``reference_checked`` or ``reference_failed``.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID

from backend.app.probing.knowledge.schemas import MapperResult
from backend.app.probing.probes.schema import Ladder, ProbePackage, ProbeReview, ReferenceCheck

logger = logging.getLogger(__name__)


APPROVAL_BLOCKED_REFERENCE = (
    "approval is disabled: the reference check failed, so a stated expected value may be wrong. "
    "You can still reject this probe."
)


class ApprovalError(ValueError):
    pass


class ProbeNotFound(KeyError):
    pass


class ProbeRepository(Protocol):
    def save_mapper_result(self, result: MapperResult) -> None: ...

    def get_mapper_result(self, segment_id: UUID) -> MapperResult | None: ...

    def save_ladder(self, segment_id: UUID, ladder: Ladder) -> None: ...

    def get_ladder(self, segment_id: UUID, tag_id: str) -> Ladder | None: ...

    def get_probe(self, probe_id: UUID) -> ProbePackage: ...

    def attach_review(self, probe_id: UUID, review: ProbeReview) -> ProbePackage: ...

    def attach_reference_check(self, probe_id: UUID, check: ReferenceCheck) -> ProbePackage: ...

    def approve(self, probe_id: UUID, by: str) -> ProbePackage: ...

    def reject(self, probe_id: UUID, by: str) -> ProbePackage: ...


class InMemoryProbeRepository:
    def __init__(self, log_path: Path | None = None) -> None:
        self._mapper: dict[UUID, MapperResult] = {}
        self._ladders: dict[tuple[UUID, str], Ladder] = {}
        self._where: dict[UUID, tuple[UUID, str]] = {}  # probe id -> ladder key
        self._log_path = log_path
        if log_path is not None:
            self._replay(log_path)

    # --- reads -----------------------------------------------------------------------

    def get_mapper_result(self, segment_id: UUID) -> MapperResult | None:
        return self._mapper.get(segment_id)

    def get_ladder(self, segment_id: UUID, tag_id: str) -> Ladder | None:
        return self._ladders.get((segment_id, tag_id))

    def get_probe(self, probe_id: UUID) -> ProbePackage:
        key = self._where.get(probe_id)
        if key is None:
            raise ProbeNotFound(str(probe_id))
        return next(p for p in self._ladders[key].probes if p.probe_id == probe_id)

    # --- writes ------------------------------------------------------------------------

    def save_mapper_result(self, result: MapperResult) -> None:
        self._mapper[result.segment_id] = result
        self._append("mapper_result", {"result": result.model_dump(mode="json")})

    def save_ladder(self, segment_id: UUID, ladder: Ladder) -> None:
        self._put_ladder(segment_id, ladder)
        self._append("ladder", {"segment_id": str(segment_id), "ladder": ladder.model_dump(mode="json")})

    def attach_review(self, probe_id: UUID, review: ProbeReview) -> ProbePackage:
        """Store an advisory review. Approval, verification and static checks are untouched."""
        updated = self._replace(probe_id, {"review": review})
        self._append("review", {"probe_id": str(probe_id), "review": review.model_dump(mode="json")})
        return updated

    def attach_reference_check(self, probe_id: UUID, check: ReferenceCheck) -> ProbePackage:
        """Store the result of running the reference solution; sets ``verification`` to its status."""
        updated = self._replace(probe_id, self._reference_changes(probe_id, check))
        self._append("reference_check", {"probe_id": str(probe_id), "check": check.model_dump(mode="json")})
        return updated

    def _reference_changes(self, probe_id: UUID, check: ReferenceCheck) -> dict:
        changes: dict = {"reference_check": check, "verification": check.status}
        if check.status == "reference_failed" and self.get_probe(probe_id).approval == "approved":
            changes.update(approval="pending", approved_by=None)  # an approval does not survive a failed check
        return changes

    def approve(self, probe_id: UUID, by: str) -> ProbePackage:
        return self._decide(probe_id, "approved", by)

    def reject(self, probe_id: UUID, by: str) -> ProbePackage:
        return self._decide(probe_id, "rejected", by)

    def _decide(self, probe_id: UUID, decision: str, by: str, *, log: bool = True) -> ProbePackage:
        if not by or not by.strip():
            raise ApprovalError("a named person must approve or reject a probe")
        probe = self.get_probe(probe_id)
        if decision == "approved" and not probe.static_checks.passed:
            raise ApprovalError("a probe that failed its static checks cannot be approved")
        if decision == "approved" and probe.verification == "reference_failed":
            raise ApprovalError(APPROVAL_BLOCKED_REFERENCE)
        updated = self._replace(probe_id, {"approval": decision, "approved_by": by.strip()})
        if log:
            self._append("decision", {"probe_id": str(probe_id), "decision": decision, "by": by.strip()})
        return updated

    def _replace(self, probe_id: UUID, changes: dict) -> ProbePackage:
        updated = self.get_probe(probe_id).model_copy(update=changes)
        segment_id, tag_id = self._where[probe_id]
        ladder = self._ladders[(segment_id, tag_id)]
        replaced = tuple(updated if p.probe_id == probe_id else p for p in ladder.probes)
        self._ladders[(segment_id, tag_id)] = ladder.model_copy(update={"probes": replaced})
        return updated

    def _put_ladder(self, segment_id: UUID, ladder: Ladder) -> None:
        self._ladders[(segment_id, ladder.tag_id)] = ladder
        for probe in ladder.probes:
            self._where[probe.probe_id] = (segment_id, ladder.tag_id)

    # --- the event log ---------------------------------------------------------------------

    def _append(self, event: str, payload: dict) -> None:
        if self._log_path is None:
            return
        row = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError:
            logger.warning("could not append to the probe event log")

    def _replay(self, path: Path) -> None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row["event"] == "mapper_result":
                    result = MapperResult.model_validate(row["result"])
                    self._mapper[result.segment_id] = result
                elif row["event"] == "ladder":
                    self._put_ladder(UUID(row["segment_id"]), Ladder.model_validate(row["ladder"]))
                elif row["event"] == "review":
                    self._replace(UUID(row["probe_id"]), {"review": ProbeReview.model_validate(row["review"])})
                elif row["event"] == "reference_check":
                    check = ReferenceCheck.model_validate(row["check"])
                    self._replace(UUID(row["probe_id"]), self._reference_changes(UUID(row["probe_id"]), check))
                elif row["event"] == "decision":
                    self._decide(UUID(row["probe_id"]), row["decision"], row["by"], log=False)
            except (ValueError, KeyError, ApprovalError, ProbeNotFound):
                logger.warning("skipped an unreadable line %d of the probe event log", number)
