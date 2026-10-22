"""Recorded reference-check results, so a demo needs no Docker.

A result is keyed by everything that decides it: the reference code, the function,
the cases, the checker, the image and the harness that ran it. Change any of these
and the old record is simply not found. Files hold probe code and numbers the model
wrote, never candidate code.
"""

import hashlib
import json
import logging
from pathlib import Path

from backend.app.probing.probes.docker_runner import HARNESS, IMAGE
from backend.app.probing.probes.schema import ProbePackage, ReferenceCheck

logger = logging.getLogger(__name__)


def reference_key(probe: ProbePackage) -> str:
    payload = json.dumps(
        [
            probe.function_name,
            probe.reference_solution,
            [[c.case_id, list(c.args), c.expected] for c in probe.cases],
            probe.checker,
            IMAGE,
            hashlib.sha256(HARNESS.encode()).hexdigest(),
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ReferenceCheckCache:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def get(self, probe: ProbePackage) -> ReferenceCheck | None:
        path = self._directory / f"{reference_key(probe)}.json"
        try:
            return ReferenceCheck.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, probe: ProbePackage, check: ReferenceCheck) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            (self._directory / f"{reference_key(probe)}.json").write_text(
                check.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("could not write a reference-check record")
