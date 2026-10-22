"""The Probe Generator: a ladder of questions for one knowledge area.

It receives a tag id and levels, never candidate code, so its prompts hold only
taxonomy text and are safe to cache. What comes back is untrusted: it is parsed
as data, checked statically (``validation.py``) and never executed. A level whose
probe still fails the checks after one retry is left out of the ladder with a note.
"""

import json
import re
from datetime import date
from pathlib import Path
from uuid import uuid4

from backend.app.probing.ai.errors import LLMError
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.prompts import load_prompt, render
from backend.app.probing.ai.provenance import ProvenanceBuilder
from backend.app.probing.knowledge.tags import Taxonomy, default_taxonomy
from backend.app.probing.probes.schema import DEFAULT_LEVEL, LEVELS, Ladder, Level, ProbePackage
from backend.app.probing.probes.validation import check_draft

PROMPTS_DIR = Path(__file__).with_name("prompts")
PROMPT_NAME = "probe_generate_v1"
TEMPERATURE = 0.4
MAX_TOKENS = 3000

# The prompt holds only taxonomy text, so the request carries no candidate data.
ORIGIN = "fixture"

# Added after a recorded 10-minute probe asked for an optimum that a heap only approximates,
# with an expected value its own reference contradicted. Applied to the 10 and 20 minute levels
# (the 5 minute prompt is unchanged, so its saved answer still applies).
CONSTRAINT_RULES = (
    " The answer must be unique and exactly computable: for every input there is exactly one correct "
    "result, and the reference computes it exactly. Do not pick a problem where the data structure only "
    "gives an approximation or a greedy heuristic (for example scheduling or other optimisation problems "
    "that are hard in general). Every example case must obey every constraint the question states: if the "
    "question says the lists are non-empty or k is valid, no case may break that."
)

GUIDANCE: dict[int, str] = {
    5: "A warm-up: one idea, a function of at most about 10 lines, no tricky edge cases.",
    10: "A standard exercise: the idea plus one extra step, a function of about 10 to 20 lines."
    + CONSTRAINT_RULES,
    20: "A fuller problem: the idea inside a problem with several edge cases, about 20 to 35 lines."
    + CONSTRAINT_RULES,
}

PROMPT_VERSIONS: dict[int, str] = {5: PROMPT_NAME, 10: f"{PROMPT_NAME}.1", 20: f"{PROMPT_NAME}.1"}

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class ProbeGenerator:
    def __init__(
        self,
        gateway: LLMGateway,
        *,
        model: str,
        taxonomy: Taxonomy | None = None,
        today: date | None = None,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._taxonomy = taxonomy or default_taxonomy()
        self._today = today
        self._system, self._user = load_prompt(PROMPTS_DIR, PROMPT_NAME)

    async def generate_ladder(self, tag_id: str, levels: tuple[Level, ...] = LEVELS) -> Ladder:
        tag = self._taxonomy.get(tag_id)
        if tag is None or not tag.enabled:
            raise ValueError(f"unknown or disabled tag {tag_id!r}")
        probes: list[ProbePackage] = []
        notes: list[str] = []
        for level in levels:
            probe, note = await self.generate_probe(tag_id, level)
            if probe is not None:
                probes.append(probe)
            if note:
                notes.append(f"{level} min: {note}")
        return Ladder(
            tag_id=tag.id,
            tag=tag.name,
            taxonomy_version=self._taxonomy.taxonomy_version,
            default_level=DEFAULT_LEVEL,
            probes=tuple(probes),
            notes=tuple(notes),
        )

    async def generate_probe(self, tag_id: str, level: Level) -> tuple[ProbePackage | None, str]:
        """One probe, or ``None`` and a short reason (fixed text, no model output)."""
        tag = self._taxonomy.get(tag_id)
        builder = ProvenanceBuilder(
            PROMPT_VERSIONS[level],
            provider=self._gateway.client.provider,
            model=self._model,
            family=self._gateway.client.family,
        )
        hint = ""
        reason = "the model call failed"
        for attempt in range(2):
            try:
                result = await self._gateway.complete(
                    system=self._system,
                    user=self._prompt(tag, level, hint),
                    model=self._model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    origin=ORIGIN,
                )
            except LLMError as exc:
                return None, f"not generated: {type(exc).__name__}"
            builder.add(
                provider=result.provider,
                model=result.model,
                family=result.family,
                temperature=TEMPERATURE,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cache_hit=result.cache_hit,
            )
            draft = _parse_draft(result.text)
            if draft is None:
                reason, hint = "the model's output was not a JSON object", _hint(["the answer was not valid JSON"])
                continue
            checked = check_draft(draft)
            if checked.checks.passed:
                return (
                    ProbePackage(
                        probe_id=uuid4(),
                        tag_id=tag.id,
                        tag=tag.name,
                        taxonomy_version=self._taxonomy.taxonomy_version,
                        target_minutes=level,
                        question=draft["question"].strip(),
                        signature=draft["signature"].strip(),
                        function_name=draft["function_name"],
                        cases=checked.cases,
                        checker=draft.get("checker", "exact"),
                        reference_solution=draft["reference_solution"],
                        static_checks=checked.checks,
                        provenance=builder.build(today=self._today),
                        notes=("generated; tests were not executed",),
                    ),
                    "",
                )
            failed = [f"{c.name}: {c.detail}" for c in checked.checks.checks if not c.passed]
            reason, hint = "failed static checks: " + ", ".join(c.name for c in checked.checks.checks if not c.passed), _hint(failed)
        return None, f"dropped after one retry; {reason}"

    def _prompt(self, tag, level: int, hint: str) -> str:
        return render(
            self._user,
            tag_id=tag.id,
            minutes=str(level),
            tag_name=tag.name,
            area=tag.area,
            definition=tag.definition,
            guidance=GUIDANCE[level],
            retry_hint=hint,
        )


def _hint(problems: list[str]) -> str:
    return "\nYour previous answer had these problems; fix them:\n" + "\n".join(f"- {p}" for p in problems) + "\n"


def _parse_draft(text: str) -> dict | None:
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
