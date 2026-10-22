"""The difficulty reviewer: a second model's advisory opinion on a probe.

Advisory only. A review never changes a probe's approval, verification or static
checks; it is shown to the interviewer next to the probe. It reads the probe
(question, cases, reference) and reasons about it; nothing is executed. It sees
no candidate code, because probes hold none.

The reviewer is behind ``PROBING_REVIEWER_ENABLED``. It tries a chain of models in
order. A model is passed over when its call fails or its output is invalid after
one retry, and the reason is kept in the review's provenance. A model from the
same provider as the generator is passed over unless
``PROBING_ALLOW_SAME_FAMILY_REVIEWER`` is set, and is then labelled a weaker check.
If no model answers there is no review.
"""

import json
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from backend.app.probing.ai.errors import LLMError
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.prompts import fence, load_prompt, render
from backend.app.probing.ai.provenance import ProvenanceBuilder
from backend.app.probing.probes.schema import ProbePackage, ProbeReview

PROMPTS_DIR = Path(__file__).with_name("prompts")
PROMPT_NAME = "probe_review_v1"
TEMPERATURE = 0.0
MAX_TOKENS = 1500  # a reasoning model's thinking counts against this and against the provider limits

# The prompt holds only a probe, which holds no candidate data.
ORIGIN = "fixture"

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_DIFFICULTIES = {"easier", "about_right", "harder"}


class ProbeReviewer:
    def __init__(
        self,
        gateway: LLMGateway | None = None,
        *,
        model: str | None = None,
        chain: Sequence[tuple[LLMGateway, str]] | None = None,
        enabled: bool = False,
        allow_same_family: bool = False,
        today: date | None = None,
    ) -> None:
        """Either one ``gateway`` and ``model``, or a ``chain`` of (gateway, model) tried in order."""
        self._chain = tuple(chain) if chain else ((gateway, model),) if gateway and model else ()
        if not self._chain:
            raise ValueError("a reviewer needs a gateway and model, or a chain")
        self.enabled = enabled
        self._allow_same_family = allow_same_family
        self._today = today
        self._system, self._user = load_prompt(PROMPTS_DIR, PROMPT_NAME)

    async def review(self, probe: ProbePackage) -> tuple[ProbeReview | None, str]:
        """The review, or ``None`` and a short fixed-text reason."""
        if not self.enabled:
            return None, "the reviewer is switched off (PROBING_REVIEWER_ENABLED)"
        user = self._prompt(probe)
        skipped: list[str] = []
        for gateway, model in self._chain:
            same_family = gateway.client.family == probe.provenance.family
            if same_family and not self._allow_same_family:
                skipped.append(
                    f"{model}: a reviewer from the same provider as the generator is not allowed "
                    "(PROBING_ALLOW_SAME_FAMILY_REVIEWER)"
                )
                continue
            review, reason = await self._ask(gateway, model, user, same_family)
            if review is not None:
                return review.model_copy(
                    update={"provenance": review.provenance.model_copy(update={"skipped": tuple(skipped)})}
                ), ""
            skipped.append(f"{model}: {reason}")
        return None, "not reviewed: " + "; ".join(skipped)

    async def _ask(self, gateway: LLMGateway, model: str, user: str, same_family: bool):
        builder = ProvenanceBuilder(
            PROMPT_NAME, provider=gateway.client.provider, model=model, family=gateway.client.family
        )
        for _ in range(2):
            try:
                result = await gateway.complete(
                    system=self._system,
                    user=user,
                    model=model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    origin=ORIGIN,
                )
            except LLMError as exc:
                return None, type(exc).__name__
            builder.add(
                provider=result.provider,
                model=result.model,
                family=result.family,
                temperature=TEMPERATURE,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cache_hit=result.cache_hit,
            )
            parsed = _parse(result.text)
            if parsed is not None:
                label = f"Advisory difficulty review by {result.model}"
                if same_family:
                    label += " (weaker check: same provider as the generator)"
                return (
                    ProbeReview(
                        weaker_check=same_family,
                        label=label,
                        provenance=builder.build(today=self._today),
                        **parsed,
                    ),
                    "",
                )
        return None, "output invalid after one retry"

    def _prompt(self, probe: ProbePackage) -> str:
        cases = "\n".join(
            f"- {c.case_id}: args={json.dumps(list(c.args))} expected={json.dumps(c.expected)}"
            for c in probe.cases
        )
        return render(
            self._user,
            minutes=str(probe.target_minutes),
            tag_name=probe.tag,
            question=probe.question,
            signature=probe.signature,
            cases=cases,
            reference=fence(probe.reference_solution),
        )


def _parse(text: str) -> dict | None:
    """Validated fields for ``ProbeReview``, or ``None``. The output is data: extra keys are dropped."""
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    minutes, difficulty = data.get("estimated_minutes"), data.get("difficulty")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 240:
        return None
    if difficulty not in _DIFFICULTIES:
        return None
    return {
        "estimated_minutes": minutes,
        "difficulty": difficulty,
        "concerns": _strings(data.get("concerns"), 5),
        "case_problems": _strings(data.get("case_problems"), 6),
    }


def _strings(value, limit: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(v.strip()[:400] for v in value if isinstance(v, str) and v.strip())[:limit]
