"""The Knowledge Mapper: tag a segment with up to three knowledge areas.

Rules gather evidence only. The model decides the tags, in two calls (free-text
reasoning, then structured JSON), and the cross-check compares the two. Tags are
evidence about the code, never proof of anything about the candidate.

Any failure of the model path (consent refused, rate limited, bad output) falls
back to a rule-only result with a note. It does not raise to the caller.
"""

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from backend.app.probing.ai.consent import DataOrigin
from backend.app.probing.ai.errors import ConsentNotConfirmed, LLMError, RateLimited, RequestTooLarge
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.ai.prompts import fence, load_prompt, render
from backend.app.probing.ai.provenance import ProvenanceBuilder
from backend.app.probing.knowledge.crosscheck import cross_check, rule_only_ranking
from backend.app.probing.knowledge.cues import CueHit, evidence_by_tag, extract_cues
from backend.app.probing.knowledge.schemas import MappedTag, MapperResult
from backend.app.probing.knowledge.tags import Taxonomy, default_taxonomy
from backend.app.probing.schemas import Segment

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).with_name("prompts")
REASON_PROMPT = "mapper_reason_v1"
STRUCTURE_PROMPT = "mapper_structure_v1"
PROMPT_VERSION = f"{REASON_PROMPT}+{STRUCTURE_PROMPT}"

REASON_TEMPERATURE = 0.2
STRUCTURE_TEMPERATURE = 0.0
REASON_MAX_TOKENS = 1024
STRUCTURE_MAX_TOKENS = 512
MAX_TAGS = 3

_RETRY_HINT = (
    "\n\nYour previous answer was not valid. Answer again with only a JSON object whose "
    "tag ids come from the allowed list."
)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

# Fixed, data-free reasons recorded when the model path is not used.
_FALLBACK_NOTES = {
    ConsentNotConfirmed: "rule_only: consent for candidate code to a hosted model is not confirmed",
    RateLimited: "rule_only: the provider is rate limited",
    RequestTooLarge: "rule_only: the request is too large for the token cap",
}


class _InvalidOutput(ValueError):
    pass


class KnowledgeMapper:
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
        self._reason_system, self._reason_user = load_prompt(PROMPTS_DIR, REASON_PROMPT)
        self._structure_system, self._structure_user = load_prompt(PROMPTS_DIR, STRUCTURE_PROMPT)

    async def map_segment(self, segment: Segment, *, data_origin: DataOrigin) -> MapperResult:
        entry = segment.entry_function.name if segment.entry_function else None
        hits = extract_cues(segment.code, entry_function=entry)
        builder = ProvenanceBuilder(
            PROMPT_VERSION,
            provider=self._gateway.client.provider,
            model=self._model,
            family=self._gateway.client.family,
        )
        try:
            return await self._with_model(segment, data_origin, hits, entry, builder)
        except _InvalidOutput:
            note = "rule_only: the model's output was invalid after one retry"
        except LLMError as exc:
            note = next(
                (text for kind, text in _FALLBACK_NOTES.items() if isinstance(exc, kind)),
                "rule_only: the model call failed",
            )
            logger.warning("Mapper fell back to rule-only for segment %s (%s)", segment.segment_id, type(exc).__name__)
        return self._rule_only(segment, data_origin, hits, builder, note)

    # --- model path ---------------------------------------------------------------

    async def _with_model(self, segment, origin, hits, entry, builder) -> MapperResult:
        reasoning = await self._call(
            builder,
            origin,
            system=self._reason_system,
            user=self._reason_prompt(segment, hits, entry),
            temperature=REASON_TEMPERATURE,
            max_tokens=REASON_MAX_TOKENS,
        )
        chosen = await self._structure(reasoning, origin, builder)

        check = cross_check([tag_id for tag_id, _ in chosen], hits)
        evidence = evidence_by_tag(hits)
        tags = tuple(
            self._mapped_tag(rank, tag_id, check.tag_status[tag_id], evidence, why)
            for rank, (tag_id, why) in enumerate(chosen, start=1)
        )
        return MapperResult(
            taxonomy_version=self._taxonomy.taxonomy_version,
            segment_id=segment.segment_id,
            task_id=segment.task_id,
            data_origin=origin,
            tags=tags,
            ignored_strong_cues=check.ignored_strong_cues,
            overall_status=check.overall_status,
            rule_evidence=evidence,
            # The reasoning may quote candidate code, so it is not kept for that origin.
            reasoning=reasoning if origin != "candidate" else None,
            provenance=builder.build(today=self._today),
            notes=(),
        )

    async def _structure(self, reasoning: str, origin, builder) -> list[tuple[str, str]]:
        enabled = self._taxonomy.enabled_ids()
        user = render(self._structure_user, tag_ids=", ".join(enabled), analysis=reasoning)
        schema = _json_schema(enabled)
        for attempt in range(2):
            text = await self._call(
                builder,
                origin,
                system=self._structure_system,
                user=user + (_RETRY_HINT if attempt else ""),
                temperature=STRUCTURE_TEMPERATURE,
                max_tokens=STRUCTURE_MAX_TOKENS,
                json_schema=schema,
            )
            chosen = _parse_tags(text, set(enabled))
            if chosen:
                return chosen
        raise _InvalidOutput

    async def _call(self, builder, origin, *, system, user, temperature, max_tokens, json_schema=None) -> str:
        result = await self._gateway.complete(
            system=system,
            user=user,
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            origin=origin,
            json_schema=json_schema,
        )
        builder.add(
            provider=result.provider,
            model=result.model,
            family=result.family,
            temperature=temperature,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cache_hit=result.cache_hit,
        )
        return result.text

    def _reason_prompt(self, segment: Segment, hits: list[CueHit], entry: str | None) -> str:
        return render(
            self._reason_user,
            code_block=fence(segment.code),
            context="\n".join(f"- {line}" for line in segment.context) or "(none: top level of the function body or file)",
            entry=entry or "unknown",
            cues=_format_cues(hits),
            tags="\n".join(
                f"- {tag.id} | {tag.name}: {tag.definition}" for tag in self._taxonomy.enabled()
            ),
        )

    # --- rule-only fallback ------------------------------------------------------------

    def _rule_only(self, segment, origin, hits, builder, note) -> MapperResult:
        evidence = evidence_by_tag(hits)
        ranked = rule_only_ranking(hits, self._taxonomy.position)
        tags = tuple(
            self._mapped_tag(rank, tag_id, "unchecked", evidence, "rule evidence only; no model was used")
            for rank, tag_id in enumerate(ranked, start=1)
        )
        notes = (note,) if hits else (note, "no rule cues were found")
        return MapperResult(
            taxonomy_version=self._taxonomy.taxonomy_version,
            segment_id=segment.segment_id,
            task_id=segment.task_id,
            data_origin=origin,
            tags=tags,
            ignored_strong_cues=(),
            overall_status="rule_only",
            rule_evidence=evidence,
            reasoning=None,
            provenance=builder.build(today=self._today),
            notes=notes,
        )

    def _mapped_tag(self, rank, tag_id, status, evidence, why) -> MappedTag:
        tag = self._taxonomy.get(tag_id)
        return MappedTag(
            rank=rank,
            tag_id=tag_id,
            tag=tag.name,
            status=status,
            cues=evidence.get(tag_id, ()),
            anchor=dict(tag.anchor),
            anchor_note=tag.anchor_note,
            why=why,
        )


def _format_cues(hits: list[CueHit]) -> str:
    if not hits:
        return "- (none found)"
    lines = []
    for hit in hits:
        targets = ", ".join(f"{s.tag_id} ({s.strength})" for s in hit.supports)
        lines.append(f"- {hit.text} [supports: {targets}]")
    return "\n".join(lines)


def _json_schema(tag_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "maxItems": MAX_TAGS,
                "items": {
                    "type": "object",
                    "properties": {
                        "tag_id": {"type": "string", "enum": list(tag_ids)},
                        "why": {"type": "string"},
                    },
                    "required": ["tag_id", "why"],
                },
            }
        },
        "required": ["tags"],
    }


def _parse_tags(text: str, allowed: set[str]) -> list[tuple[str, str]]:
    """Valid, de-duplicated ``(tag_id, why)`` pairs, at most three, in order.

    Anything unexpected (unknown ids, extra keys, wrong types) is dropped, never
    trusted: the model's output is data.
    """
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
    except ValueError:
        return []
    items = data.get("tags") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    chosen: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tag_id, why = item.get("tag_id"), item.get("why")
        if not isinstance(tag_id, str) or tag_id not in allowed or any(tag_id == t for t, _ in chosen):
            continue
        chosen.append((tag_id, why.strip() if isinstance(why, str) else ""))
        if len(chosen) == MAX_TAGS:
            break
    return chosen
