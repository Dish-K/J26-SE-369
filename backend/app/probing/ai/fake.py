"""A deterministic model client for tests and offline demos.

Everything it returns is labelled ``provider="fake"`` and ``model="dummy-v0"``.
It sends nothing anywhere, so the consent guard lets any data origin through.

Scripted use: pass ``script`` (strings returned in order; an exception instance
is raised instead; a callable is called with ``(system, user, json_schema)``).
Without a script it answers the Knowledge Mapper prompts from the rule cues in
the prompt, so the whole pipeline can run with no network.
"""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from backend.app.probing.ai.client import LLMResult

FAKE_PROVIDER = "fake"
FAKE_MODEL = "dummy-v0"

ScriptItem = str | BaseException | Callable[[str, str, dict[str, Any] | None], str]

_STRENGTH_ORDER = {"strong": 0, "medium": 1, "weak": 2}
_SUPPORT_LINE = re.compile(r"\[supports:\s*([^\]]+)\]")
_SUPPORT_ITEM = re.compile(r"([a-z0-9_]+)\s*\((strong|medium|weak)\)")
_RANKED = re.compile(r"^RANKED:\s*(.*)$", re.MULTILINE)


class FakeClient:
    provider = FAKE_PROVIDER
    family = "fake"
    hosted = False

    def __init__(self, script: Sequence[ScriptItem] | None = None) -> None:
        self._script = list(script) if script is not None else None
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str = FAKE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "temperature": temperature,
                "json_schema": json_schema,
            }
        )
        item = self._next_item(system, user, json_schema)
        if isinstance(item, BaseException):
            raise item
        text = item(system, user, json_schema) if callable(item) else item
        return LLMResult(
            text=text,
            tokens_in=-(-(len(system) + len(user)) // 4),
            tokens_out=-(-len(text) // 4),
            model=FAKE_MODEL,
            provider=FAKE_PROVIDER,
            family=self.family,
            latency_ms=0,
        )

    def _next_item(self, system, user, json_schema) -> ScriptItem:
        if self._script is None:
            return _default_reply
        if not self._script:
            raise AssertionError("FakeClient script is exhausted")
        return self._script.pop(0)


def _default_reply(system: str, user: str, json_schema: dict[str, Any] | None) -> str:
    if json_schema is not None:
        return _structured_reply(user, json_schema)
    return _reasoning_reply(user)


def _reasoning_reply(user: str) -> str:
    """Rank the tags the prompt's cues point at; otherwise fall back to arrays."""
    best: dict[str, int] = {}
    for line in _SUPPORT_LINE.findall(user):
        for tag_id, strength in _SUPPORT_ITEM.findall(line):
            rank = _STRENGTH_ORDER[strength]
            best[tag_id] = min(best.get(tag_id, 9), rank)
    ranked = sorted(best, key=lambda tag_id: (best[tag_id], tag_id))[:3] or ["array_string"]
    return (
        "FAKE ANALYSIS (dummy-v0, not a real model). The tags below follow the "
        "rule cues only.\nRANKED: " + ", ".join(ranked)
    )


def _structured_reply(user: str, json_schema: dict[str, Any]) -> str:
    match = _RANKED.search(user)
    ids = [part.strip() for part in match.group(1).split(",") if part.strip()] if match else []
    allowed = (
        json_schema.get("properties", {})
        .get("tags", {})
        .get("items", {})
        .get("properties", {})
        .get("tag_id", {})
        .get("enum")
    )
    if allowed is not None:
        ids = [tag_id for tag_id in ids if tag_id in allowed]
    return json.dumps(
        {"tags": [{"tag_id": tag_id, "why": "fake model: follows the rule cues"} for tag_id in ids]}
    )
