"""Response cache keyed by a hash of the request.

Only the model's answer and token counts are stored, never the input text. The
gateway uses it only for data that is not candidate code.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.probing.ai.client import LLMResult


def cache_key(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    json_schema: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        [provider, model, system, user, temperature, max_tokens, json_schema],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """In memory, and on disk too when a directory is given."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._memory: dict[str, LLMResult] = {}

    def get(self, key: str) -> LLMResult | None:
        if key in self._memory:
            return self._memory[key]
        if self._directory is None:
            return None
        path = self._directory / f"{key}.json"
        try:
            result = LLMResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # missing or damaged entries are simply a miss
        self._memory[key] = result
        return result

    def put(self, key: str, result: LLMResult) -> None:
        self._memory[key] = result
        if self._directory is None:
            return
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            (self._directory / f"{key}.json").write_text(
                result.model_dump_json(), encoding="utf-8"
            )
        except OSError:
            pass  # a cache that cannot be written must not fail the call
