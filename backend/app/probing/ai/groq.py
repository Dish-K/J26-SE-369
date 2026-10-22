"""Groq client over the OpenAI-compatible chat endpoint (``httpx`` only, no vendor SDK).

The key is read from ``GROQ_API_KEY`` when the client is built and is sent in the
``Authorization`` header. It is never logged, returned, or put in an error
message or URL.
"""

import os
import re
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from backend.app.probing.ai.client import LLMResult
from backend.app.probing.ai.errors import LLMError, ProviderHTTPError
from backend.app.probing.ai.gemini import MissingApiKey

BASE_URL = "https://api.groq.com/openai/v1"
KEY_VARIABLE = "GROQ_API_KEY"
TIMEOUT_SECONDS = 60.0

#: A retry-after longer than this means a daily limit, so waiting in-process is pointless.
LONG_WAIT_SECONDS = 120.0

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def read_key(env=None) -> str:
    if env is None:
        load_dotenv(override=False)
        env = os.environ
    key = (env.get(KEY_VARIABLE) or "").strip()
    if not key:
        raise MissingApiKey(f"{KEY_VARIABLE} is not set; add it to .env")
    return key


class GroqClient:
    provider = "groq"
    family = "groq"
    hosted = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http: httpx.AsyncClient | None = None,
        timeout: float = TIMEOUT_SECONDS,
        reasoning: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """``reasoning`` maps a model id to extra body fields (reasoning tokens count toward the limits)."""
        self._key = api_key or read_key()
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._reasoning = reasoning or {}

    def __repr__(self) -> str:  # never show the key
        return "GroqClient()"

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            **self._reasoning.get(model, {}),
        }
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        try:
            response = await self._http.post(
                f"{BASE_URL}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._key}"},
            )
        except httpx.HTTPError as exc:
            # Connection problems are treated like a 503 so the gateway retries them.
            raise ProviderHTTPError(503, f"network error ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise _http_error(response)
        try:
            data = response.json()
        except ValueError:
            raise LLMError("Groq returned a reply that is not JSON") from None
        return _parse(data, model, int((time.monotonic() - started) * 1000))

    async def aclose(self) -> None:
        await self._http.aclose()


async def list_models(*, api_key: str | None = None) -> list[str]:
    """The model ids this key can use."""
    key = api_key or read_key()
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as http:
        response = await http.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {key}"})
    if response.status_code != 200:
        raise _http_error(response)
    return sorted(str(item["id"]) for item in response.json().get("data", []) if "id" in item)


def _http_error(response: httpx.Response) -> ProviderHTTPError:
    message = ""
    try:
        message = str((response.json().get("error") or {}).get("message", ""))
    except (ValueError, AttributeError):
        pass
    status = response.status_code
    retry_after = None
    try:
        retry_after = float(response.headers.get("retry-after", ""))
    except ValueError:
        pass
    # A per-day limit will not clear within a retry; report it so the next model is tried.
    daily = status == 429 and ("per day" in message or (retry_after or 0) > LONG_WAIT_SECONDS)
    if daily:
        message = "daily limit reached for this model on this key"
    return ProviderHTTPError(status, message[:300] or response.reason_phrase, retry_after, quota_zero=daily)


def _parse(data: dict[str, Any], model: str, latency_ms: int) -> LLMResult:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise LLMError("Groq returned no answer")
    choice = choices[0]
    text = _THINK.sub("", str((choice.get("message") or {}).get("content") or "")).strip()
    if not text:
        raise LLMError(f"Groq returned no text (finish_reason {choice.get('finish_reason')})")
    usage = data.get("usage") or {}
    return LLMResult(
        text=text,
        tokens_in=int(usage.get("prompt_tokens", 0)),
        # completion_tokens already includes reasoning tokens.
        tokens_out=int(usage.get("completion_tokens", 0)),
        model=model,
        provider="groq",
        family="groq",
        latency_ms=latency_ms,
    )
