"""Gemini client over the native REST API (``httpx`` only, no vendor SDK).

The key is read from ``GEMINI_API_KEY`` when the client is built and is sent in
the ``x-goog-api-key`` header. It is never logged, returned, or put in an error
message or URL.
"""

import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from backend.app.probing.ai.client import LLMResult
from backend.app.probing.ai.errors import LLMError, ProviderHTTPError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
KEY_VARIABLE = "GEMINI_API_KEY"
TIMEOUT_SECONDS = 60.0


class MissingApiKey(LLMError):
    pass


def read_key(env=None) -> str:
    if env is None:
        load_dotenv(override=False)
        env = os.environ
    key = (env.get(KEY_VARIABLE) or "").strip()
    if not key:
        raise MissingApiKey(f"{KEY_VARIABLE} is not set; add it to .env")
    return key


class GeminiClient:
    provider = "gemini"
    family = "google"
    hosted = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http: httpx.AsyncClient | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self._key = api_key or read_key()
        self._http = http or httpx.AsyncClient(timeout=timeout)

    def __repr__(self) -> str:  # never show the key
        return "GeminiClient()"

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
        config: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseJsonSchema"] = json_schema
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        started = time.monotonic()
        try:
            response = await self._http.post(
                f"{BASE_URL}/models/{model}:generateContent",
                json=body,
                headers={"x-goog-api-key": self._key},
            )
        except httpx.HTTPError as exc:
            # Connection problems are treated like a 503 so the gateway retries them.
            raise ProviderHTTPError(503, f"network error ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise _http_error(response)
        return _parse(response.json(), model, int((time.monotonic() - started) * 1000))

    async def aclose(self) -> None:
        await self._http.aclose()


async def list_models(*, api_key: str | None = None) -> list[dict[str, Any]]:
    """Models that support ``generateContent``: ``name``, display name, token limits."""
    key = api_key or read_key()
    found: list[dict[str, Any]] = []
    token = None
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as http:
        while True:
            params = {"pageSize": 200, **({"pageToken": token} if token else {})}
            response = await http.get(f"{BASE_URL}/models", params=params, headers={"x-goog-api-key": key})
            if response.status_code != 200:
                raise _http_error(response)
            data = response.json()
            for item in data.get("models", []):
                if "generateContent" in item.get("supportedGenerationMethods", []):
                    found.append(
                        {
                            "id": item["name"].removeprefix("models/"),
                            "display_name": item.get("displayName", ""),
                            "input_token_limit": item.get("inputTokenLimit"),
                            "output_token_limit": item.get("outputTokenLimit"),
                        }
                    )
            token = data.get("nextPageToken")
            if not token:
                return found


def _http_error(response: httpx.Response) -> ProviderHTTPError:
    message = ""
    try:
        message = str(response.json().get("error", {}).get("message", ""))
    except ValueError:
        pass
    status = response.status_code
    quota_zero = status == 429 and "limit: 0" in message
    if quota_zero:
        message = "quota for this model is 0 on this key (no free tier); pick another model"
    retry_after = None
    try:
        retry_after = float(response.headers.get("retry-after", ""))
    except ValueError:
        pass
    return ProviderHTTPError(
        status, message[:300] or response.reason_phrase, retry_after, quota_zero=quota_zero
    )


def _parse(data: dict[str, Any], model: str, latency_ms: int) -> LLMResult:
    candidates = data.get("candidates") or []
    if not candidates:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise LLMError(f"Gemini returned no answer ({reason})")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text and candidate.get("finishReason") not in (None, "STOP"):
        raise LLMError(f"Gemini returned no text (finishReason {candidate.get('finishReason')})")
    usage = data.get("usageMetadata") or {}
    return LLMResult(
        text=text,
        tokens_in=int(usage.get("promptTokenCount", 0)),
        # Thinking tokens are billed and counted against the limit as output.
        tokens_out=int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0)),
        model=model,
        provider="gemini",
        family="google",
        latency_ms=latency_ms,
    )
