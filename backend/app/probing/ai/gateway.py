"""The single door to a model: consent, cache, pacing, retry.

Every LLM task goes through ``LLMGateway.complete`` so none of the guards can
be skipped by accident.
"""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.app.probing.ai.cache import ResponseCache, cache_key
from backend.app.probing.ai.client import LLMClient
from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.consent import DataOrigin, check_consent, is_cacheable
from backend.app.probing.ai.errors import RateLimited
from backend.app.probing.ai.quota import DailyCounter
from backend.app.probing.ai.ratelimit import RateLimiter, estimate_tokens, retry_provider_call

logger = logging.getLogger(__name__)


class GatewayResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    tokens_in: int
    tokens_out: int
    model: str
    provider: str
    family: str
    cache_hit: bool


class LLMGateway:
    def __init__(
        self,
        client: LLMClient,
        settings: AISettings | None = None,
        *,
        limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        daily: DailyCounter | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        self.client = client
        self.settings = settings or AISettings()
        self._limiter = limiter or RateLimiter(
            self.settings.max_rpm, self.settings.max_tpm, sleep=sleep
        )
        self._cache = cache
        self._daily = daily
        self._sleep = sleep
        self._slots = asyncio.Semaphore(self.settings.concurrency)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        max_tokens: int,
        origin: DataOrigin,
        json_schema: dict[str, Any] | None = None,
    ) -> GatewayResult:
        check_consent(origin, self.client, self.settings)

        key = None
        if self._cache is not None and is_cacheable(origin):
            key = cache_key(
                provider=self.client.provider,
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
            hit = self._cache.get(key)
            if hit is not None:
                return _to_result(hit, cache_hit=True)

        # Input estimate plus the room reserved for the answer.
        await self._limiter.acquire(estimate_tokens(system) + estimate_tokens(user) + max_tokens)

        models = [model] + ([self.settings.fallback_model] if self.settings.fallback_model else [])
        result = None
        for index, name in enumerate(models):
            try:
                result = await self._call_model(name, system, user, temperature, max_tokens, json_schema)
                break
            except RateLimited as exc:
                if index == len(models) - 1:
                    raise
                logger.warning("model %s unavailable (%s); trying %s", name, type(exc).__name__, models[index + 1])
        logger.info(
            "LLM call: provider=%s model=%s tokens_in=%d tokens_out=%d",
            result.provider,
            result.model,
            result.tokens_in,
            result.tokens_out,
        )
        if key is not None:
            self._cache.put(key, result)
        return _to_result(result, cache_hit=False)

    async def _call_model(self, model, system, user, temperature, max_tokens, json_schema):
        estimate = estimate_tokens(system) + estimate_tokens(user) + max_tokens

        async def call():
            # Every HTTP attempt counts against the day, retries included.
            if self._daily is not None:
                self._daily.take(model, estimate)
            return await self.client.complete(
                system,
                user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )

        async with self._slots:
            result = await retry_provider_call(call, sleep=self._sleep)
        if self._daily is not None:
            self._daily.add_tokens(model, result.tokens_in + result.tokens_out)
        return result


def _to_result(result, *, cache_hit: bool) -> GatewayResult:
    return GatewayResult(
        text=result.text,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        model=result.model,
        provider=result.provider,
        family=result.family,
        cache_hit=cache_hit,
    )
