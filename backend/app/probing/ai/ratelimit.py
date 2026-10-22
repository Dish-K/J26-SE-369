"""Request and token pacing, and retry with backoff.

Defaults assume the worst case (a small free tier). Time is injectable so the
tests do not sleep.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypeVar

from backend.app.probing.ai.errors import (
    ProviderHTTPError,
    QuotaUnavailable,
    RateLimited,
    RequestTooLarge,
)

T = TypeVar("T")

WINDOW_SECONDS = 60.0
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)


def estimate_tokens(text: str) -> int:
    """Characters divided by four, rounded up. Crude on purpose."""
    return -(-len(text) // 4)


class RateLimiter:
    """Sliding one-minute window over requests and estimated tokens."""

    def __init__(
        self,
        max_rpm: int,
        max_tpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._max_rpm = max_rpm
        self._max_tpm = max_tpm
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    async def acquire(self, tokens: int) -> None:
        """Wait until a request of ``tokens`` fits, then record it."""
        if tokens > self._max_tpm:
            raise RequestTooLarge(
                f"request needs about {tokens} tokens but the per-minute cap is "
                f"{self._max_tpm}; it was not sent"
            )
        async with self._lock:
            while True:
                now = self._clock()
                self._prune(now)
                used = sum(t for _, t in self._events)
                if len(self._events) < self._max_rpm and used + tokens <= self._max_tpm:
                    self._events.append((now, tokens))
                    return
                wait = WINDOW_SECONDS - (now - self._events[0][0])
                await self._sleep(max(wait, 0.01))


async def retry_provider_call(
    call: Callable[[], Awaitable[T]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    backoff: tuple[float, ...] = BACKOFF_SECONDS,
) -> T:
    """Run ``call``; retry 429 and 5xx with backoff, honouring ``Retry-After``.

    Raises ``RateLimited`` once every retry is used. Other HTTP errors are
    raised at once, because retrying a 400 or 401 cannot help.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except ProviderHTTPError as exc:
            if exc.quota_zero:
                raise QuotaUnavailable(str(exc)) from exc
            retryable = exc.status == 429 or 500 <= exc.status < 600
            if not retryable:
                raise
            if attempt >= len(backoff):
                raise RateLimited(f"provider kept refusing: {exc}") from exc
            delay = exc.retry_after if exc.retry_after is not None else backoff[attempt]
            await sleep(max(delay, 0.0))
            attempt += 1
