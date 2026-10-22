"""A per-model daily request and token counter that survives restarts.

Google's free-tier day resets at midnight Pacific time. The Windows Python here
has no time zone database, so the Pacific date is worked out from the US
daylight-saving rule directly. Groq's day is counted from midnight UTC, which is
an assumption (its console does not say). The file holds only a date and counts
per model id: no prompts, no code, no answers.
"""

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from backend.app.probing.ai.errors import RateLimited


class DailyQuotaExhausted(RateLimited):
    """The configured number of requests for this model today is used up."""


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def pacific_date(now_utc: datetime) -> date:
    """The calendar date in US Pacific time (DST: second Sunday of March to first of November)."""
    year = now_utc.year
    start = datetime.combine(_nth_sunday(year, 3, 2), datetime.min.time(), timezone.utc).replace(hour=10)
    end = datetime.combine(_nth_sunday(year, 11, 1), datetime.min.time(), timezone.utc).replace(hour=9)
    offset = -7 if start <= now_utc < end else -8
    return (now_utc + timedelta(hours=offset)).date()


def utc_date(now_utc: datetime) -> date:
    return now_utc.astimezone(timezone.utc).date()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DailyCounter:
    def __init__(
        self,
        limit: int,
        path: Path | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
        token_limit: int | None = None,
        day_of: Callable[[datetime], date] = pacific_date,
    ) -> None:
        self._limit = limit
        self._token_limit = token_limit  # tokens per model per day; None = not counted
        self._path = path
        self._clock = clock
        self._day_of = day_of
        self._day = ""
        self._counts: dict[str, int] = {}
        self._tokens: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if self._path is None:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._day = str(data["day"])
            self._counts = {str(k): int(v) for k, v in data["counts"].items()}
            self._tokens = {str(k): int(v) for k, v in data.get("tokens", {}).items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self._day, self._counts, self._tokens = "", {}, {}  # missing or damaged: start the day fresh

    def _roll(self) -> None:
        today = self._day_of(self._clock()).isoformat()
        if today != self._day:
            self._day, self._counts, self._tokens = today, {}, {}

    def used(self, model: str) -> int:
        self._roll()
        return self._counts.get(model, 0)

    def tokens_used(self, model: str) -> int:
        self._roll()
        return self._tokens.get(model, 0)

    def take(self, model: str, estimated_tokens: int = 0) -> None:
        """Count one request, or raise ``DailyQuotaExhausted`` without counting.

        With a token limit set, a request whose estimate would pass it is refused too.
        """
        self._roll()
        if self._counts.get(model, 0) >= self._limit:
            raise DailyQuotaExhausted(f"daily request limit of {self._limit} reached for {model}")
        if self._token_limit is not None and self._tokens.get(model, 0) + estimated_tokens > self._token_limit:
            raise DailyQuotaExhausted(f"daily token limit of {self._token_limit} reached for {model}")
        self._counts[model] = self._counts.get(model, 0) + 1
        self._save()

    def add_tokens(self, model: str, tokens: int) -> None:
        """Record the tokens a finished call actually used (reasoning tokens included)."""
        if self._token_limit is None or tokens <= 0:
            return
        self._roll()
        self._tokens[model] = self._tokens.get(model, 0) + tokens
        self._save()

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {"day": self._day, "counts": self._counts}
            if self._token_limit is not None:
                data["tokens"] = self._tokens
            self._path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass  # a counter that cannot be saved must not fail the call
