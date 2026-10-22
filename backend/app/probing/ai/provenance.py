"""Provenance stamped on every LLM-derived result.

Provider, model, prompt and version, temperatures, date and token counts, so a
later reader can tell exactly how a rating or tag was produced.
"""

from datetime import date, datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    model: str
    family: str
    prompt_version: str
    temperatures: tuple[float, ...]
    date: date
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cache_hit: bool
    calls: int = Field(ge=0)
    # Models tried first and passed over, each with a short fixed-text reason (reviewer chain).
    skipped: tuple[str, ...] = ()


class ProvenanceBuilder:
    """Accumulates the calls that make up one result."""

    def __init__(
        self, prompt_version: str, *, provider: str = "none", model: str = "none", family: str = "none"
    ) -> None:
        self._prompt_version = prompt_version
        self._temperatures: list[float] = []
        self._tokens_in = 0
        self._tokens_out = 0
        self._cache_hits = 0
        self._calls = 0
        # Defaults describe a result that made no call (a rule-only fallback).
        self._provider = provider
        self._model = model
        self._family = family

    def add(
        self,
        *,
        provider: str,
        model: str,
        family: str,
        temperature: float,
        tokens_in: int,
        tokens_out: int,
        cache_hit: bool,
    ) -> None:
        self._provider, self._model, self._family = provider, model, family
        self._temperatures.append(temperature)
        self._tokens_in += tokens_in
        self._tokens_out += tokens_out
        self._cache_hits += int(cache_hit)
        self._calls += 1

    def build(self, *, today: date | None = None) -> Provenance:
        return Provenance(
            provider=self._provider,
            model=self._model,
            family=self._family,
            prompt_version=self._prompt_version,
            temperatures=tuple(self._temperatures),
            date=today or datetime.now(timezone.utc).date(),
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            # True only when nothing was sent to a provider.
            cache_hit=self._calls > 0 and self._cache_hits == self._calls,
            calls=self._calls,
        )
