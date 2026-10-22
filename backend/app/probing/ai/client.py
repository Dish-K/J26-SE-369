"""The model-client interface every provider implements."""

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LLMResult(BaseModel):
    """One completed model call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    model: str
    provider: str
    family: str
    latency_ms: int = Field(ge=0)


class LLMClient(Protocol):
    """A provider client.

    ``family`` groups models trained alike (``google``, ``openai-oss``,
    ``anthropic``); the reviewer must come from a different family than the
    generator. ``hosted`` is False only for clients that send nothing off the
    machine, which is what the consent guard keys on.
    """

    provider: str
    family: str
    hosted: bool

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...
