"""The consent guard: candidate code never reaches a hosted model by accident.

The ethics and consent wording has not been confirmed to cover third-party AI
processing, so code whose origin is ``candidate`` is refused unless the owner
sets ``PROBING_ALLOW_CANDIDATE_CODE_TO_HOSTED_LLM=true``.
"""

from typing import Literal

from backend.app.probing.ai.client import LLMClient
from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.errors import ConsentNotConfirmed

DataOrigin = Literal["fixture", "public_dataset", "owner_sample", "candidate"]

# Origins that may go to a hosted model this week.
ALLOWED_ORIGINS: frozenset[str] = frozenset({"fixture", "public_dataset", "owner_sample"})


def check_consent(origin: DataOrigin, client: LLMClient, settings: AISettings) -> None:
    """Raise ``ConsentNotConfirmed`` if this origin may not go to this client."""
    if origin not in ALLOWED_ORIGINS and origin != "candidate":
        raise ValueError(f"unknown data origin {origin!r}")
    if not client.hosted:
        return
    if origin in ALLOWED_ORIGINS:
        return
    if settings.allow_candidate_code_to_hosted_llm:
        return
    raise ConsentNotConfirmed(
        "candidate code may not be sent to a hosted model: consent for third-party "
        "AI processing is not confirmed "
        "(set PROBING_ALLOW_CANDIDATE_CODE_TO_HOSTED_LLM=true once it is)"
    )


def is_cacheable(origin: DataOrigin) -> bool:
    """Responses are stored on disk only for data that is not candidate code."""
    return origin in ALLOWED_ORIGINS
