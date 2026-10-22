"""AI settings, read from environment variables (and ``.env``).

Keys are not part of these settings. A provider client reads its own key
variable when it is built and never logs or returns it.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

_PROBING_DIR = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _number(env: Mapping[str, str], name: str, default: float, cast=int):
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


#: Free-tier pacing for the primary Gemini model, a margin below 15 RPM / 250K TPM / 500 RPD.
GEMINI_FREE_TIER = {"max_rpm": 12, "max_tpm": 200_000, "max_rpd": 450}
GEMINI_PRIMARY_MODEL = "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
GEMINI_REVIEWER_MODEL = "gemma-4-31b-it"

#: Groq free plan (console): 30 RPM, 1K RPD, 8K TPM, 200K TPD per model. Safe margins below that.
GROQ_FREE_TIER = {"max_rpm": 20, "max_tpm": 6_000, "max_rpd": 900, "max_tpd": 180_000}
#: Reviewer order: the first two are Groq models, Gemma is the last resort (same provider as the generator).
GROQ_REVIEWER_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")
#: Reasoning tokens count toward the Groq limits, so ask for little of them.
GROQ_REASONING = {
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
    "qwen/qwen3.8-27b": {"reasoning_effort": "low"},
}


@dataclass(frozen=True)
class AISettings:
    # Worst-case defaults: assume a small free tier.
    max_rpm: int = 10
    max_tpm: int = 6000
    concurrency: int = 1
    max_rpd: int | None = None  # requests per model per day; None = not counted
    max_tpd: int | None = None  # tokens per model per day; None = not counted
    fallback_model: str | None = None
    allow_candidate_code_to_hosted_llm: bool = False
    allow_same_family_reviewer: bool = False
    reviewer_enabled: bool = False
    demo_mode: str = "cache"
    cache_dir: Path = _PROBING_DIR / ".cache" / "llm"
    mapper_model: str | None = None
    generator_model: str | None = None
    reviewer_model: str | None = None

    def with_gemini_free_tier(self, env: Mapping[str, str] | None = None) -> "AISettings":
        """Free-tier pacing, except for limits the owner set explicitly in the environment."""
        env = os.environ if env is None else env
        names = {"max_rpm": "PROBING_LLM_MAX_RPM", "max_tpm": "PROBING_LLM_MAX_TPM", "max_rpd": "PROBING_LLM_MAX_RPD"}
        changes = {k: v for k, v in GEMINI_FREE_TIER.items() if not env.get(names[k])}
        if not self.fallback_model:
            changes["fallback_model"] = GEMINI_FALLBACK_MODEL
        return replace(self, **changes)

    def with_groq_free_tier(self) -> "AISettings":
        return replace(self, fallback_model=None, **GROQ_FREE_TIER)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AISettings":
        if env is None:
            load_dotenv(override=False)
            env = os.environ
        demo_mode = env.get("PROBING_DEMO_MODE", "cache").strip().lower()
        if demo_mode not in {"cache", "live"}:
            raise ValueError(f"PROBING_DEMO_MODE must be cache or live, got {demo_mode!r}")
        cache_dir = env.get("PROBING_CACHE_DIR")
        return cls(
            max_rpm=_number(env, "PROBING_LLM_MAX_RPM", cls.max_rpm),
            max_tpm=_number(env, "PROBING_LLM_MAX_TPM", cls.max_tpm),
            concurrency=_number(env, "PROBING_LLM_CONCURRENCY", cls.concurrency),
            max_rpd=_number(env, "PROBING_LLM_MAX_RPD", None),
            fallback_model=env.get("PROBING_FALLBACK_MODEL") or None,
            allow_candidate_code_to_hosted_llm=_flag(
                env, "PROBING_ALLOW_CANDIDATE_CODE_TO_HOSTED_LLM", False
            ),
            allow_same_family_reviewer=_flag(env, "PROBING_ALLOW_SAME_FAMILY_REVIEWER", False),
            reviewer_enabled=_flag(env, "PROBING_REVIEWER_ENABLED", False),
            demo_mode=demo_mode,
            cache_dir=Path(cache_dir) if cache_dir else cls.cache_dir,
            mapper_model=env.get("PROBING_MAPPER_MODEL") or None,
            generator_model=env.get("PROBING_GENERATOR_MODEL") or None,
            reviewer_model=env.get("PROBING_REVIEWER_MODEL") or None,
        )
