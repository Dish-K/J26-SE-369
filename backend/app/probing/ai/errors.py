"""Errors of the shared LLM wrapper."""


class LLMError(RuntimeError):
    """A model call failed for a reason the caller may want to handle."""


class ConsentNotConfirmed(LLMError):
    """Candidate code may not be sent to a hosted model (consent not confirmed)."""


class RateLimited(LLMError):
    """The provider kept refusing after every retry; use cached results instead."""


class RequestTooLarge(LLMError):
    """One request alone would exceed the per-minute token cap, so it was not sent."""


class ProviderHTTPError(LLMError):
    """A provider answered with an HTTP error status.

    Raised by provider clients so the gateway can retry 429 and 5xx. The message
    must never contain a request header or key.
    """

    def __init__(
        self, status: int, message: str, retry_after: float | None = None, *, quota_zero: bool = False
    ) -> None:
        super().__init__(f"provider returned HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after
        self.quota_zero = quota_zero


class QuotaUnavailable(RateLimited):
    """The provider says this model has no quota on this key (limit 0); retrying cannot help."""
