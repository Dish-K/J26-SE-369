from backend.app.probing.schemas import RangeRejectionReason


class SegmentResolutionError(ValueError):
    """A flagged line range cannot be resolved to a segment in the given code.

    ``reason`` is the rejection reason recorded for the flag; the task itself
    carries on.
    """

    def __init__(self, reason: RangeRejectionReason, message: str) -> None:
        super().__init__(message)
        self.reason: RangeRejectionReason = reason


class ProbingInputError(ValueError):
    """A row read for Segment Extraction violates the C1/C2 contract."""
