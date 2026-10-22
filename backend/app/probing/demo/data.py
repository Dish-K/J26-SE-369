"""Demo data: one constructed solution with three flagged ranges.

Nothing here comes from a real candidate. The code was written for the demo and
the flags are invented, so every page built from it says "Demo data". The reader
implements the same protocol as the Postgres one, so the real Segment Extraction
runs on it unchanged.
"""

from uuid import UUID

from backend.app.probing.schemas import FlagInput, SubmissionInput

DEMO_TASK_ID = UUID("00000000-0000-4000-8000-00000000c301")

DEMO_CODE = """import heapq


def top_k_frequent(nums, k):
    counts = {}
    for n in nums:
        counts[n] = counts.get(n, 0) + 1

    heap = []
    for value, count in counts.items():
        heapq.heappush(heap, (count, value))
        if len(heap) > k:
            heapq.heappop(heap)

    result = []
    while heap:
        result.append(heapq.heappop(heap)[1])
    result.reverse()
    return result
"""

# (line range, confidence): invented, to look like mapped classifier flags.
_FLAGS = (
    (UUID("00000000-0000-4000-8000-00000000f001"), 5, 7, 0.91),
    (UUID("00000000-0000-4000-8000-00000000f002"), 9, 13, 0.84),
    (UUID("00000000-0000-4000-8000-00000000f003"), 15, 18, 0.62),
)


class FixtureReader:
    """The demo task's flags and file, in the shape ``extract_segments`` reads."""

    async def get_task_flags(self, task_id: UUID) -> list[FlagInput]:
        if task_id != DEMO_TASK_ID:
            return []
        return [
            FlagInput(
                flag_id=flag_id,
                task_id=DEMO_TASK_ID,
                model_version="demo-flags-v0",
                category="external_ai",
                confidence=confidence,
                start_line=start,
                end_line=end,
            )
            for flag_id, start, end, confidence in _FLAGS
        ]

    async def get_submission(self, task_id: UUID) -> SubmissionInput | None:
        if task_id != DEMO_TASK_ID:
            return None
        return SubmissionInput(task_id=DEMO_TASK_ID, code_text=DEMO_CODE)


def flag_confidences() -> dict[UUID, float]:
    return {flag_id: confidence for flag_id, _, _, confidence in _FLAGS}
