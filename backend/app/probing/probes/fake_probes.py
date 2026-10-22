"""Canned probes for the fake client (offline demo and tests).

These are hand-written, labelled by the fake provider (``dummy-v0``), and exist
so the whole pipeline can run with no network. Their reference solutions were
checked by hand but are, like every probe, ``unverified``: nothing here is run.
Tags without a canned probe get a generic placeholder question.
"""

import json
import re
from typing import Any

from backend.app.probing.ai.fake import FakeClient

_TAG = re.compile(r"^TAG_ID:\s*(\S+)", re.MULTILINE)
_MINUTES = re.compile(r"^TARGET_MINUTES:\s*(\d+)", re.MULTILINE)
_REVIEW = re.compile(r"^REVIEW_REQUEST$", re.MULTILINE)


def _case(args: list[Any], expected: Any, note: str = "") -> dict[str, str]:
    return {"args_json": json.dumps(args), "expected_json": json.dumps(expected), "note": note}


def _probe(question, signature, name, reference, cases, checker="exact") -> dict[str, Any]:
    return {
        "question": question,
        "function_name": name,
        "signature": signature,
        "checker": checker,
        "reference_solution": reference,
        "cases": cases,
    }


CANNED: dict[tuple[str, int], dict[str, Any]] = {
    ("hash_table", 5): _probe(
        "Return True if any value appears at least twice in the list of integers, otherwise False.",
        "def has_duplicate(nums: list[int]) -> bool",
        "has_duplicate",
        "def has_duplicate(nums):\n    seen = set()\n    for n in nums:\n        if n in seen:\n            return True\n        seen.add(n)\n    return False\n",
        [
            _case([[1, 2, 3, 1]], True),
            _case([[1, 2, 3]], False),
            _case([[]], False, "empty"),
            _case([[5]], False, "single"),
            _case([[2, 2]], True),
            _case([[-1, -1, 0]], True, "negatives"),
        ],
    ),
    ("hash_table", 10): _probe(
        "Given a list of integers and a target, return the two indices i < j whose values add up to the target. Exactly one such pair exists.",
        "def two_sum_indices(nums: list[int], target: int) -> list[int]",
        "two_sum_indices",
        "def two_sum_indices(nums, target):\n    seen = {}\n    for j, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], j]\n        seen[n] = j\n    return []\n",
        [
            _case([[2, 7, 11, 15], 9], [0, 1]),
            _case([[3, 2, 4], 6], [1, 2]),
            _case([[3, 3], 6], [0, 1], "equal values"),
            _case([[-1, -2, -3, -4, -5], -8], [2, 4], "negatives"),
            _case([[0, 4, 3, 0], 0], [0, 3], "zeros"),
            _case([[1, 5, 5, 2], 10], [1, 2]),
        ],
    ),
    ("hash_table", 20): _probe(
        "Group the words that are anagrams of each other. Return the groups in order of the first appearance of each group, and keep the words inside a group in their original order.",
        "def group_anagrams(words: list[str]) -> list[list[str]]",
        "group_anagrams",
        "def group_anagrams(words):\n    groups = {}\n    for w in words:\n        groups.setdefault(tuple(sorted(w)), []).append(w)\n    return list(groups.values())\n",
        [
            _case([["eat", "tea", "tan", "ate", "nat", "bat"]], [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]),
            _case([[""]], [[""]], "empty string"),
            _case([["a"]], [["a"]]),
            _case([[]], [], "no words"),
            _case([["ab", "ba", "ab"]], [["ab", "ba", "ab"]], "repeated word"),
            _case([["abc", "def"]], [["abc"], ["def"]]),
        ],
    ),
    ("heap_priority_queue", 5): _probe(
        "Return the k-th largest value in the list (counting duplicates), where 1 <= k <= len(nums).",
        "def kth_largest(nums: list[int], k: int) -> int",
        "kth_largest",
        "import heapq\n\n\ndef kth_largest(nums, k):\n    heap = []\n    for n in nums:\n        heapq.heappush(heap, n)\n        if len(heap) > k:\n            heapq.heappop(heap)\n    return heap[0]\n",
        [
            _case([[3, 2, 1, 5, 6, 4], 2], 5),
            _case([[3, 2, 3, 1, 2, 4, 5, 5, 6], 4], 4),
            _case([[1], 1], 1, "single"),
            _case([[7, 7, 7], 2], 7, "duplicates"),
            _case([[-1, -5, -3], 1], -1, "negatives"),
            _case([[5, 9], 2], 5),
        ],
    ),
    ("heap_priority_queue", 10): _probe(
        "Repeatedly take the two heaviest stones and smash them together: equal stones both vanish, otherwise the lighter vanishes and the heavier loses the lighter's weight. Return the weight of the last stone, or 0 if none is left.",
        "def last_stone_weight(stones: list[int]) -> int",
        "last_stone_weight",
        "import heapq\n\n\ndef last_stone_weight(stones):\n    heap = [-s for s in stones]\n    heapq.heapify(heap)\n    while len(heap) > 1:\n        a = -heapq.heappop(heap)\n        b = -heapq.heappop(heap)\n        if a != b:\n            heapq.heappush(heap, -(a - b))\n    return -heap[0] if heap else 0\n",
        [
            _case([[2, 7, 4, 1, 8, 1]], 1),
            _case([[1]], 1, "single"),
            _case([[2, 2]], 0, "equal pair"),
            _case([[3, 7, 2]], 2),
            _case([[8, 8, 8]], 8, "three equal"),
            _case([[]], 0, "no stones"),
        ],
    ),
    ("heap_priority_queue", 20): _probe(
        "Merge k individually sorted lists of integers into one sorted list, using a heap of the current front elements.",
        "def merge_k_sorted(lists: list[list[int]]) -> list[int]",
        "merge_k_sorted",
        "import heapq\n\n\ndef merge_k_sorted(lists):\n    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]\n    heapq.heapify(heap)\n    out = []\n    while heap:\n        value, i, pos = heapq.heappop(heap)\n        out.append(value)\n        if pos + 1 < len(lists[i]):\n            heapq.heappush(heap, (lists[i][pos + 1], i, pos + 1))\n    return out\n",
        [
            _case([[[1, 4, 5], [1, 3, 4], [2, 6]]], [1, 1, 2, 3, 4, 4, 5, 6]),
            _case([[]], [], "no lists"),
            _case([[[], []]], [], "empty lists"),
            _case([[[1], [0]]], [0, 1]),
            _case([[[-2, -1], [-3]]], [-3, -2, -1], "negatives"),
            _case([[[5, 6, 7]]], [5, 6, 7], "one list"),
        ],
    ),
    ("binary_search", 5): _probe(
        "In a sorted list, return the index of the target, or the index where it would be inserted to keep the list sorted.",
        "def search_insert(nums: list[int], target: int) -> int",
        "search_insert",
        "def search_insert(nums, target):\n    lo, hi = 0, len(nums)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n",
        [
            _case([[1, 3, 5, 6], 5], 2),
            _case([[1, 3, 5, 6], 2], 1),
            _case([[1, 3, 5, 6], 7], 4, "after the end"),
            _case([[1, 3, 5, 6], 0], 0, "before the start"),
            _case([[], 3], 0, "empty"),
            _case([[2], 2], 0, "single"),
        ],
    ),
    ("binary_search", 10): _probe(
        "In a sorted list that may contain duplicates, return [first_index, last_index] of the target, or [-1, -1] if it is absent.",
        "def first_and_last(nums: list[int], target: int) -> list[int]",
        "first_and_last",
        "import bisect\n\n\ndef first_and_last(nums, target):\n    lo = bisect.bisect_left(nums, target)\n    if lo == len(nums) or nums[lo] != target:\n        return [-1, -1]\n    return [lo, bisect.bisect_right(nums, target) - 1]\n",
        [
            _case([[5, 7, 7, 8, 8, 10], 8], [3, 4]),
            _case([[5, 7, 7, 8, 8, 10], 6], [-1, -1]),
            _case([[], 0], [-1, -1], "empty"),
            _case([[1], 1], [0, 0], "single"),
            _case([[2, 2, 2], 2], [0, 2], "all equal"),
            _case([[1, 2, 3], 3], [2, 2]),
        ],
    ),
    ("binary_search", 20): _probe(
        "Each pile holds some bananas. You may eat k bananas per hour from one pile (a pile with fewer than k takes the whole hour). Return the smallest integer k that finishes every pile within h hours; h >= number of piles.",
        "def min_eating_speed(piles: list[int], h: int) -> int",
        "min_eating_speed",
        "def min_eating_speed(piles, h):\n    lo, hi = 1, max(piles)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        hours = sum(-(-p // mid) for p in piles)\n        if hours <= h:\n            hi = mid\n        else:\n            lo = mid + 1\n    return lo\n",
        [
            _case([[3, 6, 7, 11], 8], 4),
            _case([[30, 11, 23, 4, 20], 5], 30),
            _case([[30, 11, 23, 4, 20], 6], 23),
            _case([[1], 1], 1, "minimal"),
            _case([[1000000000], 2], 500000000, "large"),
            _case([[312884470], 312884469], 2, "large, tight"),
        ],
    ),
}


def _generic(tag_id: str, minutes: int) -> dict[str, Any]:
    return _probe(
        f"PLACEHOLDER from the fake model (no canned probe for '{tag_id}'): return the sum of the integers in the list.",
        "def total(nums: list[int]) -> int",
        "total",
        "def total(nums):\n    result = 0\n    for n in nums:\n        result += n\n    return result\n",
        [
            _case([[1, 2, 3]], 6),
            _case([[]], 0, "empty"),
            _case([[-1, 1]], 0),
            _case([[5]], 5),
        ],
    )


def fake_probe_reply(system: str, user: str, json_schema: dict[str, Any] | None) -> str:
    """A FakeClient script item: answers the probe-generation prompt."""
    tag = _TAG.search(user)
    minutes = _MINUTES.search(user)
    if not tag or not minutes:
        raise AssertionError("fake_probe_reply only answers the probe-generation prompt")
    tag_id, level = tag.group(1), int(minutes.group(1))
    return json.dumps(CANNED.get((tag_id, level)) or _generic(tag_id, level))


def fake_review_reply(system: str, user: str, json_schema: dict[str, Any] | None) -> str:
    """A FakeClient script item: answers the review prompt with a fixed, labelled opinion."""
    minutes = int(_MINUTES.search(user).group(1))
    return json.dumps(
        {
            "estimated_minutes": minutes,
            "difficulty": "about_right",
            "concerns": ["FAKE REVIEW (dummy-v0): a canned opinion, not a real assessment."],
            "case_problems": [],
        }
    )


class FakeDemoClient(FakeClient):
    """The fake model for the offline demo: Mapper prompts follow the cues, probe prompts get canned probes."""

    def _next_item(self, system, user, json_schema):
        if self._script is None:
            if _REVIEW.search(user):
                return fake_review_reply
            if _TAG.search(user):
                return fake_probe_reply
        return super()._next_item(system, user, json_schema)


class FakeReviewerClient(FakeDemoClient):
    """The fake reviewer: a different 'family' from the fake generator, like a real second provider."""

    family = "fake-reviewer"
