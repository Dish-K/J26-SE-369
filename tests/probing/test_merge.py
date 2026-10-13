import random
from uuid import UUID, uuid4

from backend.app.probing.errors import SegmentResolutionError
from backend.app.probing.languages import get_extractor
from backend.app.probing.merge import build_segments, choose_parent, group_touching_flags
from backend.app.probing.rejection import reject_unusable_flags
from tests.probing.fakes import make_flag, make_submission

EXTRACTOR = get_extractor("python")
TASK = uuid4()

# one statement per line: "s<i> = <i>" sits on line i + 1 (line 1 is the def)
CODE = "def f():\n" + "".join(f"    s{i} = {i}\n" for i in range(1, 31))


def fid(n: int) -> UUID:
    return UUID(int=n)


def build(flags, code=CODE, extractor=EXTRACTOR):
    return build_segments(TASK, flags, code, extractor)


# --- grouping ---------------------------------------------------------------


def test_worked_example_b_chain_merges_into_one_segment():
    a = make_flag(10, 12, task_id=TASK, confidence=0.62, flag_id=fid(1))
    b = make_flag(12, 15, task_id=TASK, confidence=0.81, flag_id=fid(2))
    c = make_flag(15, 18, task_id=TASK, confidence=0.70, flag_id=fid(3))

    segments, rejected, notes = build([c, a, b])  # input order is irrelevant

    assert rejected == [] and notes == []
    (segment,) = segments
    assert segment.parent_flag_id == b.flag_id
    assert segment.flag_ids == (fid(1), fid(2), fid(3))
    assert (segment.location.start_line, segment.location.end_line) == (10, 18)
    assert segment.code.splitlines()[0] == "s9 = 9"
    assert segment.code.splitlines()[-1].strip() == "s17 = 17"
    assert segment.extra == {} and segment.entry_function is None


def test_touching_ranges_merge_but_a_one_line_gap_does_not():
    touching = group_touching_flags(
        [make_flag(10, 12, task_id=TASK), make_flag(13, 14, task_id=TASK)]
    )
    gapped = group_touching_flags(
        [make_flag(10, 12, task_id=TASK), make_flag(14, 15, task_id=TASK)]
    )
    assert [len(g) for g in touching] == [2]
    assert [len(g) for g in gapped] == [1, 1]


def test_contained_range_is_absorbed_and_does_not_shrink_the_group():
    outer = make_flag(5, 20, task_id=TASK)
    inner = make_flag(8, 9, task_id=TASK)
    after = make_flag(21, 22, task_id=TASK)  # touches the outer range's end
    groups = group_touching_flags([inner, after, outer])
    assert len(groups) == 1 and len(groups[0]) == 3


def test_matches_naive_pairwise_connected_components():
    rng = random.Random(369)
    for _ in range(200):
        flags = []
        for _ in range(rng.randint(1, 8)):
            start = rng.randint(1, 30)
            flags.append(make_flag(start, start + rng.randint(0, 5), task_id=TASK))

        def touch(x, y):
            return x.start_line <= y.end_line + 1 and y.start_line <= x.end_line + 1

        parent = {f.flag_id: f.flag_id for f in flags}

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for x in flags:
            for y in flags:
                if touch(x, y):
                    parent[find(x.flag_id)] = find(y.flag_id)
        expected = {frozenset(f.flag_id for f in flags if find(f.flag_id) == r)
                    for r in {find(f.flag_id) for f in flags}}

        actual = {frozenset(f.flag_id for f in g) for g in group_touching_flags(flags)}
        assert actual == expected


def test_merge_is_cross_category():
    a = make_flag(10, 12, task_id=TASK, category="ide_ai", flag_id=fid(1))
    b = make_flag(12, 14, task_id=TASK, category="external_ai", flag_id=fid(2))
    (segment,), _, _ = build([a, b])
    assert segment.flag_ids == (fid(1), fid(2))


# --- parent selection ---------------------------------------------------------


def test_parent_is_highest_confidence():
    group = [
        make_flag(1, 2, task_id=TASK, confidence=0.5, flag_id=fid(1)),
        make_flag(2, 3, task_id=TASK, confidence=0.9, flag_id=fid(2)),
    ]
    assert choose_parent(group).flag_id == fid(2)


def test_parent_tie_breaks_on_earliest_start_then_lowest_flag_id():
    early = make_flag(3, 6, task_id=TASK, confidence=0.7, flag_id=fid(9))
    late = make_flag(5, 8, task_id=TASK, confidence=0.7, flag_id=fid(1))
    assert choose_parent([late, early]).flag_id == fid(9)  # earliest start wins

    same_a = make_flag(3, 6, task_id=TASK, confidence=0.7, flag_id=fid(7))
    same_b = make_flag(3, 9, task_id=TASK, confidence=0.7, flag_id=fid(4))
    assert choose_parent([same_a, same_b]).flag_id == fid(4)  # then lowest flag_id


# --- diagnostics ----------------------------------------------------------------


def test_cross_model_version_group_is_merged_and_noted():
    a = make_flag(10, 12, task_id=TASK, model_version="v1", flag_id=fid(1))
    b = make_flag(12, 14, task_id=TASK, model_version="v2", flag_id=fid(2))
    segments, _, notes = build([a, b])
    assert len(segments) == 1
    assert [(n.kind, n.flag_ids) for n in notes] == [
        ("group_spans_model_versions", (fid(1), fid(2)))
    ]


def test_same_model_version_group_has_no_note():
    a = make_flag(10, 12, task_id=TASK)
    b = make_flag(12, 14, task_id=TASK)
    assert build([a, b])[2] == []


# --- ungroup fallback -----------------------------------------------------------


class FailingExtractor:
    """Delegates to the real extractor but fails chosen (start, end) ranges."""

    language = "python"

    def __init__(self, failing_ranges):
        self._failing = set(failing_ranges)
        self._real = EXTRACTOR

    def parse(self, code):
        return self._real.parse(code)

    def find_entry_function(self, code):
        return self._real.find_entry_function(code)

    def resolve_boundary(self, code, start_line, end_line):
        if (start_line, end_line) in self._failing:
            raise SegmentResolutionError("unparseable_range", "injected failure")
        return self._real.resolve_boundary(code, start_line, end_line)


def test_unresolvable_union_dissolves_into_standalone_segments():
    a = make_flag(10, 12, task_id=TASK, flag_id=fid(1))
    b = make_flag(12, 15, task_id=TASK, flag_id=fid(2))
    c = make_flag(15, 18, task_id=TASK, flag_id=fid(3))

    segments, rejected, notes = build([a, b, c], extractor=FailingExtractor({(10, 18)}))

    assert rejected == []
    assert [(s.parent_flag_id, s.flag_ids) for s in segments] == [
        (fid(1), (fid(1),)),
        (fid(2), (fid(2),)),
        (fid(3), (fid(3),)),
    ]
    assert [(n.kind, n.flag_ids) for n in notes] == [
        ("group_ungrouped", (fid(1), fid(2), fid(3)))
    ]


def test_ungrouped_flag_that_still_fails_is_rejected_with_its_own_reason():
    a = make_flag(10, 12, task_id=TASK, flag_id=fid(1))
    b = make_flag(12, 15, task_id=TASK, flag_id=fid(2))

    segments, rejected, notes = build([a, b], extractor=FailingExtractor({(10, 15), (12, 15)}))

    assert [s.parent_flag_id for s in segments] == [fid(1)]
    assert [(r.flag_id, r.reason) for r in rejected] == [(fid(2), "unparseable_range")]
    assert [n.kind for n in notes] == ["group_ungrouped"]


def test_unresolvable_single_flag_is_rejected_without_an_ungroup_note():
    flag = make_flag(10, 12, task_id=TASK, flag_id=fid(1))
    segments, rejected, notes = build([flag], extractor=FailingExtractor({(10, 12)}))
    assert segments == [] and notes == []
    assert [r.flag_id for r in rejected] == [fid(1)]


def test_real_union_that_swallows_a_syntax_error_is_ungrouped():
    # Lines 2 and 6 are clean on their own. Flag M covers the broken `except`
    # block between them, so A-M-B form one group, but resolving the union
    # climbs to the whole `try`, which contains the syntax error.
    code = (
        "try:\n"
        "    a = 1\n"            # 2
        "except ValueError:\n"   # 3
        "    b = (1 +\n"         # 4  broken
        "else:\n"                # 5
        "    c = 3\n"            # 6
    )
    a = make_flag(2, 2, task_id=TASK, flag_id=fid(1))
    m = make_flag(3, 5, task_id=TASK, flag_id=fid(2))
    b = make_flag(6, 6, task_id=TASK, flag_id=fid(3))

    segments, rejected, notes = build([a, m, b], code=code)

    assert [(s.parent_flag_id, s.code) for s in segments] == [(fid(1), "a = 1"), (fid(3), "c = 3")]
    assert [(r.flag_id, r.reason) for r in rejected] == [(fid(2), "unparseable_range")]
    assert [(n.kind, n.flag_ids) for n in notes] == [
        ("group_ungrouped", (fid(1), fid(2), fid(3)))
    ]


# --- determinism and composition ---------------------------------------------------


def test_segment_ids_are_deterministic_and_distinct():
    flags = [make_flag(2, 3, task_id=TASK, flag_id=fid(1)),
             make_flag(20, 21, task_id=TASK, flag_id=fid(2))]
    first, _, _ = build(flags)
    second, _, _ = build(list(reversed(flags)))
    assert [s.segment_id for s in first] == [s.segment_id for s in second]
    assert first[0].segment_id != first[1].segment_id


def test_no_flags_means_no_segments():
    assert build([]) == ([], [], [])


def test_segments_are_ordered_by_position():
    late = make_flag(25, 25, task_id=TASK)
    early = make_flag(3, 3, task_id=TASK)
    segments, _, _ = build([late, early])
    assert [s.location.start_line for s in segments] == [3, 25]


def test_rejected_flag_does_not_pull_a_neighbour_into_a_merge():
    code = "def f():\n    a = 1\n\n    b = 2\n    c = 3\n"  # line 3 blank
    healthy = make_flag(4, 4, task_id=TASK, flag_id=fid(1))
    blank = make_flag(3, 3, task_id=TASK, flag_id=fid(2))  # touches healthy's range

    usable, rejected = reject_unusable_flags(
        [healthy, blank], make_submission(TASK, code), EXTRACTOR
    )
    segments, _, _ = build_segments(TASK, usable, code, EXTRACTOR)

    assert [r.flag_id for r in rejected] == [fid(2)]
    (segment,) = segments
    assert segment.flag_ids == (fid(1),)
    assert segment.code == "b = 2"
    assert (segment.location.start_line, segment.location.end_line) == (4, 4)
