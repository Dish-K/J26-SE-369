from uuid import uuid4

import pytest

from backend.app.probing.errors import SegmentResolutionError
from backend.app.probing.languages import get_extractor
from backend.app.probing.rejection import reject_unusable_flags
from tests.probing.fakes import REDACTED_AT, make_flag, make_submission

CODE = "def f(xs):\n    a = 1\n\n    b = 2\n    return a + b\n"  # 5 lines, line 3 blank
EXTRACTOR = get_extractor("python")


def run(flags, submission):
    return reject_unusable_flags(flags, submission, EXTRACTOR)


def reasons(rejected):
    return {r.flag_id: r.reason for r in rejected}


def test_worked_example_c_redacted_submission():
    task = uuid4()
    flag_d = make_flag(5, 7, task_id=task)
    usable, rejected = run([flag_d], make_submission(task, "", redacted_at=REDACTED_AT))
    assert usable == []
    assert [(r.flag_id, r.reason) for r in rejected] == [(flag_d.flag_id, "submission_redacted")]
    assert "redacted" in rejected[0].detail


def test_all_four_reasons_and_healthy_flag_survives():
    task = uuid4()
    code = "def f(xs):\n    a = 1\n\n    b = 2\n    return a + b\ndef g(:\n"  # line 6 broken
    healthy = make_flag(4, 4, task_id=task)
    beyond = make_flag(4, 40, task_id=task)
    blank = make_flag(3, 3, task_id=task)
    broken = make_flag(6, 6, task_id=task)

    usable, rejected = run([healthy, beyond, blank, broken], make_submission(task, code))

    assert usable == [healthy]
    assert reasons(rejected) == {
        beyond.flag_id: "out_of_range",
        blank.flag_id: "blank_range",
        broken.flag_id: "unparseable_range",
    }


def test_rejection_is_per_flag_not_per_task():
    task = uuid4()
    good_one = make_flag(2, 2, task_id=task)
    good_two = make_flag(5, 5, task_id=task)
    bad = make_flag(9, 9, task_id=task)
    usable, rejected = run([good_one, bad, good_two], make_submission(task, CODE))
    assert usable == [good_one, good_two]  # order preserved
    assert [r.flag_id for r in rejected] == [bad.flag_id]


def test_task_with_no_survivors_is_a_valid_empty_result():
    task = uuid4()
    flags = [make_flag(50, 60, task_id=task), make_flag(70, 71, task_id=task)]
    usable, rejected = run(flags, make_submission(task, CODE))
    assert usable == []
    assert len(rejected) == 2


def test_no_flags_is_a_valid_empty_result():
    task = uuid4()
    assert run([], make_submission(task, CODE)) == ([], [])


@pytest.mark.parametrize("code", [None, ""], ids=["null-code", "empty-code"])
def test_null_or_empty_code_is_rejected_as_redacted(code):
    task = uuid4()
    flag = make_flag(1, 1, task_id=task)
    _, rejected = run([flag], make_submission(task, code))
    assert [r.reason for r in rejected] == ["submission_redacted"]


def test_missing_submission_row_rejects_every_flag():
    task = uuid4()
    flags = [make_flag(1, 1, task_id=task), make_flag(2, 2, task_id=task)]
    usable, rejected = run(flags, None)
    assert usable == []
    assert {r.reason for r in rejected} == {"submission_redacted"}
    assert rejected[0].detail == "no task_submissions row"


def test_redaction_marker_wins_even_if_text_is_present():
    task = uuid4()
    flag = make_flag(1, 1, task_id=task)
    _, rejected = run([flag], make_submission(task, CODE, redacted_at=REDACTED_AT))
    assert [r.reason for r in rejected] == ["submission_redacted"]


def test_flag_from_another_task_is_a_programming_error():
    flag = make_flag(1, 1, task_id=uuid4())
    with pytest.raises(ValueError):
        run([flag], make_submission(uuid4(), CODE))


def test_resolution_error_carries_the_reason():
    with pytest.raises(SegmentResolutionError) as caught:
        EXTRACTOR.resolve_boundary(CODE, 3, 3)
    assert caught.value.reason == "blank_range"
    with pytest.raises(SegmentResolutionError) as caught:
        EXTRACTOR.resolve_boundary(CODE, 1, 99)
    assert caught.value.reason == "out_of_range"


@pytest.mark.parametrize(
    "code,line",
    [
        ("y = 1\nx = foo(1,\n", 2),  # parser produces an ERROR node
        ("y = 1\ndef g(:\n    pass\n", 2),  # parser inserts a MISSING ')' token
    ],
    ids=["error-node", "missing-token"],
)
def test_error_and_missing_nodes_are_unparseable(code, line):
    with pytest.raises(SegmentResolutionError) as caught:
        EXTRACTOR.resolve_boundary(code, line, line)
    assert caught.value.reason == "unparseable_range"
