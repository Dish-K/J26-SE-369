import asyncio
from uuid import UUID, uuid4

from backend.app.probing.languages import get_extractor
from backend.app.probing.service import extract_segments
from tests.probing.fakes import REDACTED_AT, InMemoryProbingReader, make_flag, make_submission

EXAMPLE_A = """def process(items):
    for item in items:
        if item.active:
            score = item.value * 2
    return score
"""


def fid(n: int) -> UUID:
    return UUID(int=n)


def run(reader, task, **kwargs):
    return asyncio.run(extract_segments(task, reader, **kwargs))


def reader_with(task, code, flags, **submission_kwargs):
    reader = InMemoryProbingReader()
    reader.add_submission(make_submission(task, code, **submission_kwargs))
    for flag in flags:
        reader.add_flag(flag)
    return reader


def test_worked_example_a_end_to_end():
    task = uuid4()
    reader = reader_with(task, EXAMPLE_A, [make_flag(4, 4, task_id=task, flag_id=fid(1))])

    result = run(reader, task)

    (segment,) = result.segments
    assert segment.code == "score = item.value * 2"
    assert segment.context == ("for item in items:", "if item.active:")
    assert segment.parent_flag_id == fid(1)
    assert segment.flag_ids == (fid(1),)
    assert segment.task_id == task
    assert segment.entry_function.name == "process"
    assert segment.entry_function.uncertain is False
    assert segment.extra == {}
    assert result.rejected_flags == () and result.notes == ()


def test_worked_example_b_end_to_end():
    task = uuid4()
    code = "def f():\n" + "".join(f"    s{i} = {i}\n" for i in range(1, 25))
    flags = [
        make_flag(10, 12, task_id=task, confidence=0.62, flag_id=fid(1)),
        make_flag(12, 15, task_id=task, confidence=0.81, flag_id=fid(2)),
        make_flag(15, 18, task_id=task, confidence=0.70, flag_id=fid(3)),
    ]
    (segment,) = run(reader_with(task, code, flags), task).segments
    assert segment.parent_flag_id == fid(2)
    assert segment.flag_ids == (fid(1), fid(2), fid(3))
    assert (segment.location.start_line, segment.location.end_line) == (10, 18)


def test_worked_example_c_redacted_submission():
    task = uuid4()
    reader = reader_with(
        task, "", [make_flag(5, 7, task_id=task, flag_id=fid(4))], redacted_at=REDACTED_AT
    )
    result = run(reader, task)
    assert result.segments == ()
    assert [(r.flag_id, r.reason) for r in result.rejected_flags] == [
        (fid(4), "submission_redacted")
    ]


def test_rejected_flag_does_not_block_the_rest_of_the_task():
    task = uuid4()
    flags = [
        make_flag(4, 4, task_id=task, flag_id=fid(1)),
        make_flag(99, 120, task_id=task, flag_id=fid(2)),  # beyond the file
    ]
    result = run(reader_with(task, EXAMPLE_A, flags), task)
    assert [s.parent_flag_id for s in result.segments] == [fid(1)]
    assert [(r.flag_id, r.reason) for r in result.rejected_flags] == [(fid(2), "out_of_range")]


def test_task_with_no_flags_is_an_empty_valid_result():
    task = uuid4()
    reader = InMemoryProbingReader()
    reader.add_submission(make_submission(task, EXAMPLE_A))
    result = run(reader, task)
    assert (result.segments, result.rejected_flags, result.notes) == ((), (), ())


def test_flags_of_other_tasks_are_ignored():
    mine, other = uuid4(), uuid4()
    reader = reader_with(mine, EXAMPLE_A, [make_flag(4, 4, task_id=mine, flag_id=fid(1))])
    reader.add_submission(make_submission(other, "x = 1\n"))
    reader.add_flag(make_flag(1, 1, task_id=other, flag_id=fid(2)))
    result = run(reader, mine)
    assert [s.parent_flag_id for s in result.segments] == [fid(1)]


def test_segments_without_functions_carry_no_entry_function():
    task = uuid4()
    reader = reader_with(task, "a = 1\nb = 2\n", [make_flag(2, 2, task_id=task)])
    (segment,) = run(reader, task).segments
    assert segment.code == "b = 2"
    assert segment.entry_function is None


def test_entry_function_failure_does_not_fail_extraction():
    class ExplodingEntry:
        language = "python"

        def __init__(self):
            self._real = get_extractor("python")

        def parse(self, code):
            return self._real.parse(code)

        def resolve_boundary(self, code, start, end):
            return self._real.resolve_boundary(code, start, end)

        def find_entry_function(self, code):
            raise RuntimeError("boom")

    task = uuid4()
    reader = reader_with(task, EXAMPLE_A, [make_flag(4, 4, task_id=task)])
    (segment,) = run(reader, task, extractor=ExplodingEntry()).segments
    assert segment.code == "score = item.value * 2"
    assert segment.entry_function is None


def test_every_segment_gets_the_same_entry_function():
    task = uuid4()
    code = "def solve(xs):\n    a = 1\n    b = 2\n\n    c = 3\n    d = 4\n    return a\n"
    flags = [make_flag(2, 2, task_id=task), make_flag(6, 6, task_id=task)]
    segments = run(reader_with(task, code, flags), task).segments
    assert len(segments) == 2
    assert {s.entry_function.name for s in segments} == {"solve"}


def test_cross_model_version_note_is_surfaced():
    task = uuid4()
    flags = [
        make_flag(4, 4, task_id=task, model_version="v1", flag_id=fid(1)),
        make_flag(4, 5, task_id=task, model_version="v2", flag_id=fid(2)),
    ]
    result = run(reader_with(task, EXAMPLE_A, flags), task)
    assert len(result.segments) == 1
    assert [(n.kind, n.flag_ids) for n in result.notes] == [
        ("group_spans_model_versions", (fid(1), fid(2)))
    ]


def test_extraction_is_deterministic():
    task = uuid4()
    flags = [make_flag(2, 2, task_id=task, flag_id=fid(1)), make_flag(4, 4, task_id=task, flag_id=fid(2))]
    first = run(reader_with(task, EXAMPLE_A, flags), task)
    second = run(reader_with(task, EXAMPLE_A, list(reversed(flags))), task)
    assert first == second


def test_reader_is_never_asked_for_the_submission_when_there_are_no_flags():
    class CountingReader(InMemoryProbingReader):
        submission_reads = 0

        async def get_submission(self, task_id):
            type(self).submission_reads += 1
            return await super().get_submission(task_id)

    assert run(CountingReader(), uuid4()).segments == ()
    assert CountingReader.submission_reads == 0
