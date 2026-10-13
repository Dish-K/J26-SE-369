import pytest

from backend.app.probing.errors import SegmentResolutionError
from backend.app.probing.languages import get_extractor


def resolve(code: str, start: int, end: int):
    return get_extractor("python").resolve_boundary(code, start, end)


EXAMPLE_A = """def process(items):
    for item in items:
        if item.active:
            score = item.value * 2
    return score
"""


def test_worked_example_a_code_and_context_are_separate():
    boundary = resolve(EXAMPLE_A, 4, 4)
    assert boundary.code == "score = item.value * 2"
    assert boundary.context == ("for item in items:", "if item.active:")
    assert boundary.location.model_dump() == {
        "start_line": 4,
        "end_line": 4,
        "start_col": 13,
        "end_col": 35,
    }


def test_function_header_is_not_context():
    assert "def process(items):" not in resolve(EXAMPLE_A, 4, 4).context


def test_top_level_statement_has_no_context():
    boundary = resolve("x = 1\ny = 2\n", 2, 2)
    assert boundary.code == "y = 2"
    assert boundary.context == ()


def test_header_only_flag_returns_whole_compound_statement():
    code = "def f(xs):\n    for x in xs:\n        print(x)\n    return 1\n"
    boundary = resolve(code, 2, 2)
    assert boundary.code == "for x in xs:\n        print(x)"
    assert boundary.context == ()  # its own header is part of code
    assert (boundary.location.start_line, boundary.location.end_line) == (2, 3)


def test_decorated_function_includes_decorators():
    code = "import functools\n\n@functools.cache\ndef fib(n):\n    return n\n"
    boundary = resolve(code, 4, 4)  # flag on the bare def line
    assert boundary.code.startswith("@functools.cache\ndef fib(n):")
    assert boundary.location.start_line == 3


def test_flag_on_decorator_line_only():
    code = "@dec\ndef f():\n    return 1\n"
    assert resolve(code, 1, 1).code == code.rstrip("\n")


def test_multi_statement_range_is_contiguous_run_not_enclosing_block():
    code = (
        "def f(a):\n"
        "    x = 1\n"     # 2
        "    y = 2\n"     # 3
        "    z = 3\n"     # 4
        "    w = 4\n"     # 5
        "    return a\n"
    )
    boundary = resolve(code, 3, 4)
    assert boundary.code == "y = 2\n    z = 3"
    assert boundary.context == ()
    assert (boundary.location.start_line, boundary.location.end_line) == (3, 4)


def test_worked_example_b_union_range_run():
    body = "".join(f"    s{i} = {i}\n" for i in range(1, 20))
    code = "def f():\n" + body  # statements on lines 2..20
    boundary = resolve(code, 10, 18)
    assert boundary.code.splitlines()[0] == "s9 = 9"
    assert boundary.code.splitlines()[-1].strip() == "s17 = 17"
    assert (boundary.location.start_line, boundary.location.end_line) == (10, 18)


def test_run_inside_nested_block_keeps_enclosing_headers():
    code = (
        "def f(xs):\n"
        "    for x in xs:\n"
        "        a = 1\n"   # 3
        "        b = 2\n"   # 4
        "        c = 3\n"   # 5
    )
    boundary = resolve(code, 3, 4)
    assert boundary.code == "a = 1\n        b = 2"
    assert boundary.context == ("for x in xs:",)


def test_range_covering_whole_compound_statement_returns_it():
    code = "def f(xs):\n    for x in xs:\n        a = 1\n        b = 2\n    return 0\n"
    boundary = resolve(code, 2, 4)
    assert boundary.code.startswith("for x in xs:")
    assert boundary.code.endswith("b = 2")
    assert boundary.context == ()


def test_elif_and_else_context_lists_clause_and_parent_header():
    code = "if a:\n    p = 0\nelif b:\n    q = 1\nelse:\n    r = 2\n"
    assert resolve(code, 4, 4).context == ("if a:", "elif b:")
    assert resolve(code, 6, 6).context == ("if a:", "else:")


def test_try_except_context():
    code = "def f():\n    try:\n        a = 1\n    except ValueError:\n        b = 2\n"
    assert resolve(code, 3, 3).context == ("try:",)
    assert resolve(code, 5, 5).context == ("try:", "except ValueError:")


def test_with_and_while_context():
    code = "with open(p) as fh:\n    while True:\n        line = fh.readline()\n"
    assert resolve(code, 3, 3).context == ("with open(p) as fh:", "while True:")


def test_context_stops_at_enclosing_function_inside_a_method():
    code = (
        "class A:\n"
        "    def m(self, xs):\n"
        "        for x in xs:\n"
        "            self.t += x\n"
    )
    boundary = resolve(code, 4, 4)
    assert boundary.code == "self.t += x"
    assert boundary.context == ("for x in xs:",)


def test_match_case_context():
    code = "match v:\n    case 1:\n        z = 1\n"
    assert resolve(code, 3, 3).context == ("match v:", "case 1:")


def test_semicolon_chain_single_line_falls_back_to_first_statement():
    boundary = resolve("x = 1; y = 2\nz = 3\n", 1, 1)
    assert boundary.code == "x = 1"
    assert (boundary.location.start_col, boundary.location.end_col) == (1, 6)


def test_semicolon_chain_in_multi_line_range_keeps_whole_run():
    boundary = resolve("x = 1; y = 2\nz = 3\n", 1, 2)
    assert boundary.code == "x = 1; y = 2\nz = 3"


def test_blank_and_comment_edges_are_trimmed_from_a_run():
    code = "def f():\n    a = 1\n\n    # note\n    b = 2\n    # tail\n    c = 3\n"
    boundary = resolve(code, 3, 6)  # blank, comment, b, comment
    assert boundary.code == "b = 2"


def test_leading_blank_lines_in_range_are_ignored():
    code = "def f():\n    a = 1\n\n    b = 2\n"
    assert resolve(code, 3, 4).code == "b = 2"


def test_multibyte_text_uses_character_columns():
    code = 's = "héllo"; t = 1\n'
    boundary = resolve(code, 1, 1)
    assert boundary.code == 's = "héllo"'
    assert (boundary.location.start_col, boundary.location.end_col) == (1, 12)


def test_crlf_source_is_supported():
    code = "def f():\r\n    a = 1\r\n    b = 2\r\n"
    assert resolve(code, 3, 3).code == "b = 2"


def test_multiline_statement_with_triple_quoted_string():
    code = 'def f():\n    s = """a\nb"""\n    return s\n'
    boundary = resolve(code, 2, 2)
    assert boundary.code == 's = """a\nb"""'
    assert (boundary.location.start_line, boundary.location.end_line) == (2, 3)


@pytest.mark.parametrize(
    "start,end", [(0, 1), (3, 2), (1, 99)], ids=["zero-start", "inverted", "past-end"]
)
def test_out_of_range_lines_raise(start, end):
    with pytest.raises(SegmentResolutionError):
        resolve("a = 1\nb = 2\n", start, end)


def test_blank_only_range_raises():
    with pytest.raises(SegmentResolutionError):
        resolve("a = 1\n\n\nb = 2\n", 2, 3)
