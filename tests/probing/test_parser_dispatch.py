import pytest

from backend.app.probing.languages import PARSERS, LanguageNotSupportedError, get_extractor
from backend.app.probing.python_extractor import (
    STATEMENT_CONTAINER_TYPES,
    PythonSegmentExtractor,
)

SAMPLE = """def process(items):
    for item in items:
        if item.active:
            score = item.value * 2
    return score
"""


def test_only_python_is_registered():
    assert set(PARSERS) == {"python"}
    assert isinstance(get_extractor("python"), PythonSegmentExtractor)


def test_unknown_language_is_rejected():
    with pytest.raises(LanguageNotSupportedError):
        get_extractor("java")


def test_parse_returns_module_tree_without_errors():
    tree = get_extractor("python").parse(SAMPLE)
    root = tree.root_node
    assert root.type == "module"
    assert not root.has_error
    assert root.children[0].type == "function_definition"


def test_parse_keeps_original_line_numbers():
    tree = get_extractor("python").parse(SAMPLE)
    function = tree.root_node.children[0]
    assert function.start_point.row == 0  # tree-sitter rows are zero-based
    assert function.end_point.row == 4


def test_parse_tolerates_syntax_errors():
    tree = get_extractor("python").parse("def broken(:\n    pass\n")
    assert tree.root_node.type == "module"
    assert tree.root_node.has_error


def test_statement_container_types():
    assert STATEMENT_CONTAINER_TYPES == {"module", "block"}
