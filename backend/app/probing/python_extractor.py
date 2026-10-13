"""Python (tree-sitter) support for C3 Segment Extraction.

Everything Python-specific lives here so a second language is an additive
entry in ``languages.PARSERS`` and never a change to this module.

Boundary resolution (brief section 5): given a flagged line range in
final-submission coordinates, find the smallest complete statement, or run of
sibling statements, covering it, and the headers of the compound statements
it sits inside.
"""

import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree

from backend.app.probing.errors import SegmentResolutionError
from backend.app.probing.schemas import EntryFunction, Location, ResolvedBoundary

LANGUAGE_NAME = "python"

# Grammar node types whose direct children are statements. A segment's
# boundary is the child statement sitting directly inside one of these.
STATEMENT_CONTAINER_TYPES: frozenset[str] = frozenset({"module", "block"})

# Compound statements and clauses whose header line(s) become ``context``.
_HEADER_TYPES: frozenset[str] = frozenset(
    {
        "if_statement",
        "elif_clause",
        "else_clause",
        "for_statement",
        "while_statement",
        "with_statement",
        "try_statement",
        "except_clause",
        "except_group_clause",
        "finally_clause",
        "match_statement",
        "case_clause",
    }
)

# Context never crosses these: the function or module level is not context.
_CONTEXT_STOP_TYPES: frozenset[str] = frozenset({"function_definition", "class_definition"})

_BLANK = b" \t\r\f\v"


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


class PythonSegmentExtractor:
    language = LANGUAGE_NAME

    def __init__(self) -> None:
        self._parser = Parser(Language(tree_sitter_python.language()))

    def parse(self, code: str) -> Tree:
        """Parse a full submitted file. The tree is returned even with syntax errors."""
        return self._parser.parse(code.encode("utf-8"))

    def resolve_boundary(self, code: str, start_line: int, end_line: int) -> ResolvedBoundary:
        """Resolve a flagged line range (1-based, inclusive) to a segment.

        ``start_line``/``end_line`` are trusted as final-submission coordinates;
        they are only bounds-checked here, never re-mapped.
        """
        source = code.encode("utf-8")
        lines = source.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise SegmentResolutionError(
                "out_of_range",
                f"line range {start_line}-{end_line} is outside the {len(lines)}-line submission",
            )

        first, last = _content_span(lines, start_line - 1, end_line - 1)
        tree = self.parse(code)
        covering = tree.root_node.named_descendant_for_point_range(first, last)
        if covering is None:
            raise SegmentResolutionError(
                "unparseable_range", f"no syntax node covers lines {start_line}-{end_line}"
            )

        statements = _boundary_statements(covering, first[0], last[0])
        if not statements:
            raise SegmentResolutionError(
                "unparseable_range", f"no statement covers lines {start_line}-{end_line}"
            )
        # Defensive, like the redacted-code check: C2 hands over finished code,
        # so a syntax error here means the data is inconsistent. Do not build a
        # probe on a region the parser could not understand. ``has_error`` is
        # true for ERROR nodes and MISSING tokens in the statement's subtree.
        if any(statement.has_error for statement in statements):
            raise SegmentResolutionError(
                "unparseable_range",
                f"lines {start_line}-{end_line} resolve to code with syntax errors",
            )

        head, tail = statements[0], statements[-1]
        return ResolvedBoundary(
            code=source[head.start_byte : tail.end_byte].decode("utf-8"),
            context=tuple(_context_headers(head, source)),
            location=Location(
                start_line=head.start_point.row + 1,
                end_line=tail.end_point.row + 1,
                start_col=_char_column(lines, head.start_point) + 1,
                end_col=_char_column(lines, tail.end_point) + 1,
            ),
        )


    def find_entry_function(self, code: str) -> EntryFunction | None:
        """Best-effort guess at the function a driver would call (brief section 8).

        Same-file, static, by name only: ``foo(...)`` and ``x.foo(...)`` both
        count as a call to any function named ``foo``. No type resolution, so
        two classes' same-named methods are indistinguishable (accepted).

        Judgment calls, all in favour of finding the function nothing else
        *inside the file's functions* depends on:
          * only calls made from within a function body are edges; module-level
            driver code (``print(solve(3))``, ``if __name__ ...``) is not;
          * a function calling itself does not count as being called;
          * dunder methods are never candidates (they are invoked implicitly),
            but the calls they make still are edges.

        Returns ``None`` when the file defines no function. Otherwise: the one
        function with no incoming call; else the largest of those (most
        statements, then most lines); else ``uncertain=True`` with the best
        guess (``reason="tied_candidates"``). If every function is called by
        another one (a cycle), the guess is the largest among the functions
        with the fewest distinct callers and is always uncertain
        (``reason="no_uncalled_function"``).
        """
        root = self.parse(code).root_node
        functions: list[Node] = []
        calls: list[Node] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "function_definition":
                functions.append(node)
            elif node.type == "call":
                calls.append(node)
            stack.extend(reversed(node.children))
        if not functions:
            return None

        names = {fn.id: _function_name(fn) for fn in functions}
        by_name: dict[str, list[Node]] = {}
        for fn in functions:
            by_name.setdefault(names[fn.id], []).append(fn)

        # function id -> ids of the distinct other functions that call it
        callers: dict[int, set[int]] = {fn.id: set() for fn in functions}
        for call in calls:
            callee = _callee_name(call)
            caller = _enclosing_function(call)
            if callee is None or caller is None:
                continue
            for target in by_name.get(callee, ()):
                if target.id != caller.id:
                    callers[target.id].add(caller.id)

        pool = [fn for fn in functions if not _is_dunder(names[fn.id])] or functions
        uncalled = [fn for fn in pool if not callers[fn.id]]
        if uncalled:
            ranked_pool, no_uncalled = uncalled, False
        else:
            # Everything is called by something (a cycle). Take the functions
            # with the fewest distinct callers, so a big, widely-used helper
            # is not preferred just for being large.
            fewest = min(len(callers[fn.id]) for fn in pool)
            ranked_pool = [fn for fn in pool if len(callers[fn.id]) == fewest]
            no_uncalled = True

        ranked = sorted(ranked_pool, key=lambda fn: (*(-v for v in _size(fn)), fn.start_byte))
        best = ranked[0]
        if no_uncalled:
            reason = "no_uncalled_function"
        elif len(ranked) > 1 and _size(ranked[1]) == _size(best):
            reason = "tied_candidates"
        else:
            reason = None
        return EntryFunction(name=names[best.id], uncertain=reason is not None, reason=reason)


def _function_name(function: Node) -> str:
    name = function.child_by_field_name("name")
    return name.text.decode("utf-8") if name is not None and name.text else ""


def _callee_name(call: Node) -> str | None:
    target = call.child_by_field_name("function")
    if target is None:
        return None
    if target.type == "attribute":
        target = target.child_by_field_name("attribute")
    if target is not None and target.type == "identifier" and target.text:
        return target.text.decode("utf-8")
    return None


def _enclosing_function(node: Node) -> Node | None:
    parent = node.parent
    while parent is not None and parent.type != "function_definition":
        parent = parent.parent
    return parent


def _size(function: Node) -> tuple[int, int]:
    """(statement count, line count); statements include nested functions' own."""
    statements = 0
    stack = list(function.children)
    while stack:
        node = stack.pop()
        if (
            node.parent is not None
            and node.parent.type in STATEMENT_CONTAINER_TYPES
            and node.type != "comment"
        ):
            statements += 1
        stack.extend(node.children)
    return statements, function.end_point.row - function.start_point.row + 1


def _content_span(
    lines: list[bytes], first_row: int, last_row: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Points (row, byte column) of the first and one-past-last non-blank bytes."""
    first: tuple[int, int] | None = None
    last: tuple[int, int] | None = None
    for row in range(first_row, last_row + 1):
        stripped = lines[row].strip(_BLANK)
        if not stripped:
            continue
        if first is None:
            first = (row, len(lines[row]) - len(lines[row].lstrip(_BLANK)))
        last = (row, len(lines[row].rstrip(_BLANK)))
    if first is None or last is None:
        raise SegmentResolutionError(
            "blank_range", f"lines {first_row + 1}-{last_row + 1} contain no code"
        )
    return first, last


def _boundary_statements(covering: Node, first_row: int, last_row: int) -> list[Node]:
    """The statement, or contiguous run of sibling statements, to use as ``code``."""
    if covering.type in STATEMENT_CONTAINER_TYPES:
        # The range spans several siblings: take the touched run, not the block.
        touched = [
            child
            for child in covering.named_children
            if child.start_point.row <= last_row and child.end_point.row >= first_row
        ]
        while len(touched) > 1 and touched[0].type == "comment":
            touched.pop(0)
        while len(touched) > 1 and touched[-1].type == "comment":
            touched.pop()
        # KNOWN EDGE CASE: ``x = 1; y = 2``. Flags are line-granular, so both
        # statements are touched; per the brief fall back to the first one when
        # the flag is a single line. Multi-line ranges keep the whole run.
        if first_row == last_row and len(touched) > 1:
            touched = touched[:1]
        return touched

    node = covering
    parent = node.parent
    while parent is not None and parent.type not in STATEMENT_CONTAINER_TYPES:
        node, parent = parent, parent.parent
    # A decorated function is reached here as ``decorated_definition`` because
    # ``function_definition`` is not directly inside a container: the
    # decorators come along without special handling.
    return [node]


def _context_headers(statement: Node, source: bytes) -> list[str]:
    """Header text of every enclosing compound statement, outermost first."""
    headers: list[str] = []
    node = statement.parent  # the container holding the segment
    while node is not None:
        parent = node.parent
        if parent is None or parent.type in _CONTEXT_STOP_TYPES:
            break
        if parent.type in _HEADER_TYPES:
            body = next((c for c in parent.children if c.type == "block"), None)
            if body is not None:
                header = source[parent.start_byte : body.start_byte].decode("utf-8").strip()
                headers.append(header)
        node = parent
    headers.reverse()
    return headers


def _char_column(lines: list[bytes], point) -> int:
    """Convert a tree-sitter byte column to a 0-based character column."""
    row, byte_column = point.row, point.column
    if row >= len(lines):
        return byte_column
    return len(lines[row][:byte_column].decode("utf-8", errors="ignore"))
