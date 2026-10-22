"""Rule evidence for the Knowledge Mapper: structural cues found with tree-sitter.

A cue is a hint, never a tag. The model decides the tags; cues only feed the
prompt and the cross-check. Finding no cue is a valid outcome.

The cue ids (C01..C25) follow the build brief's Appendix B. They were ported
from an ``ast`` prototype onto tree-sitter, because a segment is a run of
statements and may not parse as a whole module. Where this file differs from
the brief it says so next to the rule.
"""

from collections.abc import Iterator
from typing import Literal

import tree_sitter_python as tsp
from pydantic import BaseModel, ConfigDict
from tree_sitter import Language, Node, Parser

Strength = Literal["strong", "medium", "weak"]

_PARSER = Parser(Language(tsp.language()))

HASH_CTORS = {"dict", "set", "Counter", "defaultdict", "OrderedDict", "frozenset"}
HEAP_FUNCS = {"heappush", "heappop", "heapify", "heappushpop", "heapreplace", "nlargest", "nsmallest"}
BISECT_FUNCS = {"bisect", "bisect_left", "bisect_right", "insort"}
MATH_FUNCS = {"gcd", "lcm", "comb", "perm", "factorial", "isqrt", "sqrt", "log", "log2", "ceil", "floor"}
BARE_MATH_FUNCS = {"gcd", "comb", "isqrt", "factorial"}
CACHE_DECORATORS = {"cache", "lru_cache"}
MODULUS_NAMES = {"mod", "MOD"}
MODULUS_CONSTANTS = {"1000000007", "998244353"}
TREE_ATTRS = {"left", "right"}
WALK_ATTRS = {"left", "right", "children", "next"}
DISTANCE_NAMES = {"dist", "d", "inf"}


class Support(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tag_id: str
    strength: Strength


class CueHit(BaseModel):
    """One cue found in the code, and the tags it points at."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cue_id: str
    text: str  # short human string, for example "C03: heappush()"
    supports: tuple[Support, ...]


def _hit(cue_id: str, detail: str, *supports: tuple[str, Strength]) -> CueHit:
    return CueHit(
        cue_id=cue_id,
        text=f"{cue_id}: {detail}",
        supports=tuple(Support(tag_id=t, strength=s) for t, s in supports),
    )


# --- tree helpers -----------------------------------------------------------------


def _walk(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _text(node: Node | None) -> str:
    return "" if node is None else node.text.decode("utf-8", "replace")


def _unparen(node: Node | None) -> Node | None:
    while node is not None and node.type == "parenthesized_expression":
        inner = [c for c in node.children if c.is_named]
        node = inner[0] if inner else None
    return node


def _is_int(node: Node | None, value: int) -> bool:
    return node is not None and node.type == "integer" and _text(node) == str(value)


def _operator(node: Node) -> str:
    op = node.child_by_field_name("operator")
    return op.type if op is not None else ""


def _call_name(call: Node) -> str | None:
    func = call.child_by_field_name("function")
    if func is None:
        return None
    if func.type == "identifier":
        return _text(func)
    if func.type == "attribute":
        return _text(func.child_by_field_name("attribute"))
    return None


def _call_object(call: Node) -> Node | None:
    func = call.child_by_field_name("function")
    if func is not None and func.type == "attribute":
        return func.child_by_field_name("object")
    return None


def _call_arguments(call: Node) -> list[Node]:
    args = call.child_by_field_name("arguments")
    return [c for c in args.children if c.is_named] if args is not None else []


def _array_root(node: Node | None) -> str | None:
    """The name of the array a (possibly nested) subscript indexes into."""
    while node is not None and node.type == "subscript":
        node = node.child_by_field_name("value")
    return _text(node) if node is not None else None


def _inside_type_annotation(node: Node) -> bool:
    parent = node.parent
    while parent is not None:
        if parent.type == "type":
            return True
        parent = parent.parent
    return False


def _enclosing_function_name(node: Node) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type == "function_definition":
            return _text(parent.child_by_field_name("name"))
        parent = parent.parent
    return None


def _is_midpoint(node: Node) -> bool:
    """``(a + b) // 2``, ``(a + b) >> 1`` or ``a + (b - a) // 2``."""
    if node.type != "binary_operator":
        return False
    op, left, right = _operator(node), _unparen(node.child_by_field_name("left")), node.child_by_field_name("right")
    if op == "//" and _is_int(right, 2) or op == ">>" and _is_int(right, 1):
        return left is not None and left.type == "binary_operator" and _operator(left) == "+"
    if op == "+":
        right = _unparen(right)
        if right is not None and right.type == "binary_operator" and _operator(right) == "//":
            half = right.child_by_field_name("right")
            inner = _unparen(right.child_by_field_name("left"))
            return _is_int(half, 2) and inner is not None and inner.type == "binary_operator" and _operator(inner) == "-"
    return False


class _Scan:
    """One parse of the code plus the lookups several cues share."""

    def __init__(self, code: str, entry_function: str | None) -> None:
        self.source = code.encode("utf-8")
        self.root = _PARSER.parse(self.source).root_node
        self.nodes = list(_walk(self.root))
        self.by_type: dict[str, list[Node]] = {}
        for node in self.nodes:
            self.by_type.setdefault(node.type, []).append(node)
        self.midpoints = {n.id for n in self.by_type.get("binary_operator", []) if _is_midpoint(n)}
        self.entry_function = entry_function
        self.prefix_assignments: set[int] = set()  # filled by C07, read by C17

    def of(self, *types: str) -> list[Node]:
        return [n for t in types for n in self.by_type.get(t, [])]

    def calls(self) -> list[Node]:
        return self.by_type.get("call", [])

    def recursive_scopes(self) -> list[tuple[Node, str, Strength]]:
        """Scopes whose code calls the function it belongs to.

        A function that calls itself is a strong signal. When the segment
        itself calls the entry function, the segment is treated as a
        recursive body too, but only as a medium signal because the entry
        function is a guess.
        """
        scopes: list[tuple[Node, str, Strength]] = []
        for func in self.of("function_definition"):
            name = _text(func.child_by_field_name("name"))
            if any(self._calls_name(c, name) for c in _walk(func) if c.type == "call"):
                scopes.append((func, name, "strong"))
        if self.entry_function and not any(
            _text(f.child_by_field_name("name")) == self.entry_function
            for f in self.of("function_definition")
        ):
            if any(self._calls_name(c, self.entry_function) for c in self.calls()):
                scopes.append((self.root, self.entry_function, "medium"))
        return scopes

    @staticmethod
    def _calls_name(call: Node, name: str) -> bool:
        func = call.child_by_field_name("function")
        if func is None:
            return False
        if func.type == "identifier":
            return _text(func) == name
        if func.type == "attribute":
            owner = func.child_by_field_name("object")
            return (
                _text(func.child_by_field_name("attribute")) == name
                and owner is not None
                and owner.type == "identifier"
                and _text(owner) in {"self", "cls"}
            )
        return False


# --- the cues ------------------------------------------------------------------------


def _c01(scan: _Scan) -> list[CueHit]:  # weak: a dict or set is often incidental
    hits = []
    kinds = {
        "dictionary": "dict literal",
        "set": "set literal",
        "dictionary_comprehension": "dict comprehension",
        "set_comprehension": "set comprehension",
    }
    for node in scan.of(*kinds):
        hits.append(_hit("C01", kinds[node.type], ("hash_table", "weak")))
    for call in scan.calls():
        name = _call_name(call)
        if name in HASH_CTORS:
            hits.append(_hit("C01", f"{name}()", ("hash_table", "weak")))
    return hits


def _c02(scan: _Scan) -> list[CueHit]:
    hits = []
    for call in scan.calls():
        name = _call_name(call)
        if name == "sorted":
            hits.append(_hit("C02", "sorted()", ("sorting", "medium")))
        elif name == "sort" and _call_object(call) is not None:
            hits.append(_hit("C02", ".sort()", ("sorting", "medium")))
    return hits


def _c03(scan: _Scan) -> list[CueHit]:
    return [
        _hit("C03", f"{name}()", ("heap_priority_queue", "strong"))
        for call in scan.calls()
        if (name := _call_name(call)) in HEAP_FUNCS
    ]


def _c04(scan: _Scan) -> list[CueHit]:
    return [
        _hit("C04", f"{name}()", ("binary_search", "strong"))
        for call in scan.calls()
        if (name := _call_name(call)) in BISECT_FUNCS
    ]


def _c05(scan: _Scan) -> list[CueHit]:
    # Beyond the brief: also matches the overflow-safe form a + (b - a) // 2.
    for loop in scan.of("while_statement"):
        if any(n.id in scan.midpoints for n in _walk(loop)):
            return [_hit("C05", "midpoint inside a while loop", ("binary_search", "strong"))]
    return []


def _c06(scan: _Scan) -> list[CueHit]:
    return [
        _hit("C06", "accumulate()", ("prefix_sum", "strong"))
        for call in scan.calls()
        if _call_name(call) == "accumulate"
    ]


def _other_cells(node: Node, array: str, own_cell: str) -> list[Node]:
    """Cells of ``array`` read inside ``node`` other than the one being assigned."""
    return [
        n
        for n in _walk(node)
        if n.type == "subscript" and _array_root(n) == array and _text(n) != own_cell
    ]


def _c07(scan: _Scan) -> list[CueHit]:
    # Brief: s[i + 1] = s[i] + x. Fibonacci-style dp[i] = dp[i-1] + dp[i-2] has two
    # other cells and is left to C17, so a running sum is not mistaken for it.
    hits = []
    for assign in scan.of("assignment"):
        left, right = assign.child_by_field_name("left"), assign.child_by_field_name("right")
        if left is None or right is None or left.type != "subscript":
            continue
        right = _unparen(right)
        if right is None or right.type != "binary_operator" or _operator(right) != "+":
            continue
        array = _array_root(left)
        operands = [right.child_by_field_name("left"), right.child_by_field_name("right")]
        cells = [
            o for o in operands
            if o is not None and o.type == "subscript" and _array_root(o) == array and _text(o) != _text(left)
        ]
        if len(cells) == 1:
            scan.prefix_assignments.add(assign.id)
            hits.append(
                _hit("C07", "running-sum array", ("prefix_sum", "strong"), ("dynamic_programming", "weak"))
            )
    return hits


def _c08(scan: _Scan) -> list[CueHit]:
    hits = []
    for call in scan.calls():
        name = _call_name(call)
        owner = _call_object(call)
        if name in MATH_FUNCS and owner is not None and _text(owner) == "math":
            hits.append(_hit("C08", f"math.{name}()", ("math_number_theory", "medium")))
        elif name in BARE_MATH_FUNCS and call.child_by_field_name("function").type == "identifier":
            hits.append(_hit("C08", f"{name}()", ("math_number_theory", "medium")))
    return hits


def _c09(scan: _Scan) -> list[CueHit]:
    hits = []
    for node in scan.of("binary_operator"):
        if _operator(node) != "%":
            continue
        right = _unparen(node.child_by_field_name("right"))
        if right is None:
            continue
        # Beyond the brief: also the usual literal moduli.
        if (right.type == "identifier" and _text(right) in MODULUS_NAMES) or (
            right.type == "integer" and _text(right) in MODULUS_CONSTANTS
        ):
            hits.append(_hit("C09", "% modulus", ("math_number_theory", "medium")))
    return hits


def _c10(scan: _Scan) -> list[CueHit]:
    hits = []
    for node in scan.of("binary_operator", "augmented_assignment"):
        if _inside_type_annotation(node):
            continue
        op = _operator(node).rstrip("=") if node.type == "augmented_assignment" else _operator(node)
        if op in {"^", "<<"}:
            hits.append(_hit("C10a", f"{op} operator", ("bit_manipulation", "strong")))
        elif op in {"&", "|", ">>"} and node.id not in scan.midpoints:
            hits.append(_hit("C10b", f"{op} operator", ("bit_manipulation", "medium")))
    for call in scan.calls():
        if _call_name(call) == "bit_count":
            hits.append(_hit("C10a", ".bit_count()", ("bit_manipulation", "strong")))
    return hits


def _c11(scan: _Scan) -> list[CueHit]:
    return [
        _hit("C11", "deque()", ("stack_queue", "medium"))
        for call in scan.calls()
        if _call_name(call) == "deque"
    ]


def _c12(scan: _Scan) -> list[CueHit]:
    hits = []
    for call in scan.calls():
        if _call_object(call) is None:
            continue
        name = _call_name(call)
        args = _call_arguments(call)
        if name == "popleft" or (name == "pop" and len(args) == 1 and _is_int(args[0], 0)):
            detail = ".popleft()" if name == "popleft" else ".pop(0)"
            hits.append(_hit("C12", detail, ("stack_queue", "strong"), ("graph_traversal", "weak")))
    return hits


def _c13(scan: _Scan) -> list[CueHit]:
    appended: set[str] = set()
    popped: set[str] = set()
    for call in scan.calls():
        owner = _call_object(call)
        if owner is None:
            continue
        name = _call_name(call)
        if name == "append":
            appended.add(_text(owner))
        elif name == "pop" and not _call_arguments(call):
            popped.add(_text(owner))
    return [
        _hit("C13", f"{obj}.append() + {obj}.pop()", ("stack_queue", "strong"))
        for obj in sorted(appended & popped)
    ]


def _step_targets(loop: Node, operator: str) -> set[str]:
    return {
        _text(n.child_by_field_name("left"))
        for n in _walk(loop)
        if n.type == "augmented_assignment"
        and _operator(n) == operator
        and _is_int(n.child_by_field_name("right"), 1)
        and n.child_by_field_name("left").type == "identifier"
    }


def _c14(scan: _Scan) -> list[CueHit]:
    for loop in scan.of("while_statement"):
        up, down = _step_targets(loop, "+="), _step_targets(loop, "-=")
        if up - down and down - up:
            return [_hit("C14", "one index moves up, another down", ("two_pointers", "strong"))]
    return []


def _c15(scan: _Scan) -> list[CueHit]:
    for loop in scan.of("for_statement"):
        loop_vars = {_text(n) for n in _walk(loop.child_by_field_name("left")) if n.type == "identifier"}
        for inner in _walk(loop):
            if inner.type == "while_statement" and _step_targets(inner, "+=") - loop_vars:
                return [_hit("C15", "inner while advances another index", ("sliding_window", "medium"))]
    return []


def _c16(scan: _Scan) -> list[CueHit]:
    for loop in scan.of("for_statement"):
        if any(n.type == "for_statement" and n.id != loop.id for n in _walk(loop)):
            return [_hit("C16", "nested for loops", ("brute_force", "medium"))]
    return []


def _c17(scan: _Scan) -> list[CueHit]:
    # Skips statements C07 already read as a running sum (a prefix-sum table is
    # also a DP table, and one statement should not raise a strong cue for both).
    hits = []
    for assign in scan.of("assignment"):
        left, right = assign.child_by_field_name("left"), assign.child_by_field_name("right")
        if left is None or right is None or left.type != "subscript" or assign.id in scan.prefix_assignments:
            continue
        right = _unparen(right)
        if right is None:
            continue
        if right.type == "call":
            if _call_name(right) not in {"max", "min", "sum"}:
                continue
        elif right.type != "binary_operator":
            continue
        if _other_cells(right, _array_root(left), _text(left)):
            hits.append(_hit("C17", "table cell built from other cells", ("dynamic_programming", "strong")))
    return hits


def _c18(scan: _Scan) -> list[CueHit]:
    hits = []
    for decorator in scan.of("decorator"):
        expr = next((c for c in decorator.children if c.is_named), None)
        if expr is not None and expr.type == "call":
            expr = expr.child_by_field_name("function")
        name = _text(expr.child_by_field_name("attribute")) if expr is not None and expr.type == "attribute" else _text(expr)
        if name in CACHE_DECORATORS:
            hits.append(_hit("C18", f"@{name}", ("dynamic_programming", "strong")))
    return hits


def _c19(scan: _Scan) -> list[CueHit]:
    return [
        _hit("C19", f"{name}() calls itself" if s == "strong" else f"calls entry function {name}()",
             ("recursion_dc", s))
        for _, name, s in scan.recursive_scopes()
    ]


def _c20(scan: _Scan) -> list[CueHit]:
    hits = []
    for scope, name, _ in scan.recursive_scopes():
        attrs = {_text(n.child_by_field_name("attribute")) for n in _walk(scope) if n.type == "attribute"}
        touches = attrs & WALK_ATTRS
        has_subscript = any(n.type == "subscript" for n in _walk(scope))
        if not touches and not has_subscript:
            continue
        supports = [("graph_traversal", "medium")]
        if attrs & TREE_ATTRS:
            supports.append(("tree", "medium"))
        hits.append(_hit("C20", f"recursive {name}() walks a structure", *supports))
    return hits


def _c21(scan: _Scan, heap_found: bool) -> list[CueHit]:
    if not heap_found:
        return []
    names = {_text(n) for n in scan.of("identifier")}
    attrs = {_text(n.child_by_field_name("attribute")) for n in scan.of("attribute")}
    if names & DISTANCE_NAMES or "inf" in attrs:
        return [_hit("C21", "heap plus distance values", ("advanced_graph", "medium"))]
    return []


def _c22(scan: _Scan) -> list[CueHit]:
    hits = []
    for scope, name, _ in scan.recursive_scopes():
        recursive = [c.start_byte for c in _walk(scope) if c.type == "call" and scan._calls_name(c, name)]
        appends: dict[str, list[int]] = {}
        pops: dict[str, list[int]] = {}
        for call in (c for c in _walk(scope) if c.type == "call"):
            owner = _call_object(call)
            if owner is None:
                continue
            if _call_name(call) == "append":
                appends.setdefault(_text(owner), []).append(call.start_byte)
            elif _call_name(call) == "pop" and not _call_arguments(call):
                pops.setdefault(_text(owner), []).append(call.start_byte)
        for obj, starts in appends.items():
            if any(a < r < p for a in starts for r in recursive for p in pops.get(obj, [])):
                hits.append(_hit("C22", f"{obj}.append() before and {obj}.pop() after the recursive call",
                                 ("backtracking", "medium")))
                break
    return hits


def _c23(scan: _Scan) -> list[CueHit]:
    names = {_text(n) for n in scan.of("identifier")}
    next_count = sum(1 for n in scan.of("attribute") if _text(n.child_by_field_name("attribute")) == "next")
    if "ListNode" in names or next_count >= 2:
        return [_hit("C23", "linked nodes (ListNode / .next)", ("linked_list", "medium"))]
    return []


def _c24(scan: _Scan) -> list[CueHit]:
    names = {_text(n) for n in scan.of("identifier")}
    attrs = {_text(n.child_by_field_name("attribute")) for n in scan.of("attribute")}
    if "TreeNode" in names or attrs & TREE_ATTRS:
        return [_hit("C24", "tree nodes (TreeNode / .left / .right)", ("tree", "medium"))]
    return []


def _c25(scan: _Scan) -> list[CueHit]:
    hits = []
    for assign in scan.of("assignment"):
        left, right = assign.child_by_field_name("left"), assign.child_by_field_name("right")
        if left is None or right is None or left.type != "subscript" or right.type != "call":
            continue
        name = _call_name(right)
        args = _call_arguments(right)
        if (name == "find" or name == _enclosing_function_name(assign)) and len(args) == 1 \
                and _text(args[0]) == _text(left):
            hits.append(_hit("C25", "find() with path compression", ("advanced_graph", "medium")))
    return hits


def extract_cues(code: str, *, entry_function: str | None = None) -> list[CueHit]:
    """All cues found in ``code``, in cue-id order, each (id, text) once.

    ``entry_function`` is the best-effort name from Segment Extraction. A
    segment that calls it is treated as recursive.
    """
    scan = _Scan(code, entry_function)
    c03 = _c03(scan)
    hits: list[CueHit] = []
    for group in (
        _c01(scan), _c02(scan), c03, _c04(scan), _c05(scan), _c06(scan), _c07(scan),
        _c08(scan), _c09(scan), _c10(scan), _c11(scan), _c12(scan), _c13(scan), _c14(scan),
        _c15(scan), _c16(scan), _c17(scan), _c18(scan), _c19(scan), _c20(scan),
        _c21(scan, bool(c03)), _c22(scan), _c23(scan), _c24(scan), _c25(scan),
    ):
        hits.extend(group)
    seen: set[str] = set()
    unique = []
    for hit in hits:
        if hit.text not in seen:
            seen.add(hit.text)
            unique.append(hit)
    return unique


def evidence_by_tag(hits: list[CueHit]) -> dict[str, tuple[str, ...]]:
    """tag id -> the cue strings that support it, for the stored result."""
    evidence: dict[str, list[str]] = {}
    for hit in hits:
        for support in hit.supports:
            evidence.setdefault(support.tag_id, []).append(hit.text)
    return {tag_id: tuple(texts) for tag_id, texts in evidence.items()}
