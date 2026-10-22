"""Static checks on a generated probe. Nothing here runs the model's code.

The reference solution and cases are read with ``ast`` and ``json`` only.
"""

import ast
import json
from dataclasses import dataclass
from typing import Any

from backend.app.probing.probes.schema import CheckResult, ProbeCase, StaticChecks

ALLOWED_IMPORTS = frozenset(
    {"heapq", "bisect", "collections", "itertools", "functools", "math", "typing", "string", "operator"}
)
FORBIDDEN_NAMES = frozenset(
    {"eval", "exec", "compile", "open", "__import__", "input", "globals", "locals", "vars",
     "getattr", "setattr", "delattr", "breakpoint", "exit", "quit"}
)
MIN_CASES, MAX_CASES = 4, 10
MAX_QUESTION_CHARS = 1500
MAX_REFERENCE_LINES = 60


@dataclass(frozen=True)
class Checked:
    checks: StaticChecks
    cases: tuple[ProbeCase, ...]  # empty when the cases could not be read


def check_draft(draft: dict[str, Any]) -> Checked:
    """Run every check; never stops at the first failure so the report is complete."""
    results: list[CheckResult] = []

    def add(name: str, passed: bool, detail: str = "") -> bool:
        results.append(CheckResult(name=name, passed=passed, detail="" if passed else detail))
        return passed

    question = draft.get("question")
    add(
        "question_present",
        isinstance(question, str) and 20 <= len(question.strip()) <= MAX_QUESTION_CHARS,
        f"question must be 20 to {MAX_QUESTION_CHARS} characters",
    )

    name = draft.get("function_name")
    reference = draft.get("reference_solution")
    params = _signature_params(draft.get("signature"), name, add)
    tree = _parse_reference(reference, add)
    if tree is not None:
        _check_reference(tree, reference, name, params, add)

    add(
        "checker_known",
        draft.get("checker", "exact") in {"exact", "unordered", "float_tolerance"},
        "checker must be exact, unordered or float_tolerance",
    )
    cases = _read_cases(draft.get("cases"), params, add)
    return Checked(StaticChecks(passed=all(r.passed for r in results), checks=tuple(results)), cases)


def _signature_params(signature, name, add) -> list[str] | None:
    if not isinstance(signature, str) or not isinstance(name, str):
        add("signature_parses", False, "signature and function_name must be text")
        return None
    func = None
    try:
        module = ast.parse(signature.strip().rstrip(":") + ":\n    pass\n")
        if len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef):
            func = module.body[0]
    except SyntaxError:
        pass
    if func is None:
        add("signature_parses", False, "signature must be a single def line: def name(args) -> type")
        return None
    add("signature_parses", True)
    add("signature_names_function", func.name == name, "function_name must match the signature")
    return [a.arg for a in func.args.args]


def _parse_reference(reference, add) -> ast.Module | None:
    if not isinstance(reference, str) or not reference.strip():
        add("reference_parses", False, "reference solution is missing")
        return None
    try:
        tree = ast.parse(reference)
    except (SyntaxError, ValueError):
        add("reference_parses", False, "reference solution is not valid Python")
        return None
    add("reference_parses", True)
    return tree


def _check_reference(tree, source, name, params, add) -> None:
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    target = funcs.get(name) if isinstance(name, str) else None
    add("reference_defines_function", target is not None, "reference must define the named function at top level")
    if target is not None and params is not None:
        add(
            "reference_params_match",
            [a.arg for a in target.args.args] == params,
            "reference parameters must match the signature",
        )

    bad_imports: list[str] = []
    bad_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad_imports += [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                bad_imports.append(node.module or ".")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad_names.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            bad_names.append(node.attr)
    add("reference_imports_allowed", not bad_imports, f"imports not allowed: {sorted(set(bad_imports))}")
    add("reference_no_dangerous_names", not bad_names, f"names not allowed: {sorted(set(bad_names))}")
    add(
        "reference_short",
        len(source.strip().splitlines()) <= MAX_REFERENCE_LINES,
        f"reference longer than {MAX_REFERENCE_LINES} lines",
    )
    unbounded = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.While)
        and isinstance(n.test, ast.Constant)
        and n.test.value
        and not any(isinstance(c, (ast.Break, ast.Return)) for c in ast.walk(n))
    ]
    add("reference_no_unbounded_loop", not unbounded, "'while True' with no break or return")


def _read_cases(raw, params, add) -> tuple[ProbeCase, ...]:
    if not isinstance(raw, list) or not MIN_CASES <= len(raw) <= MAX_CASES:
        add("cases_count", False, f"need {MIN_CASES} to {MAX_CASES} cases")
        return ()
    add("cases_count", True)
    cases: list[ProbeCase] = []
    problems: list[str] = []
    for index, item in enumerate(raw, start=1):
        try:
            args = json.loads(item["args_json"])
            expected = json.loads(item["expected_json"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"case {index}: args_json and expected_json must be valid JSON")
            continue
        if not isinstance(args, list):
            problems.append(f"case {index}: args_json must be a JSON list of arguments")
            continue
        if params is not None and len(args) != len(params):
            problems.append(f"case {index}: expected {len(params)} arguments, got {len(args)}")
            continue
        note = item.get("note", "")
        cases.append(
            ProbeCase(
                case_id=f"c{index}",
                args=tuple(args),
                expected=expected,
                note=note if isinstance(note, str) else "",
            )
        )
    if len({json.dumps(c.args, sort_keys=True) for c in cases}) != len(cases):
        problems.append("two cases have identical arguments")
    add("cases_well_formed", not problems, "; ".join(problems))
    return tuple(cases) if not problems else ()
