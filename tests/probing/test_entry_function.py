from backend.app.probing.languages import get_extractor

EXTRACTOR = get_extractor("python")


def entry(code: str):
    result = EXTRACTOR.find_entry_function(code)
    return None if result is None else (result.name, result.uncertain)


def test_single_function_is_the_entry():
    assert entry("def solve(xs):\n    return sorted(xs)\n") == ("solve", False)


def test_called_helper_is_not_the_entry():
    code = (
        "def helper(x):\n"
        "    return x * 2\n"
        "\n"
        "def solve(xs):\n"
        "    return [helper(x) for x in xs]\n"
    )
    assert entry(code) == ("solve", False)


def test_flat_script_has_no_entry_function():
    assert entry("x = 1\nprint(x)\n") is None


def test_empty_file_has_no_entry_function():
    assert entry("") is None


def test_largest_zero_incoming_function_wins_on_statements():
    code = (
        "def small():\n"
        "    return 1\n"
        "\n"
        "def big(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        total += x\n"
        "    return total\n"
    )
    assert entry(code) == ("big", False)


def test_statement_tie_is_broken_by_line_count():
    code = (
        "def compact(a, b):\n"
        "    x = a + b\n"
        "    return x\n"
        "\n"
        "def spread(a, b):\n"
        "    x = (a +\n"
        "         b)\n"
        "    return x\n"
    )
    assert entry(code) == ("spread", False)


def test_full_tie_is_uncertain_and_picks_first_in_file():
    code = "def first(a):\n    return a\n\ndef second(a):\n    return a\n"
    assert entry(code) == ("first", True)


def test_self_recursion_does_not_count_as_being_called():
    code = "def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\n"
    assert entry(code) == ("fact", False)


def test_module_level_driver_call_is_not_an_incoming_edge():
    code = (
        "def solve(n):\n"
        "    return n + 1\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    print(solve(3))\n"
    )
    assert entry(code) == ("solve", False)


def test_mutual_recursion_only_is_uncertain():
    code = (
        "def even(n):\n"
        "    return True if n == 0 else odd(n - 1)\n"
        "\n"
        "def odd(n):\n"
        "    return False if n == 0 else even(n - 1)\n"
    )
    name, uncertain = entry(code)
    assert name in {"even", "odd"}
    assert uncertain is True


def test_method_called_through_self_is_not_the_entry():
    code = (
        "class Solution:\n"
        "    def helper(self, x):\n"
        "        return x + 1\n"
        "\n"
        "    def solve(self, xs):\n"
        "        return [self.helper(x) for x in xs]\n"
    )
    assert entry(code) == ("solve", False)


def test_method_called_through_an_instance_attribute():
    code = (
        "class Counter:\n"
        "    def bump(self):\n"
        "        return 1\n"
        "\n"
        "def run(c):\n"
        "    return c.bump()\n"
    )
    assert entry(code) == ("run", False)


def test_dunder_methods_are_never_the_entry():
    code = (
        "class Solver:\n"
        "    def __init__(self, data):\n"
        "        self.data = data\n"
        "        self.size = len(data)\n"
        "        self.seen = set()\n"
        "\n"
        "    def run(self):\n"
        "        return self.size\n"
    )
    assert entry(code) == ("run", False)


def test_calls_made_by_dunder_methods_still_count():
    code = (
        "class Solver:\n"
        "    def __init__(self):\n"
        "        self.prepare()\n"
        "\n"
        "    def prepare(self):\n"
        "        return 1\n"
        "\n"
        "    def run(self):\n"
        "        return 2\n"
    )
    # prepare is called (by __init__); run is the only uncalled non-dunder
    assert entry(code) == ("run", False)


def test_only_dunder_functions_fall_back_to_the_whole_pool():
    code = "class A:\n    def __init__(self):\n        self.x = 1\n"
    assert entry(code) == ("__init__", False)


def test_nested_helper_called_by_its_parent_is_not_a_candidate():
    code = (
        "def outer(xs):\n"
        "    def inner(x):\n"
        "        return x * 2\n"
        "    return [inner(x) for x in xs]\n"
    )
    assert entry(code) == ("outer", False)


def test_same_named_methods_in_different_classes_are_indistinguishable():
    # Accepted limitation: no type resolution. A.go is called via b.go(), and
    # because matching is by name, B.go looks called too.
    code = (
        "class A:\n"
        "    def go(self):\n"
        "        return 1\n"
        "\n"
        "class B:\n"
        "    def go(self):\n"
        "        return 2\n"
        "\n"
        "def main(b):\n"
        "    return b.go()\n"
    )
    assert entry(code) == ("main", False)


def test_heuristic_survives_a_syntax_error():
    code = "def solve(xs):\n    return sorted(xs\n"
    result = EXTRACTOR.find_entry_function(code)
    assert result is not None and result.name == "solve"


def reason(code: str):
    return EXTRACTOR.find_entry_function(code).reason


def test_reason_is_none_when_confident():
    assert reason("def solve(xs):\n    return xs\n") is None


def test_reason_for_tied_candidates():
    code = "def first(a):\n    return a\n\ndef second(a):\n    return a\n"
    assert reason(code) == "tied_candidates"


def test_reason_for_a_cycle_with_no_uncalled_function():
    code = (
        "def even(n):\n"
        "    return True if n == 0 else odd(n - 1)\n"
        "\n"
        "def odd(n):\n"
        "    return False if n == 0 else even(n - 1)\n"
    )
    assert reason(code) == "no_uncalled_function"


def test_cycle_fallback_prefers_fewest_callers_over_sheer_size():
    # Everything is called. `util` is the largest but has two callers; `a` and
    # `b` have one each, so the guess comes from them, not from `util`.
    code = (
        "def util(x):\n"
        "    y = x + 1\n"
        "    z = y * 2\n"
        "    w = z - 3\n"
        "    return w\n"
        "\n"
        "def a(n):\n"
        "    if n <= 0:\n"
        "        return util(n)\n"
        "    return b(n - 1) + util(n)\n"
        "\n"
        "def b(n):\n"
        "    return a(n - 1) + util(n)\n"
    )
    result = EXTRACTOR.find_entry_function(code)
    assert (result.name, result.uncertain, result.reason) == ("a", True, "no_uncalled_function")
