import pytest

from backend.app.probing.knowledge.cues import evidence_by_tag, extract_cues


def ids(code: str, **kwargs) -> set[str]:
    return {hit.cue_id for hit in extract_cues(code, **kwargs)}


def supports(code: str, cue_id: str, **kwargs) -> set[tuple[str, str]]:
    return {
        (s.tag_id, s.strength)
        for hit in extract_cues(code, **kwargs)
        if hit.cue_id == cue_id
        for s in hit.supports
    }


# One positive and one negative snippet per cue. (id, code that fires, code that must not)
CASES = [
    ("C01", "seen = set()\nd = {1: 2}\n", "x = [1, 2]\n"),
    ("C02", "xs = sorted(items)\nitems.sort()\n", "x = len(items)\n"),
    ("C03", "heappush(h, 3)\nx = heappop(h)\n", "h.append(3)\n"),
    ("C04", "i = bisect_left(a, x)\n", "i = a.index(x)\n"),
    ("C05", "while lo < hi:\n    mid = (lo + hi) // 2\n", "mid = (lo + hi) // 2\n"),
    ("C06", "p = list(accumulate(a))\n", "p = sum(a)\n"),
    ("C07", "pre[i + 1] = pre[i] + a[i]\n", "count[x] = count[x] + 1\n"),
    ("C08", "g = math.gcd(a, b)\nc = comb(n, k)\n", "g = other.gcd(a, b)\n"),
    ("C09", "x = (a * b) % MOD\n", "x = a % 2\n"),
    ("C10a", "x = a ^ b\ny = 1 << k\n", "x = a + b\n"),
    ("C10b", "x = a & b\ny = a | b\n", "x = a and b\n"),
    ("C11", "q = deque([1])\n", "q = list([1])\n"),
    ("C12", "x = q.popleft()\ny = q.pop(0)\n", "x = q.pop()\n"),
    ("C13", "st.append(1)\nx = st.pop()\n", "st.append(1)\nx = other.pop()\n"),
    ("C14", "while i < j:\n    i += 1\n    j -= 1\n", "while i < j:\n    i += 1\n"),
    ("C15", "for r in range(n):\n    while bad:\n        l += 1\n", "for r in range(n):\n    r2 = r\n"),
    ("C16", "for i in a:\n    for j in a:\n        pass\n", "for i in a:\n    pass\nfor j in a:\n    pass\n"),
    ("C17", "dp[i] = max(dp[i - 1], dp[i - 2])\n", "dp[i] = 5\n"),
    ("C18", "@lru_cache(maxsize=None)\ndef f(n):\n    return n\n", "@staticmethod\ndef f(n):\n    return n\n"),
    ("C19", "def fact(n):\n    return n * fact(n - 1)\n", "def fact(n):\n    return n\n"),
    ("C20", "def walk(n):\n    walk(n.left)\n    walk(n.right)\n", "def walk(n):\n    walk(n - 1)\n"),
    ("C21", "heappush(h, (d, v))\n", "heappush(h, v)\n"),
    ("C22", "def go(i):\n    path.append(i)\n    go(i + 1)\n    path.pop()\n",
     "def go(i):\n    go(i + 1)\n    path.append(i)\n"),
    ("C23", "node = ListNode(0)\nnode.next.next = None\n", "node = Other(0)\n"),
    ("C24", "root = TreeNode(1)\n", "root = Other(1)\n"),
    ("C25", "def find(x):\n    parent[x] = find(parent[x])\n    return parent[x]\n",
     "def find(x):\n    parent[x] = 3\n    return parent[x]\n"),
]


@pytest.mark.parametrize("cue_id, positive, negative", CASES, ids=[c[0] for c in CASES])
def test_each_cue_fires_on_its_pattern_and_not_on_a_near_miss(cue_id, positive, negative):
    assert cue_id in ids(positive)
    assert cue_id not in ids(negative)


def test_empty_and_plain_code_have_no_cues():
    assert extract_cues("") == []
    assert extract_cues("x = 1\ny = x + 2\n") == []
    assert evidence_by_tag([]) == {}


def test_cues_carry_tags_and_strengths():
    assert supports("heappush(h, 1)\n", "C03") == {("heap_priority_queue", "strong")}
    assert supports("d = {}\n", "C01") == {("hash_table", "weak")}
    assert supports("x = q.popleft()\n", "C12") == {("stack_queue", "strong"), ("graph_traversal", "weak")}
    assert supports("pre[i + 1] = pre[i] + a[i]\n", "C07") == {
        ("prefix_sum", "strong"),
        ("dynamic_programming", "weak"),
    }


def test_cue_text_is_short_and_readable():
    (hit,) = [h for h in extract_cues("heappush(h, 1)\n") if h.cue_id == "C03"]
    assert hit.text == "C03: heappush()"


def test_each_cue_text_appears_once():
    texts = [h.text for h in extract_cues("heappush(h, 1)\nheappush(h, 2)\n")]
    assert len(texts) == len(set(texts))


# --- the finer points ---------------------------------------------------------------


def test_running_sum_is_not_also_a_strong_dp_cue():
    code = "pre[i + 1] = pre[i] + a[i]\n"
    assert "C07" in ids(code) and "C17" not in ids(code)


def test_fibonacci_table_is_dp_not_prefix_sum():
    code = "dp[i] = dp[i - 1] + dp[i - 2]\n"
    assert "C17" in ids(code) and "C07" not in ids(code)


def test_incrementing_the_same_cell_is_neither():
    code = "count[x] = count[x] + 1\n"
    assert not ids(code) & {"C07", "C17"}


def test_overflow_safe_midpoint_counts_as_binary_search():
    assert "C05" in ids("while lo < hi:\n    mid = lo + (hi - lo) // 2\n")


def test_midpoint_shift_is_not_a_bit_manipulation_cue():
    code = "while lo < hi:\n    mid = (lo + hi) >> 1\n"
    assert "C05" in ids(code) and "C10b" not in ids(code)


def test_type_annotation_pipe_is_not_a_bit_operator():
    assert "C10b" not in ids("def f(x: int | None) -> None:\n    pass\n")


def test_set_add_inside_a_method_named_add_is_not_recursion():
    code = "class S:\n    def add(self, x):\n        self.seen.add(x)\n"
    assert "C19" not in ids(code)


def test_method_recursion_through_self_is_found():
    code = "class S:\n    def dfs(self, n):\n        self.dfs(n.next)\n"
    assert "C19" in ids(code)


def test_calling_the_entry_function_marks_the_segment_recursive_but_only_medium():
    code = "return fib(n - 1) + fib(n - 2)\n"
    assert "C19" not in ids(code)
    assert supports(code, "C19", entry_function="fib") == {("recursion_dc", "medium")}


def test_indented_slice_of_a_function_body_is_still_read():
    code = "for item in items:\n        if item.active:\n            seen = set()\n    return seen"
    assert "C01" in ids(code)


def test_syntax_error_does_not_raise():
    assert isinstance(extract_cues("def broken(:\n    heappush(h, 1)\n"), list)


def test_evidence_by_tag_groups_cue_strings():
    evidence = evidence_by_tag(extract_cues("x = q.popleft()\nseen = set()\n"))
    assert evidence["stack_queue"] == ("C12: .popleft()",)
    assert set(evidence) == {"stack_queue", "graph_traversal", "hash_table"}
