"""Event codes must match the ranges the migrations CHECK, or features go wrong without an error."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import shared_features
from shared_features import ENUMS, ExitStatus, Kind, Modifier, Origin, SourcePane

REPO_ROOT = Path(shared_features.__file__).resolve().parents[1]


def _find_migrations() -> Path:
    """Search for the file, since the migrations folder sits at different depths per layout."""
    for candidate in REPO_ROOT.rglob("002_tables.sql"):
        if "__pycache__" not in candidate.parts:
            return candidate.parent
    raise FileNotFoundError(
        f"002_tables.sql not found under {REPO_ROOT} -- enum parity cannot be "
        f"checked against a schema that is not present"
    )


MIGRATIONS_DIR = _find_migrations()
MIGRATION = MIGRATIONS_DIR / "002_tables.sql"

_BETWEEN = re.compile(r"(\w+)\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)", re.IGNORECASE)


def sql_ranges() -> dict[str, tuple[int, int]]:
    text = MIGRATION.read_text(encoding="utf-8")
    return {m.group(1).lower(): (int(m.group(2)), int(m.group(3)))
            for m in _BETWEEN.finditer(text)}


def test_the_migration_is_readable():
    assert MIGRATION.exists(), f"frozen migration missing at {MIGRATION}"
    assert sql_ranges(), "no CHECK ... BETWEEN ranges found; did the file move?"


@pytest.mark.parametrize(
    "enum, column",
    [(Kind, "kind"), (Origin, "origin"),
     (ExitStatus, "exit_status"), (SourcePane, "source_pane")],
)
def test_enum_values_match_the_sql_check_range(enum, column):
    low, high = sql_ranges()[column]
    values = sorted(int(m) for m in enum)
    assert values == list(range(low, high + 1)), (
        f"{enum.__name__} is {values} but {column} is CHECKed to {low}..{high}"
    )


def test_the_modifier_bitmask_fits_its_check_range():
    low, high = sql_ranges()["modifiers"]
    assert low == 0
    assert sum(int(f) for f in Modifier) == high, (
        "every modifier bit set must equal the CHECK upper bound"
    )
    assert [int(f) for f in Modifier] == [1, 2, 4, 8], "bits must be powers of two"


def test_sql_backed_enums_are_declared_as_such_in_the_registry():
    for name in ("Kind", "Origin", "ExitStatus", "SourcePane", "Modifier"):
        assert ENUMS[name]["authority"] == "sql"


def test_unconstrained_enums_are_declared_as_c1_authored():
    """The schema constrains none of these, so 'sql' would claim a guarantee it does not give."""
    text = MIGRATION.read_text(encoding="utf-8")
    for name, column in (("PasteOrigin", "paste_origin"), ("Visibility", "env_state")):
        assert ENUMS[name]["authority"] == "c1"
        assert not re.search(rf"CHECK\s*\([^)]*\b{column}\b[^)]*BETWEEN", text), (
            f"{column} gained a CHECK constraint -- promote it to authority: sql"
        )
    assert ENUMS["KeyClass"]["authority"] == "spec"


def test_the_migration_set_still_has_no_comment_on_statements():
    """A COMMENT ON would give the code map a second home."""
    for sql in MIGRATION.parent.glob("0*.sql"):
        assert "COMMENT ON" not in sql.read_text(encoding="utf-8").upper(), (
            f"{sql.name} added a COMMENT ON; the registry must stay the only map"
        )


MIGRATION_008 = MIGRATION.parent / "008_snapshot_version_and_retention.sql"


def test_capture_reason_matches_the_check_constraint_in_008():
    """A registry value outside 008's CHECK would fail at capture time, during an interview."""
    from shared_features import CaptureReason

    text = MIGRATION_008.read_text(encoding="utf-8")
    match = re.search(
        r"capture_reason\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)", text, re.IGNORECASE
    )
    assert match, "008 no longer constrains capture_reason"

    low, high = int(match.group(1)), int(match.group(2))
    values = [int(v) for v in CaptureReason]
    assert min(values) == low
    assert max(values) == high
    assert values == list(range(low, high + 1)), "the range must have no gaps"


def test_capture_reason_is_declared_sql_backed():
    assert ENUMS["CaptureReason"]["authority"] == "sql"
    assert ENUMS["CaptureReason"]["column"] == "code_snapshots.capture_reason"


def test_capture_reason_is_defined_once():
    """It is an enum, so it belongs in the registry and nowhere else. The
    resolution module holds only interviewer-facing wording."""
    source = (
        REPO_ROOT / "shared_features" / "code_resolution.py"
    ).read_text(encoding="utf-8")
    assert "class CaptureReason" not in source
    assert "from .feature_schema import CaptureReason" in source
