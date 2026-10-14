from backend.app.db.migrate import (
    MIGRATIONS_DIR,
    changed_since_applied,
    checksum,
    migration_files,
    pending,
)


def _applied(files):
    return {path.name: checksum(path.read_text(encoding="utf-8")) for path in files}


def test_only_numbered_files_are_migrations_in_order():
    names = [path.name for path in migration_files()]
    assert names[:8] == [
        "001_extensions.sql",
        "002_tables.sql",
        "003_hypertables.sql",
        "004_indexes.sql",
        "005_classifier_output.sql",
        "006_classifier_flag_final_locations.sql",
        "007_code_snapshots.sql",
        "008_snapshot_version_and_retention.sql",
    ]
    assert not any(name.startswith("verify_") for name in names)


def test_comments_and_blank_lines_do_not_change_the_checksum():
    sql = (MIGRATIONS_DIR / "004_indexes.sql").read_text(encoding="utf-8")
    assert checksum(sql) == checksum("-- a new note\n\n" + sql.replace("\n", "\n\n"))


def test_a_change_to_the_sql_changes_the_checksum():
    sql = (MIGRATIONS_DIR / "003_hypertables.sql").read_text(encoding="utf-8")
    assert checksum(sql) != checksum(sql.replace("1 hour", "1 day"))


def test_pending_skips_applied_files_and_keeps_the_order():
    files = migration_files()
    assert pending(files, _applied(files[:3])) == files[3:]


def test_an_applied_file_with_changed_sql_is_reported():
    files = migration_files()
    applied = _applied(files)
    assert changed_since_applied(files, applied) == []

    applied["003_hypertables.sql"] = checksum("SELECT 1;")
    assert changed_since_applied(files, applied) == ["003_hypertables.sql"]
