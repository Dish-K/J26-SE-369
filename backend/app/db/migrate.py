import asyncio
import hashlib
import re
import sys
from pathlib import Path
import asyncpg

from backend.app.db.pool import close_db_pool, create_db_pool

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_COMMENT = re.compile(r"--[^\n]*")

_LEDGER = """ CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now() ) """

class MigrationError(Exception):
    pass

def migration_files(folder: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(folder.glob("[0-9][0-9][0-9]_*.sql"))

def checksum(sql: str) -> str:
    # Comments and spacing are ignored, so only a change to the SQL itself counts.
    code = " ".join(_COMMENT.sub("", sql).split())
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def pending(files: list[Path], applied: dict[str, str]) -> list[Path]:
    return [path for path in files if path.name not in applied]

def changed_since_applied(files: list[Path], applied: dict[str, str]) -> list[str]:
    return [
        path.name
        for path in files
        if path.name in applied
        and applied[path.name] != checksum(path.read_text(encoding="utf-8"))
    ]

async def applied_migrations(connection: asyncpg.Connection) -> dict[str, str]:
    await connection.execute(_LEDGER)
    rows = await connection.fetch("SELECT filename, checksum FROM schema_migrations")
    return {row["filename"]: row["checksum"] for row in rows}

async def apply_migration(connection: asyncpg.Connection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")

    # The file and its ledger row commit together, so a failed file leaves no trace.
    async with connection.transaction():
        await connection.execute(sql)
        await connection.execute(
            "INSERT INTO schema_migrations (filename, checksum) VALUES ($1, $2)",
            path.name,
            checksum(sql),
        )

async def migrate() -> list[str]:
    files = migration_files()
    pool = await create_db_pool()

    try:
        async with pool.acquire() as connection:
            applied = await applied_migrations(connection)

            changed = changed_since_applied(files, applied)
            if changed:
                raise MigrationError(
                    f"Changed after being applied: {', '.join(changed)}. "
                    "Put the change in a new migration file instead."
                )

            done = []
            for path in pending(files, applied):
                try:
                    await apply_migration(connection, path)
                except asyncpg.PostgresError as exc:
                    raise MigrationError(f"{path.name} failed: {exc}") from exc
                print(f"Applied {path.name}")
                done.append(path.name)
            return done
    finally:
        await close_db_pool(pool)

def main() -> int:
    try:
        done = asyncio.run(migrate())
    except MigrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not done:
        print("Nothing to apply")
    return 0

if __name__ == "__main__":
    sys.exit(main())

