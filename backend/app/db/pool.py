import os

import asyncpg
from dotenv import load_dotenv

def _required_environment_value(name: str) -> str:
    value = os.getenv(name)

    if value is None or value == "":
        raise RuntimeError(f"Required environment variable is not set: {name}")

    return value

def _database_port() -> int:
    value = _required_environment_value("DB_PORT")

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError("DB_PORT must be an integer") from exc

async def create_db_pool() -> asyncpg.Pool:
    load_dotenv(override=False)

    return await asyncpg.create_pool(
        host=_required_environment_value("DB_HOST"),
        port=_database_port(),
        database=_required_environment_value("DB_NAME"),
        user=_required_environment_value("DB_USER"),
        password=_required_environment_value("DB_PASSWORD"),
        min_size=1,
        max_size=10,
    )

async def close_db_pool(pool: asyncpg.Pool) -> None:
    await pool.close()

async def check_db_connection(pool: asyncpg.Pool) -> bool:
    try:
        async with pool.acquire(timeout=5.0) as connection:
            result = await connection.fetchval("SELECT 1", timeout=5.0)
    except (
        asyncpg.PostgresError,
        asyncpg.InterfaceError,
        OSError,
        TimeoutError,
    ):
        return False

    return result == 1 