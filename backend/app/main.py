from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from backend.app.db.pool import (
    check_db_connection,
    close_db_pool,
    create_db_pool,
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = await create_db_pool()
    app.state.db_pool = pool

    try:
        yield
    finally:
        await close_db_pool(pool)

app = FastAPI(title="CodeTrace", lifespan=lifespan)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    database_is_healthy = await check_db_connection(request.app.state.db_pool)

    if not database_is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unavailable",
                "database": "unavailable",
            },
        )
    
    return {
        "status": "ok",
        "database": "ok",
    }