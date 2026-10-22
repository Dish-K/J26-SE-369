"""The C3 demo: a small separate FastAPI app, not part of the main backend.

Run from the repository root:

    python -m uvicorn backend.app.probing.demo.app:app --host 127.0.0.1 --port 8100

It uses the constructed demo solution in ``data.py`` and the fake model unless
``PROBING_DEMO_PROVIDER=gemini`` is set. Nothing is executed and nothing is
written outside ``demo/state/`` (an event log of approvals, git-ignored).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.probing.ai.errors import LLMError
from backend.app.probing.demo.service import DemoService, ReviewUnavailable
from backend.app.probing.store.repository import ApprovalError, ProbeNotFound

STATIC = Path(__file__).with_name("static")


class Decision(BaseModel):
    by: str


def create_app(service: DemoService | None = None) -> FastAPI:
    svc = service or DemoService.from_env()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await svc.load()
        yield

    app = FastAPI(title="C3 Adaptive Probing demo", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state():
        return svc.state()

    @app.get("/api/segments/{segment_id}")
    def detail(segment_id: UUID):
        return _guard(lambda: svc.detail(segment_id))

    @app.post("/api/segments/{segment_id}/map")
    async def map_segment(segment_id: UUID):
        result = await _guard_async(svc.map_segment(segment_id))
        return result.model_dump(mode="json")

    @app.post("/api/segments/{segment_id}/ladder/{tag_id}")
    async def ladder(segment_id: UUID, tag_id: str):
        result = await _guard_async(svc.generate_ladder(segment_id, tag_id))
        return result.model_dump(mode="json")

    @app.post("/api/probes/{probe_id}/approve")
    def approve(probe_id: UUID, body: Decision):
        return _guard(lambda: svc.repo.approve(probe_id, body.by).model_dump(mode="json"))

    @app.post("/api/probes/{probe_id}/review")
    async def review(probe_id: UUID):
        try:
            result = await svc.review_probe(probe_id)
        except ProbeNotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except ReviewUnavailable as exc:
            raise HTTPException(409, str(exc)) from None
        return result.model_dump(mode="json")

    @app.post("/api/probes/{probe_id}/reference-check")
    async def reference_check(probe_id: UUID):
        try:
            probe = await svc.check_reference(probe_id)
        except ProbeNotFound as exc:
            raise HTTPException(404, str(exc)) from None
        return probe.model_dump(mode="json")

    @app.get("/api/examples/flawed")
    async def flawed_example():
        return await svc.flawed_example()

    @app.post("/api/probes/{probe_id}/reject")
    def reject(probe_id: UUID, body: Decision):
        return _guard(lambda: svc.repo.reject(probe_id, body.by).model_dump(mode="json"))

    return app


def _guard(call):
    try:
        return call()
    except (LookupError, ProbeNotFound) as exc:
        raise HTTPException(404, str(exc)) from None
    except ApprovalError as exc:
        raise HTTPException(400, str(exc)) from None


async def _guard_async(awaitable):
    try:
        return await awaitable
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from None


app = create_app()
