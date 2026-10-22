"""C4 HTTP routes, registered on the existing FastAPI app in ``main.py``.

PP1 exposes the demo only: the real Phase 1-3 pipeline on the labelled
synthetic prototype, with no database. Database-backed routes for real C2
flags come with the full API phase.
"""

from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.explainability.api_schemas import AnalysisResponse, DemoFlagList
from backend.app.explainability.demo import DemoExplainability
from backend.app.explainability.errors import ExplainabilityContractError, FlagNotFoundError

router = APIRouter(prefix="/explainability", tags=["explainability"])


@lru_cache(maxsize=1)
def get_demo() -> DemoExplainability:
    """One demo instance per process, built on first use (not at import)."""
    return DemoExplainability()


@router.get("/demo/flags", response_model=DemoFlagList)
async def demo_flags(demo: DemoExplainability = Depends(get_demo)) -> DemoFlagList:
    return demo.flags()


@router.get("/demo/flags/{flag_id}/analysis", response_model=AnalysisResponse)
async def demo_analysis(
    flag_id: UUID, demo: DemoExplainability = Depends(get_demo)
) -> AnalysisResponse:
    try:
        return await demo.analysis(flag_id)
    except FlagNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ExplainabilityContractError as exc:
        # A broken C2/C4 contract is reported, never hidden behind partial data.
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": type(exc).__name__, "message": str(exc)},
        ) from exc
