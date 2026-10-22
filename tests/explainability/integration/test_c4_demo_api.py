"""PP1 demo API: real Phase 1-3 pipeline on the synthetic prototype, no database."""

import uuid
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.explainability.router import get_demo, router


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    app = FastAPI()  # only the C4 router: proves the demo needs no database
    app.include_router(router)
    get_demo.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_demo.cache_clear()


@pytest.fixture(scope="module")
def flags(client: TestClient) -> dict:
    response = client.get("/explainability/demo/flags")
    assert response.status_code == 200
    return response.json()


def test_flag_list_is_labelled_synthetic(flags: dict) -> None:
    assert flags["data_source"]["kind"] == "synthetic_prototype"
    assert "synthetic" in flags["data_source"]["label"].lower()
    assert {f["category"] for f in flags["flags"]} == {"external_ai", "ide_ai"}


@pytest.mark.parametrize("category", ["external_ai", "ide_ai"])
def test_analysis_is_complete_and_consistent(
    client: TestClient, flags: dict, category: str
) -> None:
    summary = next(f for f in flags["flags"] if f["category"] == category)
    response = client.get(f"/explainability/demo/flags/{summary['flag_id']}/analysis")
    assert response.status_code == 200
    body = response.json()

    assert body["data_source"]["kind"] == "synthetic_prototype"
    representative = body["representative_prediction"]["prediction_id"]
    assert body["flag"]["representative_prediction_id"] == representative
    assert body["feature_vector"]["feature_vector_id"] == (
        body["representative_prediction"]["feature_vector_id"]
    )
    # every explanation describes the same prediction and class
    for section in ("shap", "lime", "agreement", "stability", "faithfulness", "ecs"):
        assert body[section]["prediction_id"] == representative, section
        assert body[section]["target_class"] == category, section
    assert body["model"]["target_class"] == category

    timeline = body["timeline"]
    assert sum(w["is_representative"] for w in timeline) == 1
    assert sum(w["in_segment"] for w in timeline) == body["flag"]["window_count"]
    assert next(w for w in timeline if w["is_representative"])["prediction_id"] == representative

    assert len(body["features"]) == len(body["shap"]["features"])
    assert all(f["label"] for f in body["features"])
    assert "runs" not in body["stability"]  # summary only
    assert body["stability"]["repetitions_requested"] == 20
    assert body["ecs"]["status"] == "computed"
    assert len(body["ecs_sensitivity"]["scenarios"]) == 4
    assert set(body["planned"]) == {"counterfactuals", "behavioural_realism"}


def test_unknown_flag_is_404(client: TestClient) -> None:
    response = client.get(f"/explainability/demo/flags/{uuid.uuid4()}/analysis")
    assert response.status_code == 404


def test_malformed_flag_id_is_422(client: TestClient) -> None:
    assert client.get("/explainability/demo/flags/not-a-uuid/analysis").status_code == 422
