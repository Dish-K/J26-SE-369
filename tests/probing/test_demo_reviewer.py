"""The review endpoint of the demo page. Offline: uses the fake model.

Kept apart from test_reviewer.py because it needs backend/app/probing/demo/.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.probing.ai.config import AISettings
from backend.app.probing.demo.app import create_app
from backend.app.probing.demo.service import DemoService


# --- the demo ----------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    settings = AISettings(reviewer_enabled=True)
    service = DemoService(provider="fake", state_dir=tmp_path, settings=settings)
    with TestClient(create_app(service)) as test_client:
        yield test_client


def a_stored_probe(client):
    seg = client.get("/api/state").json()["segments"][1]
    client.post(f"/api/segments/{seg['segment_id']}/map")
    return client.post(f"/api/segments/{seg['segment_id']}/ladder/heap_priority_queue").json()["probes"][1]


def test_demo_review_endpoint_stores_an_advisory_review(client):
    assert client.get("/api/state").json()["reviewer"]["enabled"] is True
    probe = a_stored_probe(client)
    reviewed = client.post(f"/api/probes/{probe['probe_id']}/review").json()
    assert reviewed["review"]["advisory_only"] is True and reviewed["approval"] == "pending"
    again = client.post(f"/api/probes/{probe['probe_id']}/review").json()
    assert again["review"] == reviewed["review"]


def test_demo_review_is_refused_when_the_reviewer_is_off(tmp_path):
    service = DemoService(provider="fake", state_dir=tmp_path, settings=AISettings())
    with TestClient(create_app(service)) as c:
        assert c.get("/api/state").json()["reviewer"]["enabled"] is False
        probe = a_stored_probe(c)
        response = c.post(f"/api/probes/{probe['probe_id']}/review")
        assert response.status_code == 409 and "switched off" in response.json()["detail"]


def test_demo_review_of_an_unknown_probe_is_404(client):
    assert client.post("/api/probes/00000000-0000-4000-8000-0000000000aa/review").status_code == 404
