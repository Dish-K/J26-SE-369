import pytest
from fastapi.testclient import TestClient

from backend.app.probing.ai.config import AISettings
from backend.app.probing.demo.app import create_app
from backend.app.probing.demo.data import DEMO_CODE
from backend.app.probing.demo.service import DemoService


@pytest.fixture
def client(tmp_path):
    service = DemoService(provider="fake", state_dir=tmp_path, settings=AISettings())
    with TestClient(create_app(service)) as test_client:
        test_client.state_dir = tmp_path
        yield test_client


def first_segment(client, index=0):
    return client.get("/api/state").json()["segments"][index]


def test_state_lists_the_three_demo_segments_and_the_fake_banners(client):
    state = client.get("/api/state").json()
    assert state["banners"] == ["Demo data", "Fake model"]
    assert [(s["start_line"], s["end_line"]) for s in state["segments"]] == [(5, 7), (9, 13), (15, 18)]
    assert state["model"] == "dummy-v0" and all(not s["mapped"] for s in state["segments"])


def test_page_is_served_and_contains_the_required_banner_text(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Unverified: tests were not executed" in page.text


def test_map_then_generate_then_approve(client):
    heap = first_segment(client, 1)  # lines 9-13 use heapq
    mapped = client.post(f"/api/segments/{heap['segment_id']}/map").json()
    assert mapped["tags"][0]["tag_id"] == "heap_priority_queue"
    assert mapped["provenance"]["model"] == "dummy-v0" and mapped["data_origin"] == "fixture"

    ladder = client.post(f"/api/segments/{heap['segment_id']}/ladder/heap_priority_queue").json()
    assert [p["target_minutes"] for p in ladder["probes"]] == [5, 10, 20]
    probe = ladder["probes"][1]
    assert probe["verification"] == "unverified" and probe["approval"] == "pending" and probe["oracle"] == {}

    assert client.post(f"/api/probes/{probe['probe_id']}/approve", json={"by": " "}).status_code == 400
    approved = client.post(f"/api/probes/{probe['probe_id']}/approve", json={"by": "Interviewer A"}).json()
    assert approved["approval"] == "approved" and approved["verification"] == "unverified"

    detail = client.get(f"/api/segments/{heap['segment_id']}").json()
    assert detail["ladders"]["heap_priority_queue"]["probes"][1]["approved_by"] == "Interviewer A"
    assert detail["unverified_note"] == "Unverified: tests were not executed"


def test_ladder_needs_the_segment_mapped_and_the_tag_chosen(client):
    seg = first_segment(client, 1)
    assert client.post(f"/api/segments/{seg['segment_id']}/ladder/heap_priority_queue").status_code == 404
    client.post(f"/api/segments/{seg['segment_id']}/map")
    assert client.post(f"/api/segments/{seg['segment_id']}/ladder/graph_traversal").status_code == 404


def test_unknown_segment_and_probe_are_404(client):
    missing = "00000000-0000-4000-8000-0000000000aa"
    assert client.get(f"/api/segments/{missing}").status_code == 404
    assert client.post(f"/api/segments/{missing}/map").status_code == 404
    assert client.post(f"/api/probes/{missing}/approve", json={"by": "A"}).status_code == 404


def test_mapping_twice_returns_the_stored_result(client):
    seg = first_segment(client, 0)
    first = client.post(f"/api/segments/{seg['segment_id']}/map").json()
    assert client.post(f"/api/segments/{seg['segment_id']}/map").json() == first


def test_event_log_keeps_approvals_but_not_the_code(client):
    seg = first_segment(client, 1)
    client.post(f"/api/segments/{seg['segment_id']}/map")
    ladder = client.post(f"/api/segments/{seg['segment_id']}/ladder/heap_priority_queue").json()
    client.post(f"/api/probes/{ladder['probes'][0]['probe_id']}/approve", json={"by": "Interviewer A"})
    log = (client.state_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "Interviewer A" in log
    assert "counts[n] = counts.get(n, 0) + 1" not in log and DEMO_CODE.splitlines()[4] not in log


def test_cache_mode_with_gemini_never_calls_the_network(tmp_path):
    service = DemoService(provider="gemini", mode="cache", state_dir=tmp_path, cache_dir=tmp_path / "cache", settings=AISettings())
    with TestClient(create_app(service)) as c:
        assert c.get("/api/state").json()["banners"] == ["Demo data", "Cached result"]
        seg = c.get("/api/state").json()["segments"][1]
        mapped = c.post(f"/api/segments/{seg['segment_id']}/map").json()
        assert mapped["overall_status"] == "rule_only"  # no saved answer, so rules only
        assert mapped["tags"][0]["tag_id"] == "heap_priority_queue"


def test_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError):
        DemoService(provider="nope", settings=AISettings())
