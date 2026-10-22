"""The mock C2 hand-off of the demo page: what C2 sends, what Segment Extraction makes of it. Offline: fake model."""

import pytest
from fastapi.testclient import TestClient

from backend.app.probing.ai.config import AISettings
from backend.app.probing.demo.app import STATIC, create_app
from backend.app.probing.demo.data import DEMO_CODE
from backend.app.probing.demo.service import DemoService


@pytest.fixture
def client(tmp_path):
    service = DemoService(provider="fake", state_dir=tmp_path, settings=AISettings())
    with TestClient(create_app(service)) as test_client:
        yield test_client


def test_the_handoff_carries_the_flags_and_the_submitted_file(client):
    sent = client.get("/api/handoff").json()
    assert sent["code"] == DEMO_CODE and "no real candidate" in sent["source"]
    assert [(f["start_line"], f["end_line"]) for f in sent["flags"]] == [(5, 7), (9, 13), (15, 18)]
    assert {f["category"] for f in sent["flags"]} == {"external_ai"}
    assert all(0 <= f["confidence"] <= 1 for f in sent["flags"])


def test_sending_runs_segment_extraction_and_keeps_the_segment_ids(client):
    before = [s["segment_id"] for s in client.get("/api/state").json()["segments"]]
    result = client.post("/api/handoff/send", json={}).json()
    assert [(s["start_line"], s["end_line"]) for s in result["segments"]] == [(5, 7), (9, 13), (15, 18)]
    assert result["rejected"] == [] and result["notes"] == []
    assert [s["segment_id"] for s in result["segments"]] == before
    assert all(len(s["flag_ids"]) == 1 for s in result["segments"])


def test_a_bad_flag_is_rejected_with_a_reason_and_the_rest_carry_on(client):
    sent = client.get("/api/handoff", params={"include_bad_flag": True}).json()
    assert len(sent["flags"]) == 4 and (sent["flags"][3]["start_line"], sent["flags"][3]["end_line"]) == (90, 95)
    result = client.post("/api/handoff/send", json={"include_bad_flag": True}).json()
    assert len(result["segments"]) == 3
    assert [(r["flag_id"], r["reason"]) for r in result["rejected"]] == [(sent["flags"][3]["flag_id"], "out_of_range")]
    assert result["rejected"][0]["detail"]


def test_sending_again_does_not_lose_a_mapped_segment(client):
    first = client.get("/api/state").json()["segments"][0]["segment_id"]
    assert client.post(f"/api/segments/{first}/map").status_code == 200
    client.post("/api/handoff/send", json={"include_bad_flag": True})
    state = client.get("/api/state").json()
    assert next(s for s in state["segments"] if s["segment_id"] == first)["mapped"] is True


def test_the_default_hand_off_still_has_no_bad_flag(client):
    assert len(client.get("/api/handoff").json()["flags"]) == 3


def test_the_page_offers_the_hand_off_and_never_inserts_html():
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "Send to C3" in page and "Hand-off from C2" in page and "Unverified: tests were not executed" in page
    assert "innerHTML" not in page and "outerHTML" not in page and "insertAdjacentHTML" not in page
