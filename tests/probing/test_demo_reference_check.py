"""The reference-check endpoint and recorded results of the demo page. Offline: scripted runner, no Docker.

Kept apart from test_reference_check.py because it needs backend/app/probing/demo/ and its recorded files.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.probing.ai.config import AISettings
from backend.app.probing.demo import service as demo_service
from backend.app.probing.demo.app import create_app
from backend.app.probing.demo.service import DemoService
from backend.app.probing.probes.docker_runner import RunnerChoice
from backend.app.probing.probes.runner import NullRunner
from backend.app.probing.probes.schema import ReferenceCheck
from tests.probing.test_reference_check import ScriptedRunner, run


# --- the demo ----------------------------------------------------------------------------------------------------


NO_RUNNER = RunnerChoice(NullRunner(), "none", "no code runner is configured (PROBING_RUNNER)")


def demo(tmp_path, runner, provider="fake"):
    return DemoService(provider=provider, mode="cache", state_dir=tmp_path, settings=AISettings(), runner=runner)


def stored_probe(client):
    seg = client.get("/api/state").json()["segments"][1]
    client.post(f"/api/segments/{seg['segment_id']}/map")
    return client.post(f"/api/segments/{seg['segment_id']}/ladder/heap_priority_queue").json()["probes"][1]


def test_the_approve_endpoint_answers_400_with_the_reason_for_a_failed_reference(tmp_path):
    scripted = ScriptedRunner()
    with TestClient(create_app(demo(tmp_path, RunnerChoice(scripted, "docker", "scripted")))) as client:
        probe = stored_probe(client)
        scripted.values = {c["case_id"]: "wrong" for c in probe["cases"]}
        assert client.post(f"/api/probes/{probe['probe_id']}/reference-check").json()["verification"] == "reference_failed"
        refused = client.post(f"/api/probes/{probe['probe_id']}/approve", json={"by": "Dana"})
        assert refused.status_code == 400 and "approval is disabled" in refused.json()["detail"]
        assert client.post(f"/api/probes/{probe['probe_id']}/reject", json={"by": "Dana"}).json()["approval"] == "rejected"


def recorded_files():
    return sorted((demo_service.DEMO_DIR / "cache" / "reference_checks").glob("*.json"))


def test_without_a_runner_the_probe_stays_unverified_and_nothing_errors(tmp_path):
    with TestClient(create_app(demo(tmp_path, NO_RUNNER))) as client:
        assert client.get("/api/state").json()["runner"]["mode"] == "none"
        probe = stored_probe(client)
        response = client.post(f"/api/probes/{probe['probe_id']}/reference-check")
        assert response.status_code == 200
        body = response.json()
        assert body["verification"] == "unverified" and body["reference_check"] is None


def test_with_a_runner_the_check_runs_once_and_is_stored(tmp_path):
    scripted = ScriptedRunner()
    before = recorded_files()
    with TestClient(create_app(demo(tmp_path, RunnerChoice(scripted, "docker", "scripted")))) as client:
        probe = stored_probe(client)
        scripted.values = {c["case_id"]: c["expected"] for c in probe["cases"]}
        body = client.post(f"/api/probes/{probe['probe_id']}/reference-check").json()
        assert body["verification"] == "reference_checked" and body["reference_check"]["served"] == "live"
        assert body["approval"] == "pending" and body["oracle"] == {}
        again = client.post(f"/api/probes/{probe['probe_id']}/reference-check").json()
        assert again["reference_check"] == body["reference_check"] and len(scripted.calls) == 1
        assert client.post("/api/probes/00000000-0000-4000-8000-0000000000aa/reference-check").status_code == 404
    assert recorded_files() == before  # the fake provider's probes are not recorded


def test_recorded_results_serve_the_demo_segment_with_no_runner(tmp_path):
    service = demo(tmp_path, NO_RUNNER, provider="gemini")

    async def go():
        await service.load()
        segment = next(s for s in service._segments.values() if s.location.start_line == 9)
        await service.map_segment(segment.segment_id)
        ladder = await service.generate_ladder(segment.segment_id, "heap_priority_queue")
        return [await service.check_reference(p.probe_id) for p in ladder.probes]

    probes = run(go())
    assert len(probes) == 3
    for probe in probes:
        assert probe.verification == "reference_checked" and probe.reference_check.served == "recorded"
        assert all(c.passed for c in probe.reference_check.cases)
        assert probe.approval == "pending" and probe.oracle == {}


def test_the_flawed_example_is_shown_as_caught_even_in_fake_mode(tmp_path):
    with TestClient(create_app(demo(tmp_path, NO_RUNNER))) as client:
        example = client.get("/api/examples/flawed").json()
    probe = example["probe"]
    assert probe["verification"] == "reference_failed" and probe["reference_check"]["served"] == "recorded"
    bad = [c for c in probe["reference_check"]["cases"] if not c["passed"]]
    assert [(c["case_id"], c["expected"], c["got"]) for c in bad] == [("c2", 6, 5)]


def test_the_recorded_files_are_the_expected_ones():
    # 3 heap probes and the flawed example, plus the 3 hash table probes recorded by the live run of 2026-10-22
    files = recorded_files()
    assert len(files) == 7
    statuses = sorted(ReferenceCheck.model_validate_json(f.read_text(encoding="utf-8")).status for f in files)
    assert statuses == ["reference_checked"] * 4 + ["reference_failed"] * 3
