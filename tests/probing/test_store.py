import asyncio
import json
from uuid import uuid4

import pytest

from backend.app.probing.ai.config import AISettings
from backend.app.probing.ai.gateway import LLMGateway
from backend.app.probing.knowledge.mapper import KnowledgeMapper
from backend.app.probing.probes.fake_probes import FakeDemoClient
from backend.app.probing.probes.generator import ProbeGenerator
from backend.app.probing.store.repository import ApprovalError, InMemoryProbeRepository, ProbeNotFound
from tests.probing.test_mapper import segment_of

SECRET_CODE = "secret_variable_name_42 = 1\nheappush(h, 3)\nx = heappop(h)\n"


def build():
    client = FakeDemoClient()
    gateway = LLMGateway(client, AISettings(max_rpm=100_000, max_tpm=10**9))
    segment = segment_of(SECRET_CODE, 2, 3)
    mapped = asyncio.run(KnowledgeMapper(gateway, model="dummy-v0").map_segment(segment, data_origin="fixture"))
    ladder = asyncio.run(ProbeGenerator(gateway, model="dummy-v0").generate_ladder("heap_priority_queue"))
    return segment, mapped, ladder


def test_save_and_read_back():
    repo = InMemoryProbeRepository()
    segment, mapped, ladder = build()
    repo.save_mapper_result(mapped)
    repo.save_ladder(segment.segment_id, ladder)
    assert repo.get_mapper_result(segment.segment_id) == mapped
    assert repo.get_ladder(segment.segment_id, "heap_priority_queue") == ladder
    assert repo.get_ladder(segment.segment_id, "sorting") is None
    assert repo.get_mapper_result(uuid4()) is None
    assert repo.get_probe(ladder.probes[0].probe_id) == ladder.probes[0]


def test_approval_needs_a_name_and_leaves_verification_alone():
    repo = InMemoryProbeRepository()
    segment, _, ladder = build()
    repo.save_ladder(segment.segment_id, ladder)
    probe_id = ladder.probes[1].probe_id
    with pytest.raises(ApprovalError):
        repo.approve(probe_id, "  ")
    approved = repo.approve(probe_id, "Interviewer A")
    assert approved.approval == "approved" and approved.approved_by == "Interviewer A"
    assert approved.verification == "unverified"
    stored = repo.get_ladder(segment.segment_id, "heap_priority_queue")
    assert [p.approval for p in stored.probes] == ["pending", "approved", "pending"]
    assert repo.reject(probe_id, "Interviewer A").approval == "rejected"


def test_unknown_probe_is_reported():
    with pytest.raises(ProbeNotFound):
        InMemoryProbeRepository().approve(uuid4(), "x")


def test_a_probe_that_failed_its_static_checks_cannot_be_approved():
    repo = InMemoryProbeRepository()
    segment, _, ladder = build()
    failing = ladder.probes[0].model_copy(
        update={"static_checks": ladder.probes[0].static_checks.model_copy(update={"passed": False})}
    )
    repo.save_ladder(segment.segment_id, ladder.model_copy(update={"probes": (failing,)}))
    with pytest.raises(ApprovalError):
        repo.approve(failing.probe_id, "Interviewer A")


def test_event_log_restores_everything_after_a_restart(tmp_path):
    log = tmp_path / "events.jsonl"
    repo = InMemoryProbeRepository(log)
    segment, mapped, ladder = build()
    repo.save_mapper_result(mapped)
    repo.save_ladder(segment.segment_id, ladder)
    repo.approve(ladder.probes[2].probe_id, "Interviewer A")

    restarted = InMemoryProbeRepository(log)
    assert restarted.get_mapper_result(segment.segment_id) == mapped
    assert restarted.get_probe(ladder.probes[2].probe_id).approval == "approved"
    assert restarted.get_probe(ladder.probes[0].probe_id).approval == "pending"


def test_log_never_contains_the_code(tmp_path):
    log = tmp_path / "events.jsonl"
    repo = InMemoryProbeRepository(log)
    segment, mapped, ladder = build()
    repo.save_mapper_result(mapped)
    repo.save_ladder(segment.segment_id, ladder)
    assert "secret_variable_name_42" not in log.read_text(encoding="utf-8")


def test_damaged_log_lines_are_skipped(tmp_path):
    log = tmp_path / "events.jsonl"
    repo = InMemoryProbeRepository(log)
    segment, _, ladder = build()
    repo.save_ladder(segment.segment_id, ladder)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{broken\n" + json.dumps({"event": "ladder", "segment_id": "nope"}) + "\n")
    restarted = InMemoryProbeRepository(log)
    assert restarted.get_ladder(segment.segment_id, "heap_priority_queue") == ladder
