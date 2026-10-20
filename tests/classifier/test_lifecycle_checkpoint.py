from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
from uuid import uuid4

import pytest

from backend.app.classifier.flag_identity import flag_id_for_segment
from backend.app.classifier.lifecycle_checkpoint import CheckpointInvalid,capture_checkpoint,restore_checkpoint
from backend.app.classifier.segment_lifecycle import SegmentLifecycle
from tests.classifier.flag_persistence_support import make_window


def flow_for(w):
    return SegmentLifecycle(session_id=w.session_id,task_id=w.task_id,model_version=w.model_version)


def restore(doc,w):
    return restore_checkpoint(doc,session_id=w.session_id,task_id=w.task_id,model_version=w.model_version)


def resign(doc):
    doc["sha256"] = hashlib.sha256(json.dumps(doc["payload"],sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def test_empty_stream_roundtrip():
    w=make_window()
    flow=flow_for(w)
    assert restore(capture_checkpoint(flow),w).snapshot()==flow.snapshot()


def test_active_growth_roundtrip_preserves_ids_confidence_events_and_types():
    a,b=make_window(confidence=.7),make_window(1,confidence=.9,ranges=((0,2,14,20),))
    a.regions[0].edit_events[0]["time"]=datetime.now(timezone.utc)
    flow=flow_for(a)
    flow.accept_window(a)
    flow.accept_window(b)
    saved=capture_checkpoint(flow)
    copy=restore(json.dumps(saved),a)
    assert copy.snapshot()==flow.snapshot()
    assert flag_id_for_segment(copy.snapshot().active_segments[0])==flag_id_for_segment(flow.snapshot().active_segments[0])
    assert capture_checkpoint(copy)==saved


def test_restore_can_continue_with_the_next_compatible_window():
    a,b=make_window(),make_window(1)
    flow=flow_for(a)
    flow.accept_window(a)
    restored=restore(capture_checkpoint(flow),a)
    assert restored.accept_window(b)==flow.accept_window(b)
    assert restored.snapshot().active_segments[0].window_count==2


def test_new_raw_event_added_after_checkpoint_cannot_change_original_observations():
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    checkpoint=capture_checkpoint(flow)
    original=flow.snapshot()
    # Simulates current raw evidence becoming richer after original acceptance.
    w.regions[0].edit_events[0]["new_late_metadata"]="not originally accepted"
    assert restore(checkpoint,w).snapshot()==original
    assert "new_late_metadata" not in restore(checkpoint,w).snapshot().active_segments[0].members[0].edit_events[0]


def test_checkpoint_retains_clean_windows_and_independent_regions():
    windows=[make_window(ranges=((0,2,10,10),(0,2,12,12))),
             make_window(1,ranges=((0,2,11,11),)),
             make_window(2,category="no_ai"),make_window(3,category="ide_ai")]
    flow=flow_for(windows[0])
    for w in windows:flow.accept_window(w)
    again=restore(capture_checkpoint(flow),windows[0])
    assert again.snapshot()==flow.snapshot()
    assert len(again.snapshot().closed_segments)==3
    assert len(again.snapshot().active_segments)==1
    assert again.snapshot().processed_window_count==4


def test_per_step_history_is_restored_before_the_relevant_window():
    a,b=make_window(),make_window(1,ranges=((0,3,15,20),))
    history={"session_id":a.session_id,"task_id":a.task_id,"load_seq":0,"version_id":3,
             "batch_seq":1,"seq_in_batch":0,"change_index":0,"origin":0,
             "start_line":2,"end_line":2,"inserted_lines":5,"net_line_delta":5}
    flow=flow_for(a)
    flow.accept_window(a)
    flow.accept_window(b,edit_history=[history])
    assert restore(capture_checkpoint(flow),a).snapshot()==flow.snapshot()
    assert flow.snapshot().active_segments[0].version_id==3


def test_finished_state_stays_finished_on_restart():
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    flow.finish(pending_live_processing_complete=True)
    again=restore(capture_checkpoint(flow),w)
    assert again.snapshot()==flow.snapshot()
    with pytest.raises(RuntimeError,match="finished"):
        again.accept_window(make_window(1))


def test_checkpoint_does_not_claim_closed_flags_were_saved():
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    flow.finish(pending_live_processing_complete=True)
    flag_id=flag_id_for_segment(flow.snapshot().closed_segments[0])
    flow._acknowledge_persisted(flag_id)
    again=restore(capture_checkpoint(flow),w)
    assert not again.snapshot().persisted_flag_ids
    assert flag_id_for_segment(again.snapshot().pending_segments[0])==flag_id


@pytest.mark.parametrize("field,value",[("format_version",2),("format_version",True),
    ("finished","yes"),("frames",{}),("state_sha256","wrong")])
def test_invalid_content_is_rejected_even_with_recomputed_checksum(field,value):
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    doc=capture_checkpoint(flow)
    doc["payload"][field]=value
    resign(doc)
    with pytest.raises(CheckpointInvalid):restore(doc,w)


def test_checksum_detects_modified_checkpoint():
    w=make_window()
    doc=capture_checkpoint(flow_for(w))
    doc["payload"]["finished"]=True
    with pytest.raises(CheckpointInvalid,match="checksum"):
        restore(doc,w)


@pytest.mark.parametrize("field,value",[("session_id",uuid4()),("task_id",uuid4()),("model_version","other-model")])
def test_restore_requires_the_expected_stream_identity(field,value):
    w=make_window()
    doc=capture_checkpoint(flow_for(w))
    expected={"session_id":w.session_id,"task_id":w.task_id,"model_version":w.model_version}
    expected[field]=value
    with pytest.raises(CheckpointInvalid,match="another stream"):
        restore_checkpoint(doc,**expected)


def test_duplicate_json_keys_are_rejected():
    w=make_window()
    with pytest.raises(CheckpointInvalid):
        restore('{"payload":{},"payload":{},"sha256":"x"}',w)


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),object(),datetime(2026,1,1)])
def test_unsupported_or_nonfinite_evidence_is_not_silently_lost(bad):
    w=make_window()
    w.regions[0].edit_events[0]["extra"]=bad
    flow=flow_for(w)
    flow.accept_window(w)
    with pytest.raises(CheckpointInvalid):capture_checkpoint(flow)


@pytest.mark.parametrize("name",["inserted_text","code_text"])
def test_source_text_is_rejected_by_metadata_only_checkpoint(name):
    w=make_window()
    w.regions[0].edit_events[0][name]="source code"
    flow=flow_for(w)
    flow.accept_window(w)
    with pytest.raises(CheckpointInvalid,match="source-code"):
        capture_checkpoint(flow)


def test_capture_and_recovery_exports_are_detached():
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    before=deepcopy(flow.snapshot())
    inputs=flow.recovery_inputs()
    inputs.frames[0][0].regions[0].edit_events[0]["batch_seq"]=999
    doc=capture_checkpoint(flow)
    doc["payload"]["frames"].clear()
    assert flow.snapshot()==before
