import asyncio
from copy import deepcopy
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from backend.app.classifier.lifecycle_checkpoint import capture_checkpoint
from backend.app.classifier.lifecycle_recovery import (
    CheckpointConflict,RecoverableSegmentLifecycle,load_checkpoint,write_checkpoint,
)
from backend.app.classifier.segment_lifecycle import SegmentLifecycle
from tests.classifier.flag_persistence_support import make_window,stored_prediction


def flow_for(w):
    return SegmentLifecycle(session_id=w.session_id,task_id=w.task_id,model_version=w.model_version)


class Connection:
    def __init__(self,windows):
        self.predictions={w.prediction_id:stored_prediction(w) for w in windows}
        self.row=None
        self.fail_commit=False
    @asynccontextmanager
    async def transaction(self):
        before=deepcopy(self.row)
        try:yield
        except BaseException:
            self.row=before
            raise
        if self.fail_commit:
            self.fail_commit=False
            raise OSError("lost commit acknowledgement")
    async def fetch(self,sql,ids):
        return [self.predictions[i] for i in ids if i in self.predictions]
    async def fetchrow(self,sql,*args):return deepcopy(self.row)
    async def fetchval(self,sql,*args):
        if "SELECT EXISTS" in sql:return True
        task,session,model,document,*expected=args
        import json
        if sql.lstrip().startswith("INSERT"):
            if self.row is not None:return None
            revision=1
        else:
            if self.row is None or self.row["revision"]!=expected[0]:return None
            revision=self.row["revision"]+1
        self.row={"session_id":session,"revision":revision,"checkpoint":json.loads(document)}
        return revision


class Pool:
    def __init__(self,c):self.connection=c
    @asynccontextmanager
    async def acquire(self):yield self.connection


def run(c):return asyncio.run(c)


def test_write_load_and_actual_new_adapter_restore_same_active_segment():
    w=make_window()
    pool=Pool(Connection([w]))
    first=run(RecoverableSegmentLifecycle.open(pool,session_id=w.session_id,task_id=w.task_id,model_version=w.model_version,create_if_missing=True))
    expected=run(first.accept_window(w))
    again=run(RecoverableSegmentLifecycle.open(pool,session_id=w.session_id,task_id=w.task_id,model_version=w.model_version))
    assert again.snapshot()==expected and again.revision==1


def test_failed_checkpoint_write_does_not_publish_new_memory_state(monkeypatch):
    w=make_window()
    adapter=RecoverableSegmentLifecycle(object(),flow_for(w),0)
    before=adapter.snapshot()
    monkeypatch.setattr("backend.app.classifier.lifecycle_recovery.write_checkpoint",AsyncMock(side_effect=OSError("offline")))
    with pytest.raises(OSError):run(adapter.accept_window(w))
    assert adapter.snapshot()==before and adapter.revision==0


def test_original_proposed_save_can_retry_after_uncertain_commit():
    w=make_window()
    pool=Pool(Connection([w]))
    flow=flow_for(w)
    flow.accept_window(w)
    pool.connection.fail_commit=True
    with pytest.raises(OSError):run(write_checkpoint(pool,flow,expected_revision=0))
    assert pool.connection.row["revision"]==1
    assert run(write_checkpoint(pool,flow,expected_revision=0))==1
    assert pool.connection.row["revision"]==1


def test_stale_writer_cannot_replace_newer_accepted_observations():
    a,b=make_window(),make_window(1)
    pool=Pool(Connection([a,b]))
    old=flow_for(a)
    old.accept_window(a)
    assert run(write_checkpoint(pool,old,expected_revision=0))==1
    current=deepcopy(old)
    current.accept_window(b)
    assert run(write_checkpoint(pool,current,expected_revision=1))==2
    with pytest.raises(CheckpointConflict):
        run(write_checkpoint(pool,old,expected_revision=0))
    assert pool.connection.row["checkpoint"]==capture_checkpoint(current)


def test_missing_checkpoint_with_nonzero_revision_is_not_recreated():
    w=make_window()
    pool=Pool(Connection([w]))
    flow=flow_for(w)
    flow.accept_window(w)
    with pytest.raises(CheckpointConflict):run(write_checkpoint(pool,flow,expected_revision=3))
    assert pool.connection.row is None


def test_missing_checkpoint_is_not_silently_reported_as_empty_recovered_state():
    w=make_window()
    pool=Pool(Connection([w]))
    with pytest.raises(LookupError,match="explicit new-stream"):
        run(load_checkpoint(pool,session_id=w.session_id,task_id=w.task_id,model_version=w.model_version))


def test_missing_stored_prediction_blocks_checkpoint_creation():
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    pool=Pool(Connection([]))
    with pytest.raises(LookupError):run(write_checkpoint(pool,flow,expected_revision=0))
    assert pool.connection.row is None


@pytest.mark.parametrize("field,value",[("confidence",.1),("ai_positive",False),("run_id","final")])
def test_prediction_mismatch_is_rejected(field,value):
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    pool=Pool(Connection([w]))
    pool.connection.predictions[w.prediction_id][field]=value
    with pytest.raises(ValueError,match="disagree"):
        run(write_checkpoint(pool,flow,expected_revision=0))
    assert pool.connection.row is None


def test_restart_reuses_frozen_evidence_without_fetching_current_raw_events():
    w=make_window()
    pool=Pool(Connection([w]))
    flow=flow_for(w)
    flow.accept_window(w)
    run(write_checkpoint(pool,flow,expected_revision=0))
    # The test connection exposes predictions/checkpoints only; no raw events API.
    restored,revision=run(load_checkpoint(pool,session_id=w.session_id,task_id=w.task_id,model_version=w.model_version))
    assert restored.snapshot()==flow.snapshot() and revision==1


def test_guarded_finish_is_saved_and_restores_finished_state():
    w=make_window()
    pool=Pool(Connection([w]))
    adapter=RecoverableSegmentLifecycle(pool,flow_for(w),0)
    run(adapter.accept_window(w))
    with pytest.raises(ValueError):run(adapter.finish(pending_live_processing_complete=False))
    assert adapter.revision==1
    state=run(adapter.finish(pending_live_processing_complete=True))
    restored,_=run(load_checkpoint(pool,session_id=w.session_id,task_id=w.task_id,model_version=w.model_version))
    assert restored.snapshot()==state


def test_closed_save_delegates_to_existing_task13_task12_handoff(monkeypatch):
    w=make_window()
    flow=flow_for(w)
    flow.accept_window(w)
    flow.finish(pending_live_processing_complete=True)
    adapter=RecoverableSegmentLifecycle(object(),flow,1)
    mocked=AsyncMock(return_value=())
    monkeypatch.setattr("backend.app.classifier.lifecycle_recovery.persist_closed_segments",mocked)
    assert run(adapter.persist_closed())==()
    mocked.assert_awaited_once()
