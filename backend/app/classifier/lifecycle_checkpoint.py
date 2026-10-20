from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from math import isfinite
from typing import Any
from uuid import UUID

from .localization import LocalizedCodeRegion
from .segment_builder import ClassifiedWindow
from .segment_lifecycle import SegmentLifecycle


class CheckpointInvalid(ValueError):
    """A checkpoint is malformed, incompatible, or cannot reproduce its state."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _encode(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return {"t": "float", "v": value.hex()}
    if isinstance(value, UUID):
        return {"t": "uuid", "v": str(value)}
    if isinstance(value, datetime) and value.utcoffset() is not None:
        return {"t": "time", "v": value.isoformat()}
    if type(value) in (tuple, list):
        return {"t": "tuple" if isinstance(value,tuple) else "list", "v": [_encode(v) for v in value]}
    if type(value) is dict and all(type(k) is str for k in value):
        if {"inserted_text", "code_text"} & value.keys():
            raise CheckpointInvalid("Recovery checkpoints accept edit metadata, not source-code text")
        return {"t": "dict", "v": {k:_encode(v) for k,v in value.items()}}
    raise CheckpointInvalid(f"Unsupported recovery value type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is not dict or set(value) != {"t","v"}:
        raise CheckpointInvalid("Invalid tagged recovery value")
    kind, data = value["t"], value["v"]
    if kind == "float" and type(data) is str:
        result = float.fromhex(data)
        if isfinite(result):
            return result
    elif kind == "uuid" and type(data) is str:
        return UUID(data)
    elif kind == "time" and type(data) is str:
        result = datetime.fromisoformat(data)
        if result.utcoffset() is not None:
            return result
    elif kind in {"tuple","list"} and type(data) is list:
        result = [_decode(item) for item in data]
        return tuple(result) if kind == "tuple" else result
    elif kind == "dict" and type(data) is dict and all(type(k) is str for k in data):
        if {"inserted_text", "code_text"} & data.keys():
            raise CheckpointInvalid("Recovery checkpoints cannot contain source-code text")
        return {k:_decode(v) for k,v in data.items()}
    raise CheckpointInvalid("Invalid recovery value type or content")


def _visible_state(flow: SegmentLifecycle) -> Any:
    state = flow.snapshot()
    return _encode({
        "active": tuple(asdict(s) for s in state.active_segments),
        "closed": tuple(asdict(s) for s in state.closed_segments),
        "count": state.processed_window_count, "finished": state.finished,
    })


def capture_checkpoint(flow: SegmentLifecycle) -> dict[str, Any]:
    """Capture exact accepted inputs plus a fingerprint of the expected state.

    Floats are encoded losslessly. No pickle or current-telemetry query is used.
    The checksum detects accidental corruption; it is not an authentication key.
    """
    source = flow.recovery_inputs()
    payload = {
        "format_version": 1,
        "session_id": str(source.session_id), "task_id": str(source.task_id),
        "model_version": source.model_version, "finished": source.finished,
        "frames": [_encode({"window":asdict(w),"history":history}) for w,history in source.frames],
        "state_sha256": _hash(_visible_state(flow)),
    }
    return {"payload":payload,"sha256":_hash(payload)}


def _no_duplicates(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise CheckpointInvalid("Duplicate checkpoint JSON key")
        result[key] = value
    return result


def restore_checkpoint(
    checkpoint: str | dict[str, Any], *, session_id: UUID, task_id: UUID, model_version: str,
) -> SegmentLifecycle:
    """Validate and replay the ORIGINAL observations for the expected stream.

    Closed flags return pending. Re-running Task 12 checks existing saved rows
    and safely completes any unfinished writes; a checkpoint never fabricates
    database acknowledgement.
    """
    try:
        document = json.loads(checkpoint, object_pairs_hook=_no_duplicates) if isinstance(checkpoint,str) else checkpoint
        if type(document) is not dict or set(document) != {"payload","sha256"}:
            raise CheckpointInvalid("Invalid checkpoint envelope")
        payload = document["payload"]
        expected_keys = {"format_version","session_id","task_id","model_version","finished","frames","state_sha256"}
        if type(payload) is not dict or set(payload) != expected_keys:
            raise CheckpointInvalid("Invalid checkpoint fields")
        if document["sha256"] != _hash(payload):
            raise CheckpointInvalid("Checkpoint checksum mismatch")
        if type(payload["format_version"]) is not int or payload["format_version"] != 1:
            raise CheckpointInvalid("Unsupported checkpoint format version")
        if (payload["session_id"],payload["task_id"],payload["model_version"]) != (str(session_id),str(task_id),model_version):
            raise CheckpointInvalid("Checkpoint belongs to another stream")
        if type(payload["finished"]) is not bool or type(payload["frames"]) is not list:
            raise CheckpointInvalid("Invalid checkpoint lifecycle fields")
        flow = SegmentLifecycle(session_id=session_id,task_id=task_id,model_version=model_version)
        for encoded in payload["frames"]:
            frame = _decode(encoded)
            if type(frame) is not dict or set(frame) != {"window","history"}:
                raise CheckpointInvalid("Invalid checkpoint frame")
            fields = frame["window"]
            fields["regions"] = tuple(LocalizedCodeRegion(**r) for r in fields["regions"])
            flow.accept_window(ClassifiedWindow(**fields),edit_history=frame["history"])
        if payload["finished"]:
            flow.finish(pending_live_processing_complete=True)
        if payload["state_sha256"] != _hash(_visible_state(flow)):
            raise CheckpointInvalid("Checkpoint replay does not reproduce its recorded state")
        return flow
    except CheckpointInvalid:
        raise
    except (ValueError,TypeError,KeyError,AttributeError,OverflowError,RuntimeError) as exc:
        raise CheckpointInvalid("Checkpoint failed validation or replay") from exc
