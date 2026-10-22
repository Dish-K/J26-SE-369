"""The Knowledge Mapper's result (build brief, Appendix E).

Never holds the segment's code. For ``candidate`` origin it also holds no
reasoning text, because the model may quote the code in it.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.probing.ai.consent import DataOrigin
from backend.app.probing.ai.provenance import Provenance
from backend.app.probing.knowledge.crosscheck import IgnoredCue, OverallStatus, TagStatus

MAPPER_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MappedTag(_Frozen):
    rank: int = Field(ge=1, le=3)
    tag_id: str
    tag: str
    status: TagStatus
    cues: tuple[str, ...]
    anchor: dict[str, Any]
    anchor_note: str | None = None
    why: str


class MapperResult(_Frozen):
    schema_version: Literal[1] = MAPPER_SCHEMA_VERSION
    taxonomy_version: str
    segment_id: UUID
    task_id: UUID
    data_origin: DataOrigin
    tags: tuple[MappedTag, ...] = Field(max_length=3)
    ignored_strong_cues: tuple[IgnoredCue, ...]
    overall_status: OverallStatus
    rule_evidence: dict[str, tuple[str, ...]]
    reasoning: str | None
    provenance: Provenance
    notes: tuple[str, ...] = ()
