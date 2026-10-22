"""The knowledge-area catalogue (``tags.json``).

Tag ids are stable. Names, definitions and anchors may change between
taxonomy versions, and every stored result records ``taxonomy_version``.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

TAGS_PATH = Path(__file__).with_name("tags.json")


class Tag(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    area: str
    definition: str = Field(min_length=1)
    anchor: dict[str, str]
    enabled: bool
    optional: bool = False
    note: str | None = None
    anchor_note: str | None = None


class Taxonomy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    taxonomy_version: str
    source: str | None = None
    tags: tuple[Tag, ...]

    @model_validator(mode="after")
    def _ids_and_names_are_unique(self) -> "Taxonomy":
        ids = [tag.id for tag in self.tags]
        names = [tag.name for tag in self.tags]
        if len(set(ids)) != len(ids):
            raise ValueError("tag ids must be unique")
        if len(set(names)) != len(names):
            raise ValueError("tag names must be unique")
        return self

    def enabled(self) -> tuple[Tag, ...]:
        return tuple(tag for tag in self.tags if tag.enabled)

    def enabled_ids(self) -> tuple[str, ...]:
        return tuple(tag.id for tag in self.enabled())

    def get(self, tag_id: str) -> Tag | None:
        return next((tag for tag in self.tags if tag.id == tag_id), None)

    def by_name(self, name: str) -> Tag | None:
        return next((tag for tag in self.tags if tag.name == name), None)

    def position(self, tag_id: str) -> int:
        """Catalogue order, used as the last tie-break so output is deterministic."""
        for index, tag in enumerate(self.tags):
            if tag.id == tag_id:
                return index
        return len(self.tags)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or TAGS_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    return Taxonomy.model_validate(data)


@lru_cache(maxsize=1)
def default_taxonomy() -> Taxonomy:
    return load_taxonomy()
