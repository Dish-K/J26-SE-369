import json
from pathlib import Path

import pytest

from backend.app.probing.knowledge.tags import default_taxonomy, load_taxonomy
from tests.probing.test_cues import CASES
from backend.app.probing.knowledge.cues import extract_cues

PROBING = Path(__file__).resolve().parents[2] / "backend" / "app" / "probing"
LABEL_MAP = PROBING / "eval" / "label_map.json"


def test_catalogue_has_twenty_enabled_tags_and_a_disabled_design_tag():
    taxonomy = default_taxonomy()
    assert len(taxonomy.enabled()) == 20
    design = taxonomy.get("design")
    assert design is not None and design.enabled is False
    assert "design" not in taxonomy.enabled_ids()


def test_lookup_by_id_and_name():
    taxonomy = default_taxonomy()
    assert taxonomy.get("hash_table").name == "Hash Table"
    assert taxonomy.by_name("Hash Table").id == "hash_table"
    assert taxonomy.get("nope") is None


def test_four_tags_say_they_have_no_curriculum_anchor():
    noted = {t.id for t in default_taxonomy().tags if t.anchor_note}
    assert noted == {"two_pointers", "sliding_window", "prefix_sum", "bit_manipulation"}


def test_duplicate_ids_are_rejected(tmp_path):
    data = json.loads((PROBING / "knowledge" / "tags.json").read_text(encoding="utf-8"))
    data["tags"].append(dict(data["tags"][0], name="Another Name"))
    path = tmp_path / "tags.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_taxonomy(path)


def test_every_tag_a_cue_can_point_at_exists_and_is_enabled():
    enabled = set(default_taxonomy().enabled_ids())
    pointed = {
        s.tag_id
        for _, positive, _ in CASES
        for hit in extract_cues(positive)
        for s in hit.supports
    }
    assert pointed <= enabled


def test_label_map_only_names_tags_in_the_catalogue():
    names = {tag.name for tag in default_taxonomy().enabled()}
    label_map = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
    assert label_map["taxonomy_version"] == default_taxonomy().taxonomy_version
    for label, rule in label_map["labels"].items():
        for name in ([rule["primary"]] if rule["primary"] else []) + rule["acceptable"]:
            assert name in names, (label, name)
