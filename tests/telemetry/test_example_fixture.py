"""example_fixture.json must be exactly what the extractor produces today."""

import json

from shared_features import SCHEMA_VERSION, frame_from_payloads
from tools.regenerate_example_fixture import FIXTURE, build_fixture, render


def _stored() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_matches_a_fresh_extraction():
    # On failure, find out why the extractor's output changed before regenerating.
    assert render(build_fixture(_stored())) == FIXTURE.read_text(encoding="utf-8")


def test_fixture_payloads_load_at_the_package_version():
    payloads = _stored()["feature_payloads"]
    assert {p["schema_version"] for p in payloads} == {SCHEMA_VERSION}
    assert len(frame_from_payloads(payloads)) == len(payloads)
