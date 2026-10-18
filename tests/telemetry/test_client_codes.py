"""The browser's event codes must be exactly what the registry says today."""

from shared_features import KeyClass, Kind, Modifier
from tools.generate_client_codes import OUTPUT, render


def test_codes_file_matches_the_registry():
    # On failure, run python tools/generate_client_codes.py and commit the result.
    assert OUTPUT.read_text(encoding="utf-8") == render()


def test_the_codes_capture_uses_have_the_registry_numbers():
    text = OUTPUT.read_text(encoding="utf-8")
    assert f"  keydown: {int(Kind.keydown)}," in text
    assert f"  backspace: {int(KeyClass.backspace)}," in text
    assert f"  shift: {int(Modifier.shift)}," in text
