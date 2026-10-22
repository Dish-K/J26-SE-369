"""Versioned prompt templates stored as text files.

A template file holds the system text, a line ``=== USER ===``, then the user
text. Placeholders look like ``{{NAME}}`` and are filled in a single pass, so a
value that itself contains ``{{...}}`` is never expanded again (code under
analysis can contain anything).
"""

import re
from pathlib import Path

_SEPARATOR = "=== USER ==="
_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


def load_prompt(directory: Path, name: str) -> tuple[str, str]:
    """Return ``(system, user_template)`` for ``<directory>/<name>.txt``."""
    text = (directory / f"{name}.txt").read_text(encoding="utf-8")
    system, separator, user = text.partition(f"\n{_SEPARATOR}\n")
    if not separator:
        raise ValueError(f"prompt {name!r} has no {_SEPARATOR!r} line")
    return system.strip(), user.strip()


def render(template: str, **values: str) -> str:
    def fill(match: re.Match) -> str:
        key = match.group(1)
        if key.lower() not in values:
            raise KeyError(f"prompt placeholder {key} has no value")
        return values[key.lower()]

    return _PLACEHOLDER.sub(fill, template)


def fence(code: str) -> str:
    """A fenced block whose fence is longer than any backtick run in the code."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}python\n{code}\n{ticks}"
