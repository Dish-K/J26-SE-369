"""Language-keyed dispatch for Segment Extraction.

Only Python is populated for this build. Adding a language later is one new
entry here plus its own extractor module; the Python entry is not touched.
"""

from typing import Protocol

from tree_sitter import Tree

from backend.app.probing.python_extractor import LANGUAGE_NAME, PythonSegmentExtractor
from backend.app.probing.schemas import EntryFunction, ResolvedBoundary


class LanguageNotSupportedError(ValueError):
    """No segment extractor is registered for the requested language."""


class SegmentExtractor(Protocol):
    """What the language-neutral pipeline needs from a per-language extractor."""

    language: str

    def parse(self, code: str) -> Tree: ...

    def resolve_boundary(
        self, code: str, start_line: int, end_line: int
    ) -> ResolvedBoundary: ...

    def find_entry_function(self, code: str) -> EntryFunction | None: ...


PARSERS: dict[str, SegmentExtractor] = {
    LANGUAGE_NAME: PythonSegmentExtractor(),
}


def get_extractor(language: str) -> SegmentExtractor:
    try:
        return PARSERS[language]
    except KeyError:
        raise LanguageNotSupportedError(
            f"no segment extractor for language {language!r}; supported: {sorted(PARSERS)}"
        ) from None
