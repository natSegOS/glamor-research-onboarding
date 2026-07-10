"""Loaders for the pre-registered decision-logic lexicons in data/lexicons/.

These lexicons encode controlled-vocabulary lists that are part of the study's
pre-registered scoring and construction protocol (design/04 §4.5, §4.7). By
living in data files rather than in source code they are:
  - auditable: a reviewer can diff them against the design document;
  - versionable: the PROVENANCE.json sidecar records the freeze date and
    rationale for each list;
  - behavior-preserving: the regex produced by compile_phrase_regex is
    semantically identical to the inline regex it replaces.

See data/lexicons/PROVENANCE.json for the provenance record.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import Sequence


_LEXICONS_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"


def _read_lexicon_lines(name: str) -> list[str]:
    """Read data/lexicons/<name>, stripped, skipping blank lines and '#' comments."""
    path = _LEXICONS_DIR / name
    with open(path, encoding="utf-8") as fh:
        stripped_lines = [raw_line.strip() for raw_line in fh]
    return [line for line in stripped_lines if line and not line.startswith("#")]


def load_word_lexicon(name: str) -> frozenset[str]:
    """Load a word lexicon by filename from data/lexicons/ as a lowercased
    frozenset, suitable for fast membership tests."""
    words = frozenset(line.lower() for line in _read_lexicon_lines(name))
    if not words:
        raise ValueError(f"lexicon {name!r} is empty or contains only comments")
    return words


def load_phrase_lexicon(name: str) -> list[str]:
    """Load a phrase lexicon by filename from data/lexicons/ as a list of
    stripped strings in file order."""
    entries = _read_lexicon_lines(name)
    if not entries:
        raise ValueError(f"lexicon {name!r} is empty or contains only comments")
    return entries


def compile_phrase_regex(entries: Sequence[str],
                         word_boundary: bool = True,
                         flags: int = re.IGNORECASE) -> re.Pattern:
    """Build a compiled alternation regex from a sequence of phrase strings.

    Each entry is re.escape-d before joining, so entries are treated as literal
    phrases (not patterns). Word boundaries are added around the alternation
    group when word_boundary=True (the default).
    """
    alternation = "|".join(re.escape(entry) for entry in entries)
    if word_boundary:
        pattern = rf"\b(?:{alternation})\b"
    else:
        pattern = rf"(?:{alternation})"
    return re.compile(pattern, flags)
