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


def load_word_lexicon(name: str) -> frozenset[str]:
    """Load a word lexicon by filename from data/lexicons/.

    Returns a frozenset of lowercase stripped words. Lines beginning with '#'
    and blank lines are ignored. Suitable for fast membership tests.
    """
    path = _LEXICONS_DIR / name
    words: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.lower())
    if not words:
        raise ValueError(f"lexicon {name!r} is empty or contains only comments")
    return frozenset(words)


def load_phrase_lexicon(name: str) -> list[str]:
    """Load a phrase lexicon by filename from data/lexicons/.

    Lines beginning with '#' and blank lines are ignored. Entries are returned
    as a list of stripped strings in file order.
    """
    path = _LEXICONS_DIR / name
    entries: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
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
