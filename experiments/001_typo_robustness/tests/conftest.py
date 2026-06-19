"""Shared pytest fixtures.

The package is installed (pip install -e .), so tests import the flat modules
directly (e.g. `import regimes`, `from perturbation import ...`).
"""

from __future__ import annotations

import pytest

import regimes


@pytest.fixture
def is_word():
    """The demo-wordlist predicate, sufficient for hermetic tests."""
    return regimes.make_is_word()


@pytest.fixture
def small_vocabulary_is_word():
    """A tiny explicit vocabulary, for tests that need to control exactly which
    strings count as words."""
    return regimes.make_is_word(
        {"cat", "cot", "cab", "car", "bat", "bad", "bag", "the", "france", "finance"})


class FakeTokenizer:
    """A deterministic stand-in for a model tokenizer. Splits on whitespace and
    fragments longer or non-alphabetic tokens into more pieces, so fragmentation
    contrasts are exercisable without loading a real tokenizer."""

    def encode(self, text: str):
        tokens = []
        for word in text.split():
            piece_count = 1 + (len(word) - 1) // 4 + (0 if word.isalpha() else 1)
            tokens.extend([word] * piece_count)
        return tokens


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()
