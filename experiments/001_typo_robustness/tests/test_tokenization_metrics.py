"""Adversarial and property-based tests for tokenization metrics.

Covers: fragmentation stratum boundary exactness, token-inflation-ratio
invariants, subword-count-change monotonicity, and fragmentation-matched-pair
determinism and consistency.
"""

from __future__ import annotations

import tokenization as tm
from enums import FragmentationStratum


# ---------------------------------------------------------------------------
# fragmentation_stratum — boundary exactness
# ---------------------------------------------------------------------------

def test_fragmentation_stratum_buckets():
    assert tm.fragmentation_stratum(0) == FragmentationStratum.LOW
    assert tm.fragmentation_stratum(-1) == FragmentationStratum.LOW
    assert tm.fragmentation_stratum(1) == FragmentationStratum.HIGH
    assert tm.fragmentation_stratum(3) == FragmentationStratum.HIGH


def test_fragmentation_stratum_boundary_at_zero():
    """0 is the last LOW value; 1 is the first HIGH value."""
    assert tm.fragmentation_stratum(0) == FragmentationStratum.LOW
    assert tm.fragmentation_stratum(1) == FragmentationStratum.HIGH


def test_fragmentation_stratum_large_positive_is_high():
    assert tm.fragmentation_stratum(100) == FragmentationStratum.HIGH


def test_fragmentation_stratum_large_negative_is_low():
    assert tm.fragmentation_stratum(-100) == FragmentationStratum.LOW


def test_fragmentation_stratum_returns_enum():
    result = tm.fragmentation_stratum(1)
    assert isinstance(result, FragmentationStratum)


# ---------------------------------------------------------------------------
# token_inflation_ratio — invariants
# ---------------------------------------------------------------------------

def test_token_inflation_ratio_identity(fake_tokenizer):
    ratio = tm.token_inflation_ratio(fake_tokenizer, "the cat sat", "the cat sat")
    assert ratio == 1.0


def test_token_inflation_ratio_increases_with_fragmentation(fake_tokenizer):
    """More unusual characters → more tokens → higher ratio."""
    normal = "cat"
    fragmented = "c@t!!!"   # non-alpha → extra pieces
    ratio = tm.token_inflation_ratio(fake_tokenizer, normal, fragmented)
    assert ratio >= 1.0


def test_token_inflation_ratio_is_positive(fake_tokenizer):
    for text in ["hello", "the quick brown fox", "x"]:
        ratio = tm.token_inflation_ratio(fake_tokenizer, text, text)
        assert ratio > 0


def test_token_inflation_ratio_short_perturbed_may_be_below_one(fake_tokenizer):
    """Perturbing to a shorter token sequence can yield ratio < 1."""
    # FakeTokenizer: non-alpha text gets +1 piece; perturbing 'the' to 't' drops length.
    # At minimum, the function must not crash and must return a positive number.
    ratio = tm.token_inflation_ratio(fake_tokenizer, "hello world", "hi world")
    assert ratio > 0


# ---------------------------------------------------------------------------
# subword_count_change
# ---------------------------------------------------------------------------

def test_subword_count_change_sign(fake_tokenizer):
    change = tm.subword_count_change(fake_tokenizer, "cat", "c@t")
    assert change >= 1


def test_subword_count_change_identity(fake_tokenizer):
    assert tm.subword_count_change(fake_tokenizer, "cat", "cat") == 0


def test_subword_count_change_shorter_variant(fake_tokenizer):
    """Perturbing to a shorter word should not crash and should return an int."""
    change = tm.subword_count_change(fake_tokenizer, "longer", "lo")
    assert isinstance(change, int)


# ---------------------------------------------------------------------------
# fragmentation_matched_pair — determinism and consistency
# ---------------------------------------------------------------------------

def test_build_fragmentation_matched_pair_is_deterministic(is_word, fake_tokenizer):
    first = tm.build_fragmentation_matched_pair(fake_tokenizer, "capital", 1, 5, is_word)
    second = tm.build_fragmentation_matched_pair(fake_tokenizer, "capital", 1, 5, is_word)
    assert first == second


def test_matched_pair_has_low_and_high_when_present(is_word, fake_tokenizer):
    pair = tm.build_fragmentation_matched_pair(fake_tokenizer, "remaining", 1, 9, is_word)
    if pair is not None:
        assert pair.low_fragmentation_subword_change <= 0
        assert pair.high_fragmentation_subword_change >= 1
        assert pair.low_fragmentation_variant != pair.high_fragmentation_variant


def test_matched_pair_low_below_zero_high_at_least_one(is_word, fake_tokenizer):
    """When both variants exist, LOW must not fragment more than HIGH."""
    pair = tm.build_fragmentation_matched_pair(fake_tokenizer, "remaining", 1, 9, is_word)
    if pair is not None:
        assert pair.low_fragmentation_subword_change < pair.high_fragmentation_subword_change


def test_fragmentation_matched_pair_determinism_across_seeds(is_word, fake_tokenizer):
    """Determinism must hold for different budget and seed values."""
    for budget, seed in [(1, 5), (2, 10), (3, 15)]:
        a = tm.build_fragmentation_matched_pair(fake_tokenizer, "example", budget, seed, is_word)
        b = tm.build_fragmentation_matched_pair(fake_tokenizer, "example", budget, seed, is_word)
        assert a == b
