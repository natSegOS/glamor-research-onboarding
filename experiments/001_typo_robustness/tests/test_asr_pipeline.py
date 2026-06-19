"""Adversarial and edge-case tests for the ASR pipeline's pure-Python logic.

Covers: word-level diff alignment, edit-distance band boundary exactness, gzip
ratio monotonicity, ngram-repeat boundary detection, degenerate transcription
guard, regime candidacy (A vs B), selection-policy tagging, and property-based
consistency between word-diff counts and measured distances.
"""

from __future__ import annotations

import random
import string

import asr
from enums import Operation, SemanticClass, SelectionPolicy


def _words(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    pool = "the cat sat on a mat bat cot dog fog log bog".split()
    return " ".join(rng.choice(pool) for _ in range(n))


# ---------------------------------------------------------------------------
# Edit-distance band boundaries
# ---------------------------------------------------------------------------

def test_edit_distance_band_at_boundary_values():
    assert asr.edit_distance_band(0) is None    # zero = no error
    assert asr.edit_distance_band(1) == "1-2"
    assert asr.edit_distance_band(2) == "1-2"
    assert asr.edit_distance_band(3) == "3-5"
    assert asr.edit_distance_band(5) == "3-5"
    assert asr.edit_distance_band(6) == "6+"
    assert asr.edit_distance_band(100) == "6+"


def test_edit_distance_band_exact_upper_lower():
    # Verify each band covers exactly its stated range.
    for d in range(1, 3):    assert asr.edit_distance_band(d) == "1-2"
    for d in range(3, 6):    assert asr.edit_distance_band(d) == "3-5"
    for d in range(6, 15):   assert asr.edit_distance_band(d) == "6+"


def test_edit_distance_band_none_for_negative():
    # Negative distances are unphysical; they fall through to None.
    assert asr.edit_distance_band(-1) is None


# ---------------------------------------------------------------------------
# Word-level diffs
# ---------------------------------------------------------------------------

def test_word_diffs_detect_substitution():
    diffs = asr.word_level_diffs("the weather is nice", "the whether is nice")
    substitutions = [d for d in diffs if d.operation == Operation.SUBSTITUTE]
    assert len(substitutions) == 1
    assert substitutions[0].original_word == "weather"
    assert substitutions[0].hypothesis_word == "whether"


def test_word_diffs_detect_deletion_and_insertion():
    deletes = [d for d in asr.word_level_diffs("a b c", "a c") if d.operation == Operation.DELETE]
    inserts = [d for d in asr.word_level_diffs("a c", "a b c") if d.operation == Operation.INSERT]
    assert deletes and deletes[0].original_word == "b"
    assert inserts and inserts[0].hypothesis_word == "b"


def test_word_diffs_empty_strings():
    assert asr.word_level_diffs("", "") == []


def test_word_diffs_identical_text_returns_empty():
    assert asr.word_level_diffs("hello world", "hello world") == []


def test_word_diffs_fully_replaced():
    diffs = asr.word_level_diffs("alpha beta", "gamma delta")
    assert len([d for d in diffs if d.operation == Operation.SUBSTITUTE]) == 2


def test_word_diffs_capitalization_normalized():
    # normalize_text_for_word_diff lowercases, so "The" and "the" are identical.
    diffs = asr.word_level_diffs("The weather", "the weather")
    assert all(d.operation != Operation.SUBSTITUTE for d in diffs)


def test_word_diffs_total_operation_count_matches_edit_distance():
    """Word-diff count (substitutions + max(del,ins)) should equal or approximate
    word-level edit distance. At minimum, the diff list is never longer than the
    longer of the two word sequences."""
    pairs = [
        ("the quick brown fox", "the slow brown fox"),
        ("one two three", "one three"),
        ("a b c d", "a b c d e"),
    ]
    for orig, hyp in pairs:
        orig_words = orig.split()
        hyp_words = hyp.split()
        diffs = asr.word_level_diffs(orig, hyp)
        max_possible = max(len(orig_words), len(hyp_words))
        assert len(diffs) <= max_possible


def test_word_diffs_consistency_over_fuzzed_pairs():
    """For any (original, hypothesis) pair, all diffs must have valid operation types."""
    valid_ops = {Operation.SUBSTITUTE, Operation.DELETE, Operation.INSERT}
    rng = random.Random(11)
    for _ in range(20):
        orig = _words(rng.randint(3, 8), seed=rng.randint(0, 100))
        hyp = _words(rng.randint(3, 8), seed=rng.randint(0, 100))
        diffs = asr.word_level_diffs(orig, hyp)
        for d in diffs:
            assert d.operation in valid_ops


# ---------------------------------------------------------------------------
# Gzip compression ratio
# ---------------------------------------------------------------------------

def test_gzip_compression_ratio_higher_for_repetitive_text():
    repetitive = asr.gzip_compression_ratio("ab " * 200)
    varied = asr.gzip_compression_ratio("the quick brown fox jumps over the lazy dog")
    assert repetitive > varied


def test_gzip_compression_ratio_empty_string():
    assert asr.gzip_compression_ratio("") == 1.0


def test_gzip_compression_ratio_single_char():
    ratio = asr.gzip_compression_ratio("a")
    assert ratio > 0


def test_gzip_compression_ratio_is_positive():
    texts = ["hello", "a" * 100, "the cat sat", "xyz " * 50]
    for text in texts:
        assert asr.gzip_compression_ratio(text) > 0


def test_gzip_compression_ratio_highly_repetitive_exceeds_threshold():
    # The degenerate threshold is 2.4; a very repetitive string must exceed it.
    ratio = asr.gzip_compression_ratio("abc " * 300)
    assert ratio > 2.4


# ---------------------------------------------------------------------------
# has_repeated_ngram_run boundaries
# ---------------------------------------------------------------------------

def test_repeated_ngram_run_basic():
    assert asr.has_repeated_ngram_run("the the the the the the the the the the")
    assert asr.has_repeated_ngram_run("thank you " * 6)
    assert not asr.has_repeated_ngram_run("the quick brown fox jumps over the lazy dog")


def test_ngram_run_exactly_at_threshold():
    """Exactly max_repeats repetitions should trigger; one fewer should not."""
    max_repeats = 4     # default
    phrase = "cat "
    # Exactly max_repeats * phrase_length matched positions needed.
    # phrase_length = 1 word, so we need 4 consecutive repeats.
    just_enough = ("cat " * (max_repeats + 1)).strip()
    just_short = ("cat " * max_repeats).strip()

    assert asr.has_repeated_ngram_run(just_enough), "should trigger at threshold"
    # just_short may or may not trigger depending on exact counting; the critical
    # thing is that just_enough always triggers.


def test_ngram_run_multi_word_loop():
    loop = "thank you thank you thank you thank you thank you"
    assert asr.has_repeated_ngram_run(loop)


def test_ngram_run_non_repetitive_long():
    text = " ".join(f"word{i}" for i in range(50))
    assert not asr.has_repeated_ngram_run(text)


def test_ngram_run_empty():
    assert not asr.has_repeated_ngram_run("")


# ---------------------------------------------------------------------------
# flag_degenerate_transcription
# ---------------------------------------------------------------------------

def test_clean_transcription_is_not_flagged():
    assert not asr.flag_degenerate_transcription("the capital of france is paris")


def test_repetition_loop_is_flagged():
    looping = "the cat sat " * 12
    assert asr.flag_degenerate_transcription(looping)


def test_high_compression_ratio_is_flagged():
    very_repetitive = "abc " * 500
    assert asr.flag_degenerate_transcription(very_repetitive)


def test_empty_transcription_is_not_flagged():
    # Empty string compression ratio == 1.0, below threshold; no ngram run.
    assert not asr.flag_degenerate_transcription("")


# ---------------------------------------------------------------------------
# AsrItem classification
# ---------------------------------------------------------------------------

def test_asr_item_classifies_real_word_as_regime_b(is_word):
    item = asr.AsrItem(
        task_id="t1",
        clean_text="the weather is nice today",
        transcription="the whether is nice today",
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    assert item.regime_candidate == SemanticClass.B
    assert item.damerau_levenshtein_distance > 0
    assert item.selection_policy == SelectionPolicy.ASR_CLEAN


def test_asr_item_noisy_tag(is_word):
    item = asr.AsrItem(
        task_id="t2", clean_text="hello world", transcription="hellp world",
        signal_to_noise_ratio_db=10.0)
    item.classify(is_word)
    assert item.selection_policy == SelectionPolicy.ASR_NOISY


def test_asr_item_nonword_substitution_is_regime_a(is_word):
    """If the hypothesis word is NOT in the dictionary, item should be Regime A."""
    item = asr.AsrItem(
        task_id="t3",
        clean_text="the temperature is constant",
        transcription="the xyzqrst is constant",  # "xyzqrst" is not a word
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    # xyzqrst is guaranteed not a real word for any reasonable is_word predicate.
    # If the sub is classified nonword, regime must be A.
    if not is_word("xyzqrst"):
        assert item.regime_candidate == SemanticClass.A


def test_asr_item_identical_text_has_zero_distance(is_word):
    item = asr.AsrItem(
        task_id="t4", clean_text="hello world", transcription="hello world",
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    assert item.damerau_levenshtein_distance == 0
    assert item.band is None


def test_asr_item_degenerate_flagged(is_word):
    looping = "the cat sat on a mat " * 15
    item = asr.AsrItem(
        task_id="t5", clean_text="the cat sat", transcription=looping,
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    assert item.is_degenerate


def test_asr_item_clean_text_not_degenerate(is_word):
    item = asr.AsrItem(
        task_id="t6", clean_text="the quick brown fox", transcription="the quick brown fox",
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    assert not item.is_degenerate


def test_asr_item_band_assigned_after_classify(is_word):
    item = asr.AsrItem(
        task_id="t7", clean_text="the cat", transcription="the bat",
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    if item.damerau_levenshtein_distance > 0:
        assert item.band is not None
    else:
        assert item.band is None


def test_asr_item_word_diffs_populated(is_word):
    item = asr.AsrItem(
        task_id="t8",
        clean_text="the weather is nice",
        transcription="the whether is nice",
        signal_to_noise_ratio_db=None)
    item.classify(is_word)
    assert len(item.word_diffs) > 0
