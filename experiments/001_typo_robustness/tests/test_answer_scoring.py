"""Adversarial and edge-case tests for answer extraction and scoring.

Covers: multi-delimiter precedence, negative/currency/comma numbers, float
tolerance boundary, integer exact-match, MCQ letter range, priority ordering
(explicit > line-leading > standalone), refusal-with-answer, clarification
override, empty/whitespace input, case-folding, and overall scoring dispatch.
"""

from __future__ import annotations

import math

import pytest

import scoring
from enums import ParseStatus, TaskFamily


# ---------------------------------------------------------------------------
# Reasoning answer extraction — happy path
# ---------------------------------------------------------------------------

def test_reasoning_prefers_hash_delimited_answer():
    assert scoring.extract_reasoning_answer("blah 5 blah\n#### 19") == 19.0


def test_reasoning_falls_back_to_last_number():
    assert scoring.extract_reasoning_answer("the answer is 7 then 19") == 19.0


def test_reasoning_handles_currency_and_commas():
    assert scoring.extract_reasoning_answer("Total: $1,234") == 1234.0


def test_reasoning_returns_none_without_number():
    assert scoring.extract_reasoning_answer("no digits here") is None


# ---------------------------------------------------------------------------
# Reasoning answer extraction — adversarial / edge cases
# ---------------------------------------------------------------------------

def test_multiple_hash_delimiters_last_wins():
    """When there are multiple '####' lines, the LAST one is authoritative."""
    assert scoring.extract_reasoning_answer("#### 5\n#### 19") == 19.0


def test_negative_number_extracted():
    assert scoring.extract_reasoning_answer("#### -7") == -7.0


def test_number_with_trailing_decimal():
    assert scoring.extract_reasoning_answer("answer: 3.14") == 3.14


def test_number_zero():
    assert scoring.extract_reasoning_answer("#### 0") == 0.0


def test_empty_string_returns_none():
    assert scoring.extract_reasoning_answer("") is None


def test_whitespace_only_returns_none():
    assert scoring.extract_reasoning_answer("   \n\t  ") is None


def test_hash_delimiter_with_currency_and_commas():
    assert scoring.extract_reasoning_answer("#### $1,234.50") == 1234.50


def test_last_number_fallback_when_no_delimiter():
    # Among several numbers, fallback picks the LAST one.
    assert scoring.extract_reasoning_answer("3 cats and 7 dogs") == 7.0


def test_hash_takes_precedence_over_last_number():
    # "#### 5" comes before the "12" but must win because it's hash-delimited.
    assert scoring.extract_reasoning_answer("text 12 more\n#### 5") == 5.0


# ---------------------------------------------------------------------------
# MCQ extraction — happy path
# ---------------------------------------------------------------------------

def test_mcq_prefers_explicit_marker():
    assert scoring.extract_multiple_choice_answer("I think A but Answer: C") == "C"


def test_mcq_line_leading_letter():
    assert scoring.extract_multiple_choice_answer("B) Photosynthesis") == "B"


def test_mcq_respects_option_count():
    # 'J' is invalid when there are only 4 options.
    assert scoring.extract_multiple_choice_answer("The answer is J", option_count=4) is None


# ---------------------------------------------------------------------------
# MCQ extraction — adversarial / edge cases
# ---------------------------------------------------------------------------

def test_mcq_letter_at_max_option_count_is_valid():
    # J is valid when there are 10 options (the full A–J range).
    assert scoring.extract_multiple_choice_answer("Answer: J", option_count=10) == "J"


def test_mcq_last_explicit_marker_wins():
    """The LAST explicit marker is returned."""
    assert scoring.extract_multiple_choice_answer("Answer: A ... Answer: C") == "C"


def test_mcq_case_insensitive_marker():
    assert scoring.extract_multiple_choice_answer("ANSWER: B") == "B"
    assert scoring.extract_multiple_choice_answer("answer is c") == "C"


def test_mcq_no_valid_letter_returns_none():
    # Must use text with no A-J letters that would accidentally match.
    # "no response provided" — no a-j uppercase, no "answer <letter>" pattern.
    assert scoring.extract_multiple_choice_answer("no response provided at all") is None


def test_mcq_empty_string_returns_none():
    assert scoring.extract_multiple_choice_answer("") is None


def test_mcq_explicit_beats_line_leading():
    text = "B) is wrong\nAnswer: D"
    assert scoring.extract_multiple_choice_answer(text) == "D"


def test_mcq_letter_k_always_invalid():
    # K is never in the A–J alphabet for any option count.
    assert scoring.extract_multiple_choice_answer("Answer: K") is None


def test_mcq_option_count_one_only_accepts_a():
    assert scoring.extract_multiple_choice_answer("Answer: A", option_count=1) == "A"
    assert scoring.extract_multiple_choice_answer("Answer: B", option_count=1) is None


# ---------------------------------------------------------------------------
# Parse-status taxonomy
# ---------------------------------------------------------------------------

def test_clarification_detected():
    result = scoring.score_reasoning("Could you clarify what you mean?", 19)
    assert result.parse_status == ParseStatus.CLARIFICATION
    assert result.is_correct == 0


def test_refusal_detected_only_without_answer():
    refusal = scoring.score_reasoning("I cannot help with that.", 19)
    assert refusal.parse_status == ParseStatus.REFUSAL

    hedged_but_answered = scoring.score_reasoning("I can't be sure, but #### 19", 19)
    assert hedged_but_answered.parse_status == ParseStatus.VALID
    assert hedged_but_answered.is_correct == 1


def test_unparseable_status():
    result = scoring.score_reasoning("I have no idea", 19)
    assert result.parse_status == ParseStatus.UNPARSEABLE


def test_clarification_phrase_variants():
    """All clarification phrases in the lexicon must trigger the status."""
    triggers = [
        "did you mean France?",
        "do you mean the city?",
        "could you clarify the question?",
        "can you clarify what you want?",
        "please clarify your intent",
        "not sure what you mean by that",
        "what do you mean exactly?",
    ]
    for phrase in triggers:
        result = scoring.score_reasoning(phrase, 19)
        assert result.parse_status == ParseStatus.CLARIFICATION, (
            f"phrase {phrase!r} did not trigger CLARIFICATION")


def test_refusal_phrase_variants():
    """All refusal phrases (without a parseable answer) must trigger REFUSAL."""
    triggers = [
        "I cannot assist with math homework.",
        "I can't help with this.",
        "I won't answer this question.",
        "I am unable to solve this.",
        "I'm unable to provide an answer.",
        "I refuse to respond.",
        "As an AI, I don't have opinions.",
    ]
    for phrase in triggers:
        result = scoring.score_reasoning(phrase, 19)
        assert result.parse_status == ParseStatus.REFUSAL, (
            f"phrase {phrase!r} did not trigger REFUSAL (got {result.parse_status})")


def test_refusal_with_answer_is_valid():
    """A refusal phrase followed by a parseable answer should still score as VALID."""
    result = scoring.score_reasoning("I won't, but if I had to: #### 19", 19)
    assert result.parse_status == ParseStatus.VALID
    assert result.is_correct == 1


def test_clarification_with_answer_is_still_clarification():
    """Clarification + parseable number: clarification wins."""
    result = scoring.score_reasoning("Did you mean 19? The answer might be 19.", 19)
    assert result.parse_status == ParseStatus.CLARIFICATION


# ---------------------------------------------------------------------------
# Reasoning scoring — tolerance and integer exact-match
# ---------------------------------------------------------------------------

def test_score_integer_exact_match_not_fuzzy():
    assert scoring.score_reasoning("#### 18", 19).is_correct == 0
    assert scoring.score_reasoning("#### 19", 19).is_correct == 1


def test_float_gold_uses_tolerance():
    # Use a non-integer gold so the tolerance path is taken.
    # 3.1415930 is within 1e-6 of 3.1415927.
    gold = 3.1415927
    result = scoring.score_reasoning("#### 3.1415930", gold)
    assert result.is_correct == 1


def test_float_gold_just_outside_tolerance():
    # 3.142 is well outside 1e-6 of 3.1415927.
    gold = 3.1415927
    result = scoring.score_reasoning("#### 3.142", gold)
    assert result.is_correct == 0


def test_integer_gold_rejects_fractional_match():
    # 19.5 != 19 for integer gold.
    assert scoring.score_reasoning("#### 19.5", 19).is_correct == 0


def test_gold_zero_is_scoreable():
    assert scoring.score_reasoning("#### 0", 0).is_correct == 1
    assert scoring.score_reasoning("#### 1", 0).is_correct == 0


def test_negative_gold():
    assert scoring.score_reasoning("#### -5", -5).is_correct == 1
    assert scoring.score_reasoning("#### 5", -5).is_correct == 0


# ---------------------------------------------------------------------------
# MCQ scoring
# ---------------------------------------------------------------------------

def test_mcq_correct_letter():
    result = scoring.score_multiple_choice("Answer: C", "C")
    assert result.is_correct == 1


def test_mcq_wrong_letter():
    result = scoring.score_multiple_choice("Answer: A", "C")
    assert result.is_correct == 0


def test_mcq_gold_case_insensitive():
    # Gold provided as lowercase; extraction always uppercases.
    result = scoring.score_multiple_choice("Answer: C", "c")
    assert result.is_correct == 1


def test_mcq_unparseable():
    # "no response provided" has no A-J letters that could match; genuinely unparseable.
    result = scoring.score_multiple_choice("no response provided at all", "C")
    assert result.parse_status == ParseStatus.UNPARSEABLE
    assert result.is_correct == 0


# ---------------------------------------------------------------------------
# Scoring dispatch
# ---------------------------------------------------------------------------

def test_score_dispatch_reasoning():
    result = scoring.score("#### 19", 19, TaskFamily.GSM_SYMBOLIC_SYNTHETIC)
    assert result.is_correct == 1


def test_score_dispatch_mcq():
    result = scoring.score("Answer: C", "C", TaskFamily.MMLU_PRO)
    assert result.is_correct == 1


def test_score_dispatch_reasoning_official():
    assert scoring.score("#### 42", 42, TaskFamily.GSM_SYMBOLIC_OFFICIAL).is_correct == 1


def test_score_dispatch_gsm8k():
    assert scoring.score("#### 7", 7, TaskFamily.GSM8K).is_correct == 1


def test_score_dispatch_mmlu():
    assert scoring.score("Answer: A", "A", TaskFamily.MMLU).is_correct == 1


def test_score_dispatch_mcq_demo():
    assert scoring.score("Answer: B", "B", TaskFamily.MCQ_DEMO).is_correct == 1


def test_score_dispatch_unknown_raises():
    with pytest.raises((ValueError, KeyError)):
        scoring.score("#### 1", 1, "unknown_family_xyz")


# ---------------------------------------------------------------------------
# ScoreResult structure
# ---------------------------------------------------------------------------

def test_score_result_structure_valid():
    result = scoring.score_reasoning("#### 19", 19)
    assert result.parse_status == ParseStatus.VALID
    assert result.is_correct == 1
    assert result.parsed_answer is not None


def test_score_result_structure_unparseable():
    result = scoring.score_reasoning("nonsense text", 19)
    assert result.parse_status == ParseStatus.UNPARSEABLE
    assert result.is_correct == 0
    assert result.parsed_answer is None


def test_score_result_is_correct_is_binary():
    """is_correct must be exactly 0 or 1, never a boolean or other value."""
    for gen, gold in [("#### 5", 5), ("#### 6", 5), ("nonsense", 5)]:
        r = scoring.score_reasoning(gen, gold)
        assert r.is_correct in (0, 1)
        assert type(r.is_correct) is int
