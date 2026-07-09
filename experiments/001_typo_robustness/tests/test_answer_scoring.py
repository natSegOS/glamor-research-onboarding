"""Answer extraction and scoring (src/scoring.py): reasoning/MCQ answer
extraction rules, the inline (structural) parse-status classifier, tolerance
vs exact-match dispatch, and per-TaskFamily scoring dispatch.

Each extraction/scoring rule is a fixed, named business rule (hash-delimiter
precedence, currency stripping, explicit-marker-beats-line-leading, ...) —
not an algorithmic property that holds over an open-ended input space — so
the "whole category of cases" here is a parametrized truth table per
function, covering expected, edge, and adversarial rows together, rather than
one assertion per test.
"""

from __future__ import annotations

import pytest

import scoring
from enums import ParseStatus, TaskFamily


# ---------------------------------------------------------------------------
# Reasoning answer extraction: value, and precedence between rules.
# ---------------------------------------------------------------------------

REASONING_EXTRACTION_CASES = [
    # (generation_text, expected_value, case_id)
    ("blah 5 blah\n#### 19", 19.0, "hash_delimited_preferred"),
    ("the answer is 7 then 19", 19.0, "falls_back_to_last_number"),
    ("Total: $1,234", 1234.0, "currency_and_commas"),
    ("no digits here", None, "no_number_returns_none"),
    ("#### 5\n#### 19", 19.0, "multiple_hash_delimiters_last_wins"),
    ("#### -7", -7.0, "negative_number"),
    ("answer: 3.14", 3.14, "trailing_decimal"),
    ("#### 0", 0.0, "zero"),
    ("", None, "empty_string"),
    ("   \n\t  ", None, "whitespace_only"),
    ("#### $1,234.50", 1234.50, "hash_with_currency_and_commas"),
    ("3 cats and 7 dogs", 7.0, "last_number_fallback_among_several"),
    ("text 12 more\n#### 5", 5.0, "hash_beats_earlier_last_number"),
]


class TestReasoningExtraction:

    @pytest.mark.parametrize(
        "generation_text,expected_value", [(c[0], c[1]) for c in REASONING_EXTRACTION_CASES],
        ids=[c[2] for c in REASONING_EXTRACTION_CASES])
    def test_extraction_rules(self, generation_text, expected_value):
        assert scoring.extract_reasoning_answer(generation_text)[0] == expected_value


# ---------------------------------------------------------------------------
# MCQ answer extraction: letter, option-count bounds, and marker precedence.
# ---------------------------------------------------------------------------

MCQ_EXTRACTION_CASES = [
    # (generation_text, option_count, expected_letter, case_id)
    ("I think A but Answer: C", None, "C", "explicit_marker_preferred"),
    ("B) Photosynthesis", None, "B", "line_leading_letter"),
    ("The answer is J", 4, None, "letter_beyond_option_count_invalid"),
    ("Answer: J", 10, "J", "letter_at_max_option_count_valid"),
    ("Answer: A ... Answer: C", None, "C", "last_explicit_marker_wins"),
    ("ANSWER: B", None, "B", "case_insensitive_marker_upper"),
    ("answer is c", None, "C", "case_insensitive_marker_lower"),
    ("no response provided at all", None, None, "no_valid_letter_returns_none"),
    ("", None, None, "empty_string_returns_none"),
    ("B) is wrong\nAnswer: D", None, "D", "explicit_beats_line_leading"),
    ("Answer: K", None, None, "letter_k_always_invalid"),
    ("Answer: A", 1, "A", "option_count_one_accepts_a"),
    ("Answer: B", 1, None, "option_count_one_rejects_b"),
]


class TestMultipleChoiceExtraction:

    @pytest.mark.parametrize(
        "generation_text,option_count,expected_letter",
        [(c[0], c[1], c[2]) for c in MCQ_EXTRACTION_CASES],
        ids=[c[3] for c in MCQ_EXTRACTION_CASES])
    def test_extraction_rules(self, generation_text, option_count, expected_letter):
        kwargs = {} if option_count is None else {"option_count": option_count}
        assert scoring.extract_multiple_choice_answer(generation_text, **kwargs)[0] == expected_letter


# ---------------------------------------------------------------------------
# Parse-status taxonomy — the inline (structural) classifier.
#
# It returns only VALID or UNPARSEABLE. The full four-way taxonomy (adding
# CLARIFICATION and REFUSAL) is assigned by the post-stage linguistic
# classifier; see test_linguistic_parse_status.py.
# ---------------------------------------------------------------------------

INLINE_PARSE_STATUS_CASES = [
    # (generation_text, gold, expected_status, expected_is_correct, case_id)
    ("I have no idea", 19, ParseStatus.UNPARSEABLE, 0, "no_parseable_answer"),
    ("#### 19", 19, ParseStatus.VALID, 1, "answer_found"),
    ("Could you clarify what you mean?", 19, ParseStatus.UNPARSEABLE, 0,
     "clarification_surface_form_is_unparseable_not_clarification"),
    ("I cannot help with that.", 19, ParseStatus.UNPARSEABLE, 0,
     "refusal_surface_form_is_unparseable_not_refusal"),
    ("I won't, but if I had to: #### 19", 19, ParseStatus.VALID, 1,
     "refusal_followed_by_an_answer_is_still_valid"),
    ("Did you mean 19? The answer might be 19.", 19, ParseStatus.VALID, 1,
     "classifier_is_purely_structural_ignores_surrounding_text"),
]


class TestInlineParseStatusClassifier:

    @pytest.mark.parametrize(
        "generation_text,gold,expected_status,expected_is_correct",
        [(c[0], c[1], c[2], c[3]) for c in INLINE_PARSE_STATUS_CASES],
        ids=[c[4] for c in INLINE_PARSE_STATUS_CASES])
    def test_classification_rules(self, generation_text, gold, expected_status, expected_is_correct):
        result = scoring.score_reasoning(generation_text, gold)
        assert result.parse_status == expected_status
        assert result.is_correct == expected_is_correct


# ---------------------------------------------------------------------------
# Reasoning scoring — integer exact-match vs. float tolerance.
# ---------------------------------------------------------------------------

REASONING_SCORING_CASES = [
    # (generation_text, gold, expected_is_correct, case_id)
    ("#### 18", 19, 0, "integer_mismatch"),
    ("#### 19", 19, 1, "integer_exact_match"),
    ("#### 3.1415930", 3.1415927, 1, "float_within_tolerance"),          # 1e-6 apart
    ("#### 3.142", 3.1415927, 0, "float_outside_tolerance"),
    ("#### 19.5", 19, 0, "integer_gold_rejects_fractional_match"),
    ("#### 0", 0, 1, "zero_gold_correct"),
    ("#### 1", 0, 0, "zero_gold_incorrect"),
    ("#### -5", -5, 1, "negative_gold_correct"),
    ("#### 5", -5, 0, "negative_gold_incorrect"),
]


class TestReasoningScoring:

    @pytest.mark.parametrize(
        "generation_text,gold,expected_is_correct",
        [(c[0], c[1], c[2]) for c in REASONING_SCORING_CASES],
        ids=[c[3] for c in REASONING_SCORING_CASES])
    def test_scoring_rules(self, generation_text, gold, expected_is_correct):
        assert scoring.score_reasoning(generation_text, gold).is_correct == expected_is_correct


# ---------------------------------------------------------------------------
# MCQ scoring.
# ---------------------------------------------------------------------------

class TestMultipleChoiceScoring:

    @pytest.mark.parametrize("generation_text,gold,expected_is_correct", [
        ("Answer: C", "C", 1),
        ("Answer: A", "C", 0),
        ("Answer: C", "c", 1),   # gold provided lowercase; extraction always uppercases
    ], ids=["correct_letter", "wrong_letter", "gold_case_insensitive"])
    def test_scoring_rules(self, generation_text, gold, expected_is_correct):
        assert scoring.score_multiple_choice(generation_text, gold).is_correct == expected_is_correct

    def test_unparseable_generation_is_incorrect_not_an_error(self):
        result = scoring.score_multiple_choice("no response provided at all", "C")
        assert result.parse_status == ParseStatus.UNPARSEABLE
        assert result.is_correct == 0


# ---------------------------------------------------------------------------
# Scoring dispatch: every TaskFamily routes to the right scorer.
# ---------------------------------------------------------------------------

class TestScoringDispatch:

    @pytest.mark.parametrize("generation_text,gold,task_family", [
        ("#### 19", 19, TaskFamily.GSM_SYMBOLIC_SYNTHETIC),
        ("Answer: C", "C", TaskFamily.MMLU_PRO),
        ("#### 42", 42, TaskFamily.GSM_SYMBOLIC_OFFICIAL),
        ("#### 7", 7, TaskFamily.GSM8K),
        ("Answer: A", "A", TaskFamily.MMLU),
        ("Answer: B", "B", TaskFamily.MCQ_DEMO),
    ], ids=lambda value: value.value if isinstance(value, TaskFamily) else None)
    def test_every_task_family_dispatches_and_scores_correctly(self, generation_text, gold, task_family):
        assert scoring.score(generation_text, gold, task_family).is_correct == 1

    def test_unknown_task_family_raises(self):
        with pytest.raises((ValueError, KeyError)):
            scoring.score("#### 1", 1, "unknown_family_xyz")


# ---------------------------------------------------------------------------
# ScoreResult structure.
# ---------------------------------------------------------------------------

class TestScoreResultStructure:

    def test_valid_result_carries_a_parsed_answer(self):
        result = scoring.score_reasoning("#### 19", 19)
        assert result.parse_status == ParseStatus.VALID
        assert result.is_correct == 1
        assert result.parsed_answer is not None

    def test_unparseable_result_has_no_parsed_answer(self):
        result = scoring.score_reasoning("nonsense text", 19)
        assert result.parse_status == ParseStatus.UNPARSEABLE
        assert result.is_correct == 0
        assert result.parsed_answer is None

    @pytest.mark.parametrize("generation_text,gold", [
        ("#### 5", 5), ("#### 6", 5), ("nonsense", 5)])
    def test_is_correct_is_always_exactly_zero_or_one(self, generation_text, gold):
        result = scoring.score_reasoning(generation_text, gold)
        assert result.is_correct in (0, 1)
        assert type(result.is_correct) is int
