"""Tests for GSM-Symbolic template parsing and instance parameter extraction.

Covers:
- parse_gsm_symbolic_template: {param,value} parsing, answer expression eval,
  gold-answer validation, both '#answer:' and '#answer =' separator forms,
  fraction-word params, verbal-multiplier params.
- extract_instance_parameters: int params, str params, multi-word str params,
  currency-symbol prefixes, repeated params with article variation, fraction-word
  and verbal-multiplier str params, named-group backreference robustness.
- serialize_parameters / deserialize_parameters: lossless Fraction round-trip.
- FRACTION_WORDS / VERBAL_MULTIPLIER_WORDS: representative entries.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from tasks.reasoning import (
    FRACTION_WORDS,
    VERBAL_MULTIPLIER_WORDS,
    deserialize_parameters,
    extract_instance_parameters,
    parse_gsm_symbolic_template,
    serialize_parameters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qa(question_part: str, answer_expr: str, *, separator: str = ":") -> str:
    """Build a minimal question_annotated string."""
    return f"{question_part}\n\n#answer{separator} {answer_expr}"


# ---------------------------------------------------------------------------
# parse_gsm_symbolic_template
# ---------------------------------------------------------------------------

class TestParseGsmSymbolicTemplate:
    def test_basic_int_params(self):
        qa = _qa("{name,Alice} has {n,5} apples and gives away {k,2}.", "n - k")
        t = parse_gsm_symbolic_template({"question_annotated": qa})
        assert t is not None
        assert t.answer_function(n=5, k=2) == 3

    def test_gold_validation_passes(self):
        qa = _qa("{name,Alice} buys {a,3} boxes of {b,4} each.", "a * b",
                 separator=":")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": 12})
        assert t is not None

    def test_gold_validation_fails_returns_none(self):
        qa = _qa("{name,Alice} buys {a,3} boxes of {b,4} each.", "a * b")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": 99})
        assert t is None

    def test_gold_none_skips_validation(self):
        qa = _qa("{name,Alice} has {n,5} items.", "n * 2")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": None})
        assert t is not None  # no validation with gold_answer=None

    def test_answer_equals_separator(self):
        qa = _qa("{x,10} items at {price,3} each.", "x * price", separator=" =")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": 30})
        assert t is not None
        assert t.answer_function(x=10, price=3) == 30

    def test_fraction_param(self):
        qa = _qa("{name,Bob} eats {frac,half} of {n,10} cookies.", "int(n * frac)")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": 5})
        assert t is not None
        assert t.answer_function(frac=Fraction(1, 2), n=10) == 5

    def test_missing_answer_section_returns_none(self):
        qa = "{name,Alice} has {n,5} apples."
        t = parse_gsm_symbolic_template({"question_annotated": qa})
        assert t is None

    def test_empty_question_annotated_returns_none(self):
        t = parse_gsm_symbolic_template({"question_annotated": ""})
        assert t is None

    def test_question_format_strips_meta_sections(self):
        qa = _qa("{n,3} + {m,4} equals what?", "n + m") + "\n\n#init:\n- $n = range(1,10)"
        t = parse_gsm_symbolic_template({"question_annotated": qa})
        assert t is not None
        assert "#init" not in t.question_format
        assert "{n}" in t.question_format

    def test_sandboxed_eval_blocks_builtins(self):
        # With gold_answer=None validation is skipped, so the template parses;
        # the sandbox only fires when answer_function is actually called.
        qa = _qa("{n,3} items.", "__import__('os').system('echo hi')")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": None})
        assert t is not None
        # Calling the function must raise NameError (no builtins in scope).
        with pytest.raises((NameError, TypeError)):
            t.answer_function(n=3)

    def test_verbal_multiplier_default(self):
        qa = _qa("{n,5} girls, {mult,twice} as many boys.", "int(n * mult)")
        t = parse_gsm_symbolic_template({"question_annotated": qa, "gold_answer": None})
        assert t is not None
        # "twice" is not in FRACTION_WORDS so it stays as str in template defaults;
        # the answer_function would fail with str arithmetic — that's expected at
        # parse time (gold=None skips validation).


# ---------------------------------------------------------------------------
# extract_instance_parameters
# ---------------------------------------------------------------------------

class TestExtractInstanceParameters:
    """Each test constructs a minimal question_annotated (template) and a
    corresponding question_text (HF instance), then checks extraction."""

    def _simple_qa(self, fmt: str, answer: str, separator: str = ":") -> str:
        return f"{fmt}\n\n#answer{separator} {answer}"

    # --- basic int extraction ---

    def test_single_int_param(self):
        qa = self._simple_qa("{name,Alice} has {n,5} apples.", "n")
        params = extract_instance_parameters(
            qa, "Bob has 12 apples.", gold_answer=12)
        assert params == {"name": "Bob", "n": 12}

    def test_multiple_int_params(self):
        qa = self._simple_qa("{name,Alice} buys {a,3} boxes of {b,4} each.", "a * b")
        params = extract_instance_parameters(
            qa, "Carla buys 6 boxes of 8 each.", gold_answer=48)
        assert params == {"name": "Carla", "a": 6, "b": 8}

    def test_int_with_comma_separator(self):
        qa = self._simple_qa("{name,Alice} earns ${n,1000} per month.", "n")
        params = extract_instance_parameters(
            qa, "Bob earns $3,500 per month.", gold_answer=3500)
        assert params is not None
        assert params["n"] == 3500

    # --- str params ---

    def test_str_param_name(self):
        qa = self._simple_qa("{name,Alice} has {n,5} items.", "n")
        params = extract_instance_parameters(qa, "DeShawn has 7 items.", gold_answer=7)
        assert params is not None
        assert params["name"] == "DeShawn"

    def test_multi_word_str_param(self):
        # Use a gender-neutral literal so the template literal matches exactly.
        qa = self._simple_qa("{name,Alice} enjoys {hobby,her knitting} daily.", "5")
        params = extract_instance_parameters(
            qa, "Carlos enjoys his painting daily.", gold_answer=5)
        assert params is not None
        assert params["hobby"] == "his painting"

    def test_currency_symbol_str_param(self):
        # {cur,$} — currency symbol extracted as a str param (not a letter).
        qa = self._simple_qa(
            "{name,Alice} earns {cur,$}{n,50}/hour.",
            "n * 8",
        )
        params = extract_instance_parameters(
            qa, "Bob earns $120/hour.", gold_answer=960)
        assert params is not None
        assert params["cur"] == "$"
        assert params["n"] == 120

    # --- repeated params ---

    def test_repeated_int_param_backreference(self):
        qa = self._simple_qa("{n,5} apples. He eats {k,2} leaving {n,5} - {k,2}.", "n - k")
        params = extract_instance_parameters(
            qa, "8 apples. He eats 3 leaving 8 - 3.", gold_answer=5)
        assert params == {"n": 8, "k": 3}

    def test_repeated_str_param_article_variation(self):
        # Template: "hire {profession}" and "{profession} charges ..."
        # Instance: "hire an accountant" and "the accountant charges ..."
        # The second occurrence differs only in article — should still extract.
        qa = self._simple_qa(
            "{name,Jackie} can hire {profession,an accountant}."
            " {profession,an accountant} charges {fee,90}.",
            "fee",
        )
        params = extract_instance_parameters(
            qa, "Olivia can hire a lawyer. The lawyer charges 150.", gold_answer=150)
        assert params is not None
        assert params["fee"] == 150

    # --- fraction / multiplier conversion ---

    def test_fraction_word_extraction(self):
        qa = self._simple_qa("{name,Alice} eats {frac,half} of {n,10} cookies.", "int(n * frac)")
        params = extract_instance_parameters(
            qa, "Bob eats a quarter of 20 cookies.", gold_answer=5)
        assert params is not None
        assert params["frac"] == Fraction(1, 4)
        assert params["n"] == 20

    def test_fraction_hyphenated_extraction(self):
        qa = self._simple_qa("{name,Alice} finishes {frac,one-third} of {n,9} tasks.", "int(n * frac)")
        params = extract_instance_parameters(
            qa, "Bob finishes one-fifth of 15 tasks.", gold_answer=3)
        assert params is not None
        assert params["frac"] == Fraction(1, 5)

    def test_verbal_multiplier_twice(self):
        qa = self._simple_qa(
            "There are {n,6} girls. There are {mult,twice} as many boys.", "int(n * mult)")
        params = extract_instance_parameters(
            qa, "There are 44 girls. There are three times as many boys.", gold_answer=132)
        assert params is not None
        assert params["mult"] == 3

    def test_verbal_multiplier_double(self):
        qa = self._simple_qa("{name,A} has {n,5} items and {mult,twice} that.", "int(n * (1 + mult))")
        params = extract_instance_parameters(
            qa, "Bob has 10 items and double that.", gold_answer=30)
        assert params is not None
        assert params["mult"] == 2

    def test_verbal_multiplier_quintuple(self):
        qa = self._simple_qa("{name,A} multiplied by {mult,twice}.", "int(5 * mult)")
        params = extract_instance_parameters(
            qa, "Bob multiplied by quintuple.", gold_answer=25)
        assert params is not None
        assert params["mult"] == 5

    def test_cardinal_number_word_extraction(self):
        qa = self._simple_qa("{name,Alice} buys {n,seven} boxes each weighing {w,5} kg.", "int(n * w)")
        params = extract_instance_parameters(
            qa, "Bob buys seven boxes each weighing 3 kg.", gold_answer=21)
        assert params is not None
        assert params["n"] == 7
        assert params["w"] == 3

    # --- answer separator forms ---

    def test_answer_equals_separator_extraction(self):
        qa = "{n,10} items.\n\n#answer = n * 2"
        params = extract_instance_parameters(qa, "5 items.", gold_answer=10)
        assert params is not None
        assert params["n"] == 5

    # --- failure cases ---

    def test_returns_none_on_gold_mismatch(self):
        qa = self._simple_qa("{name,Alice} has {n,5} apples.", "n")
        params = extract_instance_parameters(qa, "Bob has 12 apples.", gold_answer=99)
        assert params is None

    def test_returns_none_on_no_match(self):
        qa = self._simple_qa("{name,Alice} buys {n,5} {item,apple}s.", "n")
        params = extract_instance_parameters(
            qa, "COMPLETELY UNRELATED TEXT!!!", gold_answer=5)
        assert params is None


# ---------------------------------------------------------------------------
# serialize_parameters / deserialize_parameters (lossless round-trip)
# ---------------------------------------------------------------------------

class TestParameterCodec:
    def test_int_passthrough(self):
        p = {"a": 3, "b": 7}
        assert deserialize_parameters(serialize_parameters(p)) == p

    def test_str_passthrough(self):
        p = {"name": "Alice", "item": "apple"}
        assert deserialize_parameters(serialize_parameters(p)) == p

    def test_fraction_lossless(self):
        p = {"frac": Fraction(1, 3), "n": 9}
        wire = serialize_parameters(p)
        assert wire["frac"] == {"__fraction__": [1, 3]}
        assert deserialize_parameters(wire) == p

    def test_mixed(self):
        p = {"n": 10, "frac": Fraction(2, 3), "name": "Bob"}
        assert deserialize_parameters(serialize_parameters(p)) == p

    def test_empty_dict(self):
        assert serialize_parameters({}) == {}
        assert deserialize_parameters({}) == {}


# ---------------------------------------------------------------------------
# FRACTION_WORDS and VERBAL_MULTIPLIER_WORDS coverage
# ---------------------------------------------------------------------------

class TestWordDictionaries:
    @pytest.mark.parametrize("word,expected", [
        ("half",           Fraction(1, 2)),
        ("one-third",      Fraction(1, 3)),
        ("one-fifth",      Fraction(1, 5)),
        ("a quarter",      Fraction(1, 4)),
        ("two-thirds",     Fraction(2, 3)),
        ("three quarters", Fraction(3, 4)),
    ])
    def test_fraction_words(self, word, expected):
        assert FRACTION_WORDS[word] == expected

    @pytest.mark.parametrize("word,expected", [
        ("once",        1),
        ("twice",       2),
        ("double",      2),
        ("two times",   2),
        ("thrice",      3),
        ("triple",      3),
        ("three times", 3),
        ("quadruple",   4),
        ("quintuple",   5),
        ("seven",       7),
        ("twelve",      12),
        ("twenty",      20),
    ])
    def test_verbal_multiplier_words(self, word, expected):
        assert VERBAL_MULTIPLIER_WORDS[word] == expected
