"""Task item construction (src/tasks/): the synthetic reasoning generator's
gold-recomputation guarantee, scope-span slicing, MCQ option-letter
conversion and demo items, enum coercion at the JSONL loader boundary, and
GSM-Symbolic template parsing / instance-parameter extraction / the
parameter codec's lossless Fraction round-trip.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from tasks import reasoning as tasks_reasoning
from tasks import multiple_choice as mcq
from tasks.reasoning import (
    FRACTION_WORDS,
    VERBAL_MULTIPLIER_WORDS,
    deserialize_parameters,
    extract_instance_parameters,
    parse_gsm_symbolic_template,
    serialize_parameters,
)
from enums import TaskFamily


# ---------------------------------------------------------------------------
# Synthetic reasoning generator.
# ---------------------------------------------------------------------------

class TestSyntheticReasoningGenerator:

    def test_is_deterministic(self):
        first = tasks_reasoning.generate_synthetic_reasoning_items(10, seed=1)
        second = tasks_reasoning.generate_synthetic_reasoning_items(10, seed=1)
        assert [item.question_text for item in first] == [item.question_text for item in second]

    def test_every_items_gold_matches_its_own_templates_answer_function(self):
        for item in tasks_reasoning.generate_synthetic_reasoning_items(100, seed=7):
            recomputed = int(item.template.answer_function(**item.parameters))
            assert recomputed == item.gold_answer, (
                f"item {item.task_id}: template gives {recomputed}, gold is {item.gold_answer}")
            assert item.gold_answer >= 0

    def test_every_items_key_terms_include_every_parameter_value(self):
        for item in tasks_reasoning.generate_synthetic_reasoning_items(20, seed=5):
            assert item.key_terms, f"item {item.task_id} has empty key_terms"
            for value in item.parameters.values():
                assert str(value) in item.key_terms, (
                    f"operand value {value!r} not in key_terms {item.key_terms!r}")

    def test_supports_regime_c_operand_swap(self):
        item = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=5)[0]
        assert item.supports_regime_c_operand_swap

    def test_task_ids_are_unique(self):
        items = tasks_reasoning.generate_synthetic_reasoning_items(20, seed=9)
        ids = [item.task_id for item in items]
        assert len(ids) == len(set(ids))

    def test_task_family_is_the_synthetic_enum_member(self):
        for item in tasks_reasoning.generate_synthetic_reasoning_items(5, seed=1):
            assert item.task_family == TaskFamily.GSM_SYMBOLIC_SYNTHETIC


class TestReasoningChatExemplars:

    def test_every_exemplar_solution_parses_under_the_frozen_scorer(self):
        import scoring
        from enums import ExtractionTier
        for _problem, solution in tasks_reasoning.REASONING_CHAT_EXEMPLARS:
            answer, tier = scoring.extract_reasoning_answer(solution)
            assert tier == ExtractionTier.HASH_DELIMITED
            assert answer is not None

    def test_every_exemplar_turn_repeats_the_instruction_scaffold(self):
        for user_message, _assistant_message in (
                tasks_reasoning.REASONING_CHAT_EXEMPLAR_TURNS):
            assert user_message.startswith(tasks_reasoning.REASONING_INSTRUCTION)

    def test_exemplar_problems_never_appear_in_the_item_prompt(self):
        item = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=4)[0]
        for problem, _solution in tasks_reasoning.REASONING_CHAT_EXEMPLARS:
            assert problem not in item.full_prompt

    def test_full_prompt_ends_with_the_reminder_after_the_question(self):
        item = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=4)[0]
        assert item.full_prompt.endswith(
            tasks_reasoning.REASONING_INSTRUCTION_REMINDER)
        assert (item.full_prompt.index(item.question_text)
                < item.full_prompt.rindex(
                    tasks_reasoning.REASONING_INSTRUCTION_REMINDER))


class TestReasoningScopeSpans:

    def test_instruction_and_content_spans_slice_out_the_right_text(self):
        item = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=4)[0]
        instruction_start, instruction_end = item.scope_spans["instruction"]
        assert item.full_prompt[instruction_start:instruction_end] == item.instruction
        content_start, content_end = item.scope_spans["content"]
        assert item.full_prompt[content_start:content_end] == item.question_text

    def test_instruction_and_content_spans_never_overlap(self):
        for item in tasks_reasoning.generate_synthetic_reasoning_items(10, seed=6):
            instruction_start, instruction_end = item.scope_spans["instruction"]
            content_start, content_end = item.scope_spans["content"]
            assert instruction_end <= content_start or content_end <= instruction_start, (
                f"item {item.task_id}: instruction [{instruction_start},{instruction_end}) "
                f"overlaps content [{content_start},{content_end})")

    def test_every_span_falls_within_the_full_prompt(self):
        for item in tasks_reasoning.generate_synthetic_reasoning_items(10, seed=8):
            full = item.full_prompt
            for scope_name, (start, end) in item.scope_spans.items():
                assert 0 <= start <= end <= len(full), (
                    f"scope {scope_name!r}: [{start},{end}) out of range for prompt of length {len(full)}")


# ---------------------------------------------------------------------------
# MCQ option-letter conversion and demo items.
# ---------------------------------------------------------------------------

class TestOptionsSequenceToLetterDict:

    @pytest.mark.parametrize("options,expected", [
        (["Oxygen", "Nitrogen", "Carbon"], {"A": "Oxygen", "B": "Nitrogen", "C": "Carbon"}),
    ], ids=["three_options"])
    def test_small_option_sets(self, options, expected):
        assert mcq._options_sequence_to_letter_dict(options) == expected

    def test_ten_options_covers_the_full_a_through_j_range(self):
        options = [f"Option{i}" for i in range(10)]
        result = mcq._options_sequence_to_letter_dict(options)
        assert set(result.keys()) == set("ABCDEFGHIJ")
        assert result["A"] == "Option0"
        assert result["J"] == "Option9"


class TestDemonstrationMcqItems:

    def test_every_item_is_well_formed(self):
        items = mcq.make_demonstration_multiple_choice_items()
        assert len(items) == 5
        for item in items:
            assert item.gold_letter in item.options
            assert item.option_count == len(item.options) >= 2
            for letter in item.options:
                assert f"{letter}." in item.full_prompt

    def test_content_scope_span_slices_out_the_content_text(self):
        item = mcq.make_demonstration_multiple_choice_items()[0]
        content_start, content_end = item.scope_spans["content"]
        assert item.full_prompt[content_start:content_end] == item.content_text

    def test_task_ids_are_unique(self):
        items = mcq.make_demonstration_multiple_choice_items()
        ids = [item.task_id for item in items]
        assert len(ids) == len(set(ids))

    def test_task_family_is_the_demo_enum_member(self):
        for item in mcq.make_demonstration_multiple_choice_items():
            assert item.task_family == TaskFamily.MCQ_DEMO


# ---------------------------------------------------------------------------
# JSONL loaders coerce a plain string task_family into the TaskFamily enum.
# ---------------------------------------------------------------------------

def test_reasoning_jsonl_loader_coerces_task_family(tmp_path):
    record = {
        "task_id": "t1",
        "question_text": "How many apples?",
        "full_prompt": "Q: How many apples?\nA:",
        "instruction": "Q:",
        "gold_answer": 5,
        "key_terms": ["5"],
        "parameters": {"a": 5},
        "scope_spans": {"instruction": [0, 2], "content": [3, 20]},
        "task_family": str(TaskFamily.GSM_SYMBOLIC_SYNTHETIC),    # plain string
        "source": str(TaskFamily.GSM_SYMBOLIC_SYNTHETIC),
        "supports_regime_c_operand_swap": False,
    }
    jsonl_path = tmp_path / "reasoning.jsonl"
    jsonl_path.write_text(json.dumps(record) + "\n")

    items = tasks_reasoning.load_reasoning_jsonl(jsonl_path)
    assert len(items) == 1
    assert items[0].task_family == TaskFamily.GSM_SYMBOLIC_SYNTHETIC
    assert isinstance(items[0].task_family, TaskFamily)


def test_multiple_choice_jsonl_loader_coerces_task_family(tmp_path):
    # The MCQ loader schema uses "question"/"answer", not "question_text"/"gold_letter".
    record = {
        "question": "Which is the capital of France?",
        "options": {"A": "Paris", "B": "Berlin"},
        "answer": "A",
        "task_family": str(TaskFamily.MMLU_PRO),   # plain string — must be coerced
        "key_terms": ["capital", "France"],
        "category": "Geography",
    }
    jsonl_path = tmp_path / "mcq.jsonl"
    jsonl_path.write_text(json.dumps(record) + "\n")

    items = mcq.load_multiple_choice_jsonl(jsonl_path)
    assert len(items) == 1
    assert items[0].task_family == TaskFamily.MMLU_PRO
    assert isinstance(items[0].task_family, TaskFamily)


# ---------------------------------------------------------------------------
# GSM-Symbolic template parsing (question_annotated -> ReasoningTemplate).
# ---------------------------------------------------------------------------

def _annotated_question(question_part: str, answer_expr: str, *, separator: str = ":") -> str:
    return f"{question_part}\n\n#answer{separator} {answer_expr}"


class TestParseGsmSymbolicTemplate:

    def test_basic_int_params(self):
        annotated = _annotated_question(
            "{name,Alice} has {n,5} apples and gives away {k,2}.", "n - k")
        template = parse_gsm_symbolic_template({"question_annotated": annotated})
        assert template is not None
        assert template.answer_function(n=5, k=2) == 3

    def test_gold_validation_passes(self):
        annotated = _annotated_question("{name,Alice} buys {a,3} boxes of {b,4} each.", "a * b")
        template = parse_gsm_symbolic_template({"question_annotated": annotated, "gold_answer": 12})
        assert template is not None

    def test_gold_validation_failure_returns_none(self):
        annotated = _annotated_question("{name,Alice} buys {a,3} boxes of {b,4} each.", "a * b")
        template = parse_gsm_symbolic_template({"question_annotated": annotated, "gold_answer": 99})
        assert template is None

    def test_gold_answer_none_skips_validation(self):
        annotated = _annotated_question("{name,Alice} has {n,5} items.", "n * 2")
        template = parse_gsm_symbolic_template({"question_annotated": annotated, "gold_answer": None})
        assert template is not None

    def test_answer_equals_separator_form(self):
        annotated = _annotated_question("{x,10} items at {price,3} each.", "x * price", separator=" =")
        template = parse_gsm_symbolic_template({"question_annotated": annotated, "gold_answer": 30})
        assert template is not None
        assert template.answer_function(x=10, price=3) == 30

    def test_fraction_valued_parameter(self):
        annotated = _annotated_question("{name,Bob} eats {frac,half} of {n,10} cookies.", "int(n * frac)")
        template = parse_gsm_symbolic_template({"question_annotated": annotated, "gold_answer": 5})
        assert template is not None
        assert template.answer_function(frac=Fraction(1, 2), n=10) == 5

    def test_missing_answer_section_returns_none(self):
        template = parse_gsm_symbolic_template(
            {"question_annotated": "{name,Alice} has {n,5} apples."})
        assert template is None

    def test_empty_question_annotated_returns_none(self):
        assert parse_gsm_symbolic_template({"question_annotated": ""}) is None

    def test_question_format_strips_meta_sections(self):
        annotated = (_annotated_question("{n,3} + {m,4} equals what?", "n + m")
                     + "\n\n#init:\n- $n = range(1,10)")
        template = parse_gsm_symbolic_template({"question_annotated": annotated})
        assert template is not None
        assert "#init" not in template.question_format
        assert "{n}" in template.question_format

    def test_sandboxed_eval_blocks_builtins(self):
        # gold_answer=None skips validation, so the template parses; the
        # sandbox only fires when answer_function is actually called.
        annotated = _annotated_question("{n,3} items.", "__import__('os').system('echo hi')")
        template = parse_gsm_symbolic_template({"question_annotated": annotated, "gold_answer": None})
        assert template is not None
        with pytest.raises((NameError, TypeError)):
            template.answer_function(n=3)


# ---------------------------------------------------------------------------
# Instance-parameter extraction (question_annotated + question_text -> params).
# ---------------------------------------------------------------------------

class TestExtractInstanceParameters:
    """Each test constructs a minimal question_annotated (template) and a
    corresponding question_text (HF instance), then checks extraction."""

    @pytest.mark.parametrize("annotated,instance_text,gold_answer,expected_params", [
        ("{name,Alice} has {n,5} apples.\n\n#answer: n",
         "Bob has 12 apples.", 12, {"name": "Bob", "n": 12}),
        ("{name,Alice} buys {a,3} boxes of {b,4} each.\n\n#answer: a * b",
         "Carla buys 6 boxes of 8 each.", 48, {"name": "Carla", "a": 6, "b": 8}),
        ("{n,5} apples. He eats {k,2} leaving {n,5} - {k,2}.\n\n#answer: n - k",
         "8 apples. He eats 3 leaving 8 - 3.", 5, {"n": 8, "k": 3}),
    ], ids=["single_int_param", "multiple_int_params", "repeated_int_param_backreference"])
    def test_exact_parameter_dict_extraction(self, annotated, instance_text, gold_answer, expected_params):
        assert extract_instance_parameters(annotated, instance_text, gold_answer=gold_answer) == expected_params

    def test_currency_and_comma_stripped_from_int_param(self):
        annotated = "{name,Alice} earns ${n,1000} per month.\n\n#answer: n"
        params = extract_instance_parameters(annotated, "Bob earns $3,500 per month.", gold_answer=3500)
        assert params["n"] == 3500

    def test_multi_word_str_param(self):
        annotated = "{name,Alice} enjoys {hobby,her knitting} daily.\n\n#answer: 5"
        params = extract_instance_parameters(
            annotated, "Carlos enjoys his painting daily.", gold_answer=5)
        assert params["hobby"] == "his painting"

    def test_currency_symbol_extracted_as_a_str_param(self):
        annotated = "{name,Alice} earns {cur,$}{n,50}/hour.\n\n#answer: n * 8"
        params = extract_instance_parameters(annotated, "Bob earns $120/hour.", gold_answer=960)
        assert params["cur"] == "$"
        assert params["n"] == 120

    def test_repeated_str_param_survives_article_variation(self):
        # Template: "hire {profession}" ... "{profession} charges ...".
        # Instance: "hire an accountant" ... "the accountant charges ..." —
        # differs only in article, must still extract the same value.
        annotated = ("{name,Jackie} can hire {profession,an accountant}."
                     " {profession,an accountant} charges {fee,90}.\n\n#answer: fee")
        params = extract_instance_parameters(
            annotated, "Olivia can hire a lawyer. The lawyer charges 150.", gold_answer=150)
        assert params["fee"] == 150

    @pytest.mark.parametrize("annotated,instance_text,gold_answer,param_name,expected_value", [
        ("{name,Alice} eats {frac,half} of {n,10} cookies.\n\n#answer: int(n * frac)",
         "Bob eats a quarter of 20 cookies.", 5, "frac", Fraction(1, 4)),
        ("{name,Alice} finishes {frac,one-third} of {n,9} tasks.\n\n#answer: int(n * frac)",
         "Bob finishes one-fifth of 15 tasks.", 3, "frac", Fraction(1, 5)),
        ("There are {n,6} girls. There are {mult,twice} as many boys.\n\n#answer: int(n * mult)",
         "There are 44 girls. There are three times as many boys.", 132, "mult", 3),
        ("{name,A} has {n,5} items and {mult,twice} that.\n\n#answer: int(n * (1 + mult))",
         "Bob has 10 items and double that.", 30, "mult", 2),
        ("{name,A} multiplied by {mult,twice}.\n\n#answer: int(5 * mult)",
         "Bob multiplied by quintuple.", 25, "mult", 5),
        ("{name,Alice} buys {n,seven} boxes each weighing {w,5} kg.\n\n#answer: int(n * w)",
         "Bob buys seven boxes each weighing 3 kg.", 21, "n", 7),
    ], ids=["fraction_word", "fraction_hyphenated", "verbal_multiplier_twice",
            "verbal_multiplier_double", "verbal_multiplier_quintuple", "cardinal_number_word"])
    def test_fraction_and_multiplier_word_conversion(
            self, annotated, instance_text, gold_answer, param_name, expected_value):
        params = extract_instance_parameters(annotated, instance_text, gold_answer=gold_answer)
        assert params[param_name] == expected_value

    def test_answer_equals_separator_form(self):
        params = extract_instance_parameters("{n,10} items.\n\n#answer = n * 2", "5 items.", gold_answer=10)
        assert params["n"] == 5

    def test_gold_mismatch_returns_none(self):
        annotated = "{name,Alice} has {n,5} apples.\n\n#answer: n"
        assert extract_instance_parameters(annotated, "Bob has 12 apples.", gold_answer=99) is None

    def test_no_match_returns_none(self):
        annotated = "{name,Alice} buys {n,5} {item,apple}s.\n\n#answer: n"
        assert extract_instance_parameters(annotated, "COMPLETELY UNRELATED TEXT!!!", gold_answer=5) is None


# ---------------------------------------------------------------------------
# serialize_parameters / deserialize_parameters — lossless round-trip.
# ---------------------------------------------------------------------------

class TestParameterCodec:

    @pytest.mark.parametrize("parameters", [
        {"a": 3, "b": 7},
        {"name": "Alice", "item": "apple"},
        {"n": 10, "frac": Fraction(2, 3), "name": "Bob"},
        {},
    ], ids=["ints", "strings", "mixed_int_fraction_string", "empty"])
    def test_round_trip_is_lossless(self, parameters):
        assert deserialize_parameters(serialize_parameters(parameters)) == parameters

    def test_fraction_wire_format_is_a_tagged_pair(self):
        wire = serialize_parameters({"frac": Fraction(1, 3), "n": 9})
        assert wire["frac"] == {"__fraction__": [1, 3]}

    @given(int_values=st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=5),
           fraction_values=st.dictionaries(
               st.text(min_size=1, max_size=8),
               st.integers(min_value=1, max_value=1000).flatmap(
                   lambda denominator: st.integers().map(lambda n: Fraction(n, denominator))),
               max_size=5))
    @settings(max_examples=100)
    def test_round_trip_is_lossless_across_a_random_parameter_sphere(self, int_values, fraction_values):
        # Disjoint key namespaces so int and Fraction values never collide.
        parameters = {f"i_{key}": value for key, value in int_values.items()}
        parameters.update({f"f_{key}": value for key, value in fraction_values.items()})
        assert deserialize_parameters(serialize_parameters(parameters)) == parameters


class TestWordDictionaries:

    @pytest.mark.parametrize("word,expected", [
        ("half", Fraction(1, 2)), ("one-third", Fraction(1, 3)), ("one-fifth", Fraction(1, 5)),
        ("a quarter", Fraction(1, 4)), ("two-thirds", Fraction(2, 3)), ("three quarters", Fraction(3, 4)),
    ])
    def test_fraction_words(self, word, expected):
        assert FRACTION_WORDS[word] == expected

    @pytest.mark.parametrize("word,expected", [
        ("once", 1), ("twice", 2), ("double", 2), ("two times", 2), ("thrice", 3), ("triple", 3),
        ("three times", 3), ("quadruple", 4), ("quintuple", 5), ("seven", 7), ("twelve", 12), ("twenty", 20),
    ])
    def test_verbal_multiplier_words(self, word, expected):
        assert VERBAL_MULTIPLIER_WORDS[word] == expected
