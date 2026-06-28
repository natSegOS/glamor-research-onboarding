"""Adversarial and property-based tests for task modules.

Covers: gold-template correctness across many fuzzed items, key-term operand
coverage, scope span slicing, MCQ option-letter dict conversion, and
enum-coercion at the JSONL loader boundary.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tasks import reasoning as tasks_reasoning
from tasks import multiple_choice as mcq
from enums import TaskFamily


# ---------------------------------------------------------------------------
# Synthetic reasoning generator — correctness
# ---------------------------------------------------------------------------

def test_synthetic_reasoning_items_are_deterministic():
    first = tasks_reasoning.generate_synthetic_reasoning_items(10, seed=1)
    second = tasks_reasoning.generate_synthetic_reasoning_items(10, seed=1)
    assert [item.question_text for item in first] == [item.question_text for item in second]


def test_synthetic_reasoning_gold_matches_template():
    for item in tasks_reasoning.generate_synthetic_reasoning_items(40, seed=2):
        recomputed = int(item.template.answer_function(**item.parameters))
        assert recomputed == item.gold_answer
        assert item.gold_answer >= 0


def test_synthetic_reasoning_gold_matches_template_extended():
    """Property: gold matches template for ALL items across large generation."""
    for item in tasks_reasoning.generate_synthetic_reasoning_items(100, seed=7):
        recomputed = int(item.template.answer_function(**item.parameters))
        assert recomputed == item.gold_answer, (
            f"item {item.task_id}: template gives {recomputed}, gold is {item.gold_answer}")


def test_synthetic_reasoning_items_carry_key_terms():
    item = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=3)[0]
    assert item.key_terms
    for value in item.parameters.values():
        assert str(value) in item.key_terms, (
            f"operand value {value!r} not in key_terms {item.key_terms!r}")


def test_synthetic_reasoning_all_items_carry_key_terms():
    """Every generated item must have key_terms; every parameter appears."""
    for item in tasks_reasoning.generate_synthetic_reasoning_items(20, seed=5):
        assert item.key_terms, f"item {item.task_id} has empty key_terms"
        for val in item.parameters.values():
            assert str(val) in item.key_terms


def test_reasoning_scope_spans_cover_instruction_and_content():
    item = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=4)[0]
    spans = item.scope_spans
    instruction_start, instruction_end = spans["instruction"]
    assert item.full_prompt[instruction_start:instruction_end] == item.instruction
    content_start, content_end = spans["content"]
    assert item.full_prompt[content_start:content_end] == item.question_text


def test_scope_spans_are_non_overlapping():
    """Instruction and content spans must not overlap."""
    for item in tasks_reasoning.generate_synthetic_reasoning_items(10, seed=6):
        spans = item.scope_spans
        i_s, i_e = spans["instruction"]
        c_s, c_e = spans["content"]
        # The two spans must not overlap.
        assert i_e <= c_s or c_e <= i_s, (
            f"item {item.task_id}: instruction [{i_s},{i_e}) overlaps content [{c_s},{c_e})")


def test_scope_spans_slice_full_prompt():
    """Both spans must fall within the full prompt."""
    for item in tasks_reasoning.generate_synthetic_reasoning_items(10, seed=8):
        full = item.full_prompt
        for key, (s, e) in item.scope_spans.items():
            assert 0 <= s <= e <= len(full), (
                f"scope {key!r}: [{s},{e}) out of range for prompt of length {len(full)}")


def test_synthetic_items_support_regime_c_official_do_not():
    synthetic = tasks_reasoning.generate_synthetic_reasoning_items(1, seed=5)[0]
    assert synthetic.supports_regime_c_operand_swap


def test_synthetic_items_have_unique_task_ids():
    items = tasks_reasoning.generate_synthetic_reasoning_items(20, seed=9)
    ids = [item.task_id for item in items]
    assert len(ids) == len(set(ids)), "task_id values are not unique"


def test_synthetic_items_have_task_family_enum():
    items = tasks_reasoning.generate_synthetic_reasoning_items(5, seed=1)
    for item in items:
        assert item.task_family == TaskFamily.GSM_SYMBOLIC_SYNTHETIC


# ---------------------------------------------------------------------------
# MMLU-Pro option conversion + demo items
# ---------------------------------------------------------------------------

def test_options_sequence_to_letter_dict():
    result = mcq._options_sequence_to_letter_dict(["Oxygen", "Nitrogen", "Carbon"])
    assert result == {"A": "Oxygen", "B": "Nitrogen", "C": "Carbon"}


def test_options_sequence_to_letter_dict_ten_options():
    options = [f"Option{i}" for i in range(10)]
    result = mcq._options_sequence_to_letter_dict(options)
    assert set(result.keys()) == set("ABCDEFGHIJ")
    assert result["A"] == "Option0"
    assert result["J"] == "Option9"


def test_demo_mcq_items_well_formed():
    items = mcq.make_demonstration_multiple_choice_items()
    assert len(items) == 5
    for item in items:
        assert item.gold_letter in item.options
        assert item.option_count >= 2
        for letter in item.options:
            assert f"{letter}." in item.full_prompt


def test_mcq_scope_spans():
    item = mcq.make_demonstration_multiple_choice_items()[0]
    spans = item.scope_spans
    content_start, content_end = spans["content"]
    assert item.full_prompt[content_start:content_end] == item.content_text


def test_demo_mcq_gold_letter_in_options():
    """Gold letter must always be a key in item.options."""
    for item in mcq.make_demonstration_multiple_choice_items():
        assert item.gold_letter in item.options, (
            f"gold_letter {item.gold_letter!r} not in options {item.options}")


def test_demo_mcq_option_count_matches_options():
    for item in mcq.make_demonstration_multiple_choice_items():
        assert item.option_count == len(item.options), (
            f"option_count {item.option_count} != len(options) {len(item.options)}")


def test_demo_mcq_items_have_unique_task_ids():
    items = mcq.make_demonstration_multiple_choice_items()
    ids = [item.task_id for item in items]
    assert len(ids) == len(set(ids))


def test_demo_mcq_items_have_enum_task_family():
    for item in mcq.make_demonstration_multiple_choice_items():
        assert item.task_family == TaskFamily.MCQ_DEMO


# ---------------------------------------------------------------------------
# JSONL loader — enum coercion at boundary
# ---------------------------------------------------------------------------

def test_load_reasoning_jsonl_coerces_task_family():
    """Plain strings in JSONL must be coerced to TaskFamily enum members."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        record = {
            "task_id": "t1",
            "question_text": "How many apples?",
            "full_prompt": "Q: How many apples?\nA:",
            "instruction": "Q:",
            "gold_answer": 5,
            "key_terms": ["5"],
            "parameters": {"a": 5},
            "scope_spans": {"instruction": [0, 2], "content": [3, 20]},
            "task_family": "gsm_symbolic_synthetic",    # plain string
            "source": "gsm_symbolic_synthetic",
            "supports_regime_c_operand_swap": False,
        }
        f.write(json.dumps(record) + "\n")
        fname = f.name

    items = tasks_reasoning.load_reasoning_jsonl(Path(fname))
    assert len(items) == 1
    assert items[0].task_family == TaskFamily.GSM_SYMBOLIC_SYNTHETIC
    assert isinstance(items[0].task_family, TaskFamily)


def test_load_multiple_choice_jsonl_coerces_task_family():
    """Plain strings in MCQ JSONL must coerce to TaskFamily.
    The loader schema uses 'question' (not 'question_text') and 'answer' (not 'gold_letter').
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        record = {
            "question": "Which is the capital of France?",
            "options": {"A": "Paris", "B": "Berlin"},
            "answer": "A",
            "task_family": "mmlu_pro",   # plain string — must be coerced
            "key_terms": ["capital", "France"],
            "category": "Geography",
        }
        f.write(json.dumps(record) + "\n")
        fname = f.name

    items = mcq.load_multiple_choice_jsonl(Path(fname))
    assert len(items) == 1
    assert items[0].task_family == TaskFamily.MMLU_PRO
    assert isinstance(items[0].task_family, TaskFamily)
