"""Adversarial and metamorphic tests for the three-regime construction.

Covers: Regime A nonword guarantee across many seeds, Regime B distinct-real-word
guarantee, Regime C gold recomputation and operand visibility, MCQ option
permutation logic, derived-seed determinism, and all documented error paths.
"""

from __future__ import annotations

from enums import Operation, SemanticClass
import regimes as semantic_regimes
from perturbation import PerturbationError
from tasks import generate_synthetic_reasoning_items
from tasks.multiple_choice import make_demonstration_multiple_choice_items


# ---------------------------------------------------------------------------
# Derived seed
# ---------------------------------------------------------------------------

def test_derived_seed_is_deterministic_and_varies():
    assert semantic_regimes.derived_seed(1, "a", 2) == semantic_regimes.derived_seed(1, "a", 2)
    assert semantic_regimes.derived_seed(1, "a", 2) != semantic_regimes.derived_seed(1, "a", 3)


def test_derived_seed_varies_with_base_seed():
    assert semantic_regimes.derived_seed(1, "a", 0) != semantic_regimes.derived_seed(2, "a", 0)


def test_derived_seed_varies_with_label():
    assert semantic_regimes.derived_seed(1, "A", 0) != semantic_regimes.derived_seed(1, "B", 0)


def test_derived_seed_returns_int():
    seed_value = semantic_regimes.derived_seed(42, "X", 7)
    assert isinstance(seed_value, int)


# ---------------------------------------------------------------------------
# Regime A — nonword guarantee
# ---------------------------------------------------------------------------

def test_regime_a_produces_nonword(is_word):
    perturbed, edits, metadata = semantic_regimes.make_regime_a_nonword_typo(
        "France", Operation.SUBSTITUTE, 1, 7, is_word)
    assert metadata["regime"] == SemanticClass.A
    _, edited_after = metadata["edited_words"][0]
    assert not is_word(edited_after)
    assert len(edits) > 0
    assert perturbed != "France"


def test_regime_a_is_deterministic(is_word):
    first = semantic_regimes.make_regime_a_nonword_typo("capital", Operation.SUBSTITUTE, 2, 11, is_word)
    second = semantic_regimes.make_regime_a_nonword_typo("capital", Operation.SUBSTITUTE, 2, 11, is_word)
    assert first[0] == second[0]
    assert [e.to_dict() for e in first[1]] == [e.to_dict() for e in second[1]]


def test_regime_a_nonword_holds_across_many_seeds(is_word):
    """Every seed that succeeds must produce a nonword."""
    successes = 0
    for seed in range(40):
        try:
            _, _, metadata = semantic_regimes.make_regime_a_nonword_typo(
                "capital", Operation.SUBSTITUTE, 2, seed, is_word)
            _, edited_after = metadata["edited_words"][0]
            assert not is_word(edited_after), (
                f"seed={seed}: edited_after={edited_after!r} is a real word")
            successes += 1
        except PerturbationError:
            pass
    assert successes >= 10, "too few successful seeds for Regime A"


def test_regime_a_metadata_keys(is_word):
    _, _, metadata = semantic_regimes.make_regime_a_nonword_typo(
        "France", Operation.SUBSTITUTE, 1, 7, is_word)
    assert "regime" in metadata
    assert "edited_words" in metadata
    assert "damerau_levenshtein_distance" in metadata
    assert metadata["damerau_levenshtein_distance"] >= 1


def test_regime_a_edit_script_reconstructs(is_word):
    from perturbation import apply_edit_script
    original = "France"
    perturbed, edits, _ = semantic_regimes.make_regime_a_nonword_typo(
        original, Operation.SUBSTITUTE, 1, 7, is_word)
    assert apply_edit_script(original, edits) == perturbed


# ---------------------------------------------------------------------------
# Regime B — distinct real-word guarantee
# ---------------------------------------------------------------------------

def test_regime_b_produces_distinct_real_word(small_vocabulary_is_word):
    _, _, metadata = semantic_regimes.make_regime_b_real_word_shift(
        "the cat sat", 3, small_vocabulary_is_word)
    assert metadata["regime"] == SemanticClass.B
    before, after = metadata["edited_words"][0]
    assert small_vocabulary_is_word(after)
    assert after.lower() != before.lower()


def test_regime_b_is_deterministic(small_vocabulary_is_word):
    first = semantic_regimes.make_regime_b_real_word_shift("the cat sat", 3, small_vocabulary_is_word)
    second = semantic_regimes.make_regime_b_real_word_shift("the cat sat", 3, small_vocabulary_is_word)
    assert first[0] == second[0]


def test_regime_b_metadata_keys(small_vocabulary_is_word):
    _, _, metadata = semantic_regimes.make_regime_b_real_word_shift(
        "the cat sat", 3, small_vocabulary_is_word)
    assert "regime" in metadata
    assert "edited_words" in metadata
    assert "damerau_levenshtein_distance" in metadata


def test_regime_b_real_word_holds_across_seeds(small_vocabulary_is_word):
    successes = 0
    for seed in range(20):
        try:
            _, _, metadata = semantic_regimes.make_regime_b_real_word_shift(
                "the cat sat on the bat", seed, small_vocabulary_is_word)
            before, after = metadata["edited_words"][0]
            assert small_vocabulary_is_word(after), (
                f"seed={seed}: after={after!r} is not a real word")
            assert after.lower() != before.lower(), (
                f"seed={seed}: before and after are the same word")
            successes += 1
        except PerturbationError:
            pass
    assert successes >= 5


# ---------------------------------------------------------------------------
# Regime C reasoning — gold recomputation
# ---------------------------------------------------------------------------

def test_regime_c_reasoning_recomputes_gold():
    items = generate_synthetic_reasoning_items(4, seed=1)
    item = items[0]
    perturbed_text, _, metadata = semantic_regimes.make_regime_c_reasoning_operand_swap(item, 5)

    assert metadata["regime"] == SemanticClass.C
    assert metadata["new_gold_answer"] != metadata["old_gold_answer"]
    swapped_parameters = dict(item.parameters)
    swapped_parameters[metadata["swapped_parameter"]] = metadata["new_value"]
    assert item.template is not None
    assert int(item.template.answer_function(**swapped_parameters)) == metadata["new_gold_answer"]
    assert str(metadata["new_value"]) in perturbed_text


def test_regime_c_gold_differs_from_original_across_items():
    """For every synthetic item, the swapped gold must differ from the original."""
    items = generate_synthetic_reasoning_items(8, seed=2)
    for i, item in enumerate(items):
        if not item.supports_regime_c_operand_swap:
            continue
        for seed in range(5):
            try:
                _, _, metadata = semantic_regimes.make_regime_c_reasoning_operand_swap(
                    item, seed)
                assert metadata["new_gold_answer"] != metadata["old_gold_answer"], (
                    f"item {i} seed {seed}: new gold equals old gold")
                break
            except PerturbationError:
                pass


def test_regime_c_operand_appears_in_text():
    """The swapped value must be present as a string in the perturbed text."""
    items = generate_synthetic_reasoning_items(4, seed=3)
    for item in items:
        if not item.supports_regime_c_operand_swap:
            continue
        perturbed_text, _, metadata = semantic_regimes.make_regime_c_reasoning_operand_swap(item, 1)
        assert str(metadata["new_value"]) in perturbed_text
        break


def test_regime_c_metadata_keys():
    items = generate_synthetic_reasoning_items(4, seed=1)
    _, _, metadata = semantic_regimes.make_regime_c_reasoning_operand_swap(items[0], 5)
    assert "regime" in metadata
    assert "swapped_parameter" in metadata
    assert "new_value" in metadata
    assert "old_gold_answer" in metadata
    assert "new_gold_answer" in metadata


# ---------------------------------------------------------------------------
# Regime C MCQ — option permutation
# ---------------------------------------------------------------------------

def test_regime_c_mcq_option_permutation_changes_gold():
    items = make_demonstration_multiple_choice_items()
    item = items[0]
    _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=1)
    assert metadata["regime"] == SemanticClass.C
    assert metadata["new_gold_letter"] != metadata["old_gold_letter"]
    assert metadata["old_gold_letter"] == item.gold_letter


def test_regime_c_mcq_option_permutation_gold_tracked_by_content():
    """The new gold letter must point to the original gold option's text."""
    items = make_demonstration_multiple_choice_items()
    item = items[0]
    _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=1)
    old_gold_content = item.options[item.gold_letter]
    assert metadata["new_options"][metadata["new_gold_letter"]] == old_gold_content


def test_regime_c_mcq_option_permutation_is_deterministic():
    items = make_demonstration_multiple_choice_items()
    item = items[0]
    result_1 = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=5)
    result_2 = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=5)
    assert result_1[0] == result_2[0]
    assert result_1[2]["new_gold_letter"] == result_2[2]["new_gold_letter"]


def test_regime_c_mcq_option_permutation_content_in_output():
    """The question text is unchanged; all option texts appear in the new content."""
    items = make_demonstration_multiple_choice_items()
    item = items[0]
    new_content, _, _ = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=2)
    assert item.question in new_content
    for text in item.options.values():
        assert text in new_content


def test_regime_c_mcq_option_permutation_metadata_keys():
    items = make_demonstration_multiple_choice_items()
    item = items[0]
    _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=3)
    for key in ("regime", "old_gold_letter", "new_gold_letter",
                "old_options", "new_options", "damerau_levenshtein_distance"):
        assert key in metadata


def test_regime_c_mcq_option_permutation_dl_distance():
    items = make_demonstration_multiple_choice_items()
    item = items[0]
    _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=4)
    assert metadata["damerau_levenshtein_distance"] >= 1


def test_regime_c_mcq_option_permutation_all_demo_items():
    """Every demo item must produce a valid permutation for some seed."""
    items = make_demonstration_multiple_choice_items()
    for item in items:
        for seed in range(5):
            try:
                _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=seed)
                assert metadata["new_gold_letter"] != item.gold_letter
                break
            except PerturbationError:
                pass
        else:
            raise AssertionError(f"no valid permutation found for item {item.task_id}")


# ---------------------------------------------------------------------------
# Cross-regime metadata consistency
# ---------------------------------------------------------------------------

def test_all_regime_metadata_have_dl_distance(is_word):
    # Regime A
    _, _, meta_a = semantic_regimes.make_regime_a_nonword_typo(
        "France", Operation.SUBSTITUTE, 1, 7, is_word)
    assert isinstance(meta_a["damerau_levenshtein_distance"], int)
    assert meta_a["damerau_levenshtein_distance"] >= 1
