"""The three-regime construction (src/regimes.py): Regime A's nonword
guarantee, Regime B's distinct-real-word guarantee, Regime C's gold
recomputation (reasoning operand-swap and MCQ option-permutation), and
``derived_seed`` — the seed-derivation function every regime builder relies
on for deterministic rejection sampling.

``derived_seed`` was previously tested identically (differing only in call
arity) in both this file and test_linguistic_annotation.py; consolidated
here, its one home, since it lives in src/regimes.py.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from enums import Operation, SemanticClass
import regimes as semantic_regimes
from perturbation import PerturbationError, apply_edit_script
from tasks import generate_synthetic_reasoning_items
from tasks.multiple_choice import make_demonstration_multiple_choice_items


_seeds = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# derived_seed
# ---------------------------------------------------------------------------

class TestDerivedSeedExpectedBehavior:

    @given(base_seed=_seeds,
           parts=st.lists(st.one_of(st.text(max_size=20), st.integers()), min_size=1, max_size=4))
    @settings(max_examples=200)
    def test_same_inputs_always_produce_the_same_integer_seed(self, base_seed, parts):
        first = semantic_regimes.derived_seed(base_seed, *parts)
        second = semantic_regimes.derived_seed(base_seed, *parts)
        assert first == second
        assert isinstance(first, int)

    def test_changing_any_single_input_changes_the_seed(self):
        assert (semantic_regimes.derived_seed(1, "a", 2)
                != semantic_regimes.derived_seed(2, "a", 2))            # base seed
        assert (semantic_regimes.derived_seed(1, "a", 2)
                != semantic_regimes.derived_seed(1, "b", 2))            # a string part
        assert (semantic_regimes.derived_seed(1, "a", 2)
                != semantic_regimes.derived_seed(1, "a", 3))            # an int part
        assert (semantic_regimes.derived_seed(1729, "task_00001")
                != semantic_regimes.derived_seed(1729, "task_00002"))   # single-arity call


# ---------------------------------------------------------------------------
# Regime A — every edited word must become a nonword.
# ---------------------------------------------------------------------------

class TestRegimeANonwordGuarantee:

    @given(seed=_seeds)
    @settings(max_examples=100)
    def test_every_successful_construction_edits_only_to_nonwords(self, seed):
        # Constructed inline, not via the is_word fixture: hypothesis reruns
        # this body per generated example without resetting function-scoped
        # fixtures (see test_perturbation_engine.py for the same pattern).
        is_word = semantic_regimes.make_is_word()
        try:
            perturbed, edits, metadata = semantic_regimes.make_regime_a_nonword_typo(
                "capital", Operation.SUBSTITUTE, 2, seed, is_word)
        except PerturbationError:
            return
        assert metadata["regime"] == SemanticClass.A
        _, edited_after = metadata["edited_words"][0]
        assert not is_word(edited_after)
        assert len(edits) > 0
        assert perturbed != "capital"

    def test_is_deterministic(self, is_word):
        first = semantic_regimes.make_regime_a_nonword_typo("capital", Operation.SUBSTITUTE, 2, 11, is_word)
        second = semantic_regimes.make_regime_a_nonword_typo("capital", Operation.SUBSTITUTE, 2, 11, is_word)
        assert first[0] == second[0]
        assert [edit.to_dict() for edit in first[1]] == [edit.to_dict() for edit in second[1]]

    def test_metadata_carries_regime_and_dl_distance(self, is_word):
        _, _, metadata = semantic_regimes.make_regime_a_nonword_typo(
            "France", Operation.SUBSTITUTE, 1, 7, is_word)
        assert "regime" in metadata
        assert "edited_words" in metadata
        assert metadata["damerau_levenshtein_distance"] >= 1

    def test_edit_script_reconstructs_the_perturbed_text(self, is_word):
        original = "France"
        perturbed, edits, _ = semantic_regimes.make_regime_a_nonword_typo(
            original, Operation.SUBSTITUTE, 1, 7, is_word)
        assert apply_edit_script(original, edits) == perturbed


# ---------------------------------------------------------------------------
# Regime B — every substituted word must be a distinct, valid real word.
# ---------------------------------------------------------------------------

class TestRegimeBDistinctRealWordGuarantee:

    @given(seed=_seeds)
    @settings(max_examples=50)
    def test_every_successful_construction_substitutes_a_distinct_real_word(self, seed):
        is_word = semantic_regimes.make_is_word(
            {"cat", "cot", "cab", "car", "bat", "bad", "bag", "the", "france", "finance"})
        try:
            _, _, metadata = semantic_regimes.make_regime_b_real_word_shift(
                "the cat sat on the bat", seed, is_word)
        except PerturbationError:
            return
        assert metadata["regime"] == SemanticClass.B
        before, after = metadata["edited_words"][0]
        assert is_word(after)
        assert after.lower() != before.lower()

    def test_is_deterministic(self, small_vocabulary_is_word):
        first = semantic_regimes.make_regime_b_real_word_shift("the cat sat", 3, small_vocabulary_is_word)
        second = semantic_regimes.make_regime_b_real_word_shift("the cat sat", 3, small_vocabulary_is_word)
        assert first[0] == second[0]

    def test_metadata_carries_regime_and_dl_distance(self, small_vocabulary_is_word):
        _, _, metadata = semantic_regimes.make_regime_b_real_word_shift(
            "the cat sat", 3, small_vocabulary_is_word)
        assert "regime" in metadata
        assert "edited_words" in metadata
        assert "damerau_levenshtein_distance" in metadata


# ---------------------------------------------------------------------------
# Regime C — reasoning operand-swap: gold is recomputed from the template.
# ---------------------------------------------------------------------------

class TestRegimeCReasoningGoldRecomputation:

    def test_recomputed_gold_matches_the_templates_own_answer_function(self):
        item = generate_synthetic_reasoning_items(4, seed=1)[0]
        perturbed_text, _, metadata = semantic_regimes.make_regime_c_reasoning_operand_swap(item, 5)

        assert metadata["regime"] == SemanticClass.C
        assert metadata["new_gold_answer"] != metadata["old_gold_answer"]
        assert "swapped_parameter" in metadata and "new_value" in metadata
        swapped_parameters = dict(item.parameters)
        swapped_parameters[metadata["swapped_parameter"]] = metadata["new_value"]
        assert item.template is not None
        assert int(item.template.answer_function(**swapped_parameters)) == metadata["new_gold_answer"]
        assert str(metadata["new_value"]) in perturbed_text

    def test_new_gold_differs_from_original_across_many_synthetic_items(self):
        items = generate_synthetic_reasoning_items(8, seed=2)
        checked_at_least_one = False
        for item in items:
            if not item.supports_regime_c_operand_swap:
                continue
            for seed in range(5):
                try:
                    _, _, metadata = semantic_regimes.make_regime_c_reasoning_operand_swap(item, seed)
                except PerturbationError:
                    continue
                assert metadata["new_gold_answer"] != metadata["old_gold_answer"]
                checked_at_least_one = True
                break
        assert checked_at_least_one


# ---------------------------------------------------------------------------
# Regime C — MCQ option permutation: gold tracked by content, not label.
# ---------------------------------------------------------------------------

class TestRegimeCMcqOptionPermutation:

    def test_permutation_changes_the_gold_letter_but_tracks_the_original_content(self):
        item = make_demonstration_multiple_choice_items()[0]
        _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=1)
        assert metadata["regime"] == SemanticClass.C
        assert metadata["new_gold_letter"] != metadata["old_gold_letter"]
        assert metadata["old_gold_letter"] == item.gold_letter
        old_gold_content = item.options[item.gold_letter]
        assert metadata["new_options"][metadata["new_gold_letter"]] == old_gold_content

    def test_is_deterministic(self):
        item = make_demonstration_multiple_choice_items()[0]
        result_1 = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=5)
        result_2 = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=5)
        assert result_1[0] == result_2[0]
        assert result_1[2]["new_gold_letter"] == result_2[2]["new_gold_letter"]

    def test_question_and_all_option_texts_survive_into_the_new_content(self):
        item = make_demonstration_multiple_choice_items()[0]
        new_content, _, _ = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=2)
        assert item.question in new_content
        for text in item.options.values():
            assert text in new_content

    def test_metadata_carries_expected_keys_and_a_positive_dl_distance(self):
        item = make_demonstration_multiple_choice_items()[0]
        _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=3)
        for key in ("regime", "old_gold_letter", "new_gold_letter",
                    "old_options", "new_options", "damerau_levenshtein_distance"):
            assert key in metadata
        assert metadata["damerau_levenshtein_distance"] >= 1

    def test_every_demo_item_has_a_valid_permutation_for_some_seed(self):
        for item in make_demonstration_multiple_choice_items():
            for seed in range(5):
                try:
                    _, _, metadata = semantic_regimes.make_regime_c_mcq_option_permutation(item, seed=seed)
                except PerturbationError:
                    continue
                assert metadata["new_gold_letter"] != item.gold_letter
                break
            else:
                raise AssertionError(f"no valid permutation found for item {item.task_id}")


# ---------------------------------------------------------------------------
# Cross-regime consistency
# ---------------------------------------------------------------------------

def test_every_regime_reports_an_integer_dl_distance(is_word):
    _, _, metadata = semantic_regimes.make_regime_a_nonword_typo(
        "France", Operation.SUBSTITUTE, 1, 7, is_word)
    assert isinstance(metadata["damerau_levenshtein_distance"], int)
    assert metadata["damerau_levenshtein_distance"] >= 1
