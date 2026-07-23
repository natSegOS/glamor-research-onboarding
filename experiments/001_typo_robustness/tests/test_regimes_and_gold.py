"""Semantic-regime construction and gold-answer integrity.

Each test guards a CLASS of failures in the layer that turns clean items into
regime-labeled perturbations: mislabeled regimes or rows scored against the
wrong gold answer.
"""

from __future__ import annotations

import pytest

import regimes
import tokenization
from enums import FragmentationStratum, Operation, SelectionPolicy, SemanticClass
from perturbation import PerturbationError
from perturbation.engine import _cmu_homophone_neighbors
from tasks.reasoning import ReasoningItem, ReasoningTemplate
from tasks.multiple_choice import make_demonstration_multiple_choice_items


REALISTIC_SENTENCE = (
    "Miguel uses 2 pads of paper a week. If there are 30 sheets on a pad, "
    "how many sheets does he use every month?")


class TestDerivedSeeds:

    def test_derived_seed_is_deterministic_and_input_sensitive(self):
        """Rejection sampling relies on derived seeds being pure functions of
        their parts; collisions or drift would make 'same seed, same output'
        false across machines and re-runs."""
        assert regimes.derived_seed(1, "A", 0) == regimes.derived_seed(1, "A", 0)
        distinct_inputs = {
            regimes.derived_seed(1, "A", 0), regimes.derived_seed(1, "A", 1),
            regimes.derived_seed(1, "B", 0), regimes.derived_seed(2, "A", 0)}
        assert len(distinct_inputs) == 4


class TestRegimeAConstruction:

    def test_every_edited_word_is_a_nonword_and_construction_is_deterministic(
            self, is_word):
        """Regime A's definition: edited words are NONwords. An edited word
        that is a real word is a mislabeled Regime-B item contaminating the
        primary endpoint."""
        perturbed, edits, metadata = regimes.make_regime_a_nonword_typo(
            REALISTIC_SENTENCE, Operation.SUBSTITUTE, 2, 42, is_word)
        for _before, after in metadata["edited_words"]:
            assert not is_word(after)
        assert metadata["regime"] == SemanticClass.A
        perturbed_again, _, _ = regimes.make_regime_a_nonword_typo(
            REALISTIC_SENTENCE, Operation.SUBSTITUTE, 2, 42, is_word)
        assert perturbed_again == perturbed

    def test_filler_insertion_bypasses_nonword_test_with_frozen_particles_only(
            self, is_word):
        """Fillers ARE real words but are Regime A by construction; the
        builder must use only the frozen 4-particle set and count budget in
        particles."""
        _perturbed, _edits, metadata = regimes.make_regime_a_filler_insertion(
            REALISTIC_SENTENCE, 2, 7)
        assert len(metadata["inserted_fillers"]) == 2
        frozen_particles = {"uh", "um", "like", "so"}
        assert set(metadata["inserted_fillers"]) <= frozen_particles

    def test_whitespace_merge_that_lands_on_a_real_word_is_rejected(self):
        """'a part' → 'apart' is a REAL word: the merge must be rejection-
        sampled away, and with no other merge position the builder must fail
        loudly rather than emit a mislabeled Regime-A row."""
        vocabulary_where_merge_is_a_word = regimes.make_is_word(
            {"a", "part", "apart"})
        with pytest.raises(PerturbationError):
            regimes.make_regime_a_nonword_typo(
                "a part", Operation.DELETE, 1, 5,
                vocabulary_where_merge_is_a_word,
                selection_policy=SelectionPolicy.WHITESPACE)

    def test_whitespace_merge_produces_a_nonword_fusion_when_available(self, is_word):
        """The missed-space condition's contract: the fused token is a nonword
        recorded in the regime metadata."""
        _perturbed, _edits, metadata = regimes.make_regime_a_nonword_typo(
            REALISTIC_SENTENCE, Operation.DELETE, 1, 5, is_word,
            selection_policy=SelectionPolicy.WHITESPACE)
        for _before, after in metadata["edited_words"]:
            assert not is_word(after) and " " not in after


class TestRegimeBConstruction:

    def test_real_word_shift_yields_distinct_valid_words(self, is_word):
        """Regime B's definition: every substituted word is a DIFFERENT valid
        word: a nonword here is a mislabeled Regime-A item."""
        _perturbed, _edits, metadata = regimes.make_regime_b_real_word_shift(
            REALISTIC_SENTENCE, 11, is_word, edit_budget=1)
        for before, after in metadata["edited_words"]:
            assert is_word(after) and after.lower() != before.lower()
        assert metadata["phonetic_only"] is False

    def test_phonetic_only_shift_draws_exclusively_from_cmu_homophones(self, is_word):
        """The homophone condition's label depends on the pool being exactly
        the CMU same-pronunciation set; metadata must record the restriction."""
        pytest.importorskip("pronouncing")
        _perturbed, _edits, metadata = regimes.make_regime_b_real_word_shift(
            REALISTIC_SENTENCE, 11, is_word, edit_budget=1, phonetic_only=True)
        (before, after), = metadata["edited_words"]
        assert after.lower() in _cmu_homophone_neighbors(before, is_word)
        assert metadata["phonetic_only"] is True

    def test_no_homophone_bearing_word_fails_loudly(self, small_vocabulary_is_word):
        """An item with no homophone-bearing word must raise (→ exclusion
        sidecar), never silently widen to orthographic neighbors."""
        with pytest.raises(PerturbationError):
            regimes.make_regime_b_real_word_shift(
                "the cat sat", 3, small_vocabulary_is_word,
                edit_budget=1, phonetic_only=True)


def _reasoning_item_with_template(parameters: dict, answer_function,
                                  question_text: str,
                                  gold_answer: int) -> ReasoningItem:
    template = ReasoningTemplate(
        template_id="crafted", question_format=question_text,
        answer_function=answer_function, operand_ranges={})
    return ReasoningItem(
        task_id="crafted_00001", task_family="gsm_symbolic_synthetic",
        source="gsm_symbolic_synthetic", question_text=question_text,
        instruction="Solve.", gold_answer=gold_answer,
        template=template, parameters=parameters)


class TestRegimeCGoldIntegrity:

    def test_operand_swap_recomputes_gold_through_the_answer_function(self):
        """Regime C's whole point: the new gold is COMPUTED, not assumed. The
        swapped value must appear in the text and the metadata gold must equal
        answer_function(**new parameters)."""
        item = _reasoning_item_with_template(
            {"a": 7, "b": 3}, lambda a, b: a * b,
            "Ana buys 7 boxes of 3 pens. How many pens?", 21)
        perturbed, _edits, metadata = regimes.make_regime_c_reasoning_operand_swap(item, 5)
        new_parameters = dict(item.parameters)
        new_parameters[metadata["swapped_parameter"]] = metadata["new_value"]
        assert metadata["new_gold_answer"] == item.template.answer_function(**new_parameters)
        assert str(metadata["new_value"]) in perturbed
        assert metadata["new_gold_answer"] != item.gold_answer

    def test_regime_c_gold_of_zero_survives_the_request_builder(self, fake_tokenizer):
        """Falsy-zero regression: an operand swap CAN legitimately produce a
        new gold of 0. The request builder must carry 0 forward. The
        pre-2026-07-20 `or`-chain silently fell back to the OLD gold and
        scored the row against the wrong answer."""
        from pipeline.experiment import PerturbationCondition, _build_synthetic_requests

        # answer = a - 3: swapping a → 3 yields gold 0 (allowed by the
        # builder: != old gold, integer, >= 0). Force it by trying seeds.
        item = _reasoning_item_with_template(
            {"a": 5}, lambda a: a - 3, "Sam has 5 stickers minus three. Count?", 2)
        condition = PerturbationCondition(
            name="regime_c", semantic_class=SemanticClass.C, edit_budgets=[0])

        found_zero_gold_request = False
        for seed in range(200):
            requests, _exclusions = _build_synthetic_requests(
                item, condition, item.gold_answer, item.full_prompt,
                regimes.make_is_word(), fake_tokenizer, seed)
            for request in requests:
                assert request.gold_answer is not None
                if request.gold_answer == 0:
                    found_zero_gold_request = True
            if found_zero_gold_request:
                break
        assert found_zero_gold_request, (
            "no seed produced a zero-gold swap, test construction broke")

    def test_mcq_permutation_tracks_gold_by_content_not_letter(self):
        """The permuted gold letter must point at the ORIGINAL gold content
        and the option multiset must be unchanged: anything else double-
        breaks the over-robustness control."""
        item = make_demonstration_multiple_choice_items()[0]
        _content, _edits, metadata = regimes.make_regime_c_mcq_option_permutation(item, 9)
        assert metadata["new_gold_letter"] != metadata["old_gold_letter"]
        assert (metadata["new_options"][metadata["new_gold_letter"]]
                == item.options[item.gold_letter])
        assert sorted(metadata["new_options"].values()) == sorted(item.options.values())

    def test_mcq_permutation_rejects_single_option_items(self):
        """A one-option item cannot change its gold letter; silently returning
        it unpermuted would be a fake Regime-C row."""
        item = make_demonstration_multiple_choice_items()[0]
        item.options = {"A": item.options["A"]}
        item.gold_letter = "A"
        with pytest.raises(PerturbationError):
            regimes.make_regime_c_mcq_option_permutation(item, 9)

    def test_items_without_numeric_operands_fail_loudly(self):
        """No numeric parameter → no swap; a silent no-op would masquerade as
        a meaning-changing control that changed nothing."""
        item = _reasoning_item_with_template(
            {"name": "Ana"}, lambda name: 0, "Ana is here. Who is here?", 0)
        with pytest.raises(PerturbationError):
            regimes.make_regime_c_reasoning_operand_swap(item, 5)


class TestFragmentationCounterfactual:

    def test_matched_pair_strata_and_distance_are_exact(self, fake_tokenizer, is_word):
        """Method A's identification: Low ≤ 0 and High ≥ 1 subword change,
        both variants at EXACTLY the requested edit distance from the word.
        Any slack collapses the 'same word, same budget' counterfactual."""
        from perturbation import damerau_levenshtein_distance

        pair = tokenization.build_fragmentation_matched_pair(
            fake_tokenizer, "question", 1, 42, is_word)
        assert pair is not None
        assert pair.low_fragmentation_subword_change <= 0
        assert pair.high_fragmentation_subword_change >= 1
        for variant in (pair.low_fragmentation_variant, pair.high_fragmentation_variant):
            assert damerau_levenshtein_distance("question", variant) == 1
        pair_again = tokenization.build_fragmentation_matched_pair(
            fake_tokenizer, "question", 1, 42, is_word)
        assert (pair_again.low_fragmentation_variant, pair_again.high_fragmentation_variant) == (
            pair.low_fragmentation_variant, pair.high_fragmentation_variant)

    def test_candidate_words_are_longest_first_deduplicated_and_capped(self, is_word):
        """Pair-aware selection tries the richest variant spaces first; a
        wrong order or an uncapped list regresses either yield or build time."""
        candidates = tokenization.ordered_counterfactual_candidate_words(
            REALISTIC_SENTENCE, is_word)
        lengths = [len(word) for word in candidates]
        assert lengths == sorted(lengths, reverse=True)
        assert len(candidates) == len(set(candidates))
        assert len(candidates) <= tokenization.MAXIMUM_COUNTERFACTUAL_CANDIDATE_WORDS

    def test_variant_application_shares_the_extractor_word_notion(self, is_word):
        """The pilot crashed on 'Python' inside 'Python3': the candidate
        extractor ([A-Za-z]+) sees the word 'Python' there, but \\b-based
        application could not match it (digits are word characters to \\b)
        and raised mid-run. The contract is that anything the extractor can
        select, the applier can apply: letter-lookarounds on both sides."""
        text = "only Python3 here"
        candidates = tokenization.ordered_counterfactual_candidate_words(
            text, lambda word: True)
        assert "Python" in candidates
        perturbed, index = tokenization.apply_counterfactual_variant(
            text, "Python", "Pythom")
        assert perturbed == "only Pythom3 here"
        assert perturbed[index:index + len("Pythom")] == "Pythom"
        with pytest.raises(PerturbationError):
            tokenization.apply_counterfactual_variant("no such word", "Python", "Pythom")
