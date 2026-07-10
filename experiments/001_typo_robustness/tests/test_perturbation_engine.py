"""The perturbation engine's contract (src/perturbation/engine.py module
docstring): determinism, budget exactness, identity at k=0, protected spans,
edit-script reconstruction, and policy fidelity — plus the Damerau-Levenshtein
(OSA) distance metric these regimes are built on.

Each contract clause gets expected-behavior, edge-case, and adversarial-input
coverage. Hypothesis properties replace what used to be hand-rolled
``random.Random(seed)`` loops over a handful of iterations — the same
contract, checked over a far wider sphere of inputs, in far less code.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, example, settings, strategies as st
from rapidfuzz.distance import OSA

import regimes
from enums import Operation, SelectionPolicy, Scope, SemanticClass, Unit
from perturbation import (
    PerturbationError,
    apply_edit_script,
    damerau_levenshtein_distance,
    keyboard_neighbors_of,
    perturb,
)


SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog"
ALL_CHARACTER_OPERATIONS = (
    Operation.SUBSTITUTE, Operation.DELETE, Operation.INSERT, Operation.TRANSPOSE)

# Reusable hypothesis strategies for "a sphere of realistic perturbation inputs".
_alphabetic_words = st.lists(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8),
    min_size=1, max_size=8,
).map(" ".join).filter(lambda text: text.strip() != "")
_character_operations = st.sampled_from(ALL_CHARACTER_OPERATIONS)
_seeds = st.integers(min_value=0, max_value=2**31 - 1)


# ---------------------------------------------------------------------------
# Clause 1 — Determinism: same (text, seed) always yields byte-identical output.
# ---------------------------------------------------------------------------

class TestDeterminismExpectedBehavior:

    @given(text=_alphabetic_words, operation=_character_operations, seed=_seeds)
    @settings(max_examples=200)
    def test_same_inputs_yield_identical_output_and_edit_script(self, text, operation, seed):
        try:
            first_text, first_edits = perturb(
                text, operation, Unit.CHAR, Scope.ANYWHERE, 2,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
            second_text, second_edits = perturb(
                text, operation, Unit.CHAR, Scope.ANYWHERE, 2,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
        except PerturbationError:
            return  # an impossible budget for this text is fine either way
        assert first_text == second_text
        assert [edit.to_dict() for edit in first_edits] == [edit.to_dict() for edit in second_edits]

    def test_different_seeds_usually_produce_different_output(self):
        outputs = {
            perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)[0]
            for seed in range(8)
        }
        assert len(outputs) > 1


# ---------------------------------------------------------------------------
# Clause 2 — Budget exactness: exactly edit_budget edits, or PerturbationError.
# Clause 3 — Identity at k=0: an edit budget of zero returns the input unchanged.
# ---------------------------------------------------------------------------

class TestBudgetExpectedBehavior:

    @pytest.mark.parametrize("operation", list(ALL_CHARACTER_OPERATIONS))
    @pytest.mark.parametrize("edit_budget", [1, 2, 4])
    def test_exactly_edit_budget_edits_are_applied(self, operation, edit_budget):
        _, edits = perturb(
            SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, edit_budget,
            SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 7)
        assert len(edits) == edit_budget

    @pytest.mark.parametrize("operation", list(ALL_CHARACTER_OPERATIONS))
    def test_zero_budget_is_identity_for_every_operation(self, operation):
        perturbed, edits = perturb(
            SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, 0,
            SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed=42)
        assert perturbed == SAMPLE_TEXT
        assert edits == []


class TestBudgetEdgeCases:

    def test_budget_exceeding_eligible_positions_raises(self):
        with pytest.raises(PerturbationError):
            perturb("ab", Operation.DELETE, Unit.CHAR, Scope.ANYWHERE, 50,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)

    def test_budget_exhausted_by_a_single_character_raises(self):
        with pytest.raises(PerturbationError):
            perturb("a", Operation.DELETE, Unit.CHAR, Scope.ANYWHERE, 2,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)


class TestBudgetAdversarialInputs:

    def test_negative_edit_budget_raises_rather_than_misbehaving(self):
        with pytest.raises(PerturbationError):
            perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, -1,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)

    def test_empty_text_with_nonzero_budget_raises(self):
        with pytest.raises(PerturbationError):
            perturb("", Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 1,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)


# ---------------------------------------------------------------------------
# Clause 4 — Protected spans survive index-shifting under insertion/deletion.
# ---------------------------------------------------------------------------

class TestProtectedSpansExpectedBehavior:

    @pytest.mark.parametrize("text,protected_words", [
        ("The quick brown fox jumps over the lazy dog", ["brown"]),
        ("alpha beta gamma delta", ["alpha", "delta"]),
    ])
    def test_explicitly_protected_words_are_never_edited(self, text, protected_words):
        protected_spans = [
            (text.index(word), text.index(word) + len(word)) for word in protected_words]
        successful_runs = 0
        for seed in range(40):
            try:
                perturbed, _ = perturb(
                    text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed,
                    protected_spans=protected_spans)
            except PerturbationError:
                continue  # impossible if protecting every eligible word
            successful_runs += 1
            for word in protected_words:
                assert word in perturbed
        assert successful_runs > 0

    @pytest.mark.parametrize("text,numbers", [
        ("Add 12 and 7 to get the sum", ["12", "7"]),
        ("If 100 cats each eat 25 fish", ["100", "25"]),
    ])
    def test_numeric_tokens_are_protected_by_default(self, text, numbers):
        successful_runs = 0
        for seed in range(40):
            try:
                perturbed, _ = perturb(
                    text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
            except PerturbationError:
                continue
            successful_runs += 1
            for number in numbers:
                assert number in perturbed
        assert successful_runs > 30


# ---------------------------------------------------------------------------
# Clause 5 — Reconstructibility: apply_edit_script(original, script) == perturbed.
# ---------------------------------------------------------------------------

class TestReconstructionExpectedBehavior:

    @given(text=_alphabetic_words, operation=_character_operations, seed=_seeds)
    @settings(max_examples=200)
    def test_edit_script_reconstructs_the_perturbed_text(self, text, operation, seed):
        try:
            perturbed, edits = perturb(
                text, operation, Unit.CHAR, Scope.ANYWHERE, 2,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
        except PerturbationError:
            return
        assert apply_edit_script(text, edits) == perturbed
        # A JSON-round-tripped script (dicts, not Edit objects) must replay identically.
        assert apply_edit_script(text, [edit.to_dict() for edit in edits]) == perturbed


class TestReconstructionEdgeCases:

    def test_empty_edit_script_is_identity(self):
        assert apply_edit_script(SAMPLE_TEXT, []) == SAMPLE_TEXT


# ---------------------------------------------------------------------------
# Clause 6 — Policy fidelity: each selection policy only ever produces the
# kind of candidate it promises.
# ---------------------------------------------------------------------------

class TestPolicyFidelityExpectedBehavior:

    @given(seed=_seeds)
    @settings(max_examples=100)
    def test_keyboard_neighbor_substitutions_are_always_qwerty_adjacent(self, seed):
        try:
            _, edits = perturb(
                SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 1,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
        except PerturbationError:
            return
        for edit in edits:
            assert edit.after in keyboard_neighbors_of(edit.before)

    def test_keyboard_neighbors_cover_every_qwerty_letter(self):
        for letter in string.ascii_lowercase:
            neighbors = keyboard_neighbors_of(letter)
            assert len(neighbors) >= 1, f"'{letter}' has no keyboard neighbors"
            assert letter not in neighbors, f"'{letter}' is its own neighbor"

    def test_informative_word_policy_edits_only_the_supplied_key_terms(self):
        text = "Buy 5 boxes with cats inside"
        _, edits = perturb(
            text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANSWER_CRITICAL, 1,
            SelectionPolicy.INFORMATIVE_WORD, SemanticClass.A, 9, key_terms=["cats"])
        assert edits[0].word_before == "cats"

    @given(seed=_seeds)
    @settings(max_examples=50)
    def test_real_word_policy_always_produces_a_distinct_valid_word(self, seed):
        # Constructed inline (not via a pytest fixture): hypothesis reruns
        # this test body per generated example without resetting
        # function-scoped fixtures, and this predicate is stateless/pure
        # enough that constructing it fresh here is both correct and cheap.
        is_word = regimes.make_is_word(
            {"cat", "cot", "cab", "car", "bat", "bad", "bag", "the", "france", "finance"})
        try:
            _, edits = perturb(
                "the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
                SelectionPolicy.REAL_WORD, SemanticClass.B, seed, is_word=is_word)
        except PerturbationError:
            return
        edit = edits[0]
        assert is_word(edit.word_after)
        assert edit.word_after != edit.word_before


class TestPolicyFidelityAdversarialInputs:

    def test_real_word_policy_without_is_word_raises(self):
        with pytest.raises(PerturbationError):
            perturb("the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
                    SelectionPolicy.REAL_WORD, SemanticClass.B, 1)

    def test_exception_from_a_caller_supplied_is_word_propagates(self):
        # A buggy or malicious is_word predicate must surface its own error,
        # never be silently absorbed by the engine — the same "easy to catch"
        # standard the rest of the codebase was audited against.
        def broken_is_word(_token):
            raise RuntimeError("is_word blew up")

        with pytest.raises(RuntimeError, match="is_word blew up"):
            perturb("the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
                    SelectionPolicy.REAL_WORD, SemanticClass.B, 1, is_word=broken_is_word)


# ---------------------------------------------------------------------------
# Damerau-Levenshtein (OSA) distance — the metric regimes and measured_dl
# are built on. damerau_levenshtein_distance now delegates to rapidfuzz for
# speed; every property and equivalence check below applies equally to
# either implementation, which is the point.
# ---------------------------------------------------------------------------

class TestDistanceExpectedBehavior:

    def test_basic_single_edit_cases(self):
        assert damerau_levenshtein_distance("cat", "cat") == 0
        assert damerau_levenshtein_distance("cat", "cot") == 1       # substitution
        assert damerau_levenshtein_distance("cat", "ct") == 1        # deletion
        assert damerau_levenshtein_distance("cat", "cast") == 1      # insertion
        assert damerau_levenshtein_distance("ab", "ba") == 1         # adjacent transposition
        assert damerau_levenshtein_distance("ac", "ca") == 1         # not double-counted as 2
        assert damerau_levenshtein_distance("abc", "xyz") == 3

    @given(first_string=st.text(max_size=25), second_string=st.text(max_size=25),
           third_string=st.text(max_size=25))
    @settings(max_examples=300)
    def test_metric_properties_hold_across_a_random_string_sphere(
            self, first_string, second_string, third_string):
        # Non-negativity and identity.
        assert damerau_levenshtein_distance(first_string, first_string) == 0
        distance_ab = damerau_levenshtein_distance(first_string, second_string)
        assert distance_ab >= 0
        # Symmetry.
        assert distance_ab == damerau_levenshtein_distance(second_string, first_string)
        # Bounded by the longer string's length.
        assert distance_ab <= max(len(first_string), len(second_string))
        # Triangle inequality.
        distance_ac = damerau_levenshtein_distance(first_string, third_string)
        distance_cb = damerau_levenshtein_distance(third_string, second_string)
        assert distance_ab <= distance_ac + distance_cb


class TestDistanceEdgeCases:

    def test_empty_strings(self):
        assert damerau_levenshtein_distance("", "") == 0
        assert damerau_levenshtein_distance("", "abc") == 3
        assert damerau_levenshtein_distance("abc", "") == 3


def _reference_osa_distance(first_string: str, second_string: str) -> int:
    """Pure-Python Optimal String Alignment distance — the metric
    ``damerau_levenshtein_distance`` has always actually computed (not true,
    unrestricted Damerau-Levenshtein; see the test below). Kept as a
    dependency-free oracle so a future change to the production
    implementation (currently ``rapidfuzz``) is provably still computing the
    same metric, not silently drifting to a different one.
    """
    first_length, second_length = len(first_string), len(second_string)
    two_rows_back = [0] * (second_length + 1)
    previous_row = list(range(second_length + 1))
    current_row = [0] * (second_length + 1)

    for i in range(1, first_length + 1):
        current_row[0] = i
        first_character = first_string[i - 1]
        for j in range(1, second_length + 1):
            second_character = second_string[j - 1]
            substitution_cost = 0 if first_character == second_character else 1
            best_distance = min(
                previous_row[j] + 1,
                current_row[j - 1] + 1,
                previous_row[j - 1] + substitution_cost,
            )
            is_adjacent_transposition = (
                i > 1 and j > 1
                and first_character == second_string[j - 2]
                and first_string[i - 2] == second_character
            )
            if is_adjacent_transposition:
                best_distance = min(best_distance, two_rows_back[j - 2] + 1)
            current_row[j] = best_distance
        two_rows_back, previous_row, current_row = previous_row, current_row, two_rows_back

    return previous_row[second_length]


class TestDistanceAdversarialInputs:

    @given(first_string=st.text(max_size=30), second_string=st.text(max_size=30))
    @example(first_string="", second_string="")
    @example(first_string="ac", second_string="ca")            # adjacent transposition
    @example(first_string="CA", second_string="ABC")           # OSA != true DL here
    @example(first_string="café", second_string="cafe")        # unicode
    @example(first_string="a" * 30, second_string="a" * 30)    # identical, long
    @example(first_string="a" * 30, second_string="b" * 30)    # maximally different
    @settings(max_examples=500)
    def test_matches_reference_oracle_across_random_and_pinned_inputs(
            self, first_string, second_string):
        assert (OSA.distance(first_string, second_string)
                == _reference_osa_distance(first_string, second_string)
                == damerau_levenshtein_distance(first_string, second_string))


# ---------------------------------------------------------------------------
# Regression guard: the Regime-B neighbor-generation functions that returned
# unbounded, multi-gigabyte caches and SIGKILL-ed a real pilot run partway
# through (src/perturbation/engine.py) must never be cached again.
# ---------------------------------------------------------------------------

def test_large_neighbor_generators_are_never_cached():
    from perturbation import engine as perturbation_engine

    assert not hasattr(perturbation_engine._damerau_levenshtein_one_neighbors, "cache_info")
    assert not hasattr(perturbation_engine._damerau_levenshtein_band_neighbors, "cache_info")
