"""Perturbation-engine contract tests: every clause of the engine's contract,
for every selection policy, under correct, degenerate, and adversarial inputs.

Each test guards a CLASS of failures. Breaking any of them invalidates the
perturbation provenance story of the paper: edits that don't replay, budgets
that drift, policies that draw outside their advertised pools, or protected
numeric spans that get corrupted (which would silently turn intent-preserving
Regime-A items into meaning-changing ones).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies

from enums import Operation, SelectionPolicy, Scope, SemanticClass, Unit
from perturbation import (
    PerturbationError,
    apply_edit_script,
    damerau_levenshtein_distance,
    keyboard_neighbors_of,
    perturb,
)
from perturbation.engine import (
    _DISCOURSE_PARTICLE_VALUES,
    _cmu_homophone_neighbors,
    _damerau_levenshtein_band_neighbors,
    _damerau_levenshtein_one_neighbors,
)


import regimes

# Module-level predicate for hypothesis-driven tests: @given forbids
# function-scoped fixtures (each example must not share fixture state).
DEMO_IS_WORD = regimes.make_is_word()

REALISTIC_SENTENCE = (
    "Miguel uses 2 pads of paper a week. If there are 30 sheets on a pad, "
    "how many sheets does he use every month?")

# Every legal (policy, operation) combination the pipeline can request,
# with the structural inputs each policy needs.
_POLICY_OPERATION_COMBINATIONS = [
    (SelectionPolicy.KEYBOARD_NEIGHBOR, Operation.SUBSTITUTE),
    (SelectionPolicy.KEYBOARD_NEIGHBOR, Operation.DELETE),
    (SelectionPolicy.KEYBOARD_NEIGHBOR, Operation.INSERT),
    (SelectionPolicy.KEYBOARD_NEIGHBOR, Operation.TRANSPOSE),
    (SelectionPolicy.INFORMATIVE_WORD, Operation.SUBSTITUTE),
    (SelectionPolicy.REAL_WORD, Operation.SUBSTITUTE),
    (SelectionPolicy.HOMOPHONE, Operation.SUBSTITUTE),
    (SelectionPolicy.WHITESPACE, Operation.INSERT),   # split
    (SelectionPolicy.WHITESPACE, Operation.DELETE),   # merge (missed-space)
    (SelectionPolicy.FILLER_WORD, Operation.INSERT),
]


def _perturb_realistic(policy, operation, seed, is_word, edit_budget=1):
    return perturb(
        REALISTIC_SENTENCE, operation, Unit.CHAR, Scope.ANYWHERE, edit_budget,
        policy, SemanticClass.A, seed,
        key_terms=["Miguel", "paper", "sheets"], is_word=is_word,
        max_word_distance=2)


class TestEngineContractAcrossAllPolicies:

    @pytest.mark.parametrize("policy,operation", _POLICY_OPERATION_COMBINATIONS,
                             ids=lambda value: str(value))
    @settings(max_examples=15, deadline=None)
    @given(seed=strategies.integers(min_value=0, max_value=10_000))
    def test_determinism_budget_and_reconstruction_hold_together(
            self, policy, operation, seed):
        """Same inputs → byte-identical output; exactly k edits; the edit
        script replays to the perturbed text. Breaking this breaks the
        released provenance: a reviewer could no longer reproduce any
        perturbed prompt from the edit scripts."""
        is_word = DEMO_IS_WORD
        try:
            perturbed_first, edits_first = _perturb_realistic(
                policy, operation, seed, is_word)
        except PerturbationError:
            return  # legitimately unsatisfiable for this seed (e.g. empty pool)

        perturbed_second, edits_second = _perturb_realistic(
            policy, operation, seed, is_word)

        assert perturbed_first == perturbed_second
        assert [edit.to_dict() for edit in edits_first] == [
            edit.to_dict() for edit in edits_second]
        assert len(edits_first) == 1
        assert perturbed_first != REALISTIC_SENTENCE
        assert apply_edit_script(REALISTIC_SENTENCE, edits_first) == perturbed_first
        # JSON round-trip of the script must replay identically too.
        assert apply_edit_script(
            REALISTIC_SENTENCE,
            [edit.to_dict() for edit in edits_first]) == perturbed_first

    @pytest.mark.parametrize("operation", [
        Operation.SUBSTITUTE, Operation.DELETE, Operation.INSERT, Operation.TRANSPOSE])
    def test_zero_budget_is_identity(self, operation, is_word):
        """k=0 must return the input untouched — a nonzero-effect 'clean'
        arm would corrupt every matched pair in the study."""
        perturbed, edits = perturb(
            REALISTIC_SENTENCE, operation, Unit.CHAR, Scope.ANYWHERE, 0,
            SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 7, is_word=is_word)
        assert perturbed == REALISTIC_SENTENCE and edits == []

    @settings(max_examples=25, deadline=None)
    @given(seed=strategies.integers(min_value=0, max_value=10_000),
           edit_budget=strategies.integers(min_value=1, max_value=3))
    def test_protected_spans_survive_shifting_indices(self, seed, edit_budget):
        """Characters inside a protected span are never edited even as
        insertions/deletions shift indices around them. Breaking this lets a
        'typo' touch the operands the design promises are frozen."""
        is_word = DEMO_IS_WORD
        text = "The answer to the question about France is exactly 42 points."
        protected_word = "France"
        protected_start = text.index(protected_word)
        protected_spans = [(protected_start, protected_start + len(protected_word))]

        for operation in (Operation.SUBSTITUTE, Operation.DELETE, Operation.INSERT):
            try:
                perturbed, _edits = perturb(
                    text, operation, Unit.CHAR, Scope.ANYWHERE, edit_budget,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed,
                    protected_spans=protected_spans, is_word=is_word)
            except PerturbationError:
                continue
            assert protected_word in perturbed
            # Numeric tokens are protected by default (design/04 §4.7).
            assert "42" in perturbed

    @settings(max_examples=25, deadline=None)
    @given(seed=strategies.integers(min_value=0, max_value=10_000))
    def test_keyboard_policy_draws_only_qwerty_neighbors_case_preserved(self, seed):
        """Every substitution must come from the QWERTY adjacency graph with
        case preserved — the citable MulTypo replacement operation. Breaking
        this severs the keyboard-plausibility claim of Regime A."""
        is_word = DEMO_IS_WORD
        _perturbed, edits = _perturb_realistic(
            SelectionPolicy.KEYBOARD_NEIGHBOR, Operation.SUBSTITUTE, seed, is_word)
        for edit in edits:
            assert edit.after in keyboard_neighbors_of(edit.before)
            assert edit.after.isupper() == edit.before.isupper()


class TestPolicyPoolFidelity:

    def test_real_word_policy_produces_distinct_valid_words(
            self, small_vocabulary_is_word):
        """The real_word pool must contain only in-vocabulary words distinct
        from the source — otherwise Regime B items are mislabeled."""
        perturbed, edits = perturb(
            "the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
            SelectionPolicy.REAL_WORD, SemanticClass.B, 3,
            is_word=small_vocabulary_is_word, max_word_distance=1)
        (edit,) = edits
        assert small_vocabulary_is_word(edit.word_after)
        assert edit.word_after.lower() != edit.word_before.lower()
        assert edit.word_after in perturbed

    def test_homophone_policy_draws_only_cmu_exact_homophones(self, is_word):
        """The HOMOPHONE pool must be exactly the CMU same-pronunciation set —
        an orthographic neighbor sneaking in would mislabel the pure
        acoustic-confusion condition."""
        pytest.importorskip("pronouncing")
        text = "If there are thirty sheets left over"
        perturbed, edits = perturb(
            text, Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
            SelectionPolicy.HOMOPHONE, SemanticClass.B, 11, is_word=is_word)
        (edit,) = edits
        assert edit.word_after.lower() in _cmu_homophone_neighbors(
            edit.word_before, is_word)
        assert edit.word_after in perturbed

    def test_homophone_policy_fails_loudly_when_no_word_has_a_homophone(
            self, small_vocabulary_is_word):
        """With no homophone-bearing word the policy must raise — silently
        substituting an orthographic neighbor would corrupt the condition."""
        with pytest.raises(PerturbationError):
            perturb("the cat sat", Operation.SUBSTITUTE, Unit.WORD,
                    Scope.ANYWHERE, 1, SelectionPolicy.HOMOPHONE,
                    SemanticClass.B, 5, is_word=small_vocabulary_is_word)

    def test_filler_policy_inserts_only_frozen_particles_as_standalone_tokens(
            self, is_word):
        """Fillers must come from the frozen 4-particle set and land between
        words — any other token or position voids the 'definitionally
        intent-preserving' claim."""
        perturbed, edits = _perturb_realistic(
            SelectionPolicy.FILLER_WORD, Operation.INSERT, 13, is_word,
            edit_budget=2)
        assert len(edits) == 2
        for edit in edits:
            assert edit.word_after in _DISCOURSE_PARTICLE_VALUES
            assert edit.after == edit.word_after + " "
            assert f" {edit.word_after} " in perturbed

    def test_whitespace_merge_records_the_merged_token(self, is_word):
        """The missed-space merge must record the fused token as word_after —
        that field is what lets the Regime-A builder reject merges that land
        on real words ('a part' → 'apart')."""
        perturbed, edits = _perturb_realistic(
            SelectionPolicy.WHITESPACE, Operation.DELETE, 3, is_word)
        (edit,) = edits
        left_word, right_word = edit.word_before.split(" ")
        assert edit.word_after == left_word + right_word
        assert edit.word_after in perturbed


class TestDistanceOracle:

    @staticmethod
    def _reference_osa_distance(first: str, second: str) -> int:
        """Dependency-free OSA dynamic program — the oracle the C-optimized
        rapidfuzz delegate is held to."""
        rows, columns = len(first) + 1, len(second) + 1
        table = [[0] * columns for _ in range(rows)]
        for row in range(rows):
            table[row][0] = row
        for column in range(columns):
            table[0][column] = column
        for row in range(1, rows):
            for column in range(1, columns):
                substitution_cost = 0 if first[row - 1] == second[column - 1] else 1
                table[row][column] = min(
                    table[row - 1][column] + 1,
                    table[row][column - 1] + 1,
                    table[row - 1][column - 1] + substitution_cost)
                can_transpose = (
                    row > 1 and column > 1
                    and first[row - 1] == second[column - 2]
                    and first[row - 2] == second[column - 1])
                if can_transpose:
                    table[row][column] = min(
                        table[row][column], table[row - 2][column - 2] + 1)
        return table[-1][-1]

    @pytest.mark.parametrize("first,second,expected", [
        ("CA", "ABC", 3),          # OSA ≠ true DL (true DL would give 2)
        ("Frnace", "France", 1),   # adjacent transposition counts as ONE edit
        ("", "abc", 3),
        ("same", "same", 0),
    ])
    def test_pinned_distances_including_the_osa_vs_true_dl_separator(
            self, first, second, expected):
        """The codebase computes OSA (restricted DL). The 'CA'→'ABC' case is
        the canonical separator: silently switching to true DL would change
        every measured_dl in the released data."""
        assert damerau_levenshtein_distance(first, second) == expected

    @settings(max_examples=200, deadline=None)
    @given(first=strategies.text(alphabet="abcde", max_size=8),
           second=strategies.text(alphabet="abcde", max_size=8))
    def test_distance_matches_pure_python_oracle_on_random_sphere(
            self, first, second):
        """rapidfuzz's OSA must agree with the dependency-free DP everywhere —
        this is what makes the C dependency auditable."""
        assert damerau_levenshtein_distance(first, second) == (
            self._reference_osa_distance(first, second))


class TestAdversarialInputs:

    @pytest.mark.parametrize("text,operation,policy", [
        ("", Operation.SUBSTITUTE, SelectionPolicy.KEYBOARD_NEIGHBOR),
        ("12345 678 $9.99", Operation.SUBSTITUTE, SelectionPolicy.KEYBOARD_NEIGHBOR),
        ("a b", Operation.DELETE, SelectionPolicy.KEYBOARD_NEIGHBOR),  # only 1-letter words
        ("aa bb cc", Operation.TRANSPOSE, SelectionPolicy.KEYBOARD_NEIGHBOR),  # identical adjacents
        ("word", Operation.DELETE, SelectionPolicy.WHITESPACE),  # no inter-word space
    ], ids=["empty", "all-numeric", "one-letter-words", "identical-adjacent", "no-space"])
    def test_unsatisfiable_inputs_raise_instead_of_corrupting(
            self, text, operation, policy, is_word):
        """Inputs with no eligible edit position must raise PerturbationError
        (→ exclusion sidecar), never return corrupted or unchanged text as if
        perturbed — silent failures here poison matched pairs."""
        with pytest.raises(PerturbationError):
            perturb(text, operation, Unit.CHAR, Scope.ANYWHERE, 1,
                    policy, SemanticClass.A, 3, is_word=is_word)

    def test_budget_larger_than_eligible_positions_raises(self, is_word):
        """A budget the text cannot absorb must fail loudly — partial
        application would break budget exactness."""
        with pytest.raises(PerturbationError):
            perturb("hi", Operation.DELETE, Unit.CHAR, Scope.ANYWHERE, 3,
                    SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 3,
                    is_word=is_word)

    def test_negative_budget_and_unknown_vocabulary_values_are_rejected(self, is_word):
        """Malformed requests (negative budget, unknown operation/policy/scope
        strings) must be rejected at the boundary, not half-executed."""
        with pytest.raises(PerturbationError):
            perturb(REALISTIC_SENTENCE, Operation.SUBSTITUTE, Unit.CHAR,
                    Scope.ANYWHERE, -1, SelectionPolicy.KEYBOARD_NEIGHBOR,
                    SemanticClass.A, 3, is_word=is_word)
        for bad_call in (
            dict(operation="explode"), dict(selection_policy="rainbow"),
            dict(scope="everywhere-ish"),
        ):
            arguments = dict(
                operation=Operation.SUBSTITUTE,
                selection_policy=SelectionPolicy.KEYBOARD_NEIGHBOR,
                scope=Scope.ANYWHERE) | bad_call
            with pytest.raises((PerturbationError, ValueError)):
                perturb(REALISTIC_SENTENCE, arguments["operation"], Unit.CHAR,
                        arguments["scope"], 1, arguments["selection_policy"],
                        SemanticClass.A, 3, is_word=is_word)

    def test_scope_preconditions_are_enforced(self, is_word):
        """answer_critical without key_terms, and content/instruction without
        scope_spans, must raise — guessing a scope would silently perturb the
        wrong region."""
        with pytest.raises(PerturbationError):
            perturb(REALISTIC_SENTENCE, Operation.SUBSTITUTE, Unit.CHAR,
                    Scope.ANSWER_CRITICAL, 1, SelectionPolicy.KEYBOARD_NEIGHBOR,
                    SemanticClass.A, 3, is_word=is_word)
        with pytest.raises(PerturbationError):
            perturb(REALISTIC_SENTENCE, Operation.SUBSTITUTE, Unit.CHAR,
                    Scope.CONTENT, 1, SelectionPolicy.KEYBOARD_NEIGHBOR,
                    SemanticClass.A, 3, is_word=is_word)

    @settings(max_examples=60, deadline=None)
    @given(text=strategies.text(max_size=60),
           seed=strategies.integers(min_value=0, max_value=1_000))
    def test_arbitrary_unicode_never_crashes_the_engine(self, text, seed):
        """On arbitrary text (emoji, RTL, control chars) the engine either
        satisfies its full contract or raises PerturbationError — no third
        outcome, no exception of any other type."""
        is_word = DEMO_IS_WORD
        try:
            perturbed, edits = perturb(
                text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 1,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed,
                is_word=is_word)
        except PerturbationError:
            return
        assert perturbed != text
        assert apply_edit_script(text, edits) == perturbed


class TestCacheRegressions:

    def test_neighbor_generators_stay_uncached(self):
        """The DL-band generators can return ~1e5 strings per word; caching
        them once exhausted host memory and SIGKILLed a run. They must never
        grow a functools cache again (the small derived-result lookup is the
        cacheable layer)."""
        assert not hasattr(_damerau_levenshtein_one_neighbors, "cache_info")
        assert not hasattr(_damerau_levenshtein_band_neighbors, "cache_info")
