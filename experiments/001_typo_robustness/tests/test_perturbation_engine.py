"""Adversarial, metamorphic, and edge-case tests for the perturbation engine.

Covers: determinism, budget exactness, identity at k=0, protected spans,
numeric-token protection, edit-script reconstruction, policy fidelity, Damerau-
Levenshtein metric properties, and whitespace/word-unit operations.
"""

from __future__ import annotations

import random
import string

import pytest

from enums import Operation, SelectionPolicy, Scope, SemanticClass, Unit
from perturbation import (
    PerturbationError,
    apply_edit_script,
    damerau_levenshtein_distance,
    perturb,
)
from perturbation import keyboard_neighbors_of


SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog"
_ALL_CHAR_OPS = (Operation.SUBSTITUTE, Operation.DELETE, Operation.INSERT, Operation.TRANSPOSE)


def _random_alpha_text(rng: random.Random, min_len: int = 8, max_len: int = 40) -> str:
    length = rng.randint(min_len, max_len)
    words = []
    while sum(len(w) for w in words) < length:
        wlen = rng.randint(3, 8)
        words.append("".join(rng.choice(string.ascii_lowercase) for _ in range(wlen)))
    return " ".join(words)


# ---------------------------------------------------------------------------
# Clause 1: Determinism
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_output():
    for operation in _ALL_CHAR_OPS:
        first_text, first_edits = perturb(
            SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, 3,
            SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 42)
        second_text, second_edits = perturb(
            SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, 3,
            SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 42)
        assert first_text == second_text
        assert [e.to_dict() for e in first_edits] == [e.to_dict() for e in second_edits]


def test_different_seed_usually_differs():
    outputs = {perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                       SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)[0] for seed in range(8)}
    assert len(outputs) > 1


def test_determinism_across_many_texts_and_seeds():
    rng = random.Random(99)
    for seed in range(20):
        text = _random_alpha_text(rng)
        try:
            a, _ = perturb(text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                           SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
            b, _ = perturb(text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                           SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
            assert a == b
        except PerturbationError:
            pass  # impossible budgets are fine


# ---------------------------------------------------------------------------
# Clause 2: Budget exactness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation", list(_ALL_CHAR_OPS))
@pytest.mark.parametrize("edit_budget", [1, 2, 4])
def test_edit_budget_is_respected(operation, edit_budget):
    _, edits = perturb(
        SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, edit_budget,
        SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 7)
    assert len(edits) == edit_budget


def test_impossible_budget_raises():
    with pytest.raises(PerturbationError):
        perturb("ab", Operation.DELETE, Unit.CHAR, Scope.ANYWHERE, 50,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)


def test_budget_exhaustion_on_single_char():
    with pytest.raises(PerturbationError):
        perturb("a", Operation.DELETE, Unit.CHAR, Scope.ANYWHERE, 2,
                SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)


def test_budget_zero_emits_no_edits():
    for operation in _ALL_CHAR_OPS:
        _, edits = perturb(SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, 0,
                           SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)
        assert len(edits) == 0


# ---------------------------------------------------------------------------
# Clause 3: Identity at k=0
# ---------------------------------------------------------------------------

def test_zero_budget_is_identity():
    perturbed, edits = perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 0,
                               SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 1)
    assert perturbed == SAMPLE_TEXT
    assert edits == []


def test_zero_budget_identity_all_operations():
    for operation in _ALL_CHAR_OPS:
        perturbed, edits = perturb(SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, 0,
                                   SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed=42)
        assert perturbed == SAMPLE_TEXT
        assert edits == []


# ---------------------------------------------------------------------------
# Clause 4: Protected spans
# ---------------------------------------------------------------------------

def test_protected_span_is_never_edited():
    start = SAMPLE_TEXT.index("brown")
    protected = [(start, start + len("brown"))]
    for seed in range(40):
        perturbed, _ = perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 3,
                               SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed,
                               protected_spans=protected)
        assert "brown" in perturbed


def test_multiple_protected_spans():
    text = "alpha beta gamma delta"
    start_alpha = text.index("alpha")
    start_delta = text.index("delta")
    protected = [(start_alpha, start_alpha + 5), (start_delta, start_delta + 5)]
    for seed in range(30):
        try:
            perturbed, _ = perturb(text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                                   SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed,
                                   protected_spans=protected)
            assert "alpha" in perturbed
            assert "delta" in perturbed
        except PerturbationError:
            pass  # impossible if only "beta gamma" eligible


def test_numeric_tokens_protected_by_default():
    text = "Add 12 and 7 to get the sum"
    successful_runs = 0
    for seed in range(40):
        try:
            perturbed, _ = perturb(text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                                   SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
        except PerturbationError:
            continue
        assert "12" in perturbed and "7" in perturbed
        successful_runs += 1
    assert successful_runs > 30


def test_multiple_numeric_tokens_all_protected():
    text = "If 100 cats each eat 25 fish"
    for seed in range(25):
        try:
            perturbed, _ = perturb(text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 3,
                                   SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
        except PerturbationError:
            continue
        assert "100" in perturbed and "25" in perturbed


# ---------------------------------------------------------------------------
# Clause 5: Reconstruction — metamorphic property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation", list(_ALL_CHAR_OPS))
def test_edit_script_reconstructs_output(operation):
    for seed in range(20):
        perturbed, edits = perturb(SAMPLE_TEXT, operation, Unit.CHAR, Scope.ANYWHERE, 3,
                                   SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
        assert apply_edit_script(SAMPLE_TEXT, edits) == perturbed


def test_edit_script_reconstructs_from_dicts():
    perturbed, edits = perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANYWHERE, 2,
                               SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 3)
    as_dicts = [e.to_dict() for e in edits]
    assert apply_edit_script(SAMPLE_TEXT, as_dicts) == perturbed


def test_reconstruction_holds_over_fuzzed_texts():
    rng = random.Random(7)
    for seed in range(25):
        text = _random_alpha_text(rng)
        for operation in _ALL_CHAR_OPS:
            try:
                perturbed, edits = perturb(text, operation, Unit.CHAR, Scope.ANYWHERE, 2,
                                           SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
                assert apply_edit_script(text, edits) == perturbed
            except PerturbationError:
                pass


def test_reconstruction_empty_edit_list():
    assert apply_edit_script(SAMPLE_TEXT, []) == SAMPLE_TEXT


# ---------------------------------------------------------------------------
# Clause 6: Policy fidelity
# ---------------------------------------------------------------------------

def test_keyboard_neighbor_substitution_uses_adjacent_keys():
    perturbed, edits = perturb("the quick brown fox", Operation.SUBSTITUTE, Unit.CHAR,
                               Scope.ANYWHERE, 1, SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, 5)
    edit = edits[0]
    assert edit.after in keyboard_neighbors_of(edit.before)


def test_keyboard_neighbor_always_adjacent_over_many_seeds():
    """Every keyboard-neighbor substitution must produce an actual neighbor."""
    for seed in range(30):
        try:
            _, edits = perturb(SAMPLE_TEXT, Operation.SUBSTITUTE, Unit.CHAR,
                               Scope.ANYWHERE, 1, SelectionPolicy.KEYBOARD_NEIGHBOR, SemanticClass.A, seed)
            for edit in edits:
                assert edit.after in keyboard_neighbors_of(edit.before), (
                    f"seed={seed}: '{edit.before}' -> '{edit.after}' not a neighbor")
        except PerturbationError:
            pass


def test_keyboard_neighbors_of_covers_qwerty():
    """Spot-check some known QWERTY neighborhoods."""
    # 'a' neighbors on a standard QWERTY: q, w, s, z (and possibly x)
    a_neighbors = keyboard_neighbors_of("a")
    assert len(a_neighbors) >= 2
    assert all(len(c) == 1 for c in a_neighbors)

    # Every letter of the alphabet must have at least one neighbor.
    for ch in string.ascii_lowercase:
        neighbors = keyboard_neighbors_of(ch)
        assert len(neighbors) >= 1, f"'{ch}' has no keyboard neighbors"
        for n in neighbors:
            assert n != ch, f"'{ch}' is its own neighbor"


def test_informative_word_edits_only_key_terms():
    text = "Buy 5 boxes with cats inside"
    perturbed, edits = perturb(text, Operation.SUBSTITUTE, Unit.CHAR, Scope.ANSWER_CRITICAL, 1,
                               SelectionPolicy.INFORMATIVE_WORD, SemanticClass.A, 9,
                               key_terms=["cats"])
    assert edits[0].word_before == "cats"


def test_real_word_policy_requires_is_word():
    with pytest.raises(PerturbationError):
        perturb("the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
                SelectionPolicy.REAL_WORD, SemanticClass.B, 1)


def test_real_word_policy_produces_valid_word(small_vocabulary_is_word):
    perturbed, edits = perturb("the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
                               SelectionPolicy.REAL_WORD, SemanticClass.B, 3,
                               is_word=small_vocabulary_is_word)
    edit = edits[0]
    assert small_vocabulary_is_word(edit.word_after)
    assert edit.word_after != edit.word_before


def test_real_word_substitution_changes_text():
    """The resulting text must actually differ from the original."""
    def is_word(w):
        return w.lower() in {"cat", "bat", "the", "sat", "cot"}
    _, edits = perturb("the cat sat", Operation.SUBSTITUTE, Unit.WORD, Scope.ANYWHERE, 1,
                       SelectionPolicy.REAL_WORD, SemanticClass.B, 1, is_word=is_word)
    assert edits[0].word_before != edits[0].word_after


# ---------------------------------------------------------------------------
# Damerau-Levenshtein — metric properties
# ---------------------------------------------------------------------------

def test_damerau_levenshtein_basic_cases():
    assert damerau_levenshtein_distance("cat", "cat") == 0
    assert damerau_levenshtein_distance("cat", "cot") == 1     # substitution
    assert damerau_levenshtein_distance("cat", "ct") == 1      # deletion
    assert damerau_levenshtein_distance("cat", "cast") == 1    # insertion
    assert damerau_levenshtein_distance("cat", "act") == 1     # transposition
    assert damerau_levenshtein_distance("abc", "xyz") == 3


def test_damerau_levenshtein_identity():
    rng = random.Random(42)
    for seed in range(30):
        text = _random_alpha_text(rng, 3, 15)
        assert damerau_levenshtein_distance(text, text) == 0


def test_damerau_levenshtein_symmetry():
    rng = random.Random(17)
    for seed in range(30):
        a = _random_alpha_text(rng, 3, 12)
        b = _random_alpha_text(rng, 3, 12)
        assert damerau_levenshtein_distance(a, b) == damerau_levenshtein_distance(b, a)


def test_damerau_levenshtein_triangle_inequality():
    rng = random.Random(5)
    for _ in range(20):
        a = _random_alpha_text(rng, 3, 8)
        b = _random_alpha_text(rng, 3, 8)
        c = _random_alpha_text(rng, 3, 8)
        assert (damerau_levenshtein_distance(a, c)
                <= damerau_levenshtein_distance(a, b) + damerau_levenshtein_distance(b, c))


def test_damerau_levenshtein_single_substitution():
    assert damerau_levenshtein_distance("abc", "aXc") == 1


def test_damerau_levenshtein_single_deletion():
    assert damerau_levenshtein_distance("abc", "ac") == 1


def test_damerau_levenshtein_single_insertion():
    assert damerau_levenshtein_distance("ac", "abc") == 1


def test_damerau_levenshtein_single_transposition():
    assert damerau_levenshtein_distance("ab", "ba") == 1


def test_damerau_levenshtein_empty_strings():
    assert damerau_levenshtein_distance("", "") == 0
    assert damerau_levenshtein_distance("", "abc") == 3
    assert damerau_levenshtein_distance("abc", "") == 3


def test_damerau_levenshtein_transposition_not_counted_as_two():
    # Pure Levenshtein would count "ca" from "ac" as 2 (del + ins), but
    # Damerau-Levenshtein counts it as 1 transposition.
    assert damerau_levenshtein_distance("ac", "ca") == 1


def test_damerau_levenshtein_distance_bounded_by_length():
    rng = random.Random(3)
    for _ in range(20):
        a = _random_alpha_text(rng, 2, 10)
        b = _random_alpha_text(rng, 2, 10)
        dist = damerau_levenshtein_distance(a, b)
        assert dist <= max(len(a), len(b))
