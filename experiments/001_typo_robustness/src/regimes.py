"""Semantic-regime construction: turning a clean item into a perturbed one whose
relationship to the intended meaning is known and controlled.

Provenance
----------
The three-regime separation is the framing contribution (design/01 §1.4,
design/02 §2.4). It exists to prevent the most common critique of typo-
robustness work (that the "typo" silently changed the question) by separating:

    Regime A  intent-preserving nonword typo
              The edit creates an invalid word; intent is preserved.
              Example: "France" -> "Frnace". Desired behavior: answer unchanged.

    Regime B  context-recoverable real-word shift
              The edit creates a different VALID word, but context still recovers
              the intent. Example: "France" -> "Finance". Motivated by ASR
              acoustic confusion (the dominant real-world real-word-shift error
              type), so the candidate pool includes phonetic homophones
              ("weather"/"whether") alongside the orthographic DL band. See
              make_regime_b_real_word_shift below.

    Regime C  meaning-changing control
              The edit changes the intended question, so the gold answer changes
              too. For reasoning items we swap a numeric operand and RECOMPUTE
              the gold from the template; for MCQ we permute the option labels
              deterministically so the correct answer maps to a different letter.
              This tests for over-invariance (clinging to the old answer when the
              meaning changed). design/02 §2.4, design/04 §4.7.

Regime C scope limitation
-------------------------
Regime C is restricted to perturbations where the new gold answer is
computationally deterministic:
  - Reasoning: operand swap (synthetic-only, template-derived gold recomputation).
  - MCQ: option permutation (all items, no annotation required; gold tracked by
    content, not label).

Arbitrary meaning-changing perturbations (e.g., swapping named entities in
free-text questions) are excluded because the new gold cannot be verified
without semantic understanding. This is a scope limitation, documented in
design/04 §4.7.

Determinism
-----------
Every builder uses rejection sampling over *derived* seeds, so the same
(item, base_seed) always yields the same output even though it may try several
candidate edits internally. The engine's determinism guarantee is preserved.
"""

from __future__ import annotations

import hashlib
import random
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from enums import Operation, SelectionPolicy, Scope, SemanticClass, Unit
from perturbation import (
    Edit,
    PerturbationError,
    damerau_levenshtein_distance,
    perturb,
)
from tasks.reasoning import TEMPLATE_EVALUATION_FAILURE_EXCEPTIONS


_DEMO_WORDLIST_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "wordlists" / "demo_wordlist.txt"
)

# Candidate operand deltas for Regime C's reasoning operand swap: a step size
# (never zero, so the operand always changes) times a scale factor, giving a
# spread of small-to-moderate magnitude changes to search over.
_OPERAND_DELTA_STEPS: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
_OPERAND_DELTA_SCALES: tuple[int, ...] = (1, 2, 5)


def load_wordlist(path: Optional[Path] = None) -> set[str]:
    """Load a set of lowercase words from a newline-delimited file. Defaults to
    the bundled demo wordlist, which exists for smoke tests only. The real
    study pins a full English dictionary (see make_is_word)."""
    resolved_path = Path(path) if path else _DEMO_WORDLIST_PATH
    return {line.strip().lower()
            for line in resolved_path.read_text().splitlines() if line.strip()}


# Punctuation stripped from a token's edges before a dictionary lookup, so
# "France." and "(France)" are recognised as the same word as "France".
PUNCTUATION_TO_STRIP = ".,;:!?()'\""


@dataclass(frozen=True)
class WordSetPredicate:
    """A picklable, hashable ``is_word(token) -> bool`` predicate over a fixed
    vocabulary.

    A plain closure (the prior implementation) can't cross into a worker
    process via ``pickle`` (only module-level callables can), which is what
    makes ``build_requests`` (``pipeline/experiment.py``) parallelizable
    across a ``ProcessPoolExecutor``. Frozen + a ``frozenset`` field also
    keeps it hashable, which every ``lru_cache``-decorated function in
    ``perturbation.engine`` that takes ``is_word`` as an argument relies on.
    """
    vocabulary: frozenset[str]

    def __call__(self, token: str) -> bool:
        return token.lower().strip(PUNCTUATION_TO_STRIP) in self.vocabulary


def make_is_word(words: Optional[set[str]] = None) -> Callable[[str], bool]:
    """Return an ``is_word(token) -> bool`` predicate.

    Pass an explicit word set for hermetic tests. With no argument it loads the
    bundled demo wordlist, which is for pipeline smoke tests only. For the real
    study, build the predicate from a pinned full dictionary (for example
    hunspell en_US, or the `wordfreq` vocabulary) so that "is this a real word?"
    is decided by a real lexicon (design/04 §4.7).
    """
    vocabulary = words if words is not None else load_wordlist()
    return WordSetPredicate(frozenset(vocabulary))


def derived_seed(base_seed: int, *parts) -> int:
    """Deterministically derive a child seed from a base seed and arbitrary
    string parts, via SHA-256. This lets a builder try many candidate edits
    (attempt 0, 1, 2, ...) while remaining a pure function of its inputs."""
    digest_source = "|".join(str(part) for part in (base_seed,) + parts)
    digest = hashlib.sha256(digest_source.encode()).hexdigest()
    return int(digest[:12], 16)


# ---------------------------------------------------------------------------
# Regime A: intent-preserving nonword typo.
# ---------------------------------------------------------------------------

def make_regime_a_nonword_typo(
        text: str,
        operation: Operation,
        edit_budget: int,
        seed: int,
        is_word: Callable[[str], bool],
        selection_policy: SelectionPolicy = SelectionPolicy.KEYBOARD_NEIGHBOR,
        scope: Scope = Scope.ANYWHERE,
        scope_spans: Optional[dict] = None,
        protected_spans=None,
        key_terms=None,
        max_attempts: int = 64,
) -> tuple[str, list[Edit], dict]:
    """Rejection-sample edits until every edited word is a NONWORD, satisfying
    the Regime A definition. Deterministic: attempts iterate over derived seeds
    of (seed, "A", attempt)."""
    last_error: Optional[PerturbationError] = None

    for attempt in range(max_attempts):
        attempt_seed = derived_seed(seed, SemanticClass.A, attempt)
        try:
            perturbed_text, edits = perturb(
                text, operation, Unit.CHAR, scope, edit_budget,
                selection_policy, SemanticClass.A, attempt_seed,
                protected_spans=protected_spans,
                key_terms=key_terms, scope_spans=scope_spans)
        except PerturbationError as error:
            last_error = error
            continue

        changed_words = [
            (word_before, word_after)
            for word_before, word_after in
            ((edit.word_before, edit.word_after) for edit in edits)
            if word_after and word_before.lower() != word_after.lower()
        ]

        every_changed_word_is_a_nonword = (
            bool(changed_words)
            and all(not is_word(word_after) for _, word_after in changed_words)
        )
        if every_changed_word_is_a_nonword:
            metadata = {
                "regime": SemanticClass.A,
                "attempt": attempt,
                "seed_used": attempt_seed,
                "edited_words": changed_words,
                "damerau_levenshtein_distance": damerau_levenshtein_distance(text, perturbed_text),
            }
            return perturbed_text, edits, metadata

        last_error = PerturbationError("an edited word landed on a real word")

    raise PerturbationError(
        f"Regime A construction failed after {max_attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Regime A variant: discourse-particle filler-word insertion.
# ---------------------------------------------------------------------------

def make_regime_a_filler_insertion(
        text: str,
        edit_budget: int,
        seed: int,
        scope: Scope = Scope.ANYWHERE,
        scope_spans: Optional[dict] = None,
        protected_spans=None,
) -> tuple[str, list[Edit], dict]:
    """Insert ``edit_budget`` discourse-particle tokens (the frozen set
    {"uh", "um", "like", "so"}) at eligible inter-word positions.

    Intent-preservation is definitional (particles carry no propositional
    content), so no rejection sampling or is_word check is needed.  The
    nonword test is explicitly bypassed: filler tokens ARE valid English words
    but are treated as Regime A because they cannot change the question's meaning
    by construction.

    Returns (perturbed_text, edits, metadata).
    """
    attempt_seed = derived_seed(seed, SemanticClass.A, "filler", 0)
    perturbed_text, edits = perturb(
        text, Operation.INSERT, Unit.CHAR, scope, edit_budget,
        SelectionPolicy.FILLER_WORD, SemanticClass.A, attempt_seed,
        protected_spans=protected_spans,
        scope_spans=scope_spans,
        exclude_numeric_tokens=True,
    )
    inserted_fillers = [edit.word_after for edit in edits if edit.word_after]
    metadata = {
        "regime": SemanticClass.A,
        "attempt": 0,
        "seed_used": attempt_seed,
        "inserted_fillers": inserted_fillers,
        "damerau_levenshtein_distance": damerau_levenshtein_distance(text, perturbed_text),
    }
    return perturbed_text, edits, metadata


# ---------------------------------------------------------------------------
# Regime B: context-recoverable real-word shift.
# ---------------------------------------------------------------------------

def make_regime_b_real_word_shift(
        text: str,
        seed: int,
        is_word: Callable[[str], bool],
        scope: Scope = Scope.ANYWHERE,
        scope_spans: Optional[dict] = None,
        protected_spans=None,
        max_attempts: int = 64,
        edit_budget: int = 1,
        max_word_distance: int = 2,
        phonetic_only: bool = False,
) -> tuple[str, list[Edit], dict]:
    """Substitute ``edit_budget`` word(s) for valid-word neighbors, using the
    union of an orthographic band (DL ≤ ``max_word_distance``) and phonetic
    homophones (via the CMU Pronouncing Dictionary, when the ``pronouncing``
    package is available).

    The wider candidate pool better reflects the dominant ASR confusion type
    (acoustic look-alikes) while the distinct-valid-word post-condition is
    preserved: every substituted word must be a recognised word distinct from
    the original.  ``max_word_distance=2`` is the Regime B default; call with
    ``max_word_distance=1`` to restore the original orthographic-only DL=1
    behaviour for ablation comparisons.

    ``phonetic_only`` restricts the candidate pool to CMU exact homophones,
    the pure acoustic-confusion proxy condition (SelectionPolicy.HOMOPHONE;
    crosswalks to the HIVE voice arm's clean+homophone operator). Items with
    no homophone-bearing word raise PerturbationError and land in the
    exclusion sidecar, which records that condition's yield.
    """
    last_error: Optional[PerturbationError] = None
    selection_policy = (SelectionPolicy.HOMOPHONE if phonetic_only
                        else SelectionPolicy.REAL_WORD)

    for attempt in range(max_attempts):
        attempt_seed = derived_seed(seed, SemanticClass.B, attempt)
        try:
            perturbed_text, edits = perturb(
                text, Operation.SUBSTITUTE, Unit.WORD, scope, edit_budget,
                selection_policy, SemanticClass.B, attempt_seed,
                protected_spans=protected_spans,
                scope_spans=scope_spans, is_word=is_word,
                max_word_distance=max_word_distance)
        except PerturbationError as error:
            last_error = error
            continue

        all_changed_words_are_distinct_real_words = all(
            is_word(edit.word_after) and edit.word_after.lower() != edit.word_before.lower()
            for edit in edits
        )
        if all_changed_words_are_distinct_real_words:
            edited_words = [(edit.word_before, edit.word_after) for edit in edits]
            metadata = {
                "regime": SemanticClass.B,
                "attempt": attempt,
                "seed_used": attempt_seed,
                "edited_words": edited_words,
                "damerau_levenshtein_distance": damerau_levenshtein_distance(text, perturbed_text),
                "max_word_distance": max_word_distance,
                "phonetic_only": phonetic_only,
            }
            return perturbed_text, edits, metadata

    raise PerturbationError(
        f"Regime B construction failed after {max_attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Regime C: meaning-changing controls, with the new gold known by construction.
# ---------------------------------------------------------------------------

def make_regime_c_reasoning_operand_swap(
        reasoning_item,
        seed: int,
        max_attempts: int = 32,
) -> tuple[str, list[Edit], dict]:
    """Swap one numeric operand in a templated reasoning item and RECOMPUTE the
    gold answer from the template's own answer function, so the new gold is
    known exactly (design/04 §4.7). ``reasoning_item`` is a
    tasks.reasoning.ReasoningItem.

    Only an operand whose digit string appears exactly once in the text is
    swapped, so the textual replacement is guaranteed to hit the intended
    operand even when two parameters happen to share a value.
    """
    random_generator = random.Random(derived_seed(seed, SemanticClass.C, reasoning_item.task_id))

    parameters = dict(reasoning_item.parameters)
    numeric_parameter_keys = [key for key, value in parameters.items() if isinstance(value, int)]
    if not numeric_parameter_keys:
        raise PerturbationError("no numeric operand to swap")

    uniquely_locatable_keys = [
        key for key in sorted(numeric_parameter_keys)
        if len(re.findall(rf"(?<!\d){re.escape(str(parameters[key]))}(?!\d)",
                          reasoning_item.question_text)) == 1
    ]
    if not uniquely_locatable_keys:
        raise PerturbationError("no uniquely-locatable numeric operand")

    key_to_swap = random_generator.choice(uniquely_locatable_keys)
    old_value = parameters[key_to_swap]

    new_value = None
    new_gold = None
    for _ in range(max_attempts):
        delta = (random_generator.choice(_OPERAND_DELTA_STEPS)
                 * random_generator.choice(_OPERAND_DELTA_SCALES))
        candidate_value = old_value + delta
        if candidate_value <= 0 or candidate_value == old_value:
            continue

        candidate_parameters = dict(parameters)
        candidate_parameters[key_to_swap] = candidate_value
        try:
            candidate_gold = reasoning_item.template.answer_function(**candidate_parameters)
        except TEMPLATE_EVALUATION_FAILURE_EXCEPTIONS:
            continue  # this candidate operand value doesn't fit the template

        answer_actually_changed = (
            candidate_gold != reasoning_item.gold_answer
            and candidate_gold == int(candidate_gold)
            and candidate_gold >= 0
        )
        if answer_actually_changed:
            new_value = candidate_value
            new_gold = int(candidate_gold)
            break

    if new_value is None:
        raise PerturbationError("could not find an answer-changing operand swap")

    old_value_string, new_value_string = str(old_value), str(new_value)
    match = re.search(rf"(?<!\d){re.escape(old_value_string)}(?!\d)", reasoning_item.question_text)
    if match is None:
        raise PerturbationError(f"operand {old_value_string} not found in question text")

    perturbed_text = (
        reasoning_item.question_text[:match.start()]
        + new_value_string
        + reasoning_item.question_text[match.end():]
    )
    edit = Edit(Operation.WORD_SUBSTITUTE, match.start(),
                before=old_value_string, after=new_value_string,
                word_before=old_value_string, word_after=new_value_string)

    metadata = {
        "regime": SemanticClass.C,
        "swapped_parameter": key_to_swap,
        "old_value": old_value,
        "new_value": new_value,
        "old_gold_answer": reasoning_item.gold_answer,
        "new_gold_answer": new_gold,
        "damerau_levenshtein_distance":
            damerau_levenshtein_distance(reasoning_item.question_text, perturbed_text),
    }
    return perturbed_text, [edit], metadata


def make_regime_c_mcq_option_permutation(
        mcq_item,
        seed: int,
        max_attempts: int = 32,
) -> tuple[str, list[Edit], dict]:
    """Permute the option labels of an MCQ item deterministically and return the
    perturbed content_text plus the updated gold letter.

    The gold answer is tracked by content, not label: the new gold letter is
    whichever label the shuffled options assign to the original gold option text.
    This requires no human annotation and works for every MCQ item regardless of
    content (contrast: negation-based Regime C, which required per-item
    annotation of ``gold_letter_if_negated`` and was semantically underdetermined
    for general knowledge questions).

    Option-order sensitivity: Pezeshkpour & Hruschka (2023, arXiv:2308.11483)
    report 13–75% performance gaps under option reordering. A model that
    relies on option position rather than content will fail this control.

    Only permutations that move the gold content to a different label are
    accepted, so the gold letter is guaranteed to change (design/04 §4.7).

    ``mcq_item`` is a tasks.multiple_choice.MultipleChoiceItem. It must have
    at least two options; single-option items raise PerturbationError.
    """
    letters = list(mcq_item.options.keys())
    option_texts = list(mcq_item.options.values())

    if len(letters) < 2:
        raise PerturbationError(
            f"MCQ item {mcq_item.task_id!r} has fewer than 2 options; "
            "permutation impossible")

    old_gold_letter = mcq_item.gold_letter
    old_gold_content = mcq_item.options[old_gold_letter]

    rng = random.Random(derived_seed(seed, SemanticClass.C, mcq_item.task_id))

    new_options: Optional[dict] = None
    new_gold_letter: Optional[str] = None

    for _ in range(max_attempts):
        shuffled_texts = option_texts.copy()
        rng.shuffle(shuffled_texts)
        candidate_options = dict(zip(letters, shuffled_texts))
        candidate_gold_letter = next(
            (letter for letter, text in candidate_options.items()
             if text == old_gold_content),
            None)
        if (candidate_gold_letter is not None
                and candidate_gold_letter != old_gold_letter):
            new_options = candidate_options
            new_gold_letter = candidate_gold_letter
            break

    if new_options is None or new_gold_letter is None:
        raise PerturbationError(
            f"could not find an option permutation that changes the gold label "
            f"for item {mcq_item.task_id!r}")

    old_content = mcq_item.content_text
    old_rendered_options = "\n".join(
        f"{letter}. {text}" for letter, text in mcq_item.options.items())
    new_rendered_options = "\n".join(
        f"{letter}. {text}" for letter, text in new_options.items())
    new_content = f"{mcq_item.question}\n{new_rendered_options}"

    # Represent the permutation as a single structural edit for provenance.
    # word_before/after record the gold letter's positional change.
    options_block_start = len(mcq_item.question) + 1   # +1 for the newline
    edit = Edit(
        Operation.WORD_SUBSTITUTE,
        options_block_start,
        before=old_rendered_options,
        after=new_rendered_options,
        word_before=old_gold_letter,
        word_after=new_gold_letter,
    )

    metadata = {
        "regime": SemanticClass.C,
        "old_gold_letter": old_gold_letter,
        "new_gold_letter": new_gold_letter,
        "old_options": dict(mcq_item.options),
        "new_options": new_options,
        "damerau_levenshtein_distance": damerau_levenshtein_distance(old_content, new_content),
    }
    return new_content, [edit], metadata
