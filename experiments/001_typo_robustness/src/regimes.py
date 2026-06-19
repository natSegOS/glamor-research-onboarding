"""Semantic-regime construction: turning a clean item into a perturbed one whose
relationship to the intended meaning is known and controlled.

Provenance
----------
The three-regime separation is the framing contribution (design/01 §1.4,
design/02 §2.4). It exists to prevent the most common critique of typo-
robustness work — that the "typo" silently changed the question — by separating:

    Regime A  intent-preserving nonword typo
              The edit creates an invalid word; intent is preserved.
              Example: "France" -> "Frnace". Desired behavior: answer unchanged.

    Regime B  context-recoverable real-word shift
              The edit creates a different VALID word, but context still recovers
              the intent. Example: "France" -> "Finance". This is the dominant
              ASR error type (acoustic confusion), so Regime B is co-primary with
              Regime A, and its main population comes from asr.py. The
              synthetic builder here is for cross-validation.

    Regime C  meaning-changing control
              The edit changes the intended question, so the gold answer changes
              too. For reasoning items we swap a numeric operand and RECOMPUTE
              the gold from the template; for MCQ we insert a negation that flips
              the answer. This tests for over-invariance (clinging to the old
              answer when the meaning changed). design/02 §2.4, design/04 §4.7.

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

from pathlib import Path
from typing import Callable, Optional

from enums import Operation, SelectionPolicy, Scope, SemanticClass, Unit
from perturbation import (
    Edit,
    PerturbationError,
    apply_edit_script,
    damerau_levenshtein_distance,
    perturb,
)


_DEMO_WORDLIST_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "wordlists" / "demo_wordlist.txt"
)


def load_wordlist(path: Optional[Path] = None) -> set[str]:
    """Load a set of lowercase words from a newline-delimited file. Defaults to
    the bundled demo wordlist, which exists for smoke tests only — the real
    study pins a full English dictionary (see make_is_word)."""
    resolved_path = Path(path) if path else _DEMO_WORDLIST_PATH
    return {line.strip().lower()
            for line in resolved_path.read_text().splitlines() if line.strip()}


def make_is_word(words: Optional[set[str]] = None) -> Callable[[str], bool]:
    """Return an ``is_word(token) -> bool`` predicate.

    Pass an explicit word set for hermetic tests. With no argument it loads the
    bundled demo wordlist, which is for pipeline smoke tests only. For the real
    study, build the predicate from a pinned full dictionary (for example
    hunspell en_US, or the `wordfreq` vocabulary) so that "is this a real word?"
    is decided by a real lexicon (design/04 §4.7).
    """
    vocabulary = words if words is not None else load_wordlist()

    def is_word(token: str) -> bool:
        return token.lower().strip(".,;:!?()'\"") in vocabulary

    return is_word


def derived_seed(base_seed: int, *parts) -> int:
    """Deterministically derive a child seed from a base seed and arbitrary
    string parts, via SHA-256. This lets a builder try many candidate edits
    (attempt 0, 1, 2, ...) while remaining a pure function of its inputs."""
    digest_source = "|".join(str(part) for part in (base_seed,) + parts)
    digest = hashlib.sha256(digest_source.encode()).hexdigest()
    return int(digest[:12], 16)


# ---------------------------------------------------------------------------
# Regime A — intent-preserving nonword typo.
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
# Regime B — context-recoverable real-word shift (synthetic cross-validation
# arm; the primary Regime B population comes from asr.py).
# ---------------------------------------------------------------------------

def make_regime_b_real_word_shift(
        text: str,
        seed: int,
        is_word: Callable[[str], bool],
        scope: Scope = Scope.ANYWHERE,
        scope_spans: Optional[dict] = None,
        protected_spans=None,
        max_attempts: int = 64,
) -> tuple[str, list[Edit], dict]:
    """Substitute one word for a valid-word neighbor at Damerau-Levenshtein
    distance 1 (the WikiTypos-style real-word shift)."""
    last_error: Optional[PerturbationError] = None

    for attempt in range(max_attempts):
        attempt_seed = derived_seed(seed, SemanticClass.B, attempt)
        try:
            perturbed_text, edits = perturb(
                text, Operation.SUBSTITUTE, Unit.WORD, scope, 1,
                SelectionPolicy.REAL_WORD, SemanticClass.B, attempt_seed,
                protected_spans=protected_spans,
                scope_spans=scope_spans, is_word=is_word)
        except PerturbationError as error:
            last_error = error
            continue

        edit = edits[0]
        result_is_a_distinct_real_word = (
            is_word(edit.word_after)
            and edit.word_after.lower() != edit.word_before.lower()
        )
        if result_is_a_distinct_real_word:
            metadata = {
                "regime": SemanticClass.B,
                "attempt": attempt,
                "seed_used": attempt_seed,
                "edited_words": [(edit.word_before, edit.word_after)],
                "damerau_levenshtein_distance": damerau_levenshtein_distance(text, perturbed_text),
            }
            return perturbed_text, edits, metadata

    raise PerturbationError(
        f"Regime B construction failed after {max_attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Regime C — meaning-changing controls, with the new gold known by construction.
# ---------------------------------------------------------------------------

def make_regime_c_reasoning_operand_swap(
        reasoning_item,
        seed: int,
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
    for _ in range(32):
        delta = random_generator.choice([-3, -2, -1, 1, 2, 3]) * random_generator.choice([1, 2, 5])
        candidate_value = old_value + delta
        if candidate_value <= 0 or candidate_value == old_value:
            continue

        candidate_parameters = dict(parameters)
        candidate_parameters[key_to_swap] = candidate_value
        try:
            candidate_gold = reasoning_item.template.answer_function(**candidate_parameters)
        except Exception:
            continue

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


_NEGATABLE_VERB = re.compile(r"\b(is|are|was|were|does|do|can|will)\b(?! not)")


def make_regime_c_mcq_negation(
        question: str,
        gold_letter: str,
        gold_letter_if_negated: Optional[str],
        seed: int,
) -> tuple[str, list[Edit], dict]:
    """Insert a negation after the first copula/auxiliary verb in an MCQ
    question. The item must carry a known ``gold_letter_if_negated`` (only
    negation-flippable items are eligible; design/04 §4.7)."""
    if not gold_letter_if_negated or gold_letter_if_negated == gold_letter:
        raise PerturbationError("item is not negation-flippable")

    match = _NEGATABLE_VERB.search(question)
    if match is None:
        raise PerturbationError("no negatable verb found")

    insertion_point = match.end()
    perturbed_question = question[:insertion_point] + " not" + question[insertion_point:]

    edit = Edit(Operation.INSERT, insertion_point,
                before="", after=" not",
                word_before=match.group(0), word_after=match.group(0) + " not")

    metadata = {
        "regime": SemanticClass.C,
        "old_gold_answer": gold_letter,
        "new_gold_answer": gold_letter_if_negated,
        "damerau_levenshtein_distance": damerau_levenshtein_distance(question, perturbed_question),
    }
    return perturbed_question, [edit], metadata

