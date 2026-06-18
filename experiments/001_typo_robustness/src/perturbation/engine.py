"""engine.py — The complete perturbation engine for the typo-robustness study.

Implements the perturbation state vector r = (operation, unit, scope,
edit_budget, selection_policy, regime, seed) from design/02 §2.3, and
exposes a single public entry point: perturb().

Guarantees (each enforced and unit-tested in tests/test_perturb.py):
  1. Determinism      — same arguments + seed → byte-identical output and edit script.
  2. Budget exactness — exactly edit_budget primitive edits applied, or PerturbationError.
  3. Identity at k=0  — edit_budget=0 returns text unchanged with an empty edit script.
  4. Protected spans  — no edit ever touches a protected span.
  5. Reconstructibility — apply_edit_script(text, edit_script) == perturbed_text.
  6. Policy fidelity  — keyboard_neighbor draws only QWERTY neighbors; real_word
                        results are always valid words; informative_word edits only key terms.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Sequence

from .keyboard import ALPHABET, keyboard_neighbors


# -- Enumerations --

class Operation(str, Enum):
    """Damerau–Levenshtein primitive edit operation (design/02 §2.2)."""

    INSERT = "insert"
    DELETE = "delete"
    SUBSTITUTE = "substitute"
    TRANSPOSE = "transpose"


class SelectionPolicy(str, Enum):
    """Character/word selection policy controlling which edits are generated."""

    UNIFORM = "uniform"
    KEYBOARD_NEIGHBOR = "keyboard_neighbor"
    INFORMATIVE_WORD = "informative_word"
    REAL_WORD = "real_word"
    WHITESPACE = "whitespace"
    ASR_TRANSCRIPTION = "asr_transcription"


class Scope(str, Enum):
    """Which part of the prompt is eligible for edits."""

    INSTRUCTION = "instruction"
    CONTENT = "content"
    ANSWER_CRITICAL = "answer_critical"
    ANYWHERE = "anywhere"


class Unit(str, Enum):
    """Granularity of the edit operation."""

    CHAR = "char"
    WORD = "word"
    SPAN = "span"


class Regime(str, Enum):
    """Semantic regime of the perturbation (design/02 §2.4)."""

    A = "A"   # intent-preserving nonword typos
    B = "B"   # context-recoverable real-word shift
    C = "C"   # meaning-changing control


# -- Edit record --

@dataclass
class Edit:
    """One primitive edit recorded in current-string coordinates at application time.

    Storing coordinates at application time (not original-string coordinates)
    means the script replays deterministically when applied in order.
    """

    op: str   # insert | delete | substitute | transpose | word_substitute
    index: int   # character index in the string at moment of application
    before: str = ""   # character(s) removed / pre-state
    after: str = ""   # character(s) added / post-state
    word_index: int = - 1   # index of the affected word at moment of application
    word_before: str = ""
    word_after: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PerturbationError(Exception):
    """Raised when a requested perturbation cannot be applied exactly."""


_RECONST_ERR = "reconstruction mismatch at {0}"


def apply_edit_script(text: str, edits: Sequence[Edit | dict]) -> str:
    """Replay an edit script over text; must reproduce perturbed_text exactly."""

    result = text

    for edit in edits:
        entry = edit.to_dict() if isinstance(edit, Edit) else dict(edit)
        op, index = entry["op"], entry["index"]

        if op == "substitute":
            assert result[index] == entry["before"], _RECONST_ERR.format(index)
            result = result[:index] + entry["after"] + result[index + 1:]

        elif op == "insert":
            result = result[:index] + entry["after"] + result[index:]

        elif op == "delete":
            assert result[index] == entry["before"], _RECONST_ERR.format(index)
            result = result[:index] + result[index + 1:]

        elif op == "transpose":
            assert result[index:index + 2] == entry["before"], _RECONST_ERR.format(index)
            result = result[:index] + entry["after"] + result[index + 2:]

        elif op == "word_substitute":
            word_length = len(entry["before"])
            assert result[index:index + word_length] == entry["before"], _RECONST_ERR.format(index)
            result = result[:index] + entry["after"] + result[index + word_length:]

        else:
            raise PerturbationError(f"unknown op in script: {op!r}")

    return result


def edited_words(edits: Sequence[Edit]) -> list[tuple[str, str]]
    """Return (word_before, word_after) pairs for each edit, for regime checks."""

    return [(edit.word_before, edit.word_after) for edit in edits]


# -- Damerau-Levenshtein distance --

def damerau_levenshtein(a: str, b: str) -> int:
    """Standard Damerau–Levenshtein distance including adjacent transposition."""

    len_a, len_b = len(a), len(b)

    distance_matrix = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        distance_matrix[i][0] = i
    for j in range(len_b + 1):
        distance_matrix[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            substitution_cost = 0 if a[i - 1] == b[j - 1] else 1
            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,   # deletion
                distance_matrix[i][j - 1] + 1,   # insertion
                distance_matrix[i - 1][j - 1] + substitution_cost   # substitution
            )

            is_transposition = (
                i > 1 and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            )

            if is_transposition:
                distance_matrix[i][j] = min(
                    distance_matrix[i][j],
                    distance_matrix[i - 2][j - 2] + 1
                )

    return distance_matrix[len_a][len_b]


def single_edit_candidates(word: str) -> set[str]:
    """All Damerau–Levenshtein distance-1 letter variants of word."""

    candidates: set[str] = set()

    for i in range(len(word)):
        for character in ALPHABET:
            if character != word[i].lower():
                substituted = (
                    word[:i]
                    + (character.upper() if word[i].isupper() else character)
                    + word[i + 1:]
                )

                # subsitution
                candidates.add(substituted)

        # deletion
        candidates.add(word[:i] + word[i + 1:])

    for i in range(len(word) + 1):
        for character in ALPHABET:
            # insertion
            candidates.add(word[:i] + character + word[i:])

    for i in range(len(word) - 1):
        if word[i] != word[i + 1]:
            candidates.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])

    candidates.discard(word)

    return candidates


# -- Enum validation helper --

def _resolve(enum_class, value, param_name):
    """Coerce value into enum_class, raising PerturbationError on failure."""

    try:
        return enum_class(value)
    except ValueError:
        raise PerturbationError(f"unknown {param_name} {value!r}")


# -- Eligibility helpers (internal) --

def _word_spans(chars: list) -> list[tuple[int, int]]:
    """(start, end) current-index spans of whitespace-delimited words."""

    spans: list[tuple[int, int]] = []
    span_start = None

    for i, (character, _) in enumerate(chars):
        if not character.isspace():
            if span_start is None:
                span_start = i
        else:
            if span_start is not None:
                spans.append((span_start, i))
                span_start = None

    if span_start is not None:
        spans.append((span_start, len(chars)))

    return spans


def _word_at(chars: list, index: int) -> tuple[int, str]:
    """(word_index, word_string) containing current char index, or (-1, '')."""

    for word_index, (span_start, span_end) in enumerate(_word_spans(chars)):
        if span_start <= index < span_end:
            return word_index, "".join(ch for ch, _ in chars[span_start:span_end])

    return -1, ""


def _protected_ids(
    protected_spans: Sequence[tuple[int, int]] | None,
) -> set[int]:
    """Original-text character ids that must never be touched."""

    original_ids: set[int] = set()

    for span_start, span_end in (protected_spans or []):
        original_ids.update(range(span_start, span_end))

    return original_ids


def _scope_ids(
    text: str,
    scope: Scope,
    scope_spans: dict[str, tuple[int, int]] | None,
    key_terms: Sequence[str] | None,
) -> set[int]:
    """Original character-id set eligible under the requested scope."""

    total_length = len(text)

    if scope == Scope.ANYWHERE:
        return set(range(total_length))

    if scope == Scope.ANSWER_CRITICAL:
        if not key_terms:
            raise PerturbationError("scope=ANSWER_CRITICAL requires key_terms")

        eligible_ids: set[int] = set()

        for term in key_terms:
            for match in re.finditer(re.escape(term), text):
                eligible_ids.update(range(match.start(), match.end()))

        if not eligible_ids:
            raise PerturbationError("no key_terms found in text")

        return eligible_ids

    if not scope_spans or scope.value not in scope_spans:
        raise PerturbationError(
            f"scope={scope.value!r} requires scope_spans[{scope.value!r}]=(start, end)"
        )

    span_start, span_end = scope_spans[scope.value]
    return set(range(span_start, span_end))


def _numeric_ids(text: str) -> set[int]:
    """Original ids of characters in purely-numeric tokens.

    Protected by default in Regime A: corrupting a number changes the answer,
    making the edit meaning-changing rather than intent-preserving.
    """

    original_ids: set[int] = set()

    for match in re.finditer(r"\S+", text):
        token = match.group(0).strip(".,;:!?()")

        if token and re.fullmatch(r"[\d,.$%]+", token):
            original_ids.update(range(match.start(), match.end()))

    return original_ids


def _is_eligible(
    chars: list,
    allowed_original_ids: set[int],
    locked_original_ids: set[int],
) -> Callable[[int], bool]:
    """Return a predicate: is the character at current index i eligible for editing?"""

    def check(i: int) -> bool:
        _, original_id = chars[i]

        if original_id is not None and original_id in locked_original_ids:
            return False

        if original_id is not None and original_id not in allowed_original_ids:
            return False

        return True

    return check


def _letter_positions(chars: list, eligible: Callable[[int], bool]) -> list[int]:
    return [i for i, (ch, _) in enumerate(chars) if ch.isalpha() and eligible(i)]


def _space_positions(chars: list, eligible: Callable[[int], bool]) -> list[int]:
    return [i for i, (ch, _) in enumerate(chars) if ch == " " and eligible(i)]


# -- Character pool registry (internal) --

def _uniform_pool(character: str) -> list[str]:
    return list(ALPHABET.upper() if character.isupper() else ALPHABET)


def _keyboard_pool(character: str) -> list[str]:
    return list(keyboard_neighbors(character))


def _keyboard_insert_pool(anchor_character: str) -> list[str]:
    """Keyboard neighbors of anchor plus the anchor itself (double-typing error)."""

    return list(keyboard_neighbors(anchor_character) + anchor_character)


def _uniform_insert_pool(anchor_character: str) -> list[str]:
    return list(ALPHABET.upper() if anchor_character.isupper() else ALPHABET)


# Maps selection_policy value to (substitude_pool_fn, insert_pool_fn)

# real_word and whitespace are dispatched separately in perturb
_CHAR_POOL_REGISTRY: dict[str, tuple[
    Callable[[str], list[str]],
    Callable[[str], list[str]],
]] = {
    SelectionPolicy.KEYBOARD_NEIGHBOR.value: (_keyboard_pool, _keyboard_insert_pool),
    SelectionPolicy.UNIFORM.value: (_uniform_pool, _uniform_insert_pool),
    SelectionPolicy.INFORMATIVE_WORD.value: (_keyboard_pool, _keyboard_insert_pool),
}

# -- Primitive operations (internal) --

def _apply_substitute(
    chars: list,
    rng: random.Random,
    eligible: Callable[[int], bool],
    character_pool_fn: Callable[[str], list[str]],
) -> Edit:
    """Substitute one eligible letter with a character from the pool."""

    candidates = _letter_positions(chars, eligible)
    rng.shuffle(candidates)

    for i in candidates:
        old_character = chars[i][0]
        pool = [ch for ch in character_pool_fn(old_character) if ch != old_character]

        if not pool:
            continue

        new_character = rng.choice(pool)
        word_index, word_before = _word_at(chars, i)
        chars[i][0] = new_character
        _, word_after = _word_at(chars, i)

        return Edit(
            "substitute",
            i,
            before=old_character,
            after=new_character,
            word_index=word_index,
            word_before=word_before,
            word_after=word_after
        )

    raise PerturbationError("no eligible substitution position")


def _apply_delete(
    chars: list,
    rng: random.Random,
    eligible: Callable[[int], bool],
) -> Edit:
    """Delete one eligible letter, only from words of length >= 2."""

    candidates = [
        i for i in _letter_positions(chars, eligible)
        if len(_word_at(chars, i)[1]) >= 2
    ]

    if not candidates:
        raise PerturbationError("no eligible deletion position")

    i = rng.choice(candidates)
    old_character = chars[i][0]
    word_index, word_before = _word_at(chars, i)
    del chars[i]

    _, word_after = _word_at(chars, min(i, len(chars) - 1)) if chars else (-1, "")

    return Edit(
        "delete",
        i,
        before=old_character,
        after="",
        word_index=word_index,
        word_before=word_before,
        word_after=word_after,
    )


def _apply_insert(
    chars: list,
    rng: random.Random,
    eligible: Callable[[int], bool],
    character_pool_fn: Callable[[str], list[str]],
) -> Edit:
    """Insert a character adjacent to an eligible anchor letter."""

    anchors = _letter_positions(chars, eligible)
    rng.shuffle(anchors)

    for i in anchors:
        anchor_character = chars[i][0]
        pool = list(character_pool_fn(anchor_character))

        new_character = rng.choice(pool)
        word_index, word_before = _word_at(chars, i)

        insertion_offset = rng.choice([0, 1])
        insertion_index = i + insertion_offset
        chars.insert(insertion_index, [new_character, None])

        _, word_after = _word_at(chars, insertion_index)

        return Edit(
            "insert",
            insertion_index,
            before="",
            after=new_character,
            word_index=word_index,
            word_before=word_before,
            word_after=word_after,
        )

    raise PerturbationError("no eligible insertion anchor")


def _apply_transpose(
    chars: list,
    rng: random.Random,
    eligible: Callable[[int], bool],
) -> Edit:
    """Swap two adjacent eligible letters that differ."""

    candidates = []

    for i in range(len(chars) - 1):
        left, right = chars[i][0], chars[i + 1][0]
        both_are_letters = left.isalpha() and right.isalpha()
        both_are_eligible = eligible(i) and eligible(i + 1)

        if both_are_letters and left != right and both_are_eligible:
            candidates.append(i)

    if not candidates:
        raise PerturbationError("no eligible transposition position")

    i = rng.choice(candidates)
    left, right = chars[i][0], chars[i + 1][0]
    word_index, word_before = _word_at(chars, i)

    chars[i][0], chars[i + 1][0] = right, left
    _, word_after = _word_at(chars, i)

    return Edit(
        "transpose",
        i,
        before=left + right,
        after=right + left,
        word_index=word_index,
        word_before=word_before,
        word_after=word_after,
    )


def _apply_whitespace(
    chars: list,
    rng: random.Random,
    eligible: Callable[[int], bool],
    operation: Operation,
) -> Edit:
    """Whitespace split (insert space mid-word) or merge (delete space between words)."""

    if operation in (Operation.INSERT, Operation.SUBSTITUTE, Operation.TRANSPOSE):
        candidates = []

        for span_start, span_end in _word_spans(chars):
            for i in range(span_start + 1, span_end):
                if eligible(i):
                    candidates.append(i)

        if not candidates:
            raise PerturbationError("no eligible whitespace-split position")
        i = rng.choice(candidates)
        word_index, word_before = _word_at(chars, i)
        chars.insert(i, [" ", None])

        return Edit(
            "insert",
            i,
            before="",
            after=" ",
            word_index=word_index,
            word_before=word_before,
            word_after="",
        )

    # DELETE: merge two words by removing space between them
    candidates = _space_positions(chars, eligible)

    if not candidates:
        raise PerturbationError("no eligible whitespace-merge position")

    i = rng.choice(candidates)
    word_index, word_before = _word_at(chars, max(0, i - 1))
    del chars[i]

    return Edit(
        "delete",
        i,
        before=" ",
        after="",
        word_index=word_index,
        word_before=word_before,
        word_after="",
    )


def _apply_real_word(
    chars: list,
    rng: random.Random,
    eligible: Callable[[int], bool],
    is_word: Callable[[str], bool] | None,
) -> Edit:
    """Word-level substitution to a valid-dictionary DL-1 neighbor (Regime B)."""

    if is_word is None:
        raise PerturbationError("real_word policy requires an is_word callable")

    def _span_qualifies(span_start: int, span_end: int) -> bool:
        all_positions_eligible = all(eligible(i) for i in range(span_start, span_end))
        all_positions_letters = all(chars[i][0].isalpha() for i in range(span_start, span_end))
        long_enough = span_end - span_start >= 3

        return all_positions_eligible and all_positions_letters and long_enough

    eligible_spans = [
        (span_start, span_end)
        for span_start, span_end in _word_spans(chars)
        if _span_qualifies(span_start, span_end)
    ]

    rng.shuffle(eligible_spans)

    for span_start, span_end in eligible_spans:
        word = "".join(ch for ch, _ in chars[span_start:span_end])
        if not is_word(word.lower()):
            continue

        word_candidates = sorted(
            candidate
            for candidate in single_edit_candidates(word)
            if is_word(candidate.lower()) and candidate.lower() != word.lower()
        )

        if not word_candidates:
            continue

        new_word = rng.choice(word_candidates)
        word_index, _ = _word_at(chars, span_start)
        del chars[span_start:span_end]

        for offset, character in enumerate(new_word):
            chars.insert(span_start + offset, [character, None])

        return Edit(
            "word_substitute",
            span_start,
            before=word,
            after=new_word,
            word_index=word_index,
            word_before=word,
            word_after=new_word,
        )


# -- Public entry point --

def perturb(
    text: str,
    operation: Operation | str,
    unit: Unit | str,
    scope: Scope | str,
    edit_budget: int,
    selection_policy: SelectionPolicy | str,
    regime: Regime | str,
    seed: int,
    *,
    protected_spans: Sequence[tuple[int, int]] | None = None,
    key_terms: Sequence[str] | None = None,
    scope_spans: dict[str, tuple[int, int]] | None = None,
    is_word: Callable[[str], bool] | None = None,
    exclude_numeric: bool = True,
) -> tuple[str, list[Edit]]:
    """Apply exactly edit_budget primitive edits to text.

    Arguments:
        text             — the clean prompt string to perturb.
        operation        — DL primitive: insert, delete, substitute, or transpose.
        unit             — granularity: char, word, or span (metadata; char is default).
        scope            — which part of the prompt is eligible: anywhere, content,
                           instruction, or answer_critical.
        edit_budget      — exact number of primitive edits to apply (k in the design).
        selection_policy — how edits are chosen: keyboard_neighbor, uniform,
                           informative_word, real_word, or whitespace.
        regime           — target semantic regime: A (nonword), B (real-word), C (meaning-change).
        seed             — integer RNG seed for full reproducibility.

    Keyword arguments:
        protected_spans  — character ranges in the ORIGINAL text that must never be edited.
        key_terms        — required for informative_word policy and answer_critical scope.
        scope_spans      — required for instruction or content scope: maps scope name to
                           (start, end) character offsets in the full prompt.
        is_word          — required for real_word policy: callable(word) -> bool.
        exclude_numeric  — if True (default), numeric tokens are also protected.

    Returns:
        (perturbed_text, edit_script) — the modified string and the ordered list of
        Edit records that exactly reproduce it via apply_edit_script().
    """

    op = _resolve(Operation, operation, "operation")
    policy = _resolve(SelectionPolicy, selection_policy, "selection_policy")
    scope_num = _resolve(Scope, scope, "scope")
    _resolve(Unit, unit, "unit")
    _resolve(Regime, regime, "regime")

    if policy == SelectionPolicy.ASR_TRANSCRIPTION:
        raise PerturbationError(
            "asr_transcription items are produced by asr.py, not perturb()"
        )

    if edit_budget < 0:
        raise PerturbationError("edit_budget must be >= 0")
    if edit_budget == 0:
        return text, []

    rng = random.Random(seed)
    allowed_original_ids = _scope_ids(text, scope_enum, scope_spans, key_terms)

    if policy == SelectionPolicy.INFORMATIVE_WORD:
        if not key_terms:
            raise PerturbationError("informative_word policy requires key_terms")

        allowed_original_ids &= _scope_ids(text, Scope.ANSWER_CRITICAL, None, key_terms)

    locked_original_ids = _protected_ids(protected_spans)
    if exclude_numeric:
        locked_original_ids |= _numeric_ids(text)

    chars: list = [[ch, i] for i, ch in enumerate(text)]
    edits: list[Edit] = []

    for _ in range(edit_budget):
        eligible = _is_eligible(chars, allowed_original_ids, locked_original_ids)

        if policy == SelectionPolicy.REAL_WORD:
            edit = _apply_real_word(chars, rng, eligible, is_word)

        elif policy == SelectionPolicy.WHITESPACE:
            edit = _apply_whitespace(chars, rng, eligible, op)

        else:
            substitute_pool_fn, insert_pool_fn = _CHAR_POOL_REGISTRY[policy.value]

            if op == Operation.SUBSTITUTE:
                edit = _apply_substitute(chars, rng, eligible, substitute_pool_fn)
            elif op == Operation.DELETE:
                edit = _apply_delete(chars, rng, eligible)
            elif op == Operation.INSERT:
                edit = _apply_insert(chars, rng, eligible, insert_pool_fn)
            elif op == Operation.TRANSPOSE:
                edit = _apply_transpose(chars, rng, eligible)
            else:
                raise PerturbationError(f"unhandled operation {operation!r}")

        edits.append(edit)

    out = "".join(ch for ch, _ in chars)

    if out == text:
        raise PerturbationError("edits produced no net change (degenerate)")

    return out, edits

