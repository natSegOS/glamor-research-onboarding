"""The perturbation engine: the four primitive edits over Damerau-Levenshtein
edit space, applied deterministically with a fully reconstructible edit script.

Provenance
----------
The primitive edit basis {insertion, deletion, substitution, transposition} is
the Damerau-Levenshtein operation set (Damerau, 1964): the three Levenshtein
operations plus adjacent transposition, which is included because transposition
is a common human typing error. The keyboard-neighbor policy follows MulTypo
(Liu et al., 2025). See docs/PROVENANCE.md §2.1–2.3 and design/02 §2.2.

Contract (each clause is enforced by tests/test_perturbation_engine.py)
-----------------------------------------------------------------------
1. Determinism      same arguments + same seed -> byte-identical output and
                    edit script.
2. Budget exactness exactly ``edit_budget`` primitive edits are applied, or a
                    PerturbationError is raised if that is impossible.
3. Identity at k=0  an edit budget of zero returns the input unchanged.
4. Protected spans  characters inside a protected span are never edited, even
                    as surrounding indices shift under insertion/deletion.
5. Reconstructible  apply_edit_script(original, script) == perturbed output.
6. Policy fidelity  keyboard_neighbor draws only QWERTY-adjacent keys;
                    real_word produces only valid dictionary words;
                    informative_word edits only the supplied key terms.

How protected spans survive shifting indices
---------------------------------------------
The engine works on a list of [character, original_index] pairs. The
original_index is the character's position in the *original* string (None for
characters the engine inserts). Eligibility is decided against original indices,
so an insertion or deletion earlier in the string cannot accidentally expose a
protected character to a later edit — the protection travels with the character,
not with its current position.
"""

from __future__ import annotations

import random
import re

from dataclasses import dataclass, asdict
from typing import Callable, Optional, Sequence

from enums import EnglishDiscourseParticle, Operation, SelectionPolicy, Scope, Unit, SemanticClass
from perturbation.keyboard import keyboard_neighbors_of


LOWERCASE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


# The four primitive Damerau-Levenshtein operations (insert, delete, substitute,
# transpose). WORD_SUBSTITUTE is excluded here — it is not a DL primitive, but
# a convenience op for whole-word replacement recorded distinctly in edit scripts.
PRIMITIVE_OPERATIONS: tuple[Operation, ...] = (
    Operation.INSERT, Operation.DELETE, Operation.SUBSTITUTE, Operation.TRANSPOSE)

SELECTION_POLICIES: tuple[SelectionPolicy, ...] = (
    SelectionPolicy.KEYBOARD_NEIGHBOR,  # QWERTY-adjacent letter (MulTypo replacement)
    SelectionPolicy.INFORMATIVE_WORD,   # keyboard_neighbor, but restricted to key terms
    SelectionPolicy.REAL_WORD,          # whole-word substitution to a valid-word neighbor
    SelectionPolicy.WHITESPACE,         # dormant: split/merge (kept for old-JSONL replay)
    SelectionPolicy.FILLER_WORD,        # discourse-particle insertion (Workstream 3)
    SelectionPolicy.ASR_TRANSCRIPTION,  # produced by asr.py, never by this engine
)

# Ordered tuple of discourse particle values for random selection.  The source
# enumeration (EnglishDiscourseParticle) documents the literature citations;
# this tuple is sorted by enum definition order to keep selection deterministic
# across Python versions.
_DISCOURSE_PARTICLE_VALUES: tuple[str, ...] = tuple(
    particle.value for particle in EnglishDiscourseParticle
)

PERTURBATION_SCOPES: tuple[Scope, ...] = (
    Scope.INSTRUCTION, Scope.CONTENT, Scope.ANSWER_CRITICAL, Scope.ANYWHERE)


# ---------------------------------------------------------------------------
# Phonetic-confusion infrastructure (lazy; requires the ``pronouncing`` package).
# ---------------------------------------------------------------------------

# ``None`` means "not yet checked"; True / False means checked and available
# or unavailable, respectively.  Set once per process by _load_phoneme_index.
_phoneme_index_available: Optional[bool] = None

# Inverse index: stress-stripped phoneme tuple -> list of dictionary words.
# Populated by _load_phoneme_index on first call.
_phoneme_sequence_to_words: dict = {}


def _load_phoneme_index() -> bool:
    """Build the CMU Pronouncing Dictionary inverse index on first use.

    Strips lexical-stress markers (the digits 0, 1, 2 appended to vowel
    phonemes) before indexing, so "rain" and "rein" share the same phoneme
    key regardless of stress pattern.

    Returns True when the index was successfully built, False when the
    ``pronouncing`` package is absent (offline runs continue with the
    orthographic Damerau-Levenshtein band only).  Results are cached for the
    lifetime of the process.
    """

    global _phoneme_index_available, _phoneme_sequence_to_words

    if _phoneme_index_available is not None:
        return _phoneme_index_available

    try:
        import pronouncing  # noqa: PLC0415 — lazy import by design

        raw_pronunciations = pronouncing.pronunciations
        if not raw_pronunciations:
            _phoneme_index_available = False
            return False

        phoneme_to_words_index: dict = {}

        for dictionary_word, space_separated_phones in raw_pronunciations:
            stress_stripped_phoneme_key = tuple(
                re.sub(r"\d", "", phoneme)
                for phoneme in space_separated_phones.split()
            )
            phoneme_to_words_index.setdefault(
                stress_stripped_phoneme_key, []).append(dictionary_word)

        _phoneme_sequence_to_words = phoneme_to_words_index
        _phoneme_index_available = True

    except ImportError:
        _phoneme_index_available = False

    return _phoneme_index_available


def _cmu_homophone_neighbors(
        word: str,
        is_word: Optional[Callable[[str], bool]]) -> list[str]:
    """Return valid-word exact homophones of ``word`` from the CMU Pronouncing
    Dictionary.

    These are the most ecologically valid Regime-B candidates: acoustic
    confusions where an ASR system outputs a different word with an identical
    phoneme sequence (e.g. "weather" / "whether", "there" / "their").

    Falls back silently to an empty list when the ``pronouncing`` package is
    absent, the word has no CMUdict entry, or ``is_word`` is not supplied — in
    those cases the orthographic Damerau-Levenshtein band still applies.
    """

    if is_word is None or not _load_phoneme_index():
        return []

    try:
        import pronouncing  # noqa: PLC0415 — lazy import by design
    except ImportError:
        return []

    homophone_candidates: set[str] = set()

    for space_separated_phones in pronouncing.phones_for_word(word.lower()):
        stress_stripped_phoneme_key = tuple(
            re.sub(r"\d", "", phoneme)
            for phoneme in space_separated_phones.split()
        )
        for candidate_word in _phoneme_sequence_to_words.get(
                stress_stripped_phoneme_key, []):
            if candidate_word != word.lower() and is_word(candidate_word):
                homophone_candidates.add(candidate_word)

    return sorted(homophone_candidates)


class PerturbationError(Exception):
    """Raised when the requested perturbation cannot be applied exactly as
    specified (for example, an edit budget larger than the number of eligible
    positions, or a policy whose preconditions are not met)."""


@dataclass
class Edit:
    """One primitive edit, recorded in the coordinates of the string *at the
    moment the edit was applied*, so the script replays deterministically when
    the edits are applied in order.

    The character-level fields (``index``, ``before``, ``after``) are what
    ``apply_edit_script`` replays. The word-level fields are diagnostic: they
    let the regime builders check whether an edited word became a nonword
    (Regime A) or another valid word (Regime B) without re-parsing the string.
    """

    operation: Operation         # insert | delete | substitute | transpose | word_substitute
    index: int                   # character index in the string when applied
    before: str = ""             # character(s) removed or pre-edit state
    after: str = ""              # character(s) added or post-edit state

    word_index: int = -1         # index of the affected word when applied
    word_before: str = ""        # the affected word before the edit
    word_after: str = ""         # the affected word after the edit

    def to_dict(self) -> dict:
        return asdict(self)


def damerau_levenshtein_distance(first_string: str, second_string: str) -> int:
    """The Damerau-Levenshtein distance: minimum number of insertions,
    deletions, substitutions, and adjacent transpositions to turn one string
    into the other (Damerau, 1964)."""
    first_length, second_length = len(first_string), len(second_string)

    distance = [[0] * (second_length + 1) for _ in range(first_length + 1)]

    for i in range(first_length + 1):
        distance[i][0] = i
    for j in range(second_length + 1):
        distance[0][j] = j

    for i in range(1, first_length + 1):
        for j in range(1, second_length + 1):

            substitution_cost = 0 if first_string[i - 1] == second_string[j - 1] else 1

            distance[i][j] = min(
                distance[i - 1][j] + 1,                       # deletion
                distance[i][j - 1] + 1,                       # insertion
                distance[i - 1][j - 1] + substitution_cost,   # substitution
            )

            is_adjacent_transposition = (
                i > 1 and j > 1
                and first_string[i - 1] == second_string[j - 2]
                and first_string[i - 2] == second_string[j - 1]
            )
            if is_adjacent_transposition:
                distance[i][j] = min(distance[i][j], distance[i - 2][j - 2] + 1)

    return distance[first_length][second_length]


# ---------------------------------------------------------------------------
# Internal helpers operating on the [character, original_index] representation.
# A "cell" below is one [character, original_index] pair.
# ---------------------------------------------------------------------------

def _word_spans(character_cells: list) -> list[tuple[int, int]]:
    """Return the (start, end) current-index spans of whitespace-delimited
    words in the current character-cell list."""
    spans: list[tuple[int, int]] = []
    word_start: Optional[int] = None

    for current_index, (character, _original_index) in enumerate(character_cells):
        if not character.isspace():
            if word_start is None:
                word_start = current_index
        else:
            if word_start is not None:
                spans.append((word_start, current_index))
                word_start = None

    if word_start is not None:
        spans.append((word_start, len(character_cells)))

    return spans


def _word_containing_index(character_cells: list, target_index: int) -> tuple[int, str]:
    """Return the (word_index, word_text) of the word containing the character
    at ``target_index``, or (-1, "") if that index is not inside a word."""
    for word_index, (start, end) in enumerate(_word_spans(character_cells)):
        if start <= target_index < end:
            word_text = "".join(character for character, _ in character_cells[start:end])
            return word_index, word_text
    return -1, ""


def _original_indices_in_protected_spans(
        protected_spans: Optional[Sequence[tuple[int, int]]]) -> set[int]:
    """Flatten a list of (start, end) protected spans into the set of original
    character indices they cover."""
    protected_indices: set[int] = set()
    for start, end in (protected_spans or []):
        protected_indices.update(range(start, end))
    return protected_indices


def _original_indices_eligible_for_scope(
        text: str,
        scope: Scope,
        scope_spans: Optional[dict],
        key_terms: Optional[Sequence[str]]) -> set[int]:
    """Return the set of original character indices that the requested scope
    permits editing.

    - ANYWHERE        : every index.
    - ANSWER_CRITICAL : indices covered by any occurrence of a key term.
    - INSTRUCTION /
      CONTENT         : the (start, end) span supplied by the task loader in
                        ``scope_spans`` (the engine cannot guess where the
                        instruction ends and the content begins).
    """
    text_length = len(text)

    if scope == Scope.ANYWHERE:
        return set(range(text_length))

    if scope == Scope.ANSWER_CRITICAL:
        if not key_terms:
            raise PerturbationError("scope 'answer_critical' requires key_terms")
        eligible_indices: set[int] = set()
        for term in key_terms:
            for match in re.finditer(re.escape(term), text):
                eligible_indices.update(range(match.start(), match.end()))
        if not eligible_indices:
            raise PerturbationError("none of the key_terms were found in the text")
        return eligible_indices

    # INSTRUCTION or CONTENT require explicit spans from the task loader.
    scope_key = str(scope)
    if not scope_spans or scope_key not in scope_spans:
        raise PerturbationError(
            f"scope {scope!r} requires scope_spans[{scope!r}] = (start, end)")
    start, end = scope_spans[scope_key]
    return set(range(start, end))


def _original_indices_of_numeric_tokens(text: str) -> set[int]:
    """Return the original indices of characters inside purely-numeric tokens.

    Numeric tokens are protected by default in Regime A: corrupting a number
    changes the answer, which would turn an intent-preserving typo into a
    meaning-changing edit (design/04 §4.7). Regime C, which deliberately changes
    numbers, does not go through this engine for the operand swap.
    """
    numeric_indices: set[int] = set()
    for match in re.finditer(r"\S+", text):
        stripped_token = match.group(0).strip(".,;:!?()")
        token_is_numeric = bool(stripped_token) and bool(re.fullmatch(r"[\d,.$%]+", stripped_token))
        if token_is_numeric:
            numeric_indices.update(range(match.start(), match.end()))
    return numeric_indices


def _make_eligibility_test(
        character_cells: list,
        allowed_original_indices: set[int],
        locked_original_indices: set[int]) -> Callable[[int], bool]:
    """Return a predicate ``is_eligible(current_index)`` that decides whether the
    character currently at ``current_index`` may be edited. Inserted characters
    (original_index is None) are always eligible; original characters are
    eligible only if they are in the allowed set and not in the locked set."""

    def is_eligible(current_index: int) -> bool:
        _character, original_index = character_cells[current_index]

        character_was_inserted = original_index is None
        if character_was_inserted:
            return True

        if original_index in locked_original_indices:
            return False
        if original_index not in allowed_original_indices:
            return False
        return True

    return is_eligible


def _eligible_letter_positions(character_cells: list, is_eligible) -> list[int]:
    return [index for index, (character, _) in enumerate(character_cells)
            if character.isalpha() and is_eligible(index)]


def _eligible_space_positions(character_cells: list, is_eligible) -> list[int]:
    return [index for index, (character, _) in enumerate(character_cells)
            if character == " " and is_eligible(index)]


# ---------------------------------------------------------------------------
# The public entry point.
# ---------------------------------------------------------------------------

def perturb(
        text: str,
        operation: Operation,
        unit: Unit,
        scope: Scope,
        edit_budget: int,
        selection_policy: SelectionPolicy,
        semantic_class: SemanticClass,
        seed: int,
        protected_spans: Optional[Sequence[tuple[int, int]]] = None,
        key_terms: Optional[Sequence[str]] = None,
        scope_spans: Optional[dict] = None,
        is_word: Optional[Callable[[str], bool]] = None,
        exclude_numeric_tokens: bool = True,
        max_word_distance: int = 1,
) -> tuple[str, list[Edit]]:
    """Apply exactly ``edit_budget`` primitive edits to ``text`` and return the
    perturbed string together with the edit script. See the module docstring for
    the full contract.

    Parameters mirror the perturbation state vector r = (operation, unit, scope,
    edit_budget, selection_policy, semantic_class, seed) of design/02 §2.3, plus
    the structural inputs (protected_spans, key_terms, scope_spans, is_word) that
    the task loader supplies.  ``max_word_distance`` controls the DL band used by
    the real_word policy: 1 (default, backward-compatible) or 2 (Regime B; also
    enables phonetic-homophone candidates when ``pronouncing`` is installed).
    """
    operation = Operation(operation)
    unit = Unit(unit)
    scope = Scope(scope)
    selection_policy = SelectionPolicy(selection_policy)

    if operation not in PRIMITIVE_OPERATIONS:
        raise PerturbationError(f"unknown operation {operation!r}")
    if selection_policy not in SELECTION_POLICIES:
        raise PerturbationError(f"unknown selection_policy {selection_policy!r}")
    if scope not in PERTURBATION_SCOPES:
        raise PerturbationError(f"unknown scope {scope!r}")

    if selection_policy == SelectionPolicy.ASR_TRANSCRIPTION:
        raise PerturbationError(
            "asr_transcription items are produced by asr.py, not perturb()")

    if edit_budget < 0:
        raise PerturbationError("edit_budget must be >= 0")
    if edit_budget == 0:
        return text, []

    random_generator = random.Random(seed)

    allowed_original_indices = _original_indices_eligible_for_scope(
        text, scope, scope_spans, key_terms)

    if selection_policy == SelectionPolicy.INFORMATIVE_WORD:
        if not key_terms:
            raise PerturbationError("the informative_word policy requires key_terms")
        # Restrict edits to the key terms themselves, intersected with the scope.
        allowed_original_indices &= _original_indices_eligible_for_scope(
            text, Scope.ANSWER_CRITICAL, None, key_terms)

    locked_original_indices = _original_indices_in_protected_spans(protected_spans)
    if exclude_numeric_tokens:
        locked_original_indices |= _original_indices_of_numeric_tokens(text)

    character_cells: list = [[character, index] for index, character in enumerate(text)]
    applied_edits: list[Edit] = []

    for _ in range(edit_budget):
        # Rebind eligibility against the current cell list each iteration,
        # because indices shift as characters are inserted and deleted.
        is_eligible = _make_eligibility_test(
            character_cells, allowed_original_indices, locked_original_indices)

        if selection_policy == SelectionPolicy.REAL_WORD:
            edit = _apply_real_word_substitution(
                character_cells, random_generator, is_eligible, is_word,
                max_word_distance=max_word_distance)
        elif selection_policy == SelectionPolicy.FILLER_WORD:
            edit = _apply_filler_word_insertion(
                character_cells, random_generator, is_eligible)
        elif selection_policy == SelectionPolicy.WHITESPACE:
            edit = _apply_whitespace_edit(character_cells, random_generator, is_eligible, operation)
        else:
            edit = _apply_character_edit(
                character_cells, random_generator, is_eligible, operation)

        applied_edits.append(edit)

    perturbed_text = "".join(character for character, _ in character_cells)

    if perturbed_text == text:
        raise PerturbationError("the edits produced no net change (degenerate)")

    return perturbed_text, applied_edits


# ---------------------------------------------------------------------------
# Per-operation appliers. Each mutates ``character_cells`` in place and returns
# the Edit it applied, or raises PerturbationError if no eligible position
# exists for the requested operation.
# ---------------------------------------------------------------------------

def _apply_character_edit(character_cells, random_generator, is_eligible,
                          operation) -> Edit:
    """Apply one character-level edit (substitute / delete / insert / transpose)
    under the keyboard_neighbor or informative_word policy."""

    if operation == Operation.SUBSTITUTE:
        candidate_positions = _eligible_letter_positions(character_cells, is_eligible)
        random_generator.shuffle(candidate_positions)

        for position in candidate_positions:
            original_character = character_cells[position][0]

            replacement_pool = keyboard_neighbors_of(original_character)
            replacement_pool = [c for c in replacement_pool if c != original_character]
            if not replacement_pool:
                continue

            replacement_character = random_generator.choice(replacement_pool)

            word_index, word_before = _word_containing_index(character_cells, position)
            character_cells[position][0] = replacement_character
            _, word_after = _word_containing_index(character_cells, position)

            return Edit(Operation.SUBSTITUTE, position,
                        before=original_character, after=replacement_character,
                        word_index=word_index, word_before=word_before, word_after=word_after)

        raise PerturbationError("no eligible substitution position")

    if operation == Operation.DELETE:
        # Only delete from words of length >= 2, so a deletion never erases a
        # whole one-letter word.
        candidate_positions = [
            position for position in _eligible_letter_positions(character_cells, is_eligible)
            if len(_word_containing_index(character_cells, position)[1]) >= 2
        ]
        if not candidate_positions:
            raise PerturbationError("no eligible deletion position")

        position = random_generator.choice(candidate_positions)
        deleted_character = character_cells[position][0]
        word_index, word_before = _word_containing_index(character_cells, position)

        del character_cells[position]

        if character_cells:
            _, word_after = _word_containing_index(
                character_cells, min(position, len(character_cells) - 1))
        else:
            word_after = ""

        return Edit(Operation.DELETE, position,
                    before=deleted_character, after="",
                    word_index=word_index, word_before=word_before, word_after=word_after)

    if operation == Operation.INSERT:
        anchor_positions = _eligible_letter_positions(character_cells, is_eligible)
        random_generator.shuffle(anchor_positions)

        for anchor_position in anchor_positions:
            anchor_character = character_cells[anchor_position][0]

            # Keyboard neighbors plus the anchor itself (double-typing),
            # matching MulTypo's "simultaneous keystrokes" insertion.
            insertion_pool = keyboard_neighbors_of(anchor_character) + anchor_character

            inserted_character = random_generator.choice(list(insertion_pool))

            word_index, word_before = _word_containing_index(character_cells, anchor_position)
            insertion_position = anchor_position + random_generator.choice([0, 1])
            character_cells.insert(insertion_position, [inserted_character, None])
            _, word_after = _word_containing_index(character_cells, insertion_position)

            return Edit(Operation.INSERT, insertion_position,
                        before="", after=inserted_character,
                        word_index=word_index, word_before=word_before, word_after=word_after)

        raise PerturbationError("no eligible insertion anchor")

    if operation == Operation.TRANSPOSE:
        candidate_positions = []
        for position in range(len(character_cells) - 1):
            left_character = character_cells[position][0]
            right_character = character_cells[position + 1][0]
            both_are_distinct_letters = (
                left_character.isalpha() and right_character.isalpha()
                and left_character != right_character
            )
            if both_are_distinct_letters and is_eligible(position) and is_eligible(position + 1):
                candidate_positions.append(position)

        if not candidate_positions:
            raise PerturbationError("no eligible transposition position")

        position = random_generator.choice(candidate_positions)
        left_character = character_cells[position][0]
        right_character = character_cells[position + 1][0]
        word_index, word_before = _word_containing_index(character_cells, position)

        character_cells[position][0], character_cells[position + 1][0] = right_character, left_character
        _, word_after = _word_containing_index(character_cells, position)

        return Edit(Operation.TRANSPOSE, position,
                    before=left_character + right_character,
                    after=right_character + left_character,
                    word_index=word_index, word_before=word_before, word_after=word_after)

    raise PerturbationError(f"unhandled operation {operation!r}")


def _apply_whitespace_edit(character_cells, random_generator, is_eligible, operation) -> Edit:
    """Apply a whitespace split or merge — a common human and OCR/ASR error.

    An "insert"-flavored operation splits a word by inserting a space inside it;
    a "delete"-flavored operation merges two words by deleting the space between
    them.
    """
    treat_as_split = operation in (Operation.INSERT, Operation.SUBSTITUTE, Operation.TRANSPOSE)

    if treat_as_split:
        candidate_positions = []
        for start, end in _word_spans(character_cells):
            for position in range(start + 1, end):     # strictly inside the word
                if is_eligible(position):
                    candidate_positions.append(position)
        if not candidate_positions:
            raise PerturbationError("no eligible whitespace-split position")

        position = random_generator.choice(candidate_positions)
        word_index, word_before = _word_containing_index(character_cells, position)
        character_cells.insert(position, [" ", None])

        return Edit(Operation.INSERT, position, before="", after=" ",
                    word_index=word_index, word_before=word_before, word_after="")

    # Merge: delete a space.
    candidate_positions = _eligible_space_positions(character_cells, is_eligible)
    if not candidate_positions:
        raise PerturbationError("no eligible whitespace-merge position")

    position = random_generator.choice(candidate_positions)
    word_index, word_before = _word_containing_index(character_cells, max(0, position - 1))
    del character_cells[position]

    return Edit(Operation.DELETE, position, before=" ", after="",
                word_index=word_index, word_before=word_before, word_after="")


def _apply_filler_word_insertion(character_cells, random_generator, is_eligible) -> Edit:
    """Insert one discourse-particle filler token at an eligible inter-word
    space position.

    The discourse particle set is defined by ``EnglishDiscourseParticle``
    (see enums.py) with literature citations.  A particle is inserted as
    ``<existing_space><particle> `` at a space boundary so it becomes a
    standalone token between two existing words.  The edit budget counts the
    number of particles inserted, not the number of characters, consistent with
    the edit-budget semantics used throughout the engine.

    Intent-preservation is definitional: discourse particles carry no
    propositional content, so the perturbation cannot change the question's
    meaning regardless of context (no rejection sampling needed).
    """
    candidate_positions = _eligible_space_positions(character_cells, is_eligible)
    if not candidate_positions:
        raise PerturbationError("no eligible space position for filler insertion")

    position = random_generator.choice(candidate_positions)
    filler_token = random_generator.choice(_DISCOURSE_PARTICLE_VALUES)

    # Insert " <filler> " at the chosen space. We replace the single space with
    # " <filler> " by inserting characters one at a time to the right of the
    # existing space, building " <filler> " left-to-right.
    # Representation: " " (existing) + filler_chars + " " (new trailing space).
    insertion_string = filler_token + " "  # leading space is the existing one
    for offset, ch in enumerate(insertion_string):
        character_cells.insert(position + 1 + offset, [ch, None])

    word_index, word_before = _word_containing_index(
        character_cells, max(0, position - 1))

    return Edit(Operation.INSERT, position + 1,
                before="", after=filler_token,
                word_index=word_index, word_before=word_before, word_after=filler_token)


def _damerau_levenshtein_one_neighbors(word: str) -> set[str]:
    """Return every Damerau-Levenshtein-distance-1 letter variant of ``word``
    (substitution, deletion, insertion, adjacent transposition). Used by the
    real-word policy to find valid-word neighbors."""
    variants: set[str] = set()

    for i in range(len(word)):
        for replacement in LOWERCASE_ALPHABET:                       # substitution
            if replacement != word[i].lower():
                cased = replacement.upper() if word[i].isupper() else replacement
                variants.add(word[:i] + cased + word[i + 1:])
        variants.add(word[:i] + word[i + 1:])                        # deletion

    for i in range(len(word) + 1):
        for inserted in LOWERCASE_ALPHABET:                          # insertion
            variants.add(word[:i] + inserted + word[i:])

    for i in range(len(word) - 1):                                   # transposition
        if word[i] != word[i + 1]:
            variants.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])

    variants.discard(word)
    return variants


def _damerau_levenshtein_band_neighbors(word: str, max_distance: int = 1) -> set[str]:
    """Return all string variants of ``word`` within Damerau-Levenshtein distance
    ``max_distance``.

    For ``max_distance == 1`` this is identical to
    ``_damerau_levenshtein_one_neighbors``.  For ``max_distance == 2`` the
    result is the complete DL ≤ 2 neighbourhood, computed by expanding every
    level-1 variant by one additional DL-1 step.  The original word is excluded.
    """
    level1 = _damerau_levenshtein_one_neighbors(word)
    if max_distance < 2:
        return level1
    # Level 2: expand from every level-1 variant; include both levels.
    result: set[str] = set(level1)
    for neighbor in level1:
        result |= _damerau_levenshtein_one_neighbors(neighbor)
    result.discard(word)
    return result


def _apply_real_word_substitution(
        character_cells,
        random_generator,
        is_eligible,
        is_word,
        *,
        max_word_distance: int = 1) -> Edit:
    """Replace a whole word with a valid-word neighbor.

    Candidates are drawn from the union of two pools:

    1. **Orthographic band** — all strings within DL distance ``max_word_distance``
       of the source word that are recognised as valid words.  At distance 1 this
       is the WikiTypos-style real-word shift (the original behaviour).  At
       distance 2 the band is wider, catching common substitutions like
       "France" → "Finance" that are orthographically close but not distance-1.

    2. **Phonetic homophones** — words that share the source word's CMU
       pronunciation (exact homophones such as "weather"↔"whether").  These are
       the dominant ASR confusion type and are added to the candidate pool when
       the ``pronouncing`` package is installed.  If the package is absent the
       pool falls back to the orthographic band only.

    The final candidate list is the sorted union of both pools, filtered to
    valid words distinct from the source word, and selection is deterministic
    given the random generator state.
    """
    if is_word is None:
        raise PerturbationError("the real_word policy requires an is_word callable")

    eligible_word_spans = [
        (start, end) for start, end in _word_spans(character_cells)
        if all(is_eligible(index) for index in range(start, end))
        and all(character_cells[index][0].isalpha() for index in range(start, end))
        and (end - start) >= 3
    ]
    random_generator.shuffle(eligible_word_spans)

    for start, end in eligible_word_spans:
        word = "".join(character for character, _ in character_cells[start:end])
        if not is_word(word.lower()):
            continue

        # Orthographic band (DL ≤ max_word_distance).
        orthographic = {
            candidate for candidate in
            _damerau_levenshtein_band_neighbors(word, max_word_distance)
            if is_word(candidate.lower()) and candidate.lower() != word.lower()
        }
        # Phonetic homophones (exact CMUdict pronunciation match).
        phonetic = set(_cmu_homophone_neighbors(word, is_word))
        phonetic.discard(word.lower())

        valid_word_neighbors = sorted(orthographic | phonetic)
        if not valid_word_neighbors:
            continue

        replacement_word = random_generator.choice(valid_word_neighbors)
        word_index, _ = _word_containing_index(character_cells, start)

        del character_cells[start:end]
        for offset, character in enumerate(replacement_word):
            character_cells.insert(start + offset, [character, None])

        return Edit(Operation.WORD_SUBSTITUTE, start,
                    before=word, after=replacement_word,
                    word_index=word_index, word_before=word, word_after=replacement_word)

    raise PerturbationError("no real-word substitution available")


# ---------------------------------------------------------------------------
# Reconstruction (contract clause 5).
# ---------------------------------------------------------------------------

def apply_edit_script(original_text: str, edits: Sequence) -> str:
    """Replay an edit script over ``original_text`` and return the result, which
    must equal the perturbed text the engine produced. Accepts both ``Edit``
    objects and their dict form (so a script round-tripped through JSON
    replays identically)."""
    current_text = original_text

    for edit in edits:
        fields = edit.to_dict() if isinstance(edit, Edit) else dict(edit)
        operation = fields["operation"]
        index = fields["index"]

        if operation == "substitute":
            assert current_text[index] == fields["before"], f"reconstruction mismatch at {index}"
            current_text = current_text[:index] + fields["after"] + current_text[index + 1:]

        elif operation == "insert":
            current_text = current_text[:index] + fields["after"] + current_text[index:]

        elif operation == "delete":
            assert current_text[index] == fields["before"], f"reconstruction mismatch at {index}"
            current_text = current_text[:index] + current_text[index + 1:]

        elif operation == "transpose":
            assert current_text[index:index + 2] == fields["before"], f"reconstruction mismatch at {index}"
            current_text = current_text[:index] + fields["after"] + current_text[index + 2:]

        elif operation == "word_substitute":
            length_before = len(fields["before"])
            assert current_text[index:index + length_before] == fields["before"], \
                f"reconstruction mismatch at {index}"
            current_text = current_text[:index] + fields["after"] + current_text[index + length_before:]

        else:
            raise PerturbationError(f"unknown operation in edit script: {operation!r}")

    return current_text
