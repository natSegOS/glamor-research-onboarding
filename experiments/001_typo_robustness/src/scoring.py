"""Deterministic answer extraction and binary correctness scoring.

Provenance
----------
Frozen at pre-registration; every rule here is exact and never tuned against
results (design/04 §4.2–4.5). The parse-status taxonomy distinguishes four
outcomes so that an interactional failure (the model asks for clarification or
refuses) is not silently lumped in with a wrong answer:

    valid | unparseable | clarification | refusal

Clarifications and refusals score as INCORRECT for accuracy (the conservative
choice) and are also counted separately for the invalid-or-clarification rate
(design/04 §4.5) — the dual-accounting rule.

Two parse-status detectors are provided:

  classify_parse_status
      Phrase-lexicon-based; fast and CPU-only; used by the generation runner
      for inline scoring (the smoke / pilot path).  The phrase lists in
      data/lexicons/ are the mechanism.  This is the detector the test suite
      exercises.

  classify_parse_status_with_linguistic_pipeline
      spaCy dependency-parse-based; the formal detector for the post-stage
      scoring tool (tools/score_generations.py).  No phrase list is loaded at
      runtime; structural dependency criteria detect interrogatives (clarification)
      and first-person negated clauses (refusal).  The phrase lists in
      data/lexicons/ survive as frozen validation oracles that the new rule
      must agree with on a pinned example set — they are not the runtime
      mechanism.  See the reconciliation note in the plan (Part 3).

Both detectors are deliberately conservative: the invalid-or-clarification rate
is a diagnostic metric (M9), not a primary endpoint.  Under-counting is the
safer direction.  See Aliannejadi et al. (2019) for the clarification taxonomy
and Zou et al. (2023) for the refusal-phrase methodology that the phrase lists
implement and that the linguistic rule is cross-checked against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from enums import (
    ParseStatus,
    INTERACTIONAL_FAILURE_STATUSES,
    REASONING_FAMILIES,
    MCQ_FAMILIES,
)
from tasks._shared import OPTION_LETTERS, HASH_DELIMITED_ANSWER_PATTERN

import re


_ANY_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")

# Re-use the shared hash-delimited answer pattern so the response-side scorer
# and the gold-side loader parse the same surface forms.
_HASH_DELIMITED_ANSWER = HASH_DELIMITED_ANSWER_PATTERN

# Derive the option-letter character class from the single authoritative source
# rather than hard-coding "[A-J]" in two independent regex strings.
_OPTION_LETTER_CLASS = f"[{OPTION_LETTERS}]"

_MCQ_EXPLICIT_MARKER = re.compile(
    rf"answer\s*(?:is)?\s*[:\-]?\s*\(?({_OPTION_LETTER_CLASS})\)?", re.IGNORECASE)
_MCQ_LINE_LEADING_LETTER = re.compile(
    rf"^\(?({_OPTION_LETTER_CLASS})\)?[\).:\s]", re.MULTILINE)

# ---------------------------------------------------------------------------
# Linguistic parse-status classifier constants
# (used by classify_parse_status_with_linguistic_pipeline)
# ---------------------------------------------------------------------------

# English negative auxiliary forms that spaCy does not assign dep_="neg" to
# because the negation is fused with the auxiliary at the orthographic level.
# "cannot" is the primary member; split contractions (e.g. "ca"+"n't") are
# handled by the dep_="neg" check on the "n't" token.  See Huddleston &
# Pullum (2002) CGEL §2.3 on fused negation in English modals.
_FUSED_NEGATIVE_AUXILIARY_ORTHOGRAPHIC_FORMS: frozenset[str] = frozenset({"cannot"})

# Predicative adjectives that express inability or unwillingness in first-
# person clauses ("I am unable to ...", "I am incapable of ...").
# Grounded in modal-semantics terminology: see von Fintel & Iatridou (2006)
# "Epistemic Containment" §2 for the ability/volition distinction.
_INABILITY_PREDICATIVE_ADJECTIVE_LEMMAS: frozenset[str] = frozenset({
    "unable", "unwilling", "incapable",
})

# Root verbs whose first-person use constitutes a speech act of refusal
# (performative refusals that carry no negation token of their own).
# See Searle (1969) "Speech Acts" §3.4 on performative utterances.
_REFUSAL_PERFORMATIVE_VERB_LEMMAS: frozenset[str] = frozenset({
    "refuse", "decline", "apologize",
})


@dataclass
class ScoreResult:
    parsed_answer: Optional[str]
    is_correct: int                    # 1 or 0
    parse_status: ParseStatus


def normalize_number(raw: str) -> Optional[float]:
    """Parse a possibly currency- and comma-formatted number into a float, or
    None if it does not parse."""
    cleaned = raw.strip().lstrip("$").replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_reasoning_answer(generation: str) -> Optional[float]:
    """Extract the final numeric answer from a reasoning generation.

    Priority (design/04 §4.2, matching the GSM-Symbolic / GSM8K answer format):
    the number after the LAST '####' delimiter, else the LAST number anywhere in
    the text.
    """
    hash_delimited_matches = _HASH_DELIMITED_ANSWER.findall(generation)
    if hash_delimited_matches:
        return normalize_number(hash_delimited_matches[-1])

    any_number_matches = _ANY_NUMBER.findall(generation)
    if any_number_matches:
        return normalize_number(any_number_matches[-1])

    return None


def extract_multiple_choice_answer(generation: str, option_count: int = 10) -> Optional[str]:
    """Extract the chosen option letter from an MCQ generation.

    Priority: an explicit 'answer is X' marker (last occurrence), else a
    line-leading standalone letter, else the last standalone valid letter.
    """
    valid_letters = OPTION_LETTERS[:option_count]

    explicit_marker_hits = [
        letter.upper() for letter in _MCQ_EXPLICIT_MARKER.findall(generation)
        if letter.upper() in valid_letters
    ]
    if explicit_marker_hits:
        return explicit_marker_hits[-1]

    line_leading_hits = [
        letter.upper() for letter in _MCQ_LINE_LEADING_LETTER.findall(generation)
        if letter.upper() in valid_letters
    ]
    if line_leading_hits:
        return line_leading_hits[-1]

    standalone_hits = re.findall(rf"\b([{valid_letters}])\b", generation)
    if standalone_hits:
        return standalone_hits[-1].upper()

    return None


def classify_parse_status(parsed_answer) -> ParseStatus:
    """Structural parse-status classifier for the inline / smoke scoring path.

    Returns only VALID or UNPARSEABLE: interactional-failure sub-categories
    (CLARIFICATION, REFUSAL) are assigned by the formal post-stage classifier
    classify_parse_status_with_linguistic_pipeline.  Both outcomes score
    is_correct=0 for accuracy, so the accuracy primary endpoint is identical
    between the smoke and post-stage paths; only the diagnostic
    invalid-or-clarification rate (ICR, metric M9) differs.
    """
    if parsed_answer is None:
        return ParseStatus.UNPARSEABLE
    return ParseStatus.VALID


def _sentence_is_interrogative(sentence) -> bool:
    """True if the sentence ends with a '?' punctuation token.

    Implements the clarification criterion from design/04 §4.5 / plan §1.3:
    'y contains an interrogative root clause'.  A trailing '?' is the
    canonical surface marker of an interrogative in English orthography (see
    Huddleston & Pullum, 2002, CGEL §10 on interrogative clauses).
    """
    for token in reversed(list(sentence)):
        if token.is_punct:
            return token.text == "?"
        if not token.is_space:
            break
    return False


def _sentence_expresses_first_person_refusal(sentence) -> bool:
    """True if the sentence has a first-person subject with negation or an
    inability/refusal predicate.

    Implements the refusal criterion from design/04 §4.5 / plan §1.3:
    'y contains a first-person subject governing a negated ability/volition
    modal (parser: nsubj = 1st-person PRON, aux/root lemma ∈ ability/volition
    modal with DEP=neg or negating particle)'.

    The three sub-checks cover the English surface forms:
      dep_="neg"   — split contractions ("n't"), "not", "never"
      cannot       — fused negative auxiliary (morphologically opaque to dep_)
      unable/...   — predicative inability adjectives
      refuse/...   — performative refusal verbs (no negation morpheme needed)
    """
    tokens = list(sentence)
    has_first_person_subject = any(
        token.dep_ in {"nsubj", "nsubjpass"}
        and token.text.lower() in {"i", "we"}
        for token in tokens
    )
    if not has_first_person_subject:
        return False

    if any(token.dep_ == "neg" for token in tokens):
        return True
    if any(
        token.text.lower() in _FUSED_NEGATIVE_AUXILIARY_ORTHOGRAPHIC_FORMS
        for token in tokens
    ):
        return True
    if any(
        token.lemma_.lower() in _INABILITY_PREDICATIVE_ADJECTIVE_LEMMAS
        for token in tokens
    ):
        return True
    if any(
        token.lemma_.lower() in _REFUSAL_PERFORMATIVE_VERB_LEMMAS
        and token.dep_ in {"ROOT", "relcl", "advcl", "ccomp"}
        for token in tokens
    ):
        return True
    return False


def classify_parse_status_with_linguistic_pipeline(
        generation: str,
        parsed_answer,
        linguistic_pipeline,
) -> ParseStatus:
    """Formal parse-status classifier for the post-stage scoring tool.

    Uses spaCy dependency structure to assign the full four-way taxonomy
    (design/04 §4.5, formal definition in plan §1.3).  No phrase list is
    loaded; the criteria are structural:

      - clarification ⇔ any sentence in the output is interrogative
        (ends with '?' punctuation).  See Aliannejadi et al. (2019) for the
        clarification taxonomy this formalises.
      - refusal ⇔ no parseable answer AND at least one sentence expresses
        first-person negation or inability/refusal.  The positive examples
        in Zou et al. (2023) Appendix A.2 are the frozen validation oracle
        for this rule (tests/fixtures/refusal_phrases.txt).
      - unparseable ⇔ no parseable answer and neither of the above.
      - valid ⇔ a parseable answer is present.

    The detector is deliberately conservative: clarification/refusal are
    only recognised by structural criteria, so lexically unusual surface
    forms fall into UNPARSEABLE (is_correct=0, counts toward ICR).
    Conservative under-counting is correct for a diagnostic metric; see
    the docstring of classify_parse_status for the accuracy guarantee.
    """
    if parsed_answer is not None:
        return ParseStatus.VALID

    document = linguistic_pipeline(generation)

    for sentence in document.sents:
        if _sentence_is_interrogative(sentence):
            return ParseStatus.CLARIFICATION

    for sentence in document.sents:
        if _sentence_expresses_first_person_refusal(sentence):
            return ParseStatus.REFUSAL

    return ParseStatus.UNPARSEABLE


def score_reasoning(generation: str, gold_answer: float, tolerance: float = 1e-6) -> ScoreResult:
    """Score a reasoning generation against a numeric gold answer."""
    parsed_answer = extract_reasoning_answer(generation)
    parse_status = classify_parse_status(parsed_answer)

    if parse_status in INTERACTIONAL_FAILURE_STATUSES:
        # Conservative: an interactional non-answer counts against accuracy even
        # if a number happens to appear elsewhere in the text.
        recorded_answer = None if parsed_answer is None else str(parsed_answer)
        return ScoreResult(recorded_answer, 0, parse_status)

    if parsed_answer is None:
        return ScoreResult(None, 0, parse_status)

    gold_as_float = float(gold_answer)
    if gold_as_float.is_integer():
        is_correct = int(parsed_answer == gold_as_float)
    else:
        is_correct = int(abs(parsed_answer - gold_as_float) < tolerance)

    return ScoreResult(str(parsed_answer), is_correct, parse_status)


def score_multiple_choice(generation: str, gold_letter: str, option_count: int = 10) -> ScoreResult:
    """Score a multiple-choice generation against the gold option letter."""
    parsed_answer = extract_multiple_choice_answer(generation, option_count)
    parse_status = classify_parse_status(parsed_answer)

    if parse_status in INTERACTIONAL_FAILURE_STATUSES:
        return ScoreResult(parsed_answer, 0, parse_status)

    if parsed_answer is None:
        return ScoreResult(None, 0, parse_status)

    return ScoreResult(parsed_answer, int(parsed_answer == gold_letter.upper()), parse_status)


def score(generation: str, gold_answer, task_family: str) -> ScoreResult:
    """Dispatch to the right scorer based on the task family."""
    if task_family in REASONING_FAMILIES:
        return score_reasoning(generation, gold_answer)
    if task_family in MCQ_FAMILIES:
        return score_multiple_choice(generation, gold_answer)
    raise ValueError(f"unknown task_family {task_family!r}")
