"""Deterministic answer extraction and binary correctness scoring. Frozen at
pre-registration: every rule here is exact and never tuned against results
(design/04 §4.2-4.5).

Parse status is one of valid, unparseable, clarification, or refusal, so an
interactional failure (the model asks for clarification or refuses) isn't
silently lumped in with a wrong answer. Clarifications and refusals score
INCORRECT for accuracy (the conservative choice) and are also counted
separately as the invalid-or-clarification rate (design/04 §4.5): the
dual-accounting rule.

Two parse-status detectors:

  classify_parse_status
      Phrase-lexicon-based (data/lexicons/); fast, CPU-only; used by the
      generation runner for inline scoring (smoke/pilot path). What the test
      suite exercises.

  classify_parse_status_with_linguistic_pipeline
      spaCy dependency-parse-based; the formal four-way detector used inline
      during generation (Workstream 5). Structural criteria detect
      interrogatives (clarification) and first-person negated clauses
      (refusal); no phrase list at runtime. The data/lexicons/ phrase lists
      survive only as frozen validation oracles.

Both detectors are deliberately conservative: the invalid-or-clarification
rate is a diagnostic metric (M9), not a primary endpoint, and under-counting
is the safer direction. Clarification taxonomy: Aliannejadi et al. (2019).
Refusal-phrase methodology the phrase lists implement and the linguistic
rule is cross-checked against: Zou et al. (2023).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from enums import (
    ExtractionTier,
    ParseStatus,
    INTERACTIONAL_FAILURE_STATUSES,
    REASONING_FAMILIES,
    MCQ_FAMILIES,
)
from tasks._shared import OPTION_LETTERS, HASH_DELIMITED_ANSWER_PATTERN

import re


# Includes scientific notation (1.5e10, 2E-3) so GSM answers expressed as
# powers of ten aren't silently dropped (Workstream 4).
_ANY_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")

# Shared with the gold-side loader, so both parse the same surface forms.
_HASH_DELIMITED_ANSWER = HASH_DELIMITED_ANSWER_PATTERN

# Single authoritative source instead of hard-coding "[A-J]" twice.
_OPTION_LETTER_CLASS = f"[{OPTION_LETTERS}]"

_MCQ_EXPLICIT_MARKER = re.compile(
    rf"answer\s*(?:is)?\s*[:\-]?\s*\(?({_OPTION_LETTER_CLASS})\)?", re.IGNORECASE)
_MCQ_LINE_LEADING_LETTER = re.compile(
    rf"^\(?({_OPTION_LETTER_CLASS})\)?[\).:\s]", re.MULTILINE)

# ---------------------------------------------------------------------------
# Linguistic parse-status classifier constants
# (used by classify_parse_status_with_linguistic_pipeline)
# ---------------------------------------------------------------------------

# Negative auxiliaries spaCy doesn't tag dep_="neg" (negation fused with the
# auxiliary orthographically). Split contractions ("ca"+"n't") are caught by
# the dep_="neg" check on "n't" instead. Huddleston & Pullum (2002) CGEL §2.3.
_FUSED_NEGATIVE_AUXILIARY_ORTHOGRAPHIC_FORMS: frozenset[str] = frozenset({"cannot"})

# Predicative adjectives expressing inability/unwillingness in first-person
# clauses ("I am unable to ..."). von Fintel & Iatridou (2006) §2.
_INABILITY_PREDICATIVE_ADJECTIVE_LEMMAS: frozenset[str] = frozenset({
    "unable", "unwilling", "incapable",
})

# Root verbs whose first-person use is itself a speech act of refusal, with
# no negation token needed. Searle (1969) "Speech Acts" §3.4.
_REFUSAL_PERFORMATIVE_VERB_LEMMAS: frozenset[str] = frozenset({
    "refuse", "decline", "apologize",
})


@dataclass
class ScoreResult:
    parsed_answer: Optional[str]
    is_correct: int
    parse_status: ParseStatus
    extraction_tier: ExtractionTier = ExtractionTier.UNPARSEABLE


def normalize_number(raw: str) -> Optional[float]:
    """Parse a currency/comma-formatted number string to a float, or None if
    unparseable."""
    cleaned = raw.strip().lstrip("$").replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_reasoning_answer(generation: str) -> tuple[Optional[float], ExtractionTier]:
    """Extract the final numeric answer from a reasoning generation.

    Priority (design/04 §4.2, matching the GSM-Symbolic / GSM8K answer format):
    the number after the LAST '####' delimiter, else the LAST number anywhere
    in the text.
    """
    hash_delimited_matches = _HASH_DELIMITED_ANSWER.findall(generation)
    if hash_delimited_matches:
        value = normalize_number(hash_delimited_matches[-1])
        return value, (ExtractionTier.HASH_DELIMITED if value is not None
                        else ExtractionTier.UNPARSEABLE)

    any_number_matches = _ANY_NUMBER.findall(generation)
    if any_number_matches:
        value = normalize_number(any_number_matches[-1])
        return value, (ExtractionTier.LAST_NUMBER_FALLBACK if value is not None
                        else ExtractionTier.UNPARSEABLE)

    return None, ExtractionTier.UNPARSEABLE


def extract_multiple_choice_answer(
        generation: str, option_count: int = 10) -> tuple[Optional[str], ExtractionTier]:
    """Extract the chosen option letter from an MCQ generation.

    Priority:
      1. explicit 'answer is X' / 'Answer: X' marker (last occurrence)
      2. line-leading standalone letter
      3. standalone valid letter in the **last sentence only** (restricted from
         the full generation to avoid picking up A–J letters in reasoning chains)
    """
    valid_letters = OPTION_LETTERS[:option_count]

    explicit_marker_hits = [
        letter.upper() for letter in _MCQ_EXPLICIT_MARKER.findall(generation)
        if letter.upper() in valid_letters
    ]
    if explicit_marker_hits:
        return explicit_marker_hits[-1], ExtractionTier.MCQ_EXPLICIT_MARKER

    line_leading_hits = [
        letter.upper() for letter in _MCQ_LINE_LEADING_LETTER.findall(generation)
        if letter.upper() in valid_letters
    ]
    if line_leading_hits:
        return line_leading_hits[-1], ExtractionTier.MCQ_LINE_LEADING

    # Restrict standalone scan to last sentence to avoid spurious letter
    # matches earlier in reasoning chains (Workstream 4, MMLU-Pro 10-option).
    last_sentence = generation.rsplit(".", 1)[-1]
    standalone_hits = re.findall(rf"\b([{valid_letters}])\b", last_sentence)
    if standalone_hits:
        return standalone_hits[-1].upper(), ExtractionTier.MCQ_STANDALONE_SENTENCE

    return None, ExtractionTier.UNPARSEABLE


def classify_parse_status(parsed_answer) -> ParseStatus:
    """Structural parse-status classifier for the inline / smoke scoring path.

    Returns only VALID or UNPARSEABLE: interactional-failure sub-categories
    (CLARIFICATION, REFUSAL) are assigned by the formal post-stage classifier
    classify_parse_status_with_linguistic_pipeline. Both outcomes score
    is_correct=0, so accuracy is identical between the smoke and post-stage
    paths; only the diagnostic invalid-or-clarification rate (ICR, metric M9)
    differs.
    """
    if parsed_answer is None:
        return ParseStatus.UNPARSEABLE
    return ParseStatus.VALID


def _sentence_is_interrogative(sentence) -> bool:
    """True if the sentence ends with a '?' punctuation token.

    Implements the clarification criterion (design/04 §4.5 / plan §1.3):
    'y contains an interrogative root clause'. A trailing '?' is the
    canonical surface marker of an interrogative in English orthography
    (Huddleston & Pullum, 2002, CGEL §10).
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

    Implements the refusal criterion:
    'y contains a first-person subject governing a negated ability/volition
    modal (parser: nsubj = 1st-person PRON, aux/root lemma ∈ ability/volition
    modal with DEP=neg or negating particle)'.

    The three sub-checks cover the English surface forms:
      dep_="neg"   : split contractions ("n't"), "not", "never"
      cannot       : fused negative auxiliary (morphologically opaque to dep_)
      unable/...   : predicative inability adjectives
      refuse/...   : performative refusal verbs (no negation morpheme needed)
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

    Uses spaCy dependency structure to assign the full four-way taxonomy.
    No phrase list is loaded; the criteria are structural:

      - clarification ⇔ any sentence in the output is interrogative
        (ends with '?' punctuation).
      - refusal ⇔ no parseable answer AND at least one sentence expresses
        first-person negation or inability/refusal. The frozen validation
        oracle for this rule is tests/fixtures/refusal_phrases.txt.
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
    parsed_answer, tier = extract_reasoning_answer(generation)
    parse_status = classify_parse_status(parsed_answer)

    if parse_status in INTERACTIONAL_FAILURE_STATUSES:
        # Conservative: an interactional non-answer counts against accuracy even
        # if a number happens to appear elsewhere in the text.
        recorded_answer = None if parsed_answer is None else str(parsed_answer)
        return ScoreResult(recorded_answer, 0, parse_status, ExtractionTier.UNPARSEABLE)

    if parsed_answer is None:
        return ScoreResult(None, 0, parse_status, tier)

    gold_as_float = float(gold_answer)
    if gold_as_float.is_integer():
        is_correct = int(parsed_answer == gold_as_float)
    else:
        is_correct = int(abs(parsed_answer - gold_as_float) < tolerance)

    return ScoreResult(str(parsed_answer), is_correct, parse_status, tier)


def score_multiple_choice(generation: str, gold_letter: str, option_count: int = 10) -> ScoreResult:
    """Score a multiple-choice generation against the gold option letter."""
    parsed_answer, tier = extract_multiple_choice_answer(generation, option_count)
    parse_status = classify_parse_status(parsed_answer)

    if parse_status in INTERACTIONAL_FAILURE_STATUSES:
        return ScoreResult(parsed_answer, 0, parse_status, ExtractionTier.UNPARSEABLE)

    if parsed_answer is None:
        return ScoreResult(None, 0, parse_status, tier)

    return ScoreResult(parsed_answer, int(parsed_answer == gold_letter.upper()), parse_status, tier)


def score(generation: str, gold_answer, task_family: str) -> ScoreResult:
    """Dispatch to the right scorer based on the task family."""
    if task_family in REASONING_FAMILIES:
        return score_reasoning(generation, gold_answer)
    if task_family in MCQ_FAMILIES:
        return score_multiple_choice(generation, gold_answer)
    raise ValueError(f"unknown task_family {task_family!r}")
