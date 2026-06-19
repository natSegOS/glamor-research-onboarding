"""Deterministic answer extraction and binary correctness scoring.

Provenance
----------
Frozen at pre-registration; every rule here is exact and never tuned against
results (design/04 §4.2–4.5). The parse-status taxonomy distinguishes four
outcomes so that an interactional failure (the model asks for clarification or
refuses) is not silently lumped in with a wrong answer:

    valid | unparseable | clarification | refusal

Clarifications and refusals score as INCORRECT for accuracy (the conservative
choice) and are also counted separately for the invalid/clarification rate
(design/04 §4.5) — the dual-accounting rule.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Optional

from enums import (
    ParseStatus,
    INTERACTIONAL_FAILURE_STATUSES,
    REASONING_FAMILIES,
    MCQ_FAMILIES,
)


_ANY_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_HASH_DELIMITED_ANSWER = re.compile(r"####\s*(-?\$?\d[\d,]*\.?\d*)")

_MCQ_EXPLICIT_MARKER = re.compile(
    r"answer\s*(?:is)?\s*[:\-]?\s*\(?([A-J])\)?", re.IGNORECASE)
_MCQ_LINE_LEADING_LETTER = re.compile(
    r"^\(?([A-J])\)?[\).:\s]", re.MULTILINE)

_CLARIFICATION_PHRASES = re.compile(
    r"\b(did you mean|do you mean|could you clarify|can you clarify|"
    r"please clarify|not sure what you mean|what do you mean)\b",
    re.IGNORECASE)
_REFUSAL_PHRASES = re.compile(
    r"\b(i cannot|i can't|i won't|i am unable|i'm unable|i refuse|as an ai)\b",
    re.IGNORECASE)


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
    valid_letters = "ABCDEFGHIJ"[:option_count]

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


def classify_parse_status(generation: str, parsed_answer) -> ParseStatus:
    """Classify a generation into the four-way parse-status taxonomy. A refusal
    is only recognized when there is also no parseable answer, so that a model
    that says "I can't be certain, but the answer is 19" is still scored on its
    answer."""
    if _REFUSAL_PHRASES.search(generation) and parsed_answer is None:
        return ParseStatus.REFUSAL
    if _CLARIFICATION_PHRASES.search(generation):
        return ParseStatus.CLARIFICATION
    if parsed_answer is None:
        return ParseStatus.UNPARSEABLE
    return ParseStatus.VALID


def score_reasoning(generation: str, gold_answer: float, tolerance: float = 1e-6) -> ScoreResult:
    """Score a reasoning generation against a numeric gold answer."""
    parsed_answer = extract_reasoning_answer(generation)
    parse_status = classify_parse_status(generation, parsed_answer)

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
    parse_status = classify_parse_status(generation, parsed_answer)

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

