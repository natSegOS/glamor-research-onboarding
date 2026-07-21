"""Shared constants and utility functions for the reasoning and multiple-choice task modules.

These definitions live here rather than in one of the two task modules so that
neither module has to import from the other, and so that any third module
(e.g. scoring) can import a single authoritative source without creating a
circular-import or copy-paste situation.

Every constant here is referenced in at least two places across the codebase;
keeping them here is what enforces the no-redundancy / single-source-of-truth
constraint the study's reproducibility requires.
"""

from __future__ import annotations

import re

from enums import Scope


# ---------------------------------------------------------------------------
# Option-letter alphabet for multiple-choice questions
# ---------------------------------------------------------------------------

# The full ordered alphabet of option letters used by MMLU-Pro (up to ten
# options, A through J) and standard MMLU (four options, A through D).
# All option-letter logic (label dictionaries, answer-extraction regexes,
# valid-letter checks) is derived from this string rather than scattered
# independent hard-codings of "ABCDEFGHIJ" or "[A-J]".
#
# Extends naturally: to add an eleventh option, change this one constant and
# every downstream derivation updates automatically.
OPTION_LETTERS: str = "ABCDEFGHIJ"


# ---------------------------------------------------------------------------
# Prompt structure
# ---------------------------------------------------------------------------

# The separator inserted between the task instruction and the question content
# block in the full prompt string.  Both ReasoningItem and MultipleChoiceItem
# use this separator; a single definition prevents silent divergence (a mismatch
# would shift the scope-span boundaries and misalign the perturbation engine).
INSTRUCTION_CONTENT_SEPARATOR: str = "\n\n"


def content_text_of(task_item) -> str:
    """The perturbable text of any task item: MultipleChoiceItem's
    content_text (question + rendered options), or ReasoningItem's
    question_text. Shared so every consumer that walks duck-typed task items
    agrees on this."""
    return getattr(task_item, "content_text", None) or task_item.question_text


def build_full_prompt(instruction: str, content: str, suffix: str = "") -> str:
    """Assemble the full prompt string from an instruction block and a content block.

    The instruction comes first, separated from the content by
    INSTRUCTION_CONTENT_SEPARATOR (a double newline).  The result is the string
    that gets fed to the language model.

    ``suffix``, when given, is appended after the content with the same
    separator. It sits outside both scope spans (the instruction span covers
    only ``instruction``, the content span only ``content``), so it is part of
    the fixed prompt scaffold and never perturbed.

    Both ReasoningItem and MultipleChoiceItem use this function so the separator
    and ordering are guaranteed to be identical across task types.
    """
    prompt = f"{instruction}{INSTRUCTION_CONTENT_SEPARATOR}{content}"
    if suffix:
        return f"{prompt}{INSTRUCTION_CONTENT_SEPARATOR}{suffix}"
    return prompt


def build_instruction_and_content_scope_spans(instruction: str, content: str) -> dict:
    """Return the character-level span boundaries for the instruction and content
    regions within the full prompt produced by build_full_prompt.

    The perturbation engine uses these spans to restrict edits to a named scope
    region (design/02 §2.3, §3.2).  Computing the spans from the same separator
    constant that builds the prompt ensures the boundaries are always consistent.

    Returns a dict keyed by ``str(Scope.INSTRUCTION)`` and ``str(Scope.CONTENT)``
    (the same keys the perturbation engine looks up), each mapped to a
    (start, end) character span within the full prompt.
    """
    instruction_length = len(instruction)
    content_start = instruction_length + len(INSTRUCTION_CONTENT_SEPARATOR)
    return {
        str(Scope.INSTRUCTION): (0, instruction_length),
        str(Scope.CONTENT): (content_start, content_start + len(content)),
    }


# ---------------------------------------------------------------------------
# Shared answer-format pattern
# ---------------------------------------------------------------------------

# The hash-delimited answer format ``#### <number>`` appears in:
#   - GSM8K and GSM-Symbolic gold-answer records (Cobbe et al., 2021,
#     arXiv:2110.14168; Mirzadeh et al., 2025, arXiv:2410.05229), where the
#     number is always a non-negative integer.
#   - Model generations, where the number may include a dollar sign, commas,
#     or a decimal point (e.g. "#### $1,234.50").
#
# A single compiled pattern handles all cases; callers convert to int or float
# as appropriate for their context (gold-answer loading vs. generation scoring).
HASH_DELIMITED_ANSWER_PATTERN: re.Pattern = re.compile(
    r"####\s*(-?\$?\d[\d,]*\.?\d*)"
)
