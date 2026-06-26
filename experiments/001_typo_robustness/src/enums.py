"""Controlled-vocabulary enumerations for the typo-robustness study.

Every string or discrete-numeric constant that identifies a category — operation
names, selection policies, semantic classes, parse statuses, etc. — is defined
here so that no bare string literal needs to be scattered across modules.

All enums inherit from (str, Enum) with a __str__ override, which means:
  - Members compare equal to their plain-string counterparts.
  - json.dumps serialises them as their string value without extra conversion.
  - f-strings and str() produce the value, not the Enum repr, in all Python
    versions (3.10+ supported; 3.11 changed the default, hence the override).
"""

from __future__ import annotations

from enum import Enum


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Perturbation engine vocabulary
# ---------------------------------------------------------------------------

class Operation(_StrEnum):
    """The edit operation applied to a text."""
    SUBSTITUTE = "substitute"
    DELETE = "delete"
    INSERT = "insert"
    TRANSPOSE = "transpose"
    WORD_SUBSTITUTE = "word_substitute"   # whole-word swap (Regime B / Regime C)
    ASR = "asr"                           # ASR arm tag, not an engine primitive
    NONE = "none"                         # sentinel for clean (unperturbed) rows


class SelectionPolicy(_StrEnum):
    """How replacement characters or words are chosen."""
    KEYBOARD_NEIGHBOR = "keyboard_neighbor"
    INFORMATIVE_WORD = "informative_word"
    REAL_WORD = "real_word"
    WHITESPACE = "whitespace"
    ASR_TRANSCRIPTION = "asr_transcription"   # recognised by engine but rejected
    ASR_CLEAN = "asr_clean"                   # produced by asr.py (quiet condition)
    ASR_NOISY = "asr_noisy"                   # produced by asr.py (noisy condition)
    NONE = "none"                             # sentinel for clean rows


class Scope(_StrEnum):
    """Which part of the prompt the perturbation targets."""
    INSTRUCTION = "instruction"
    CONTENT = "content"
    ANSWER_CRITICAL = "answer_critical"
    ANYWHERE = "anywhere"
    NONE = "none"     # sentinel for clean rows


class Unit(_StrEnum):
    """The granularity at which edits are counted."""
    CHAR = "char"
    WORD = "word"
    SPAN = "span"


# ---------------------------------------------------------------------------
# Semantic-regime vocabulary
# ---------------------------------------------------------------------------

class SemanticClass(_StrEnum):
    """The three semantic regimes (design/02 §2.4)."""
    A = "A"       # intent-preserving nonword typo
    B = "B"       # context-recoverable real-word shift
    C = "C"       # meaning-changing control
    CLEAN = "clean"   # sentinel for the unperturbed baseline row


# ---------------------------------------------------------------------------
# Task vocabulary
# ---------------------------------------------------------------------------

class TaskFamily(_StrEnum):
    """Which task / dataset a row belongs to.

    Primary datasets (N=600, confirmatory)
    --------------------------------------
    GSM_SYMBOLIC_OFFICIAL    apple/GSM-Symbolic (fresh reasoning)
    MMLU_PRO                 TIGER-Lab/MMLU-Pro (MCQ)

    Contamination-contrast datasets (standard benchmarks, paired with primaries)
    ---------------------------------------------------------------------------
    GSM8K                    openai/gsm8k — standard arithmetic reasoning
    MMLU                     cais/mmlu — standard MCQ (4-option)

    Offline generators (unit tests / pilot / Regime C operand swap)
    ---------------------------------------------------------------
    GSM_SYMBOLIC_SYNTHETIC   offline templated generator
    MCQ_DEMO                 5-item hardcoded smoke-test set

    Historical (backward-compat with old JSONL output only)
    --------------------------------------------------------
    GSM_SYMBOLIC             old tag written by early load_reasoning_jsonl versions;
                             not in REASONING_FAMILIES — re-tag on load if present.
    """
    GSM_SYMBOLIC = "gsm_symbolic"               # historical; avoid in new code
    GSM_SYMBOLIC_OFFICIAL = "gsm_symbolic_official"
    GSM_SYMBOLIC_SYNTHETIC = "gsm_symbolic_synthetic"
    GSM8K = "gsm8k"
    MMLU_PRO = "mmlu_pro"
    MMLU = "mmlu"
    MCQ_DEMO = "mcq_demo"


REASONING_FAMILIES: frozenset[TaskFamily] = frozenset({
    TaskFamily.GSM_SYMBOLIC_OFFICIAL,
    TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
    TaskFamily.GSM8K,
})

MCQ_FAMILIES: frozenset[TaskFamily] = frozenset({
    TaskFamily.MMLU_PRO,
    TaskFamily.MCQ_DEMO,
    TaskFamily.MMLU,
})


# ---------------------------------------------------------------------------
# Scoring / parse-status vocabulary
# ---------------------------------------------------------------------------

class ParseStatus(_StrEnum):
    """Four-way parse-status taxonomy (design/04 §4.5)."""
    VALID = "valid"
    UNPARSEABLE = "unparseable"
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"


INTERACTIONAL_FAILURE_STATUSES: frozenset[ParseStatus] = frozenset({
    ParseStatus.UNPARSEABLE,
    ParseStatus.CLARIFICATION,
    ParseStatus.REFUSAL,
})


# ---------------------------------------------------------------------------
# Tokenization vocabulary
# ---------------------------------------------------------------------------

class FragmentationStratum(_StrEnum):
    """Low / High fragmentation bucket for the mediation counterfactual."""
    LOW = "Low"
    HIGH = "High"


# ---------------------------------------------------------------------------
# Pipeline / inference vocabulary
# ---------------------------------------------------------------------------

class ConditionSource(_StrEnum):
    """How a perturbation condition's samples are produced."""
    SYNTHETIC = "synthetic"   # perturbation engine
    ASR = "asr"               # pre-built AsrItems


class Precision(_StrEnum):
    """Model weight precision / quantization scheme."""
    FP16 = "fp16"
    AWQ = "awq"
    GPTQ = "gptq"


class Decoding(_StrEnum):
    """Decoding strategy used for generation."""
    GREEDY = "greedy"
