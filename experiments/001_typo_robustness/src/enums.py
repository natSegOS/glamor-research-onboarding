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
    WHITESPACE = "whitespace"           # dormant; kept for old JSONL compatibility
    # Discourse-particle insertion: the frozen set {"uh", "um", "like", "so"}.
    # Intent is preserved definitionally (particles carry no propositional content);
    # no rejection sampling needed (Workstream 3). Novel versus prior work.
    FILLER_WORD = "filler_word"
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


class ShardType(_StrEnum):
    """Shard type labels used in the run manifest and output file names.

    The generation runner groups requests by shard type so each group can use
    the correct ``max_new_tokens`` budget (reasoning answers are much longer
    than multiple-choice answers).
    """
    REASONING = "reasoning"
    MULTIPLE_CHOICE = "multiple_choice"


class TaskType(_StrEnum):
    """High-level task category; determines which scorer is applied."""
    REASONING = "reasoning"
    MULTIPLE_CHOICE = "mcq"


class DatasetRole(_StrEnum):
    """The role of a dataset in the study design."""
    PRIMARY = "primary"
    CONTAMINATION_CONTRAST = "contamination_contrast"
    SMOKE_TEST = "smoke_test"


# ---------------------------------------------------------------------------
# Annotation vocabulary
# ---------------------------------------------------------------------------

class KeyTermRuleVersion(_StrEnum):
    """Versioned identifier for the key-term identification rule K_P(x).

    Written into every annotated item record and into
    data/items/annotation_PROVENANCE.json. Incrementing this version causes the
    build tool to refuse to overwrite a pre-registered frozen dataset without an
    explicit --force flag.

    STRUCTURAL_FILTER_WITH_TFIDF_RANKED_CANDIDATES (current)
        spaCy structural filter (POS, morphological features, dependency
        relations) with candidates ranked by TF-IDF proxy score for
        prioritisation. No arbitrary cap on the number of key terms.
        Formally grounded in Manning, Raghavan & Schütze (2008),
        Universal Dependencies (Nivre et al. 2016), and
        Huddleston & Pullum (2002).
    """
    STRUCTURAL_FILTER_WITH_TFIDF_RANKED_CANDIDATES = (
        "structural_filter_with_tfidf_ranked_candidates")


class UniversalDependenciesRelationLabel(_StrEnum):
    """Subset of Universal Dependencies syntactic relation labels used in this
    study's token-classification logic.

    Values match spaCy's ``token.dep_`` attribute strings exactly.
    The Universal Dependencies standard is published at universaldependencies.org
    (Nivre et al. 2016; de Marneffe et al. 2021).
    """
    AUXILIARY = "aux"
    AUXILIARY_PASSIVE = "auxpass"
    COPULA = "cop"
    NEGATION = "neg"


class SpacyMorphologicalDegree(_StrEnum):
    """Values of the Universal Dependencies morphological 'Degree' feature as
    returned by spaCy's ``token.morph.get("Degree")``.

    Source: universaldependencies.org/u/feat/Degree.html
    (Universal Dependencies specification, Nivre et al. 2016).
    """
    COMPARATIVE = "Cmp"
    SUPERLATIVE = "Sup"


class SpacyMorphologicalNumericType(_StrEnum):
    """Values of the Universal Dependencies morphological 'NumType' feature as
    returned by spaCy's ``token.morph.get("NumType")``.

    Ordinal numerals ("first", "second", "last") directly identify which
    object or rank is in scope, making them answer-critical in both reasoning
    and multiple-choice questions (Huddleston & Pullum 2002, §5.3).

    Source: universaldependencies.org/u/feat/NumType.html
    """
    ORDINAL = "Ord"


class SpacyMorphologicalPronounType(_StrEnum):
    """Values of the Universal Dependencies morphological 'PronType' feature as
    returned by spaCy's ``token.morph.get("PronType")``.

    Total-quantifier determiners ("each", "every", "all", "both") impose
    distributive semantics; a perturbation that changes or removes such a
    quantifier directly changes the counting structure of a problem
    (Barwise & Cooper 1981, generalised quantifiers).

    Source: universaldependencies.org/u/feat/PronType.html
    """
    TOTAL = "Tot"


class UniversalDependenciesClosedClassPartOfSpeechTag(_StrEnum):
    """Closed-class (function-word) POS tags in the Universal Dependencies tagset.

    These tags identify tokens that are grammatically required function material
    and do not carry independent referential meaning.  They are excluded from
    the NER condition of the K_P(x) key-term rule: a preposition or article
    inside a named-entity span (e.g., "the" in "the United States" or "of" in
    "State of California") is not a meaningful perturbation target — altering it
    produces ungrammatical output rather than a semantically distinct question.

    Source: Nivre et al. (2016) 'Universal Dependencies v1', LREC;
    de Marneffe et al. (2021) 'Universal Dependencies', Computational Linguistics.
    """
    ADPOSITION = "ADP"
    AUXILIARY = "AUX"
    COORDINATING_CONJUNCTION = "CCONJ"
    DETERMINER = "DET"
    PARTICLE = "PART"
    PRONOUN = "PRON"
    PUNCTUATION = "PUNCT"
    SUBORDINATING_CONJUNCTION = "SCONJ"
    WHITESPACE = "SPACE"
    OTHER = "X"


class EnglishDiscourseParticle(_StrEnum):
    """Canonical English filled-pause discourse particles used in Regime A
    filler-word insertion perturbations.

    Filled pauses (``uh``, ``um``) are the most-studied class of English
    disfluency markers (Clark & Fox Tree 2002, "Using uh and um in spontaneous
    speaking", Cognition 84(1):73–111; Shriberg 1994, "Preliminaries to a
    Theory of Speech Disfluencies", UC Berkeley dissertation).  The pragmatic
    discourse markers (``like``, ``so``) are the two most frequent English
    discourse markers used as fillers in informal speech (Jurafsky & Martin
    2024, Speech and Language Processing, §26.4 on discourse coherence and
    disfluency).  Together these four tokens are the set studied across the
    core English disfluency literature and are the canonical minimal set for
    intent-preserving filler insertion: each carries no propositional content
    in inter-word positions, so their insertion is intent-preserving by definition.
    """
    FILLED_PAUSE_UH = "uh"
    FILLED_PAUSE_UM = "um"
    DISCOURSE_MARKER_LIKE = "like"
    DISCOURSE_MARKER_SO = "so"
