"""Controlled-vocabulary enumerations for the typo-robustness study.

Every string or discrete-numeric constant that identifies a category (operation
names, selection policies, semantic classes, parse statuses, etc.) is defined
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
    NONE = "none"                         # sentinel for clean (unperturbed) rows;
                                           # other enums reuse this NONE/CLEAN pattern


class SelectionPolicy(_StrEnum):
    """How replacement characters or words are chosen."""
    KEYBOARD_NEIGHBOR = "keyboard_neighbor"
    INFORMATIVE_WORD = "informative_word"
    REAL_WORD = "real_word"
    # Regime B restricted to CMU-dictionary exact homophones (no orthographic
    # band): the pure acoustic-confusion proxy. Crosswalks to the HIVE voice
    # arm's clean+homophone operator (its Table 1, #13).
    HOMOPHONE = "homophone"
    # Whitespace split/merge. The merge direction ("missed-space") crosswalks
    # to HIVE keyboard operator #21; the split direction stays dormant.
    WHITESPACE = "whitespace"
    # Discourse-particle insertion: the frozen set {"uh", "um", "like", "so"}.
    # Intent is preserved definitionally (particles carry no propositional content);
    # no rejection sampling needed (Workstream 3). Novel versus prior work.
    FILLER_WORD = "filler_word"
    # Method A counterfactual (design/02 §2.5, design/06 §6.8): paired Low/High
    # fragmentation variants of the same word at the same edit budget.
    FRAGMENTATION_MATCHED = "fragmentation_matched"
    NONE = "none"


class Scope(_StrEnum):
    """Which part of the prompt the perturbation targets."""
    INSTRUCTION = "instruction"
    CONTENT = "content"
    ANSWER_CRITICAL = "answer_critical"
    ANYWHERE = "anywhere"
    NONE = "none"


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
    CLEAN = "clean"


# ---------------------------------------------------------------------------
# Task vocabulary
# ---------------------------------------------------------------------------

class TaskFamily(_StrEnum):
    """Which task / dataset a row belongs to.

    Primary (N=600, confirmatory): GSM_SYMBOLIC_OFFICIAL (apple/GSM-Symbolic,
    fresh reasoning), MMLU_PRO (TIGER-Lab/MMLU-Pro, MCQ).
    Contamination-contrast (standard benchmarks paired with primaries):
    GSM8K (openai/gsm8k), MMLU (cais/mmlu, 4-option).
    Offline generators (unit tests / pilot / Regime C operand swap):
    GSM_SYMBOLIC_SYNTHETIC (templated generator), MCQ_DEMO (5-item smoke set).
    GSM_SYMBOLIC is a historical, backward-compat tag from early
    load_reasoning_jsonl versions, not in REASONING_FAMILIES; re-tag on load
    if present.
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


class ExtractionTier(_StrEnum):
    """Which answer-extraction rule fired, in priority order per task type
    (Workstream 4; design/04 §4.2). Recorded on every generation row so a
    reviewer can audit which surface pattern a scored answer came from."""
    HASH_DELIMITED = "hash_delimited"                    # "#### <number>" (reasoning)
    LAST_NUMBER_FALLBACK = "last_number_fallback"         # any number in text (reasoning fallback)
    MCQ_EXPLICIT_MARKER = "mcq_explicit_marker"           # "answer is X" / "Answer: X"
    MCQ_LINE_LEADING = "mcq_line_leading"                 # letter at start of line
    MCQ_STANDALONE_SENTENCE = "mcq_standalone_sentence"   # letter in last sentence
    UNPARSEABLE = "unparseable"                           # nothing found


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

class FinishReason(_StrEnum):
    """vLLM completion finish_reason values the pipeline distinguishes.
    TRUNCATED ("length") means the max_new_tokens budget cut the generation
    off. The truncation-rate gate counts these rows."""
    STOPPED = "stop"
    TRUNCATED = "length"


class ConditionSource(_StrEnum):
    """How a perturbation condition's samples are produced."""
    SYNTHETIC = "synthetic"   # perturbation engine


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
# Statistics vocabulary
# ---------------------------------------------------------------------------

class McNemarTestMethod(_StrEnum):
    """Which McNemar variant produced a McNemarResult (design/06 §6.4)."""
    EXACT_MIDP = "exact_midp"
    ASYMPTOTIC = "asymptotic"


class SampleSizeMethod(_StrEnum):
    """Which formula mcnemar_sample_size used (design/06 §6.3)."""
    CONNOR = "connor"
    SIMPLE = "simple"


class ConvergenceMethod(_StrEnum):
    """Which rung of the pre-registered convergence ladder produced the
    confirmatory logistic GLMM (design/06 §6.6; Barr et al. 2013, pp. 275–276).

    The first four rungs fit ``lme4::glmer`` (binomial, logit link, bobyqa)
    through the rpy2 bridge, simplifying the random-effects structure one
    pre-registered step at a time; the last rung is the pure-Python
    fixed-factor logistic GLM that also serves as the offline fallback when
    no R installation is available. A rung is accepted only when the fit
    converges without a singular random-effects estimate."""
    GLMER_MAXIMAL = "glmer_maximal"
    GLMER_NO_RANDOM_CORRELATIONS = "glmer_no_random_correlations"
    GLMER_NO_MODEL_SLOPE = "glmer_no_model_slope"
    GLMER_INTERCEPTS_ONLY = "glmer_intercepts_only"
    FIXED_EFFECTS_LOGISTIC_GLM = "fixed_effects_logistic_glm"


# ---------------------------------------------------------------------------
# LLM-judge vocabulary
# ---------------------------------------------------------------------------

class JudgeClassification(_StrEnum):
    """The regime-audit judge's classification of a perturbation pair (judge.py).

    Shares its A/B/C values with SemanticClass by design (the judge is
    classifying into the same three regimes) but is a distinct vocabulary: the
    judge's own opinion, not the engine's internal state tag, and it adds
    NOT_APPLICABLE for pairs too minor or ambiguous to classify.
    """
    A = "A"
    B = "B"
    C = "C"
    NOT_APPLICABLE = "not_applicable"


class JudgeConfidence(_StrEnum):
    """The judge's self-reported confidence in its classification (judge.py)."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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
    study's token-classification logic. Values match spaCy's ``token.dep_``
    attribute strings exactly (universaldependencies.org).
    """
    AUXILIARY = "aux"
    AUXILIARY_PASSIVE = "auxpass"
    COPULA = "cop"
    NEGATION = "neg"


class SpacyMorphologicalDegree(_StrEnum):
    """Values of the Universal Dependencies morphological 'Degree' feature as
    returned by spaCy's ``token.morph.get("Degree")``
    (universaldependencies.org/u/feat/Degree.html).
    """
    COMPARATIVE = "Cmp"
    SUPERLATIVE = "Sup"


class SpacyMorphologicalNumericType(_StrEnum):
    """Values of the Universal Dependencies morphological 'NumType' feature as
    returned by spaCy's ``token.morph.get("NumType")``
    (universaldependencies.org/u/feat/NumType.html).

    Ordinal numerals ("first", "second", "last") directly identify which
    object or rank is in scope, making them answer-critical in both reasoning
    and multiple-choice questions (Huddleston & Pullum 2002, §5.3).
    """
    ORDINAL = "Ord"


class SpacyMorphologicalPronounType(_StrEnum):
    """Values of the Universal Dependencies morphological 'PronType' feature as
    returned by spaCy's ``token.morph.get("PronType")``
    (universaldependencies.org/u/feat/PronType.html).

    Total-quantifier determiners ("each", "every", "all", "both") impose
    distributive semantics; a perturbation that changes or removes such a
    quantifier directly changes the counting structure of a problem
    (Barwise & Cooper 1981, generalised quantifiers).
    """
    TOTAL = "Tot"


class UniversalDependenciesClosedClassPartOfSpeechTag(_StrEnum):
    """Closed-class (function-word) POS tags in the Universal Dependencies
    tagset: grammatically required function material with no independent
    referential meaning. Excluded from the NER condition of the K_P(x)
    key-term rule: a preposition or article inside a named-entity span (e.g.
    "the" in "the United States") is not a meaningful perturbation target,
    since altering it produces ungrammatical output, not a semantically
    distinct question.
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
    disfluency markers (Clark & Fox Tree 2002, Cognition 84(1):73–111;
    Shriberg 1994, UC Berkeley dissertation). The pragmatic discourse markers
    (``like``, ``so``) are the two most frequent English filler markers in
    informal speech (Jurafsky & Martin 2024, §26.4). Each carries no
    propositional content in inter-word positions, so insertion is
    intent-preserving by definition.
    """
    FILLED_PAUSE_UH = "uh"
    FILLED_PAUSE_UM = "um"
    DISCOURSE_MARKER_LIKE = "like"
    DISCOURSE_MARKER_SO = "so"
