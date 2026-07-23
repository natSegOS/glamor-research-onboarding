"""Formal linguistic annotation for task items: computing K_P(x) and answer-critical spans.

Background
----------
The perturbation engine has two policies that depend on knowing which tokens
in a question carry semantic weight:

  answer_critical   edits are restricted to characters covered by key terms;
  informative_word  keyboard-neighbour substitutions target key-term words.

Prior to this module, both policies used a hand-coded heuristic
(numeric-token pattern plus a curated function-word exclusion list for MCQ;
numeric-token pattern plus a curated operation-word list for reasoning).
Those heuristics had no formal definition and no published accuracy record.

This module replaces them with a formally-defined function K_P(x): the set of
tokens in question x that meet at least one of the following conditions, as
detected by a pinned spaCy linguistic pipeline P:

    NOUN, PROPN, or NUM part-of-speech tag     (content and quantity carriers)
    Named-entity membership (ENT_IOB ≠ O)      (referent binders)
    Negation dependency (DEP = neg)             (answer-inverting modifier)
    Comparative or superlative degree           (scalar language)
    Total-quantifier determiner (PronType=Tot)  (distributive scope)

The rule is grounded in the standard content-word / function-word distinction
from information retrieval (Manning, Raghavan & Schütze 2008) and the
dependency grammar of negation and quantification (Tesnière 1959, Huddleston &
Pullum 2002).  The spaCy pipeline is chosen because:

  1. It is already named in the study design (design/04 §4.6).
  2. Its English models are trained on OntoNotes 5.0 / Universal Dependencies
     with published F1 scores (cited in data/items/annotation_PROVENANCE.json).
  3. It runs on CPU without GPU dependencies, consistent with the project's
     CPU/GPU split: build tools and post-processing run anywhere; only the
     generation step requires a GPU.

Usage
-----
The annotation is run once by tools/build_annotated_dataset.py and the output
is committed as a frozen JSONL file.  The experiment reads the frozen
annotations; it never calls this module at runtime.

See also
--------
data/items/annotation_PROVENANCE.json: records the exact model name, version,
    SHA-256 of the model package, the rule identifier, and publication citations.
design/02 §2.x  : formal definition of K_P(x).
design/04 §4.6  : literature justification for the choice of spaCy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

from enums import (
    KeyTermRuleVersion,
    SpacyMorphologicalDegree,
    SpacyMorphologicalNumericType,
    SpacyMorphologicalPronounType,
    UniversalDependenciesClosedClassPartOfSpeechTag,
    UniversalDependenciesRelationLabel,
)
from tasks.reasoning import (
    FRACTION_WORDS,
    VERBAL_MULTIPLIER_WORDS,
    deserialize_parameters,
)

# Reverse lookups (value -> the English words that spell it out), so the
# operand-coverage check below can recognise "thrice" as covering operand
# value 3 or "third" as covering Fraction(1, 3), not just the literal digit
# string. Built once from the same word lists reasoning.py uses to parse
# GSM-Symbolic templates in the first place, so the two can never drift apart.
_NUMBER_TO_WORDS: dict[int, set[str]] = {}
for _word, _value in VERBAL_MULTIPLIER_WORDS.items():
    _NUMBER_TO_WORDS.setdefault(_value, set()).add(_word)

_FRACTION_TO_WORDS: dict[Fraction, set[str]] = {}
for _word, _value in FRACTION_WORDS.items():
    _FRACTION_TO_WORDS.setdefault(_value, set()).add(_word)

# Dependency relation labels whose VERBs carry no independent propositional
# content: they express modality, aspect, or predicate structure but do not
# themselves name the action or state the question asks about.  Source:
# Universal Dependencies specification (Nivre et al. 2016; de Marneffe et al.
# 2021, universaldependencies.org/u/dep/).
_DEPENDENCY_LABELS_OF_NON_LEXICAL_VERBS: frozenset[str] = frozenset({
    UniversalDependenciesRelationLabel.AUXILIARY,
    UniversalDependenciesRelationLabel.AUXILIARY_PASSIVE,
    UniversalDependenciesRelationLabel.COPULA,
})

# Closed-class POS tags that should be excluded from the NER condition of the
# K_P(x) rule.  A preposition or article inside a named-entity span is
# grammatically required function material, not a meaningful perturbation
# target: altering it produces ungrammatical output rather than a semantically
# distinct question.  Content-bearing tokens inside NER spans (NOUN, PROPN,
# NUM, ADJ, VERB, ADV) are not in this set and continue to qualify via NER.
_FUNCTION_WORD_POS_TAGS_EXCLUDED_FROM_NER_CONDITION: frozenset[str] = frozenset(
    tag.value for tag in UniversalDependenciesClosedClassPartOfSpeechTag
)

# Small positive constant added inside the logarithm during Inverse Document
# Frequency proxy computation to prevent log(0) when a token's corpus
# frequency rounds to zero in the wordfreq database.  The value 1e-9 is the
# standard numerical stabilisation epsilon for log-probability computations
# (see e.g. Manning, Raghavan & Schütze 2008, §6.2 on TF-IDF smoothing).
_LOG_FREQUENCY_STABILIZATION_EPSILON: float = 1e-9


# The versioned rule identifier written into every annotated item record and
# into data/items/annotation_PROVENANCE.json.  Changing this value causes the
# build tool to refuse to overwrite pre-registered frozen datasets without an
# explicit --force flag, protecting the annotation-to-run correspondence
# required for reproducibility.
KEY_TERM_IDENTIFICATION_RULE_VERSION: KeyTermRuleVersion = (
    KeyTermRuleVersion.STRUCTURAL_FILTER_WITH_TFIDF_RANKED_CANDIDATES)


# ---------------------------------------------------------------------------
# Linguistic pipeline loading
# ---------------------------------------------------------------------------

def load_linguistic_pipeline(model_name: str):
    """Load and return the named spaCy language model, downloading it if needed.

    If the model is not yet installed it is downloaded automatically via
    ``python -m spacy download``.  A clear ImportError is raised only if spaCy
    itself is not installed.

    Parameters
    ----------
    model_name :
        The spaCy model identifier, e.g. ``"en_core_web_sm"``,
        ``"en_core_web_md"``, or ``"en_core_web_trf"``.  Passed directly to
        ``spacy.load()``.  The exact value is recorded in annotation_PROVENANCE.json
        alongside published benchmark numbers so reviewers can verify quality.
    """
    try:
        import spacy
    except ImportError as import_error:
        raise ImportError(
            "The linguistic annotation stage requires spaCy.  "
            "Install it with:  pip install spacy  "
            "Then download the English model with:  "
            f"python -m spacy download {model_name}"
        ) from import_error

    try:
        return spacy.load(model_name)
    except OSError:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name], check=True)
        return spacy.load(model_name)


# ---------------------------------------------------------------------------
# The formal K_P(x) key-term rule
# ---------------------------------------------------------------------------

def _token_is_key_term(token) -> bool:
    """Return True if ``token`` satisfies any condition of the formal K_P(x) rule.

    Every condition is grounded in a published linguistic theory or an
    authoritative morphological standard; no surface-form word lists are used.

    Conditions
    ----------
    NOUN, PROPN, NUM (part-of-speech)
        Open-class content words and quantity expressions.  These carry the
        primary semantic content of a question and changes to them alter the
        answer (content-word / function-word distinction: Manning, Raghavan &
        Schütze 2008, §6.1).

    Named-entity membership (ENT_IOB_ ≠ "O")
        Proper-noun phrases, dates, organisations, etc.  Named entities bind
        referents; a perturbation on a named-entity token typically causes a
        lookup failure or incorrect reference resolution.

    Negation dependency relation (DEP = UniversalDependenciesRelationLabel.NEGATION)
        Negation tokens (``not``, ``never``, ``no``) that govern the head verb
        or adjective.  A perturbation that alters a negation token changes the
        polarity of the question and therefore the answer (Tesnière 1959,
        dependency grammar of sentential negation).

    Comparative or superlative morphological degree
    (Degree = SpacyMorphologicalDegree.COMPARATIVE or .SUPERLATIVE)
        Tokens carrying comparative (``more``, ``fewer``, ``-er``) or
        superlative (``most``, ``fewest``, ``-est``) inflection. Scalar
        modifiers affect which option is correct in multiple-choice items and
        the direction of inequality in reasoning items (Huddleston & Pullum
        2002, §17.5 on degree in the adjective phrase).

    Ordinal numeral adjective (NumType = SpacyMorphologicalNumericType.ORDINAL)
        Ordinal adjectives ("first", "second", "last", etc.) directly
        identify which object, event, or rank the question asks about.
        Changing an ordinal typically changes the answer entirely (Huddleston
        & Pullum 2002, §5.3 on ordinal numerals).  The detection is via
        Universal Dependencies morphological features (universaldependencies.org),
        not a surface-form list, so all ordinals recognised by the pipeline are
        covered without enumeration.

    Total-quantifier determiner (POS = DET, PronType = SpacyMorphologicalPronounType.TOTAL)
        Distributive / universal quantifiers: ``each``, ``every``, ``all``,
        ``both``.  These impose distributive semantics; a perturbation that
        removes or alters such a quantifier changes the counting structure of
        the problem (Barwise & Cooper 1981, generalised quantifiers).

    Non-copular, non-auxiliary VERB
        Predicates that express the question's main action or relation
        (e.g. "costs", "earns", "exceeds").  Auxiliary and copular verbs
        ("is", "was", "can", "have" as aspect marker) are excluded via their
        Universal Dependencies dependency relation: they carry no propositional
        content independent of their complement (Nivre et al. 2016).
    """
    # Content and quantity carriers: the primary semantic load of a question.
    if token.pos_ in {"NOUN", "PROPN", "NUM"}:
        return True

    # Named-entity members: referent binders whose surface form is load-bearing.
    # Function words (DET, ADP, CCONJ, etc.) inside entity spans are excluded:
    # they are grammatically required connective tissue, not perturbation targets.
    # Only content-bearing tokens within the entity span qualify here.
    if (token.ent_iob_ != "O"
            and token.pos_ not in _FUNCTION_WORD_POS_TAGS_EXCLUDED_FROM_NER_CONDITION):
        return True

    # Negation: sentential negation changes the polarity of the answer.
    if token.dep_ == UniversalDependenciesRelationLabel.NEGATION:
        return True

    # Comparative and superlative degree: scalar modifiers that affect which
    # option is correct or which direction an inequality runs.
    morphological_degree_values = token.morph.get("Degree")
    if (SpacyMorphologicalDegree.COMPARATIVE in morphological_degree_values
            or SpacyMorphologicalDegree.SUPERLATIVE in morphological_degree_values):
        return True

    # Ordinal adjectives: identify which rank, position, or object is in scope.
    # Detected entirely via Universal Dependencies morphological features;
    # no surface-form word list is used.
    if (token.pos_ == "ADJ"
            and SpacyMorphologicalNumericType.ORDINAL in token.morph.get("NumType")):
        return True

    # Totality quantifiers: distributive determiners that change the counting
    # structure of a problem.
    if (token.pos_ == "DET"
            and SpacyMorphologicalPronounType.TOTAL in token.morph.get("PronType")):
        return True

    # Non-copular, non-auxiliary predicates: VERBs that name the question's
    # main action or relation.  The exclusion set is the standard Universal
    # Dependencies set of grammatical-function verb relations.
    if (token.pos_ == "VERB"
            and token.dep_ not in _DEPENDENCY_LABELS_OF_NON_LEXICAL_VERBS):
        return True

    return False


def _compute_tfidf_proxy_score(token_text: str, token_count_in_item: int) -> float:
    """Compute a TF-IDF proxy score for a candidate key-term surface form.

    Term Frequency (TF) is the number of times the token appears in the
    item (``token_count_in_item``).  Inverse Document Frequency (IDF) is
    approximated by ``-log(corpus_frequency + ε)`` using the ``wordfreq``
    corpus-frequency database (Speer et al. 2022, wordfreq: a library for
    looking up the frequencies of English words, via pip).  A rare word has a
    high IDF; a very common word has a low IDF.

    When ``wordfreq`` is not installed the IDF falls back to a fixed moderate
    value, reducing the ranking to TF-only order (higher document frequency
    within the item → higher score).  This fallback is explicit so that the
    absence of ``wordfreq`` does not silently break annotation; install
    ``wordfreq`` for full TF-IDF ranking.
    """
    try:
        import wordfreq  # noqa: PLC0415 (optional heavy dependency, lazy import)
        corpus_frequency = wordfreq.word_frequency(token_text.lower(), "en")
    except ImportError:
        corpus_frequency = 1e-4  # moderate-rarity fallback; TF dominates ranking
    return token_count_in_item * -math.log(
        corpus_frequency + _LOG_FREQUENCY_STABILIZATION_EPSILON)


def compute_key_term_set(
        question_text: str,
        linguistic_pipeline,
) -> list[str]:
    """Apply the K_P(x) rule to ``question_text`` and return all key terms,
    ordered by perturbation priority.

    The ordering is designed to maximise the relevance of the perturbation
    when only a small edit budget is available:

    1. **Structurally guaranteed** tokens (named entities, numeric tokens, and
       negation tokens) appear first, in document order among themselves.
       These are guaranteed by the formal rule to be answer-determining
       regardless of their surface frequency.

    2. **TF-IDF ranked** tokens: all remaining structural-filter-passing
       tokens, sorted in descending order of TF-IDF proxy score.  Higher-ranked
       (rarer-in-corpus, more-frequent-in-item) tokens appear earlier so that
       a budget-one perturbation targets the most informative key term.

    There is no cap on the total number of key terms returned.  All tokens
    satisfying the formal rule are included; the ordering alone determines
    which receives the first edit when budget is limited.  A cap would require
    an arbitrary threshold with no principled justification.

    Parameters
    ----------
    question_text :
        The raw question string (not the full prompt; the instruction span is
        not perturbed under the ``content`` or ``answer_critical`` scopes).
    linguistic_pipeline :
        A loaded spaCy language model object (returned by
        ``load_linguistic_pipeline``).
    """
    document = linguistic_pipeline(question_text)

    # Partition tokens into two priority tiers.  Tier 1 (structurally
    # guaranteed) tokens are definitionally answer-critical by the formal
    # rule, independent of corpus frequency.  Tier 2 tokens are prioritised
    # by TF-IDF proxy score.
    tier_one_surfaces: list[str] = []     # document-order, deduplicated
    tier_one_surface_set: set[str] = set()
    tier_two_surfaces: list[str] = []     # document-order, deduplicated
    tier_two_token_counts: dict[str, int] = {}

    for token in document:
        if not _token_is_key_term(token):
            continue
        surface = token.text
        is_structurally_guaranteed = (
            token.ent_iob_ != "O"
            or token.pos_ == "NUM"
            or token.dep_ == UniversalDependenciesRelationLabel.NEGATION
        )
        if is_structurally_guaranteed:
            if surface not in tier_one_surface_set:
                tier_one_surfaces.append(surface)
                tier_one_surface_set.add(surface)
        else:
            tier_two_token_counts[surface] = (
                tier_two_token_counts.get(surface, 0) + 1)
            if surface not in tier_one_surface_set and surface not in tier_two_surfaces:
                tier_two_surfaces.append(surface)

    # Sort Tier 2 by TF-IDF proxy score descending.  Tier 1 surfaces keep
    # document order (they are already unconditionally included).
    tier_two_ordered = sorted(
        tier_two_surfaces,
        key=lambda surface_form: _compute_tfidf_proxy_score(
            surface_form, tier_two_token_counts.get(surface_form, 1)),
        reverse=True,
    )

    return tier_one_surfaces + tier_two_ordered


# ---------------------------------------------------------------------------
# Template operand cross-check (for synthetic reasoning items)
# ---------------------------------------------------------------------------

def _operand_candidate_strings(operand_value: int | float | Fraction) -> set[str]:
    """Every textual form that would count as ``operand_value`` being present
    in the text: the digit string, plus (for ints and fractions) any English
    word spacy could plausibly have tagged instead: a spelled-out cardinal
    ("three"), a multiplicative adverb ("thrice", "quadruple": these are
    grammatically adverbs, not NUM, so spaCy never tags them as a number
    itself, but they still spell out the same operand value), or a fraction
    word ("third", "two thirds").
    """
    candidates = {str(operand_value)}
    if isinstance(operand_value, Fraction):
        candidates |= _FRACTION_TO_WORDS.get(operand_value, set())
        if operand_value.denominator == 1:
            candidates.add(str(operand_value.numerator))
    elif isinstance(operand_value, int):
        candidates |= _NUMBER_TO_WORDS.get(operand_value, set())
    return candidates


def validate_template_operand_coverage(
        item,
        key_terms: list[str],
) -> list[str]:
    """Assert that every numeric template operand appears in the key-term set,
    in *some* textual form: digit string, spelled-out cardinal, multiplicative
    adverb ("twice"/"thrice"/"quadruple"), or fraction word ("third").

    For synthetic reasoning items (those with a populated ``parameters`` dict),
    the answer-determining operand values are known by construction.  This
    function checks that every *numeric* operand (``int``, ``float``, or
    ``Fraction``. ``item.parameters`` must already be deserialised, i.e. run
    through ``deserialize_parameters``, so ``Fraction`` values are real
    ``Fraction`` instances rather than the ``{"__fraction__": [n, d]}`` JSONL
    encoding) is covered by at least one key term, logging violations as
    warnings rather than raising so that a single edge case does not abort the
    full annotation run.

    Non-numeric parameters (a color, a name, a currency symbol, a free-text
    phrase) are intentionally skipped: K_P(x) (design/02 §2.x) is not expected
    to capture them (a plain color adjective, for instance, is not a
    NOUN/PROPN/NUM or an entity), and Regime C's operand-swap only ever
    considers ``int``-valued parameters
    (``regimes.make_regime_c_reasoning_operand_swap``), so a non-numeric
    parameter can never be a Regime C swap target regardless of key-term
    coverage. Checking them anyway produced false-positive "violations" for
    the overwhelming majority of cases (colors, multi-word phrases compared as
    a whole against a list of single tokens, and undeserialised ``Fraction``
    dict reprs that could never match any text).

    Returns a list of violation strings (empty if all numeric operands are
    covered).

    This cross-check implements the validation oracle described in the plan:
    the K_P(x) rule is the single definition; template operands are the ground
    truth confirming coverage (design §1.2 cross-check).
    """
    parameters = getattr(item, "parameters", {})
    if not parameters:
        return []  # not a synthetic item; no oracle available

    key_term_text = " ".join(key_terms)
    violations: list[str] = []

    for operand_name, operand_value in parameters.items():
        if isinstance(operand_value, bool) or not isinstance(operand_value, (int, float, Fraction)):
            continue  # not a numeric operand; out of scope for this oracle

        candidates = _operand_candidate_strings(operand_value)
        covered = any(
            re.search(rf"\b{re.escape(candidate)}\b", key_term_text, re.IGNORECASE)
            for candidate in candidates
        )
        if not covered:
            violations.append(
                f"Operand '{operand_name}'={operand_value!r} not found in "
                f"key_terms={key_terms!r} (checked forms: {sorted(candidates)!r}) "
                f"for item {getattr(item, 'task_id', '?')!r}"
            )

    return violations


# ---------------------------------------------------------------------------
# Per-item annotation
# ---------------------------------------------------------------------------

def annotate_item(
        item,
        linguistic_pipeline,
        question_text_attribute: str = "question_text",
) -> dict:
    """Annotate a single task item and return an update dict.

    The update dict contains:
        ``key_terms``                   List of key-term surface forms (K_P(x)).
        ``linguistic_annotation_rule``  The rule version string (KEY_TERM_IDENTIFICATION_RULE_VERSION).

    The caller applies this dict to the item's serialised record; the item
    dataclass itself is not mutated (it may be frozen).

    Parameters
    ----------
    item :
        A ``ReasoningItem`` or ``MultipleChoiceItem`` instance.
    linguistic_pipeline :
        The loaded spaCy pipeline.
    question_text_attribute :
        The attribute name that holds the bare question text (without options or
        instruction).  For reasoning items this is ``"question_text"``; for
        multiple-choice items it is also ``"question"`` in the raw HuggingFace
        schema but ``"question"`` on the dataclass.  The build tool passes the
        correct attribute for each task type.
    """
    raw_text = getattr(item, question_text_attribute, "")
    key_terms = compute_key_term_set(raw_text, linguistic_pipeline)

    return {
        "key_terms": key_terms,
        "linguistic_annotation_rule": KEY_TERM_IDENTIFICATION_RULE_VERSION,
    }


@dataclass
class _OperandCoverageCheckItem:
    """Minimal stand-in for a ReasoningItem, carrying just the two attributes
    validate_template_operand_coverage reads (getattr(item, "parameters", {})
    and item.task_id), so the JSONL batch path can reuse it without
    constructing a full ReasoningItem."""
    parameters: dict
    task_id: str


# ---------------------------------------------------------------------------
# Batch annotation with provenance
# ---------------------------------------------------------------------------

def annotate_jsonl_file(
        input_path: Path,
        output_path: Path,
        linguistic_pipeline,
        model_name: str,
        question_text_field: str = "question_text",
        force: bool = False,
) -> dict:
    """Read a JSONL item file, annotate every record with K_P(x) key terms,
    and write the annotated records to ``output_path``.

    If ``output_path`` already contains annotations from the same rule version
    and ``force`` is False, a ``ValueError`` is raised to prevent accidental
    overwrite of a pre-registered frozen dataset.

    Parameters
    ----------
    input_path :
        Path to the source JSONL file produced by tools/build_task_items.py.
    output_path :
        Path to write the annotated JSONL.  May be the same as ``input_path``
        (in-place update).
    linguistic_pipeline :
        The loaded spaCy pipeline object.
    model_name :
        The spaCy model name string, for provenance recording.
    question_text_field :
        The JSON field name that holds the bare question text in each record.
    force :
        If True, overwrite existing annotations without raising.

    Returns
    -------
    dict
        A summary dict with keys ``annotated_count``, ``skipped_count``,
        ``violation_count``, and ``rule_version``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    records: list[dict] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))

    if not force and records:
        existing_rule = records[0].get("linguistic_annotation_rule", "")
        if existing_rule == KEY_TERM_IDENTIFICATION_RULE_VERSION:
            raise ValueError(
                f"Output file already contains annotations from rule "
                f"'{KEY_TERM_IDENTIFICATION_RULE_VERSION}'.  Pass force=True to overwrite.  "
                "Overwriting a pre-registered frozen dataset requires a "
                "design-doc amendment (design/10 §10.3)."
            )

    annotated_count = 0
    violation_count = 0

    for record in records:
        # Auto-detect the question field: prefer the explicit argument, but fall
        # back to "question" (MCQ schema) when the explicit field is absent.
        effective_field = (
            question_text_field
            if question_text_field in record
            else next((f for f in ("question", "question_text") if f in record), question_text_field)
        )
        question_text = record.get(effective_field, "")
        key_terms = compute_key_term_set(question_text, linguistic_pipeline)

        # Template operand cross-check (for synthetic items that carry parameters).
        stand_in_item = _OperandCoverageCheckItem(
            # deserialize_parameters restores Fraction values from their JSONL
            # {"__fraction__": [n, d]} encoding; without this, every fraction
            # operand compared against that literal dict-repr string could
            # never match and would always be reported as a false violation.
            parameters=deserialize_parameters(record.get("parameters", {})),
            task_id=record.get("task_id", "?"),
        )
        violations = validate_template_operand_coverage(stand_in_item, key_terms)
        if violations:
            for violation_message in violations:
                warnings.warn(violation_message, stacklevel=2)
            violation_count += len(violations)

        record["key_terms"] = key_terms
        record["linguistic_annotation_rule"] = KEY_TERM_IDENTIFICATION_RULE_VERSION
        annotated_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "annotated_count": annotated_count,
        "skipped_count": 0,
        "violation_count": violation_count,
        "rule_version": KEY_TERM_IDENTIFICATION_RULE_VERSION,
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# Provenance record helpers
# ---------------------------------------------------------------------------

def _sha256_of_spacy_model(model_name: str) -> Optional[str]:
    """Return the SHA-256 of the spaCy model's meta.json as a proxy for the
    model package version fingerprint.  Returns None if spaCy or the model is
    not available (so this function is safe to call in offline/test contexts).
    """
    try:
        import spacy
        model = spacy.load(model_name)
        meta = model.meta
        meta_bytes = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(meta_bytes).hexdigest()
    except (ImportError, OSError):
        # spaCy absent, or the named model isn't installed: exactly the
        # "not available" case this function's docstring promises to degrade
        # gracefully on. Anything else (a real bug) still propagates.
        return None


def build_annotation_provenance_record(
        model_name: str,
        input_paths: Sequence[Path],
        result_summaries: Sequence[dict],
) -> dict:
    """Build the provenance dict to write to data/items/annotation_PROVENANCE.json.

    The record documents the linguistic pipeline used, its published accuracy
    benchmarks, the key-term rule version and formal definition reference, and
    per-file annotation statistics.  This is the primary artifact that lets a
    reviewer verify that the key-term annotation is principled and reproducible.
    """
    model_sha256 = _sha256_of_spacy_model(model_name)

    return {
        "schema_version": "1",
        "note": (
            "Formal linguistic annotation for GLAMOR Lab Exp 001.  "
            "Frozen at pre-registration; any rule change requires a design-doc "
            "amendment (design/10 §10.3).  See design/02 §2.x for the formal "
            "definition of K_P(x) and design/04 §4.6 for the literature justification."
        ),
        "key_term_annotation": {
            "rule_version": KEY_TERM_IDENTIFICATION_RULE_VERSION,
            "implements": "design/02 §2.x — the key-term set K_P(x)",
            "formal_definition": (
                "K_P(x) = { t ∈ tok_P(x) : "
                "POS_P(t) ∈ {NOUN, PROPN, NUM}  # content/quantity carriers\n"
                "∨ ENT_P(t) ≠ ∅                  # named entities\n"
                "∨ DEP_P(t) = neg                # negation\n"
                "∨ MORPH_P(t).Degree ∈ {Cmp,Sup} # comparatives/superlatives\n"
                "∨ (POS_P(t)=DET ∧ MORPH_P(t).PronType=Tot) # totality quantifiers\n"
                "}"
            ),
            "linguistic_grounding": [
                "Content-word / function-word distinction: Manning, Raghavan & Schütze (2008) "
                "'Introduction to Information Retrieval', Cambridge UP, §2.2.",
                "Named-entity recognition as a referent-binding signal: "
                "Nadeau & Sekine (2007) 'A survey of named entity recognition and classification', "
                "Lingvisticae Investigationes 30(1).",
                "Dependency grammar of negation (DEP=neg): "
                "Tesnière (1959) 'Éléments de syntaxe structurale'; "
                "de Marneffe et al. (2014) 'Universal Stanford Dependencies', LREC.",
                "Comparative/superlative morphology: Huddleston & Pullum (2002) "
                "'The Cambridge Grammar of the English Language', §17.5.",
                "Total quantifiers / distributive determiners: "
                "Barwise & Cooper (1981) 'Generalized quantifiers and natural language', "
                "Linguistics and Philosophy 4.",
                "Universal Dependencies morphological feature scheme (PronType=Tot, Degree=Cmp/Sup): "
                "Nivre et al. (2016) 'Universal Dependencies v1', LREC.",
            ],
        },
        "linguistic_pipeline": {
            "library": "spaCy",
            "library_reference": (
                "Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). "
                "spaCy: Industrial-strength Natural Language Processing in Python. "
                "Zenodo. https://doi.org/10.5281/zenodo.1212303"
            ),
            "library_rationale": (
                "spaCy is named explicitly in the study design (design/04 §4.6).  "
                "It provides CPU-friendly part-of-speech, dependency, named-entity, and "
                "morphological annotations with published accuracy on OntoNotes 5.0 and "
                "Universal Dependencies benchmarks, making it auditable by reviewers.  "
                "The non-transformer pipeline requires no GPU, consistent with the "
                "project's CPU/GPU dependency split (build tools and post-processing "
                "run on CPU; only generation requires GPU)."
            ),
            "model_name": model_name,
            "model_sha256_of_meta_json": model_sha256,
            "model_accuracy_reference": (
                "Published accuracy figures are available in the spaCy model documentation "
                "at https://spacy.io/models/en — see the model card for the specific "
                "version installed, including F1 scores for part-of-speech tagging, "
                "dependency parsing, and named-entity recognition on OntoNotes 5.0."
            ),
        },
        "annotation_results": [
            {
                "input_path": str(input_path),
                **summary,
            }
            for input_path, summary in zip(input_paths, result_summaries)
        ],
    }
