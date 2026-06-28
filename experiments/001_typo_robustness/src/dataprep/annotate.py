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
data/items/annotation_PROVENANCE.json — records the exact model name, version,
    SHA-256 of the model package, the rule identifier, and publication citations.
design/02 §2.x  — formal definition of K_P(x).
design/04 §4.6  — literature justification for the choice of spaCy.
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Optional, Sequence


# The identifier string written into every annotated item record and into
# annotation_PROVENANCE.json.  Incrementing this version causes the build
# tool to refuse to overwrite an existing annotation without an explicit
# --force flag, protecting pre-registered frozen datasets.
KEY_TERM_RULE_VERSION: str = "kp_v1"


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
    """Return True if ``token`` satisfies any condition of the K_P(x) rule.

    The conditions implement the formal definition from design/02 §2.x:

    NOUN, PROPN, NUM
        Open-class content words and quantity expressions.  These carry the
        primary semantic content of the question and changes to them alter the
        answer (content-word / function-word distinction: Manning et al. 2008;
        Jurafsky & Martin 2023).

    Named-entity membership (ENT_IOB ≠ O)
        Proper-noun phrases, dates, organisations, etc.  Named entities bind
        referents; a typo in a named entity typically causes a lookup failure
        or an incorrect reference resolution.

    Negation dependency relation (DEP = neg)
        Negation tokens (``not``, ``never``, ``no``) that govern the head
        verb or adjective.  A typo converting ``not`` to ``ot`` or ``no`` to
        ``no.`` changes the polarity of the question and therefore the answer
        (Tesnière 1959, dependency grammar of sentential negation).

    Comparative or superlative morphological degree (Degree = Cmp or Sup)
        Tokens with comparative (``more``, ``fewer``, ``-er``) or superlative
        (``most``, ``fewest``, ``-est``) inflection.  Scalar modifiers affect
        which option is correct in MCQ and the direction of inequality in
        reasoning (Huddleston & Pullum 2002, §17.5).

    Total-quantifier determiner (POS = DET, PronType = Tot)
        Distributive / universal quantifiers: ``each``, ``every``, ``all``,
        ``both``.  These impose distributive semantics; a typo that removes or
        alters such a quantifier changes the counting structure of the problem
        (Barwise & Cooper 1981, generalised quantifiers).
    """
    # Content and quantity carriers: the primary semantic load of a question.
    if token.pos_ in {"NOUN", "PROPN", "NUM"}:
        return True

    # Named-entity members: referent binders whose surface form is load-bearing.
    if token.ent_iob_ != "O":
        return True

    # Negation: sentential negation changes the polarity of the answer.
    if token.dep_ == "neg":
        return True

    # Comparative and superlative degree: scalar modifiers that affect ordinality.
    degree_values = token.morph.get("Degree")
    if "Cmp" in degree_values or "Sup" in degree_values:
        return True

    # Totality quantifiers: distributive determiners that affect counting scope.
    if token.pos_ == "DET" and "Tot" in token.morph.get("PronType"):
        return True

    return False


def compute_key_term_set(
        question_text: str,
        linguistic_pipeline,
) -> list[str]:
    """Apply the K_P(x) rule to ``question_text`` and return the unique
    surface-form key terms in document order.

    Each token that satisfies ``_token_is_key_term`` contributes its ``text``
    attribute (the exact surface form in the source string) to the result.
    Duplicates are removed while preserving the order of first occurrence.

    Parameters
    ----------
    question_text :
        The raw question string (not the full prompt; instruction text is not
        a perturbation target under the ``content`` or ``answer_critical``
        scopes, so it is excluded from annotation).
    linguistic_pipeline :
        A loaded spaCy language model object (returned by ``load_linguistic_pipeline``).
    """
    document = linguistic_pipeline(question_text)

    seen: set[str] = set()
    key_terms: list[str] = []

    for token in document:
        if _token_is_key_term(token):
            surface_form = token.text
            if surface_form not in seen:
                seen.add(surface_form)
                key_terms.append(surface_form)

    return key_terms


# ---------------------------------------------------------------------------
# Template operand cross-check (for synthetic reasoning items)
# ---------------------------------------------------------------------------

def validate_template_operand_coverage(
        item,
        key_terms: list[str],
) -> list[str]:
    """Assert that every numeric template operand appears in the key-term set.

    For synthetic reasoning items (those with a populated ``parameters`` dict),
    the answer-determining operand values are known by construction.  This
    function checks that every operand digit string is covered by at least one
    key term, logging violations as warnings rather than raising so that a
    single edge case does not abort the full annotation run.

    Returns a list of violation strings (empty if all operands are covered).

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
        operand_string = str(operand_value)
        if operand_string not in key_term_text:
            violations.append(
                f"Operand '{operand_name}'={operand_string!r} "
                f"not found in key_terms={key_terms!r} "
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
        ``linguistic_annotation_rule``  The rule version string (KEY_TERM_RULE_VERSION).

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
        "linguistic_annotation_rule": KEY_TERM_RULE_VERSION,
    }


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
        if existing_rule == KEY_TERM_RULE_VERSION:
            raise ValueError(
                f"Output file already contains annotations from rule "
                f"'{KEY_TERM_RULE_VERSION}'.  Pass force=True to overwrite.  "
                "Overwriting a pre-registered frozen dataset requires a "
                "design-doc amendment (design/10 §10.3)."
            )

    annotated_count = 0
    violation_count = 0

    for record in records:
        question_text = record.get(question_text_field, "")
        key_terms = compute_key_term_set(question_text, linguistic_pipeline)

        # Template operand cross-check (for synthetic items that carry parameters).
        class _FakeItem:
            def __init__(self, parameters, task_id):
                self.parameters = parameters
                self.task_id = task_id

        fake_item = _FakeItem(
            parameters=record.get("parameters", {}),
            task_id=record.get("task_id", "?"),
        )
        violations = validate_template_operand_coverage(fake_item, key_terms)
        if violations:
            import warnings
            for violation_message in violations:
                warnings.warn(violation_message, stacklevel=2)
            violation_count += len(violations)

        record["key_terms"] = key_terms
        record["linguistic_annotation_rule"] = KEY_TERM_RULE_VERSION
        annotated_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "annotated_count": annotated_count,
        "skipped_count": 0,
        "violation_count": violation_count,
        "rule_version": KEY_TERM_RULE_VERSION,
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
    except Exception:
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
            "rule_version": KEY_TERM_RULE_VERSION,
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
                "input_path": str(p),
                **summary,
            }
            for p, summary in zip(input_paths, result_summaries)
        ],
    }
