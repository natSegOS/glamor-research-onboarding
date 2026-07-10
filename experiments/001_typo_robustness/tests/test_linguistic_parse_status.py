"""spaCy-dependent linguistic analysis: the parse-status classifier
(src/scoring.py) and the K_P(x) key-term identification rule
(src/dataprep/annotate.py) — the two places this codebase reasons about
spaCy dependency/POS structure rather than phrase lists.

The linguistic parse-status classifier
(classify_parse_status_with_linguistic_pipeline) is the formal post-stage
mechanism for assigning CLARIFICATION and REFUSAL status, on top of the
inline structural classifier's VALID/UNPARSEABLE (see test_answer_scoring.py).

All tests run offline via stub classes that replicate the spaCy API surface
each piece of code uses. No spaCy installation is required to run this suite.

Test coverage:
  _sentence_is_interrogative — interrogative-sentence detection
  _sentence_expresses_first_person_refusal — first-person refusal detection
  classify_parse_status_with_linguistic_pipeline — full four-way taxonomy
  Phrase-file validation oracles — the literal phrases from
      tests/fixtures/clarification_phrases.txt and
      tests/fixtures/refusal_phrases.txt are cross-checked against the
      classifier.  Phrases that do not satisfy the structural criterion are
      documented as intentional under-detections (acceptable for the
      conservative ICR lower bound).
  _token_is_key_term / compute_key_term_set — the K_P(x) rule conditions.
  validate_template_operand_coverage — detecting uncovered template operands.
  load_reasoning_jsonl's GSM_SYMBOLIC backward-compatibility shim.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import scoring
from dataprep.annotate import (
    _token_is_key_term,
    compute_key_term_set,
    validate_template_operand_coverage,
)
from enums import ParseStatus
from tasks.reasoning import load_reasoning_jsonl, TaskFamily


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Stub spaCy API for offline testing
# ---------------------------------------------------------------------------

class _StubToken:
    """Minimal token stub matching the spaCy token API used by the classifier."""

    def __init__(
        self,
        text: str,
        dep_: str = "",
        is_punct: bool = False,
        is_space: bool = False,
        lemma_: str = "",
    ):
        self.text = text
        self.dep_ = dep_
        self.is_punct = is_punct
        self.is_space = is_space
        self.lemma_ = lemma_ or text


class _StubSentence:
    """A list of stub tokens that supports iteration."""

    def __init__(self, tokens: list[_StubToken]):
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


class _StubDocument:
    """Minimal document stub with a .sents iterator."""

    def __init__(self, sentences: list[_StubSentence]):
        self._sentences = sentences

    @property
    def sents(self):
        return iter(self._sentences)


def _document(*sentences: _StubSentence) -> _StubDocument:
    return _StubDocument(list(sentences))


def _sentence(*tokens: _StubToken) -> _StubSentence:
    return _StubSentence(list(tokens))


def _pipeline_for(*sentences: _StubSentence):
    """Return a callable stub that, when called with any text, returns the
    pre-built document.  This is the minimal spaCy pipeline API surface.
    """
    document = _document(*sentences)

    class _Stub:
        def __call__(self, _text: str) -> _StubDocument:
            return document

    return _Stub()


# ---------------------------------------------------------------------------
# _sentence_is_interrogative
# ---------------------------------------------------------------------------

class TestSentenceIsInterrogative:

    def test_sentence_ending_with_question_mark_is_interrogative(self):
        sentence = _sentence(
            _StubToken("What"),
            _StubToken("?", is_punct=True),
        )
        assert scoring._sentence_is_interrogative(sentence) is True

    def test_sentence_ending_with_period_is_not_interrogative(self):
        sentence = _sentence(
            _StubToken("Okay"),
            _StubToken(".", is_punct=True),
        )
        assert scoring._sentence_is_interrogative(sentence) is False

    def test_empty_sentence_is_not_interrogative(self):
        assert scoring._sentence_is_interrogative(_sentence()) is False

    def test_trailing_space_token_is_skipped(self):
        sentence = _sentence(
            _StubToken("Hmm"),
            _StubToken("?", is_punct=True),
            _StubToken(" ", is_space=True),
        )
        assert scoring._sentence_is_interrogative(sentence) is True

    def test_question_mark_in_middle_does_not_count(self):
        sentence = _sentence(
            _StubToken("Why", is_punct=False),
            _StubToken("?", is_punct=True),
            _StubToken("Anyway", is_punct=False),
        )
        assert scoring._sentence_is_interrogative(sentence) is False

    def test_exclamation_mark_is_not_interrogative(self):
        sentence = _sentence(
            _StubToken("No"),
            _StubToken("!", is_punct=True),
        )
        assert scoring._sentence_is_interrogative(sentence) is False


# ---------------------------------------------------------------------------
# _sentence_expresses_first_person_refusal
# ---------------------------------------------------------------------------

class TestSentenceExpressesFirstPersonRefusal:

    def _first_person_with_negation(self) -> _StubSentence:
        return _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("can"),
            _StubToken("n't", dep_="neg"),
            _StubToken("help"),
        )

    def test_first_person_nsubj_plus_neg_dep_is_refusal(self):
        assert scoring._sentence_expresses_first_person_refusal(
            self._first_person_with_negation()
        ) is True

    def test_we_nsubj_plus_neg_dep_is_refusal(self):
        sentence = _sentence(
            _StubToken("We", dep_="nsubj"),
            _StubToken("do"),
            _StubToken("n't", dep_="neg"),
            _StubToken("respond"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True

    def test_first_person_with_cannot_fused_form_is_refusal(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("cannot"),
            _StubToken("help"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True

    def test_first_person_with_inability_adjective_is_refusal(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("am"),
            _StubToken("unable", lemma_="unable"),
            _StubToken("to"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True

    def test_first_person_with_incapable_is_refusal(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("am"),
            _StubToken("incapable", lemma_="incapable"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True

    def test_first_person_with_refuse_root_is_refusal(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("refuse", dep_="ROOT", lemma_="refuse"),
            _StubToken("to"),
            _StubToken("respond"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True

    def test_first_person_with_decline_root_is_refusal(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("decline", dep_="ROOT", lemma_="decline"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True

    def test_third_person_subject_with_negation_is_not_refusal(self):
        sentence = _sentence(
            _StubToken("The", dep_="det"),
            _StubToken("system", dep_="nsubj"),
            _StubToken("does"),
            _StubToken("n't", dep_="neg"),
            _StubToken("support"),
            _StubToken("this"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is False

    def test_first_person_no_negation_or_inability_is_not_refusal(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubj"),
            _StubToken("will"),
            _StubToken("answer"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is False

    def test_empty_sentence_is_not_refusal(self):
        assert scoring._sentence_expresses_first_person_refusal(_sentence()) is False

    def test_nsubjpass_counts_as_first_person_subject(self):
        sentence = _sentence(
            _StubToken("I", dep_="nsubjpass"),
            _StubToken("am"),
            _StubToken("unable", lemma_="unable"),
        )
        assert scoring._sentence_expresses_first_person_refusal(sentence) is True


# ---------------------------------------------------------------------------
# classify_parse_status_with_linguistic_pipeline — full four-way taxonomy
# ---------------------------------------------------------------------------

class TestClassifyParseStatusWithLinguisticPipeline:

    def test_valid_when_parsed_answer_is_present(self):
        pipeline = _pipeline_for()
        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "The answer is 19.", "19.0", pipeline)
        assert status == ParseStatus.VALID

    def test_valid_does_not_invoke_pipeline_for_present_answer(self):
        """When a parsed answer is present, the pipeline is never called."""
        calls = []

        class _TrackingPipeline:
            def __call__(self, text):
                calls.append(text)
                return _StubDocument([])

        scoring.classify_parse_status_with_linguistic_pipeline(
            "19", "19.0", _TrackingPipeline())
        assert calls == []

    def test_clarification_detected_for_interrogative_sentence(self):
        pipeline = _pipeline_for(
            _sentence(
                _StubToken("Could"),
                _StubToken("you"),
                _StubToken("clarify"),
                _StubToken("?", is_punct=True),
            )
        )
        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "Could you clarify?", None, pipeline)
        assert status == ParseStatus.CLARIFICATION

    def test_refusal_detected_for_first_person_negation(self):
        pipeline = _pipeline_for(
            _sentence(
                _StubToken("I", dep_="nsubj"),
                _StubToken("ca"),
                _StubToken("n't", dep_="neg"),
                _StubToken("help"),
                _StubToken(".", is_punct=True),
            )
        )
        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "I can't help.", None, pipeline)
        assert status == ParseStatus.REFUSAL

    def test_unparseable_when_no_structural_marker_and_no_answer(self):
        pipeline = _pipeline_for(
            _sentence(
                _StubToken("Hmm"),
                _StubToken(".", is_punct=True),
            )
        )
        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "Hmm.", None, pipeline)
        assert status == ParseStatus.UNPARSEABLE

    def test_clarification_takes_precedence_over_refusal(self):
        """A sentence with both a '?' and a first-person refusal clause is
        classified as CLARIFICATION because the interrogative check runs first.
        """
        pipeline = _pipeline_for(
            _sentence(
                _StubToken("I", dep_="nsubj"),
                _StubToken("ca"),
                _StubToken("n't", dep_="neg"),
                _StubToken("help"),
                _StubToken("?", is_punct=True),
            )
        )
        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "I can't help?", None, pipeline)
        assert status == ParseStatus.CLARIFICATION

    def test_refusal_only_when_no_parsed_answer(self):
        """A first-person refusal clause with a parseable answer is VALID."""
        pipeline = _pipeline_for(
            _sentence(
                _StubToken("I", dep_="nsubj"),
                _StubToken("ca"),
                _StubToken("n't", dep_="neg"),
                _StubToken("be"),
                _StubToken("sure"),
            )
        )
        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "I can't be sure, but #### 19", "19.0", pipeline)
        assert status == ParseStatus.VALID


# ---------------------------------------------------------------------------
# Phrase-file validation oracles
#
# Load the frozen phrase files from tests/fixtures/ and cross-check the
# STRUCTURAL subset against the linguistic classifier.  Phrases that satisfy
# the structural criterion are asserted to trigger the expected status;
# phrases that don't satisfy it are documented as intentional under-detections.
#
# The phrase files are retained for auditability and cross-checking per the
# plan (Part 3 reconciliation note); they are not a runtime mechanism.
# ---------------------------------------------------------------------------

def _load_phrase_list(filename: str) -> list[str]:
    """Load non-comment, non-blank lines from a fixture phrase file."""
    path = _FIXTURES_DIR / filename
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _split_contractions(word: str) -> list[str]:
    """Split English negative contractions as spaCy does.

    spaCy splits "can't" → ["ca", "n't"], "won't" → ["wo", "n't"], etc.
    This stub replicates only the negative-contraction splits relevant to the
    first-person refusal detector.
    """
    lower = word.lower().rstrip(".,!")
    contraction_splits = {
        "can't": ("ca", "n't"), "won't": ("wo", "n't"), "don't": ("do", "n't"),
        "doesn't": ("does", "n't"), "didn't": ("did", "n't"),
        "isn't": ("is", "n't"), "aren't": ("are", "n't"),
        "wasn't": ("was", "n't"), "weren't": ("were", "n't"),
        "couldn't": ("could", "n't"), "wouldn't": ("would", "n't"),
        "shouldn't": ("should", "n't"), "haven't": ("have", "n't"),
        "hasn't": ("has", "n't"), "needn't": ("need", "n't"),
        "i'm": ("I", "'m"), "i'll": ("I", "'ll"),
        "i've": ("I", "'ve"), "i'd": ("I", "'d"),
    }
    if lower in contraction_splits:
        return list(contraction_splits[lower])
    return [word]


def _stub_pipeline_from_text(text: str):
    """Build a stub pipeline by heuristically tokenizing the text.

    This is not a real parser — it is used to verify that the STRUCTURAL
    properties (ends with ?, has first-person subject with dep_="neg") hold
    for the representative phrases tested in the oracle classes.  Handles
    the negative contractions that spaCy splits into two tokens.
    """
    sentences: list[_StubSentence] = []
    for raw_sentence in re.split(r"[.!?]", text):
        raw_sentence = raw_sentence.strip()
        if not raw_sentence:
            continue

        ends_with_question = text.rstrip().endswith("?")

        tokens: list[_StubToken] = []
        for raw_word in raw_sentence.split():
            for word in _split_contractions(raw_word):
                clean = word.lower().rstrip("?.,!")
                dep = ""
                lemma = clean
                if clean in {"i", "we"}:
                    dep = "nsubj"
                elif clean in {"n't", "not"}:
                    dep = "neg"
                elif clean == "cannot":
                    lemma = "cannot"
                elif clean in {"unable", "unwilling", "incapable"}:
                    lemma = clean
                elif clean in {"refuse", "decline", "apologize"}:
                    dep = "ROOT"
                    lemma = clean
                tokens.append(_StubToken(word, dep_=dep, lemma_=lemma))

        if ends_with_question:
            tokens.append(_StubToken("?", is_punct=True))

        sentences.append(_StubSentence(tokens))

    return _pipeline_for(*sentences)


class TestClarificationPhraseOracles:
    """Validate the linguistic classifier against representative examples derived
    from clarification_phrases.txt (tests/fixtures/).

    The fixture file contains regex patterns (partial phrases for matching).
    These tests embed representative patterns in full-sentence contexts and verify:
    (a) interrogative forms (ending with '?') → CLARIFICATION, per the structural
        criterion (interrogative sentence detection).
    (b) non-interrogative forms (statements) → UNPARSEABLE, i.e. intentional
        conservative under-detection acceptable for the diagnostic ICR lower bound.
    (c) All forms score is_correct=0 regardless of the structural classification.

    The fixture is loaded as a smoke-check to verify it exists and is non-empty.
    """

    # Interrogative sentence forms that embed clarification-phrase patterns in
    # full-sentence context ending with '?'.  These are the cases the structural
    # criterion (ends with '?') correctly identifies.
    _INTERROGATIVE_FORMS = [
        "Did you mean France?",
        "Do you mean the city?",
        "Could you clarify the question?",
        "Can you clarify what you want?",
        "What do you mean exactly?",
        "Are you asking about the first world war?",
        "Which one do you mean?",
    ]

    # Non-interrogative statement forms that embed clarification-phrase patterns
    # but do NOT end with '?'.  These are intentional under-detections: the
    # structural classifier returns UNPARSEABLE, which still scores is_correct=0.
    _STATEMENT_FORMS = [
        "please clarify your intent",
        "not sure what you mean by that",
        "more context would help",
        "i need more information",
        "the question is unclear",
    ]

    def test_fixture_file_is_nonempty(self):
        phrases = _load_phrase_list("clarification_phrases.txt")
        assert len(phrases) > 10, "clarification_phrases.txt fixture seems too short"

    def test_interrogative_forms_trigger_clarification(self):
        for phrase in self._INTERROGATIVE_FORMS:
            pipeline = _stub_pipeline_from_text(phrase)
            status = scoring.classify_parse_status_with_linguistic_pipeline(
                phrase, None, pipeline)
            assert status == ParseStatus.CLARIFICATION, (
                f"interrogative phrase {phrase!r} was not classified as "
                f"CLARIFICATION (got {status!r})")

    def test_statement_forms_are_unparseable(self):
        """Non-interrogative clarification forms are conservatively UNPARSEABLE."""
        for phrase in self._STATEMENT_FORMS:
            pipeline = _stub_pipeline_from_text(phrase)
            status = scoring.classify_parse_status_with_linguistic_pipeline(
                phrase, None, pipeline)
            assert status == ParseStatus.UNPARSEABLE, (
                f"statement phrase {phrase!r} got unexpected status {status!r}; "
                "expected UNPARSEABLE (conservative under-detection by design)")

    def test_all_forms_score_is_correct_zero(self):
        """All clarification surface forms score is_correct=0 regardless of
        whether the structural classifier returns CLARIFICATION or UNPARSEABLE.
        """
        for phrase in self._INTERROGATIVE_FORMS + self._STATEMENT_FORMS:
            result = scoring.score_reasoning(phrase, 19)
            assert result.is_correct == 0, (
                f"clarification phrase {phrase!r} incorrectly scored is_correct=1")


class TestRefusalPhraseOracles:
    """Structural subset of refusal_phrases.txt must trigger REFUSAL.

    Phrases with first-person subject + negation/inability satisfy the
    structural criterion.  Phrases like 'as an AI' (no first-person negation
    structure) are intentional under-detections.
    """

    # Phrases from refusal_phrases.txt that satisfy the structural criterion:
    # first-person subject (I/we) + negation marker or inability lemma.
    _STRUCTURAL_REFUSAL_PHRASES = [
        "I can't help with this.",
        "I won't answer this question.",
        "I refuse to respond.",
        "I cannot solve this.",
        "I am unable to solve this.",
        "I'm unable to provide an answer.",
        "I'm unable to determine this.",
    ]

    def test_structural_refusal_phrases_trigger_refusal(self):
        for phrase in self._STRUCTURAL_REFUSAL_PHRASES:
            pipeline = _stub_pipeline_from_text(phrase)
            status = scoring.classify_parse_status_with_linguistic_pipeline(
                phrase, None, pipeline)
            assert status == ParseStatus.REFUSAL, (
                f"structural refusal phrase {phrase!r} was not classified as "
                f"REFUSAL (got {status!r})")

    def test_all_oracle_refusal_phrases_score_is_correct_zero(self):
        """All phrases in the validation oracle score is_correct=0, regardless
        of whether the classifier returns REFUSAL or UNPARSEABLE.
        """
        phrases = _load_phrase_list("refusal_phrases.txt")
        for phrase in phrases:
            result = scoring.score_reasoning(phrase, 19)
            assert result.is_correct == 0, (
                f"refusal oracle phrase {phrase!r} incorrectly scored "
                f"is_correct=1")


# ---------------------------------------------------------------------------
# K_P(x) key-term identification rule (src/dataprep/annotate.py).
#
# Stub classes here are prefixed `_KeyTerm...` — distinct from the
# `_Stub*`/`_pipeline_for` names above, which stub the parse-status
# classifier's dependency-parse API surface (a different shape: sentences of
# tokens with .dep_/.is_punct/.is_space) versus the key-term rule's API
# surface (a flat token sequence with .pos_/.ent_iob_/.morph).
# ---------------------------------------------------------------------------

class _KeyTermStubToken:
    """Minimal token stub matching the spaCy token API the key-term rule uses."""

    def __init__(
        self,
        text: str,
        pos_: str = "NOUN",
        dep_: str = "",
        ent_iob_: str = "O",
        degree: list | None = None,
        pron_type: list | None = None,
    ):
        self.text = text
        self.pos_ = pos_
        self.dep_ = dep_
        self.ent_iob_ = ent_iob_
        self._degree = degree or []
        self._pron_type = pron_type or []

    class _Morph:
        def __init__(self, degree, pron_type):
            self._degree = degree
            self._pron_type = pron_type

        def get(self, feature: str) -> list:
            if feature == "Degree":
                return self._degree
            if feature == "PronType":
                return self._pron_type
            return []

    @property
    def morph(self):
        return self._Morph(self._degree, self._pron_type)


class _KeyTermStubPipeline:
    """Minimal pipeline stub: returns a pre-built token sequence for any text."""

    def __init__(self, token_sequence: list[_KeyTermStubToken]):
        self._tokens = token_sequence

    def __call__(self, text: str):
        return self._tokens


def _key_term_pipeline_for(tokens: list[_KeyTermStubToken]) -> _KeyTermStubPipeline:
    return _KeyTermStubPipeline(tokens)


class TestTokenIsKeyTerm:
    """The K_P(x) rule conditions (design/04 §4.6): a truth table of every
    feature combination that must, or must not, mark a token as a key term.
    """

    @pytest.mark.parametrize("token_kwargs,expected", [
        (dict(text="France", pos_="NOUN"), True),
        (dict(text="France", pos_="PROPN"), True),
        (dict(text="42", pos_="NUM"), True),
        # Named-entity member (IOB tag B/I), even on an otherwise-plain POS.
        (dict(text="Paris", pos_="ADJ", ent_iob_="B"), True),
        (dict(text="not", pos_="PART", dep_="neg"), True),
        (dict(text="more", pos_="ADV", degree=["Cmp"]), True),
        (dict(text="most", pos_="ADV", degree=["Sup"]), True),
        (dict(text="each", pos_="DET", pron_type=["Tot"]), True),
        # Main verbs carry the question's operation ("costs", "earns",
        # "doubles") and are answer-critical by design — only copula/
        # auxiliary verbs are excluded, not verbs in general.
        (dict(text="runs", pos_="VERB"), True),
        # A preposition with no special features is not a key term.
        (dict(text="of", pos_="ADP"), False),
        (dict(text="and", pos_="CCONJ"), False),
        # Definite article "the" has PronType=Art, not Tot.
        (dict(text="the", pos_="DET", pron_type=["Art"]), False),
        # ent_iob_="O" means outside any entity; ADJ alone carries no
        # special feature.
        (dict(text="city", pos_="ADJ", ent_iob_="O"), False),
        (dict(text="has", pos_="VERB", dep_="aux"), False),
        (dict(text="is", pos_="VERB", dep_="cop"), False),
    ], ids=[
        "noun", "proper_noun", "numeral", "named_entity_member", "negation_dependent",
        "comparative_adjective", "superlative_adjective", "totality_quantifier_determiner",
        "non_auxiliary_verb", "plain_function_word", "conjunction", "non_total_determiner",
        "entity_outside_marker", "auxiliary_verb", "copula_verb",
    ])
    def test_key_term_rule_conditions(self, token_kwargs, expected):
        assert _token_is_key_term(_KeyTermStubToken(**token_kwargs)) is expected


class TestComputeKeyTermSet:
    """compute_key_term_set returns deduplicated key terms in document order."""

    def test_returns_key_terms_in_document_order(self):
        tokens = [
            _KeyTermStubToken("If", pos_="SCONJ"),
            # A real spaCy pipeline tags a country name as a named entity (GPE);
            # ent_iob_="B" here reflects that, so "France" gets tier-1
            # (structurally guaranteed) priority ahead of the TF-IDF-ranked
            # tier-2 tokens — the same priority a numeric operand or negation
            # gets, and for the same reason: it is answer-determining
            # regardless of corpus frequency.
            _KeyTermStubToken("France", pos_="PROPN", ent_iob_="B"),
            _KeyTermStubToken("has", pos_="AUX"),
            _KeyTermStubToken("50", pos_="NUM"),
            _KeyTermStubToken("cities", pos_="NOUN"),
        ]
        key_terms = compute_key_term_set("If France has 50 cities", _key_term_pipeline_for(tokens))
        assert key_terms == ["France", "50", "cities"]

    def test_deduplicates_repeated_key_terms(self):
        tokens = [
            _KeyTermStubToken("dogs", pos_="NOUN"),
            _KeyTermStubToken("and", pos_="CCONJ"),
            _KeyTermStubToken("dogs", pos_="NOUN"),  # duplicate
        ]
        key_terms = compute_key_term_set("dogs and dogs", _key_term_pipeline_for(tokens))
        assert key_terms == ["dogs"]

    def test_empty_text_returns_empty_list(self):
        assert compute_key_term_set("", _key_term_pipeline_for([])) == []

    def test_two_calls_with_the_same_input_are_identical(self):
        tokens = [
            _KeyTermStubToken("How", pos_="ADV"),
            _KeyTermStubToken("many", pos_="ADJ"),
            _KeyTermStubToken("apples", pos_="NOUN"),
        ]
        pipeline = _key_term_pipeline_for(tokens)
        assert (compute_key_term_set("How many apples", pipeline)
                == compute_key_term_set("How many apples", pipeline))

    def test_negation_token_is_included(self):
        tokens = [
            _KeyTermStubToken("is", pos_="AUX"),
            _KeyTermStubToken("not", pos_="PART", dep_="neg"),
            _KeyTermStubToken("true", pos_="ADJ"),
        ]
        assert "not" in compute_key_term_set("is not true", _key_term_pipeline_for(tokens))

    def test_totality_quantifier_is_included(self):
        # A totality-quantifier DET is tier-2 (TF-IDF-ranked), not tier-1
        # (structurally guaranteed document order is reserved for named
        # entities, numerals, and negation — see is_structurally_guaranteed
        # in compute_key_term_set) — so only inclusion, not position, is
        # guaranteed here.
        tokens = [
            _KeyTermStubToken("each", pos_="DET", pron_type=["Tot"]),
            _KeyTermStubToken("student", pos_="NOUN"),
        ]
        key_terms = compute_key_term_set("each student", _key_term_pipeline_for(tokens))
        assert set(key_terms) == {"each", "student"}

    def test_entity_member_verb_is_included(self):
        # A token with pos_=VERB that happens to be inside a named entity
        # (unusual but theoretically possible for multi-word entities) is included.
        token = _KeyTermStubToken("United", pos_="VERB", ent_iob_="B")
        assert "United" in compute_key_term_set("United", _key_term_pipeline_for([token]))


class TestTemplateOperandCoverage:
    """validate_template_operand_coverage detects uncovered operands."""

    class _FakeItem:
        def __init__(self, parameters, task_id="test_item"):
            self.parameters = parameters
            self.task_id = task_id

    def test_all_operands_covered_returns_no_violations(self):
        item = self._FakeItem({"a": 5, "b": 3})
        assert validate_template_operand_coverage(item, ["5", "3", "boxes"]) == []

    def test_missing_operand_returns_violation(self):
        item = self._FakeItem({"a": 99, "b": 3})
        violations = validate_template_operand_coverage(item, ["3", "boxes"])  # 99 missing
        assert len(violations) == 1
        assert "99" in violations[0]

    def test_no_parameters_returns_no_violations(self):
        assert validate_template_operand_coverage(self._FakeItem(parameters={}), ["anything"]) == []

    def test_item_without_parameters_attribute_returns_no_violations(self):
        # Non-synthetic items (from JSONL) have no parameters attribute.
        class _MinimalItem:
            task_id = "jsonl_item"
        assert validate_template_operand_coverage(_MinimalItem(), ["anything"]) == []


class TestGsmSymbolicBackwardCompatShim:
    """load_reasoning_jsonl re-tags the legacy 'gsm_symbolic' task family string."""

    @pytest.mark.parametrize("task_family,expected", [
        ("gsm_symbolic", TaskFamily.GSM_SYMBOLIC_OFFICIAL),           # legacy tag, re-tagged
        ("gsm_symbolic_official", TaskFamily.GSM_SYMBOLIC_OFFICIAL),  # current tag, unchanged
        ("gsm8k", TaskFamily.GSM8K),                                  # unrelated tag, unchanged
    ], ids=["legacy_retagged", "current_tag_passthrough", "gsm8k_passthrough"])
    def test_task_family_tag_resolution(self, tmp_path, task_family, expected):
        jsonl_file = tmp_path / "item.jsonl"
        jsonl_file.write_text(
            f'{{"task_id": "t0", "task_family": "{task_family}", "source": "{task_family}", '
            '"question_text": "Solve 2+2.", "instruction": "Show your work.", '
            '"gold_answer": 4, "key_terms": [], "parameters": {}}\n',
            encoding="utf-8",
        )
        items = load_reasoning_jsonl(jsonl_file)
        assert items[0].task_family == expected
