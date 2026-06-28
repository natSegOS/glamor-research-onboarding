"""Tests for the linguistic parse-status classifier (src/scoring.py).

The linguistic classifier (classify_parse_status_with_linguistic_pipeline)
is the formal post-stage mechanism for assigning CLARIFICATION and REFUSAL
status.  It uses spaCy dependency structure rather than phrase lists.

All tests run offline via _StubSentence and _StubDocument classes that
replicate the spaCy API surface the classifier uses.  No spaCy installation
is required to run this test suite.

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
"""

from __future__ import annotations

import re
from pathlib import Path

import scoring
from enums import ParseStatus


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
