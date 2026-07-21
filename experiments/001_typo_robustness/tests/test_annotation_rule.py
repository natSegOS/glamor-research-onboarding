"""The pre-registered K_P(x) key-term annotation rule (design/04 §4.6).

Each test guards the frozen annotation contract: which tokens count as
answer-critical key terms, in what priority order, and the template-operand
oracle that cross-checks the rule against known ground truth. Breaking any of
them silently changes which words the informative_word and answer_critical
policies are allowed to touch — after the datasets were frozen against the
old rule.
"""

from __future__ import annotations

import pytest

from dataprep.annotate import (
    _token_is_key_term,
    compute_key_term_set,
    validate_template_operand_coverage,
)


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
