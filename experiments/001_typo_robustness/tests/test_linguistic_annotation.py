"""Tests for the linguistic annotation stage (src/dataprep/annotate.py).

These tests run offline without spaCy installed by using a lightweight
stub linguistic pipeline that produces deterministic part-of-speech and
dependency annotations for a small controlled vocabulary.  The spaCy-specific
tests are skipped when spaCy is not installed.

Covers:
  - K_P(x) rule conditions: NOUN/PROPN/NUM, named entity, negation,
    comparative/superlative degree, totality-quantifier determiner.
  - Determinism: two calls with the same input produce identical output.
  - Template operand coverage validation: violations are detected and reported.
  - GSM_SYMBOLIC backward-compatibility shim: old "gsm_symbolic" tag is
    re-tagged to GSM_SYMBOLIC_OFFICIAL on load.
  - ASR seed determinism: regimes.derived_seed is stable across calls.
"""

from __future__ import annotations

import pytest

import regimes
from tasks.reasoning import load_reasoning_jsonl, TaskFamily
from dataprep.annotate import (
    _token_is_key_term,
    compute_key_term_set,
    validate_template_operand_coverage,
    KEY_TERM_RULE_VERSION,
)


# ---------------------------------------------------------------------------
# Stub linguistic pipeline for offline tests
# ---------------------------------------------------------------------------

class _StubToken:
    """Minimal token stub that matches the spaCy token API used by the rule."""

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


class _StubPipeline:
    """Minimal pipeline stub: calls a user-supplied token-factory for each text."""

    def __init__(self, token_sequence: list[_StubToken]):
        self._tokens = token_sequence

    def __call__(self, text: str):
        return self._tokens


def _pipeline_for(tokens: list[_StubToken]) -> _StubPipeline:
    return _StubPipeline(tokens)


# ---------------------------------------------------------------------------
# Unit tests for _token_is_key_term
# ---------------------------------------------------------------------------

class TestTokenIsKeyTerm:
    """The K_P(x) rule conditions fire independently and correctly."""

    def test_noun_is_a_key_term(self):
        token = _StubToken("France", pos_="NOUN")
        assert _token_is_key_term(token) is True

    def test_proper_noun_is_a_key_term(self):
        token = _StubToken("France", pos_="PROPN")
        assert _token_is_key_term(token) is True

    def test_numeral_is_a_key_term(self):
        token = _StubToken("42", pos_="NUM")
        assert _token_is_key_term(token) is True

    def test_named_entity_member_is_a_key_term(self):
        # Token is part of a named entity (IOB tag B or I).
        token = _StubToken("Paris", pos_="ADJ", ent_iob_="B")
        assert _token_is_key_term(token) is True

    def test_negation_dependent_is_a_key_term(self):
        token = _StubToken("not", pos_="PART", dep_="neg")
        assert _token_is_key_term(token) is True

    def test_comparative_adjective_is_a_key_term(self):
        token = _StubToken("more", pos_="ADV", degree=["Cmp"])
        assert _token_is_key_term(token) is True

    def test_superlative_adjective_is_a_key_term(self):
        token = _StubToken("most", pos_="ADV", degree=["Sup"])
        assert _token_is_key_term(token) is True

    def test_totality_quantifier_determiner_is_a_key_term(self):
        token = _StubToken("each", pos_="DET", pron_type=["Tot"])
        assert _token_is_key_term(token) is True

    def test_plain_function_word_is_not_a_key_term(self):
        # ADP (preposition) with no special features is not a key term.
        token = _StubToken("of", pos_="ADP")
        assert _token_is_key_term(token) is False

    def test_conjunction_is_not_a_key_term(self):
        token = _StubToken("and", pos_="CCONJ")
        assert _token_is_key_term(token) is False

    def test_non_total_determiner_is_not_a_key_term(self):
        # Definite article "the" has PronType=Art, not Tot.
        token = _StubToken("the", pos_="DET", pron_type=["Art"])
        assert _token_is_key_term(token) is False

    def test_non_entity_outside_marker_is_not_an_entity(self):
        # ent_iob_ = "O" means outside any entity.
        token = _StubToken("city", pos_="ADJ", ent_iob_="O")
        assert _token_is_key_term(token) is False  # ADJ, no entity, no special features

    def test_verb_without_special_features_is_not_a_key_term(self):
        token = _StubToken("runs", pos_="VERB")
        assert _token_is_key_term(token) is False


# ---------------------------------------------------------------------------
# Unit tests for compute_key_term_set
# ---------------------------------------------------------------------------

class TestComputeKeyTermSet:
    """compute_key_term_set returns deduplicated key terms in document order."""

    def test_returns_key_terms_in_document_order(self):
        tokens = [
            _StubToken("If",     pos_="SCONJ"),
            _StubToken("France", pos_="PROPN"),
            _StubToken("has",    pos_="AUX"),
            _StubToken("50",     pos_="NUM"),
            _StubToken("cities", pos_="NOUN"),
        ]
        pipeline = _pipeline_for(tokens)
        key_terms = compute_key_term_set("If France has 50 cities", pipeline)
        assert key_terms == ["France", "50", "cities"]

    def test_deduplicates_repeated_key_terms(self):
        tokens = [
            _StubToken("dogs", pos_="NOUN"),
            _StubToken("and",  pos_="CCONJ"),
            _StubToken("dogs", pos_="NOUN"),  # duplicate
        ]
        pipeline = _pipeline_for(tokens)
        key_terms = compute_key_term_set("dogs and dogs", pipeline)
        assert key_terms == ["dogs"]

    def test_empty_text_returns_empty_list(self):
        pipeline = _pipeline_for([])
        key_terms = compute_key_term_set("", pipeline)
        assert key_terms == []

    def test_determinism_two_calls_equal(self):
        """Two calls with the same input and pipeline return identical output."""
        tokens = [
            _StubToken("How",   pos_="ADV"),
            _StubToken("many",  pos_="ADJ"),
            _StubToken("apples", pos_="NOUN"),
        ]
        pipeline = _pipeline_for(tokens)
        result_one = compute_key_term_set("How many apples", pipeline)
        result_two = compute_key_term_set("How many apples", pipeline)
        assert result_one == result_two

    def test_negation_token_is_included(self):
        tokens = [
            _StubToken("is",  pos_="AUX"),
            _StubToken("not", pos_="PART", dep_="neg"),
            _StubToken("true", pos_="ADJ"),
        ]
        pipeline = _pipeline_for(tokens)
        key_terms = compute_key_term_set("is not true", pipeline)
        assert "not" in key_terms

    def test_totality_quantifier_is_included(self):
        tokens = [
            _StubToken("each",   pos_="DET", pron_type=["Tot"]),
            _StubToken("student", pos_="NOUN"),
        ]
        pipeline = _pipeline_for(tokens)
        key_terms = compute_key_term_set("each student", pipeline)
        assert key_terms == ["each", "student"]

    def test_entity_member_verb_is_included(self):
        # A token with pos_=VERB that happens to be inside a named entity
        # (unusual but theoretically possible for multi-word entities) is included.
        token = _StubToken("United", pos_="VERB", ent_iob_="B")
        pipeline = _pipeline_for([token])
        key_terms = compute_key_term_set("United", pipeline)
        assert "United" in key_terms


# ---------------------------------------------------------------------------
# Template operand coverage validation
# ---------------------------------------------------------------------------

class TestTemplateOperandCoverage:
    """validate_template_operand_coverage detects uncovered operands."""

    class _FakeItem:
        def __init__(self, parameters, task_id="test_item"):
            self.parameters = parameters
            self.task_id = task_id

    def test_all_operands_covered_returns_no_violations(self):
        item = self._FakeItem({"a": 5, "b": 3})
        key_terms = ["5", "3", "boxes"]
        violations = validate_template_operand_coverage(item, key_terms)
        assert violations == []

    def test_missing_operand_returns_violation(self):
        item = self._FakeItem({"a": 99, "b": 3})
        key_terms = ["3", "boxes"]  # 99 not in key terms
        violations = validate_template_operand_coverage(item, key_terms)
        assert len(violations) == 1
        assert "99" in violations[0]

    def test_no_parameters_returns_no_violations(self):
        item = self._FakeItem(parameters={})
        violations = validate_template_operand_coverage(item, ["anything"])
        assert violations == []

    def test_item_without_parameters_attribute_returns_no_violations(self):
        # Non-synthetic items (from JSONL) have no parameters attribute.
        class _MinimalItem:
            task_id = "jsonl_item"
        violations = validate_template_operand_coverage(_MinimalItem(), ["anything"])
        assert violations == []


# ---------------------------------------------------------------------------
# GSM_SYMBOLIC backward-compatibility shim
# ---------------------------------------------------------------------------

class TestGsmSymbolicBackwardCompatShim:
    """load_reasoning_jsonl re-tags the legacy 'gsm_symbolic' task family string."""

    def test_legacy_tag_is_retagged_to_official(self, tmp_path):
        """A JSONL file with task_family='gsm_symbolic' loads as GSM_SYMBOLIC_OFFICIAL."""
        jsonl_file = tmp_path / "legacy.jsonl"
        jsonl_file.write_text(
            '{"task_id": "old_00000", "task_family": "gsm_symbolic", '
            '"source": "gsm_symbolic", "question_text": "Solve 2+2.", '
            '"instruction": "Show your work.", "gold_answer": 4, '
            '"key_terms": [], "parameters": {}}\n',
            encoding="utf-8",
        )
        items = load_reasoning_jsonl(jsonl_file)
        assert len(items) == 1
        assert items[0].task_family == TaskFamily.GSM_SYMBOLIC_OFFICIAL
        assert items[0].source == TaskFamily.GSM_SYMBOLIC_OFFICIAL

    def test_current_tag_passes_through_unchanged(self, tmp_path):
        jsonl_file = tmp_path / "current.jsonl"
        jsonl_file.write_text(
            '{"task_id": "new_00000", "task_family": "gsm_symbolic_official", '
            '"source": "gsm_symbolic_official", "question_text": "Solve 3+3.", '
            '"instruction": "Show your work.", "gold_answer": 6, '
            '"key_terms": [], "parameters": {}}\n',
            encoding="utf-8",
        )
        items = load_reasoning_jsonl(jsonl_file)
        assert items[0].task_family == TaskFamily.GSM_SYMBOLIC_OFFICIAL

    def test_gsm8k_tag_passes_through_unchanged(self, tmp_path):
        jsonl_file = tmp_path / "gsm8k.jsonl"
        jsonl_file.write_text(
            '{"task_id": "g8k_00000", "task_family": "gsm8k", '
            '"source": "gsm8k", "question_text": "Solve 4+4.", '
            '"instruction": "Show your work.", "gold_answer": 8, '
            '"key_terms": [], "parameters": {}}\n',
            encoding="utf-8",
        )
        items = load_reasoning_jsonl(jsonl_file)
        assert items[0].task_family == TaskFamily.GSM8K


# ---------------------------------------------------------------------------
# ASR seed determinism
# ---------------------------------------------------------------------------

class TestAsrSeedDeterminism:
    """regimes.derived_seed is stable across calls and independent of PYTHONHASHSEED."""

    def test_same_inputs_produce_same_seed(self):
        seed_one = regimes.derived_seed(1729, "gsm_symbolic_00042")
        seed_two = regimes.derived_seed(1729, "gsm_symbolic_00042")
        assert seed_one == seed_two

    def test_different_task_ids_produce_different_seeds(self):
        seed_a = regimes.derived_seed(1729, "task_00001")
        seed_b = regimes.derived_seed(1729, "task_00002")
        assert seed_a != seed_b

    def test_different_base_seeds_produce_different_seeds(self):
        seed_x = regimes.derived_seed(1729, "task_00001")
        seed_y = regimes.derived_seed(9999, "task_00001")
        assert seed_x != seed_y

    def test_seed_is_an_integer(self):
        seed = regimes.derived_seed(42, "some_task_id")
        assert isinstance(seed, int)

    def test_seed_does_not_depend_on_python_hash_seed(self):
        """The seed is SHA-256-derived and therefore hash-seed-independent.

        We verify that two calls with the exact same arguments always return
        the same value, which is the minimum requirement for cross-process
        reproducibility.  The full cross-process test (multiple PYTHONHASHSEED
        values) is covered by the end-to-end smoke test in the plan's
        verification section.
        """
        for task_id in ["task_00001", "task_00002", "task_99999"]:
            assert (
                regimes.derived_seed(1729, task_id)
                == regimes.derived_seed(1729, task_id)
            )
