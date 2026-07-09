"""The (str, Enum) contract every consumer (JSONL serialization, config
parsing, frozenset membership) depends on, plus the domain-specific
vocabulary facts (which TaskFamily is reasoning vs. MCQ, the four-way
ParseStatus taxonomy, ...).
"""

from __future__ import annotations

import json

import pytest

from enums import (
    Operation, SelectionPolicy, Scope, Unit, SemanticClass, TaskFamily,
    ParseStatus, FragmentationStratum, ConditionSource,
    Precision, Decoding,
    REASONING_FAMILIES, MCQ_FAMILIES, INTERACTIONAL_FAILURE_STATUSES,
)


_ALL_ENUM_CLASSES = [
    Operation, SelectionPolicy, Scope, Unit, SemanticClass, TaskFamily,
    ParseStatus, FragmentationStratum, ConditionSource,
    Precision, Decoding,
]


class TestStrEnumRoundTripContract:
    """Every value here is a (str, Enum): str()/equality/JSON all resolve to
    the plain string, and the plain string coerces back to the same member —
    the single mechanism every consumer relies on."""

    @pytest.mark.parametrize("enum_class", _ALL_ENUM_CLASSES, ids=lambda cls: cls.__name__)
    def test_every_member_satisfies_the_contract(self, enum_class):
        for member in enum_class:
            assert str(member) == member.value
            assert f"{member}" == member.value
            assert member == member.value
            assert member.value == member
            assert json.dumps(member) == json.dumps(member.value)
            assert enum_class(member.value) is member

    @pytest.mark.parametrize("enum_class", _ALL_ENUM_CLASSES, ids=lambda cls: cls.__name__)
    def test_an_unrecognised_value_raises(self, enum_class):
        with pytest.raises((ValueError, KeyError)):
            enum_class("__not_a_real_value__")

    def test_coercion_from_plain_string(self):
        assert Operation("substitute") is Operation.SUBSTITUTE
        assert ParseStatus("clarification") is ParseStatus.CLARIFICATION
        assert SemanticClass("A") is SemanticClass.A
        assert Precision("awq") is Precision.AWQ

    def test_json_roundtrip_inside_a_dict(self):
        payload = {
            "semantic_class": SemanticClass.A,
            "parse_status": ParseStatus.VALID,
            "operation": Operation.SUBSTITUTE,
        }
        reloaded = json.loads(json.dumps(payload))
        assert reloaded == {"semantic_class": "A", "parse_status": "valid", "operation": "substitute"}


class TestTaskFamilyFrozensetMembership:

    def test_reasoning_families_contains_only_reasoning_task_families(self):
        assert REASONING_FAMILIES == {
            TaskFamily.GSM_SYMBOLIC_OFFICIAL, TaskFamily.GSM_SYMBOLIC_SYNTHETIC, TaskFamily.GSM8K}

    def test_mcq_families_contains_only_mcq_task_families(self):
        assert MCQ_FAMILIES == {TaskFamily.MMLU_PRO, TaskFamily.MMLU, TaskFamily.MCQ_DEMO}

    def test_reasoning_and_mcq_families_are_disjoint(self):
        assert REASONING_FAMILIES.isdisjoint(MCQ_FAMILIES)

    def test_plain_string_lookup_works_via_str_enum_equality(self):
        assert "gsm_symbolic_official" in REASONING_FAMILIES
        assert "mmlu_pro" in MCQ_FAMILIES


class TestParseStatusVocabulary:

    def test_covers_exactly_four_statuses(self):
        assert set(ParseStatus) == {
            ParseStatus.VALID, ParseStatus.UNPARSEABLE, ParseStatus.CLARIFICATION, ParseStatus.REFUSAL}

    def test_interactional_failure_statuses_excludes_only_valid(self):
        assert INTERACTIONAL_FAILURE_STATUSES == {
            ParseStatus.UNPARSEABLE, ParseStatus.CLARIFICATION, ParseStatus.REFUSAL}
        assert ParseStatus.VALID not in INTERACTIONAL_FAILURE_STATUSES

    def test_plain_string_lookup_works_via_str_enum_equality(self):
        assert "unparseable" in INTERACTIONAL_FAILURE_STATUSES
        assert "valid" not in INTERACTIONAL_FAILURE_STATUSES


def test_semantic_class_covers_exactly_the_four_regimes():
    assert set(SemanticClass) == {SemanticClass.A, SemanticClass.B, SemanticClass.C, SemanticClass.CLEAN}


def test_operation_vocabulary_includes_every_primitive_and_the_none_sentinel():
    names = {member.name for member in Operation}
    assert {"SUBSTITUTE", "DELETE", "INSERT", "TRANSPOSE", "WORD_SUBSTITUTE", "NONE"} <= names


def test_every_selection_dimension_has_a_none_sentinel():
    assert Scope.NONE in Scope
    assert SelectionPolicy.NONE in SelectionPolicy
    assert Operation.NONE in Operation
