"""Comprehensive tests for the enums module.

Verifies the (str, Enum) contract, JSON serialization, round-trip coercion,
frozenset membership, and that the pre-registered vocabulary is complete.
"""

from __future__ import annotations

import json

from enums import (
    Operation, SelectionPolicy, Scope, Unit, SemanticClass, TaskFamily,
    ParseStatus, FragmentationStratum, ReasoningSource, ConditionSource,
    Precision, Decoding,
    REASONING_FAMILIES, MCQ_FAMILIES, INTERACTIONAL_FAILURE_STATUSES,
)


# ---------------------------------------------------------------------------
# Enumerate all (str, Enum) classes for generic property tests.
# ---------------------------------------------------------------------------

_ALL_ENUM_CLASSES = [
    Operation, SelectionPolicy, Scope, Unit, SemanticClass, TaskFamily,
    ParseStatus, FragmentationStratum, ReasoningSource, ConditionSource,
    Precision, Decoding,
]


# --- Property: str(member) == member.value -----------------------------------

def test_str_returns_value_for_all_members():
    for cls in _ALL_ENUM_CLASSES:
        for member in cls:
            assert str(member) == member.value, (
                f"{cls.__name__}.{member.name}: str() returned {str(member)!r}, "
                f"expected {member.value!r}")


# --- Property: equality with plain string ------------------------------------

def test_member_equals_plain_string():
    for cls in _ALL_ENUM_CLASSES:
        for member in cls:
            assert member == member.value, (
                f"{cls.__name__}.{member.name} does not equal its plain string value")
            assert member.value == member, "equality is not symmetric"


# --- Property: JSON serialization produces the plain string ------------------

def test_json_dumps_produces_string_value():
    for cls in _ALL_ENUM_CLASSES:
        for member in cls:
            dumped = json.dumps(member)
            assert dumped == json.dumps(member.value), (
                f"{cls.__name__}.{member.name} JSON-serialized to {dumped!r}, "
                f"expected {json.dumps(member.value)!r}")


def test_json_roundtrip_in_dict():
    payload = {
        "semantic_class": SemanticClass.A,
        "parse_status": ParseStatus.VALID,
        "operation": Operation.SUBSTITUTE,
    }
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)
    assert reloaded == {"semantic_class": "A", "parse_status": "valid", "operation": "substitute"}


# --- Property: round-trip via Enum(value) ------------------------------------

def test_round_trip_coercion():
    for cls in _ALL_ENUM_CLASSES:
        for member in cls:
            coerced = cls(member.value)
            assert coerced is member, (
                f"{cls.__name__}({member.value!r}) returned {coerced!r}, expected {member!r}")


def test_coercion_from_plain_string():
    assert Operation("substitute") is Operation.SUBSTITUTE
    assert ParseStatus("clarification") is ParseStatus.CLARIFICATION
    assert SemanticClass("A") is SemanticClass.A
    assert Precision("awq") is Precision.AWQ


def test_invalid_value_raises():
    import pytest
    for cls in _ALL_ENUM_CLASSES:
        with pytest.raises((ValueError, KeyError)):
            cls("__not_a_real_value__")


# --- f-string interpolation ---------------------------------------------------

def test_f_string_interpolation():
    assert f"{Operation.SUBSTITUTE}" == "substitute"
    assert f"{SemanticClass.A}" == "A"
    assert f"{ParseStatus.REFUSAL}" == "refusal"
    assert f"op={Operation.DELETE}" == "op=delete"


# --- frozenset membership contracts ------------------------------------------

def test_reasoning_families_membership():
    """REASONING_FAMILIES must contain exactly the reasoning task families."""
    assert TaskFamily.GSM_SYMBOLIC_OFFICIAL in REASONING_FAMILIES
    assert TaskFamily.GSM_SYMBOLIC_SYNTHETIC in REASONING_FAMILIES
    assert TaskFamily.GSM8K in REASONING_FAMILIES
    # MCQ families must NOT be in REASONING_FAMILIES
    assert TaskFamily.MMLU_PRO not in REASONING_FAMILIES
    assert TaskFamily.MMLU not in REASONING_FAMILIES
    assert TaskFamily.MCQ_DEMO not in REASONING_FAMILIES


def test_mcq_families_membership():
    """MCQ_FAMILIES must contain exactly the MCQ task families."""
    assert TaskFamily.MMLU_PRO in MCQ_FAMILIES
    assert TaskFamily.MMLU in MCQ_FAMILIES
    assert TaskFamily.MCQ_DEMO in MCQ_FAMILIES
    # Reasoning families must NOT be in MCQ_FAMILIES
    assert TaskFamily.GSM_SYMBOLIC_OFFICIAL not in MCQ_FAMILIES
    assert TaskFamily.GSM_SYMBOLIC_SYNTHETIC not in MCQ_FAMILIES
    assert TaskFamily.GSM8K not in MCQ_FAMILIES


def test_families_are_disjoint():
    assert REASONING_FAMILIES.isdisjoint(MCQ_FAMILIES)


def test_interactional_failure_statuses():
    assert ParseStatus.UNPARSEABLE in INTERACTIONAL_FAILURE_STATUSES
    assert ParseStatus.CLARIFICATION in INTERACTIONAL_FAILURE_STATUSES
    assert ParseStatus.REFUSAL in INTERACTIONAL_FAILURE_STATUSES
    assert ParseStatus.VALID not in INTERACTIONAL_FAILURE_STATUSES


# --- Specific vocabulary completeness ----------------------------------------

def test_operation_vocabulary():
    names = {m.name for m in Operation}
    assert {"SUBSTITUTE", "DELETE", "INSERT", "TRANSPOSE",
            "WORD_SUBSTITUTE", "ASR", "NONE"} <= names


def test_semantic_class_covers_all_regimes():
    assert {SemanticClass.A, SemanticClass.B, SemanticClass.C, SemanticClass.CLEAN} == set(SemanticClass)


def test_parse_status_four_way():
    assert {ParseStatus.VALID, ParseStatus.UNPARSEABLE,
            ParseStatus.CLARIFICATION, ParseStatus.REFUSAL} == set(ParseStatus)


def test_scope_sentinel_present():
    assert Scope.NONE in Scope
    assert SelectionPolicy.NONE in SelectionPolicy
    assert Operation.NONE in Operation


# --- Membership with plain string (via __eq__ inheritance) -------------------

def test_frozenset_lookup_with_plain_string():
    """(str, Enum) equality means a plain string can be found in a frozenset of members."""
    assert "gsm_symbolic_official" in REASONING_FAMILIES
    assert "mmlu_pro" in MCQ_FAMILIES
    assert "unparseable" in INTERACTIONAL_FAILURE_STATUSES
    assert "valid" not in INTERACTIONAL_FAILURE_STATUSES
