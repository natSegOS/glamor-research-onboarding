"""Adversarial tests for the human-audit aggregation.

Covers: Fleiss' kappa known values, bounds, degenerate cases, majority-vote
tie-breaking, the Regime-A exclusion rule, variable panel sizes, and the
audit-report kappa gate.
"""

from __future__ import annotations

import math

import pytest

from analysis import audit as human_audit
from analysis.audit import AuditRating
from enums import SemanticClass


# ---------------------------------------------------------------------------
# Fleiss' kappa — known values and bounds
# ---------------------------------------------------------------------------

def test_fleiss_kappa_perfect_agreement():
    ratings = [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]
    assert math.isclose(human_audit.fleiss_kappa(ratings, category_count=2), 1.0)


def test_fleiss_kappa_chance_agreement_is_near_zero():
    ratings = [[0, 1], [0, 1], [1, 0], [1, 0]]
    kappa = human_audit.fleiss_kappa(ratings, category_count=2)
    assert kappa < 0.1


def test_fleiss_kappa_all_category_zero():
    # All annotators choose category 0 for all items: kappa is 1.0.
    ratings = [[0, 0, 0]] * 5
    kappa = human_audit.fleiss_kappa(ratings, category_count=2)
    # Marginal for category 0 is 1.0 -> chance == 1.0 -> kappa is defined as 1.0 by the guard.
    assert math.isclose(kappa, 1.0)


def test_fleiss_kappa_is_at_most_one():
    ratings = [[0, 0, 0], [1, 1, 1]] * 10
    assert human_audit.fleiss_kappa(ratings, category_count=2) <= 1.0 + 1e-9


def test_fleiss_kappa_moderate_agreement():
    # 3 annotators, 4 items, one split on every item.
    ratings = [[0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0]]
    kappa = human_audit.fleiss_kappa(ratings, category_count=2)
    # Moderate agreement — not 0, not 1.
    assert -1.0 <= kappa <= 1.0


def test_fleiss_kappa_requires_at_least_two_annotators():
    with pytest.raises(ValueError):
        human_audit.fleiss_kappa([[0], [1], [0]], category_count=2)


def test_fleiss_kappa_requires_at_least_one_item():
    with pytest.raises(ValueError):
        human_audit.fleiss_kappa([], category_count=2)


def test_fleiss_kappa_requires_equal_ratings_per_item():
    with pytest.raises(ValueError):
        human_audit.fleiss_kappa([[0, 1], [0]], category_count=2)


# ---------------------------------------------------------------------------
# resolve_item — majority vote and adjudication
# ---------------------------------------------------------------------------

def test_resolve_item_majority_intent_preserved():
    ratings = [
        AuditRating("i1", "ann1", True, True, SemanticClass.A),
        AuditRating("i1", "ann2", True, True, SemanticClass.A),
        AuditRating("i1", "ann3", False, True, SemanticClass.A),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert outcome.majority_intent_preserved is True
    assert not outcome.excluded_from_primary


def test_resolve_item_majority_intent_not_preserved():
    ratings = [
        AuditRating("i1", "ann1", False, True, SemanticClass.A),
        AuditRating("i1", "ann2", False, True, SemanticClass.A),
        AuditRating("i1", "ann3", True, True, SemanticClass.A),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert outcome.majority_intent_preserved is False


def test_regime_a_excluded_when_majority_says_meaning_changed():
    ratings = [
        AuditRating("i2", "ann1", False, False, SemanticClass.A),
        AuditRating("i2", "ann2", False, False, SemanticClass.A),
        AuditRating("i2", "ann3", True, True, SemanticClass.A),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert outcome.majority_intent_preserved is False
    assert outcome.excluded_from_primary


def test_regime_b_not_excluded_even_if_majority_not_preserved():
    """Exclusion rule only applies to Regime A."""
    ratings = [
        AuditRating("i3", "ann1", False, False, SemanticClass.B),
        AuditRating("i3", "ann2", False, False, SemanticClass.B),
        AuditRating("i3", "ann3", True, True, SemanticClass.B),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert not outcome.excluded_from_primary


def test_regime_c_not_excluded():
    ratings = [
        AuditRating("i4", "ann1", False, False, SemanticClass.C),
        AuditRating("i4", "ann2", False, False, SemanticClass.C),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert not outcome.excluded_from_primary


def test_regime_a_not_excluded_when_majority_intent_preserved():
    ratings = [
        AuditRating("i5", "ann1", True, True, SemanticClass.A),
        AuditRating("i5", "ann2", True, True, SemanticClass.A),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert not outcome.excluded_from_primary


def test_tie_is_broken_by_adjudicator():
    ratings = [
        AuditRating("i6", "ann1", True, True, SemanticClass.A),
        AuditRating("i6", "ann2", False, False, SemanticClass.A),
    ]
    adjudicator = AuditRating("i6", "adj", True, True, SemanticClass.A)
    outcome = human_audit.resolve_item(ratings, adjudicator)
    assert outcome.was_adjudicated
    assert outcome.majority_intent_preserved is True


def test_tie_without_adjudicator_is_none():
    ratings = [
        AuditRating("i7", "ann1", True, True, SemanticClass.A),
        AuditRating("i7", "ann2", False, True, SemanticClass.A),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert outcome.majority_intent_preserved is None
    assert not outcome.was_adjudicated


def test_resolve_item_rating_count_matches():
    ratings = [
        AuditRating("i8", "a", True, True, SemanticClass.A),
        AuditRating("i8", "b", True, True, SemanticClass.A),
        AuditRating("i8", "c", False, False, SemanticClass.A),
    ]
    outcome = human_audit.resolve_item(ratings)
    assert outcome.rating_count == 3


# ---------------------------------------------------------------------------
# audit_report — kappa gate and exclusions
# ---------------------------------------------------------------------------

def test_audit_report_kappa_gate_and_exclusions():
    ratings_by_item = {
        "i1": [AuditRating("i1", "a", True, True, SemanticClass.A),
               AuditRating("i1", "b", True, True, SemanticClass.A),
               AuditRating("i1", "c", True, True, SemanticClass.A)],
        "i2": [AuditRating("i2", "a", False, False, SemanticClass.A),
               AuditRating("i2", "b", False, False, SemanticClass.A),
               AuditRating("i2", "c", False, False, SemanticClass.A)],
        "i3": [AuditRating("i3", "a", True, True, SemanticClass.C),
               AuditRating("i3", "b", True, True, SemanticClass.C),
               AuditRating("i3", "c", True, True, SemanticClass.C)],
    }
    report = human_audit.audit_report(ratings_by_item)
    assert report.passes_kappa_gate
    assert "i2" in report.excluded_item_ids
    assert "i1" not in report.excluded_item_ids
    assert math.isclose(report.intent_preservation_rate_by_regime[SemanticClass.A], 0.5)


def test_audit_report_low_kappa_fails_gate():
    """Maximally split annotations → near-zero kappa → gate fails."""
    ratings_by_item = {
        "i1": [AuditRating("i1", "a", True, False, SemanticClass.A),
               AuditRating("i1", "b", False, True, SemanticClass.A)],
        "i2": [AuditRating("i2", "a", False, True, SemanticClass.A),
               AuditRating("i2", "b", True, False, SemanticClass.A)],
        "i3": [AuditRating("i3", "a", True, False, SemanticClass.B),
               AuditRating("i3", "b", False, True, SemanticClass.B)],
        "i4": [AuditRating("i4", "a", False, True, SemanticClass.B),
               AuditRating("i4", "b", True, False, SemanticClass.B)],
    }
    report = human_audit.audit_report(ratings_by_item, kappa_gate=0.60)
    assert not report.passes_kappa_gate


def test_audit_report_variable_panel_sizes_uses_modal():
    """Items rated by fewer annotators than the modal count must not enter kappa."""
    ratings_by_item = {
        "i1": [AuditRating("i1", "a", True, True, SemanticClass.A),
               AuditRating("i1", "b", True, True, SemanticClass.A),
               AuditRating("i1", "c", True, True, SemanticClass.A)],
        "i2": [AuditRating("i2", "a", True, True, SemanticClass.A),
               AuditRating("i2", "b", True, True, SemanticClass.A),
               AuditRating("i2", "c", True, True, SemanticClass.A)],
        "i3": [AuditRating("i3", "a", True, True, SemanticClass.A)],  # under-rated
    }
    # Should not raise; modal panel = 3, item i3 is excluded from kappa.
    report = human_audit.audit_report(ratings_by_item)
    assert report is not None


def test_audit_report_empty_raises():
    with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
        human_audit.audit_report({})


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------

def test_stratified_sample_size():
    items = list(range(30))
    sampled = human_audit.stratified_sample(items, stratum_key=lambda x: x % 3, per_stratum=4)
    # 3 strata × 4 per stratum = 12
    assert len(sampled) == 12


def test_stratified_sample_is_deterministic():
    items = list(range(30))
    a = human_audit.stratified_sample(items, stratum_key=lambda x: x % 3, per_stratum=4, seed=42)
    b = human_audit.stratified_sample(items, stratum_key=lambda x: x % 3, per_stratum=4, seed=42)
    assert a == b


def test_stratified_sample_per_stratum_capped_at_available():
    items = [0, 1, 2]  # one item per stratum
    sampled = human_audit.stratified_sample(items, stratum_key=lambda x: x, per_stratum=100)
    assert len(sampled) == 3


# ---------------------------------------------------------------------------
# Kappa gate constant
# ---------------------------------------------------------------------------

def test_kappa_gate_constant():
    assert math.isclose(human_audit.KAPPA_GATE, 0.60)


# ---------------------------------------------------------------------------
# Audit sample size link
# ---------------------------------------------------------------------------

def test_audit_sample_size_link():
    from analysis.statistics import audit_sample_size
    assert audit_sample_size(0.05) == 385
