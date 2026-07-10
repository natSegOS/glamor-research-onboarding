"""Human-audit aggregation (src/analysis/audit.py): Fleiss' kappa, the
majority-vote/adjudication resolution rule with Regime A's exclusion (the
only regime where a "meaning changed" majority drops the item from primary
analysis), the audit-report kappa gate, and stratified sampling.
"""

from __future__ import annotations

import math

import pytest

from analysis import audit as human_audit
from analysis.audit import AuditRating
from enums import SemanticClass


# ---------------------------------------------------------------------------
# Fleiss' kappa — known values, bounds, and error paths.
# ---------------------------------------------------------------------------

class TestFleissKappa:

    def test_perfect_agreement_gives_kappa_one(self):
        ratings = [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]
        assert math.isclose(human_audit.fleiss_kappa(ratings, category_count=2), 1.0)

    def test_chance_level_agreement_gives_kappa_near_zero(self):
        ratings = [[0, 1], [0, 1], [1, 0], [1, 0]]
        assert human_audit.fleiss_kappa(ratings, category_count=2) < 0.1

    def test_single_category_marginal_is_defined_as_kappa_one(self):
        # Every annotator picks category 0 for every item: marginal is 1.0,
        # so chance agreement is 1.0 too; the guard defines kappa as 1.0.
        ratings = [[0, 0, 0]] * 5
        assert math.isclose(human_audit.fleiss_kappa(ratings, category_count=2), 1.0)

    def test_kappa_never_exceeds_one(self):
        ratings = [[0, 0, 0], [1, 1, 1]] * 10
        assert human_audit.fleiss_kappa(ratings, category_count=2) <= 1.0 + 1e-9

    def test_moderate_agreement_stays_within_bounds(self):
        ratings = [[0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0]]
        assert -1.0 <= human_audit.fleiss_kappa(ratings, category_count=2) <= 1.0

    @pytest.mark.parametrize("ratings", [
        [[0], [1], [0]],       # only one annotator per item
        [],                     # no items
        [[0, 1], [0]],          # unequal ratings per item
    ], ids=["needs_two_annotators", "needs_at_least_one_item", "needs_equal_ratings_per_item"])
    def test_malformed_input_raises(self, ratings):
        with pytest.raises(ValueError):
            human_audit.fleiss_kappa(ratings, category_count=2)


# ---------------------------------------------------------------------------
# resolve_item — majority vote, adjudication, and the Regime-A exclusion rule.
# ---------------------------------------------------------------------------

class TestResolveItemMajorityVote:

    def test_majority_intent_preserved_is_true(self):
        ratings = [
            AuditRating("i1", "ann1", True, True, SemanticClass.A),
            AuditRating("i1", "ann2", True, True, SemanticClass.A),
            AuditRating("i1", "ann3", False, True, SemanticClass.A),
        ]
        outcome = human_audit.resolve_item(ratings)
        assert outcome.majority_intent_preserved is True
        assert not outcome.excluded_from_primary

    def test_majority_intent_preserved_is_false(self):
        ratings = [
            AuditRating("i1", "ann1", False, True, SemanticClass.A),
            AuditRating("i1", "ann2", False, True, SemanticClass.A),
            AuditRating("i1", "ann3", True, True, SemanticClass.A),
        ]
        assert human_audit.resolve_item(ratings).majority_intent_preserved is False

    def test_rating_count_matches_the_number_of_raters(self):
        ratings = [
            AuditRating("i8", "a", True, True, SemanticClass.A),
            AuditRating("i8", "b", True, True, SemanticClass.A),
            AuditRating("i8", "c", False, False, SemanticClass.A),
        ]
        assert human_audit.resolve_item(ratings).rating_count == 3


class TestRegimeExclusionRule:
    """The exclusion-from-primary-analysis rule applies to Regime A only:
    a majority "meaning changed" verdict drops the item; Regimes B and C are
    never excluded by this rule regardless of the verdict."""

    @pytest.mark.parametrize("semantic_class,majority_preserved,expected_excluded", [
        (SemanticClass.A, False, True),
        (SemanticClass.A, True, False),
        (SemanticClass.B, False, False),
        (SemanticClass.C, False, False),
    ], ids=["A_not_preserved_is_excluded", "A_preserved_is_not_excluded",
            "B_not_preserved_is_never_excluded", "C_not_preserved_is_never_excluded"])
    def test_exclusion_depends_only_on_regime_and_majority_verdict(
            self, semantic_class, majority_preserved, expected_excluded):
        ratings = [
            AuditRating("i", "ann1", majority_preserved, majority_preserved, semantic_class),
            AuditRating("i", "ann2", majority_preserved, majority_preserved, semantic_class),
        ]
        assert human_audit.resolve_item(ratings).excluded_from_primary == expected_excluded


class TestAdjudication:

    def test_a_tie_is_broken_by_the_adjudicator(self):
        ratings = [
            AuditRating("i6", "ann1", True, True, SemanticClass.A),
            AuditRating("i6", "ann2", False, False, SemanticClass.A),
        ]
        adjudicator = AuditRating("i6", "adj", True, True, SemanticClass.A)
        outcome = human_audit.resolve_item(ratings, adjudicator)
        assert outcome.was_adjudicated
        assert outcome.majority_intent_preserved is True

    def test_a_tie_without_an_adjudicator_is_unresolved(self):
        ratings = [
            AuditRating("i7", "ann1", True, True, SemanticClass.A),
            AuditRating("i7", "ann2", False, True, SemanticClass.A),
        ]
        outcome = human_audit.resolve_item(ratings)
        assert outcome.majority_intent_preserved is None
        assert not outcome.was_adjudicated


# ---------------------------------------------------------------------------
# audit_report — the kappa gate and item exclusions.
# ---------------------------------------------------------------------------

class TestAuditReport:

    def test_kappa_gate_pass_and_regime_a_exclusion(self):
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

    def test_low_kappa_fails_the_gate(self):
        # Maximally split annotations across every item -> near-zero kappa.
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

    def test_items_rated_below_the_modal_panel_size_are_excluded_from_kappa(self):
        ratings_by_item = {
            "i1": [AuditRating("i1", "a", True, True, SemanticClass.A),
                   AuditRating("i1", "b", True, True, SemanticClass.A),
                   AuditRating("i1", "c", True, True, SemanticClass.A)],
            "i2": [AuditRating("i2", "a", True, True, SemanticClass.A),
                   AuditRating("i2", "b", True, True, SemanticClass.A),
                   AuditRating("i2", "c", True, True, SemanticClass.A)],
            "i3": [AuditRating("i3", "a", True, True, SemanticClass.A)],  # under-rated
        }
        # Must not raise; modal panel size is 3, so i3 is excluded from kappa.
        assert human_audit.audit_report(ratings_by_item) is not None

    def test_empty_input_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
            human_audit.audit_report({})


# ---------------------------------------------------------------------------
# stratified_sample.
# ---------------------------------------------------------------------------

class TestStratifiedSample:

    def test_sample_size_is_strata_count_times_per_stratum(self):
        items = list(range(30))
        sampled = human_audit.stratified_sample(items, stratum_key=lambda x: x % 3, per_stratum=4)
        assert len(sampled) == 12  # 3 strata x 4 per stratum

    def test_is_deterministic_given_a_seed(self):
        items = list(range(30))
        first = human_audit.stratified_sample(items, stratum_key=lambda x: x % 3, per_stratum=4, seed=42)
        second = human_audit.stratified_sample(items, stratum_key=lambda x: x % 3, per_stratum=4, seed=42)
        assert first == second

    def test_per_stratum_request_is_capped_at_what_is_available(self):
        items = [0, 1, 2]  # one item per stratum
        sampled = human_audit.stratified_sample(items, stratum_key=lambda x: x, per_stratum=100)
        assert len(sampled) == 3


def test_kappa_gate_constant_matches_design():
    assert math.isclose(human_audit.KAPPA_GATE, 0.60)


def test_audit_sample_size_matches_statistics_module():
    from analysis.statistics import audit_sample_size
    assert audit_sample_size(0.05) == 385
