"""The statistical machinery (src/analysis/statistics.py): sample-size
formulas, the paired outcome table, McNemar's test, the metric set, and
bootstrap confidence intervals.

Design-table values and other fixed numeric contracts stay as golden
regression checks (a property can't catch formula drift the way a pinned
expected number can); genuine algorithmic properties (p-values in [0, 1],
paired-table completeness) use hypothesis over a random input sphere instead
of the hand-rolled ``random.Random(seed)`` loops this file used to have.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies as st

from analysis import statistics as stats
from enums import McNemarTestMethod, ParseStatus, SampleSizeMethod


# ---------------------------------------------------------------------------
# Sample size — design-table values (regression: catches formula drift).
# ---------------------------------------------------------------------------

class TestSampleSizeDesignTableValues:

    def test_simple_method_matches_design_tables(self):
        assert stats.mcnemar_sample_size(0.05, 0.20, method=SampleSizeMethod.SIMPLE) == 628
        assert stats.mcnemar_sample_size(0.03, 0.10, method=SampleSizeMethod.SIMPLE) == 873
        assert stats.mcnemar_sample_size(0.05, 0.10, method=SampleSizeMethod.SIMPLE) == 314

    def test_connor_method_supports_provisional_600_per_cell(self):
        assert stats.mcnemar_sample_size(0.05, 0.19, method=SampleSizeMethod.CONNOR) <= 600

    def test_audit_sample_size_matches_design(self):
        assert stats.audit_sample_size(0.05) == 385
        assert stats.audit_sample_size(0.03) == 1068


class TestSampleSizeExpectedBehavior:

    def test_smaller_minimum_detectable_effect_requires_more_samples(self):
        n_large_mde = stats.mcnemar_sample_size(0.10, 0.25, method=SampleSizeMethod.SIMPLE)
        n_small_mde = stats.mcnemar_sample_size(0.05, 0.25, method=SampleSizeMethod.SIMPLE)
        assert n_small_mde > n_large_mde

    def test_sample_size_increases_with_discordant_rate_near_the_design_point(self):
        # n ~ p*(1-p)/d^2 at fixed d=0.05; 0.10*0.90 < 0.20*0.80 < 0.25*0.75.
        n_low = stats.mcnemar_sample_size(0.05, 0.10, method=SampleSizeMethod.SIMPLE)
        n_mid = stats.mcnemar_sample_size(0.05, 0.20, method=SampleSizeMethod.SIMPLE)
        n_high = stats.mcnemar_sample_size(0.05, 0.25, method=SampleSizeMethod.SIMPLE)
        assert n_low < n_mid < n_high

    @pytest.mark.parametrize("difference,discordant_rate", [(0.05, 0.20), (0.03, 0.10), (0.05, 0.30)])
    def test_connor_method_is_never_larger_than_simple(self, difference, discordant_rate):
        connor = stats.mcnemar_sample_size(difference, discordant_rate, method=SampleSizeMethod.CONNOR)
        simple = stats.mcnemar_sample_size(difference, discordant_rate, method=SampleSizeMethod.SIMPLE)
        assert connor <= simple

    def test_audit_sample_size_increases_as_margin_narrows(self):
        assert stats.audit_sample_size(0.03) > stats.audit_sample_size(0.10)


class TestSampleSizeAdversarialInputs:

    def test_difference_exceeding_discordant_rate_raises(self):
        with pytest.raises(ValueError):
            stats.mcnemar_sample_size(0.30, 0.20)

    def test_zero_difference_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            stats.mcnemar_sample_size(0.0, 0.20)


# ---------------------------------------------------------------------------
# Paired outcome table.
# ---------------------------------------------------------------------------

class TestPairedTable:

    @pytest.mark.parametrize("clean,perturbed,expected_counts", [
        ([1, 1, 1, 0, 0], [1, 0, 0, 1, 0], (1, 2, 1, 1)),           # mixed
        ([1] * 20, [1] * 20, (20, 0, 0, 0)),                        # all both_correct
        ([0] * 20, [0] * 20, (0, 0, 0, 20)),                        # all both_wrong
        ([1] * 15, [0] * 15, (0, 15, 0, 0)),                        # all broke
    ], ids=["mixed", "all_both_correct", "all_both_wrong", "all_broke"])
    def test_counts_match_expected_quadrant(self, clean, perturbed, expected_counts):
        table = stats.build_paired_table(clean, perturbed)
        assert (table.both_correct, table.broke, table.recovered, table.both_wrong) == expected_counts
        assert table.total == len(clean)

    @given(clean=st.lists(st.integers(0, 1), min_size=1, max_size=200),
           seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100)
    def test_the_four_quadrants_always_sum_to_n(self, clean, seed):
        rng_for_pert = __import__("random").Random(seed)
        perturbed = [rng_for_pert.randint(0, 1) for _ in clean]
        table = stats.build_paired_table(clean, perturbed)
        assert table.both_correct + table.broke + table.recovered + table.both_wrong == len(clean)
        assert table.total == len(clean)


# ---------------------------------------------------------------------------
# McNemar's test.
# ---------------------------------------------------------------------------

class TestMcNemarTest:

    @pytest.mark.parametrize("broke,recovered,expected_method", [
        (3, 1, McNemarTestMethod.EXACT_MIDP),
        (4, 2, McNemarTestMethod.EXACT_MIDP),
        (40, 10, McNemarTestMethod.ASYMPTOTIC),
        (25, 5, McNemarTestMethod.ASYMPTOTIC),
    ], ids=["few_discordant", "at_exact_boundary", "many_discordant", "past_boundary"])
    def test_exact_vs_asymptotic_dispatch(self, broke, recovered, expected_method):
        assert stats.mcnemar_test(broke=broke, recovered=recovered).method == expected_method

    def test_no_discordant_pairs_gives_p_value_one(self):
        assert stats.mcnemar_test(0, 0).p_value == 1.0

    def test_is_symmetric_in_broke_and_recovered(self):
        assert math.isclose(
            stats.mcnemar_test(broke=30, recovered=5).p_value,
            stats.mcnemar_test(broke=5, recovered=30).p_value, abs_tol=1e-9)

    @given(broke=st.integers(0, 100), recovered=st.integers(0, 100))
    @settings(max_examples=100)
    def test_p_value_is_always_in_the_unit_interval(self, broke, recovered):
        result = stats.mcnemar_test(broke, recovered)
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# The metric set.
# ---------------------------------------------------------------------------

class TestMetrics:

    @pytest.mark.parametrize("clean,perturbed,expected", [
        ([1] * 80 + [0] * 20, [1] * 60 + [0] * 40, 0.60 / 0.80),  # partial retention
        ([1] * 50, [0] * 50, 0.0),                                 # everything broke
        ([1] * 50, [1] * 50, 1.0),                                 # nothing broke
    ], ids=["partial", "all_broke", "all_retained"])
    def test_retention(self, clean, perturbed, expected):
        assert math.isclose(stats.retention(clean, perturbed), expected)

    @pytest.mark.parametrize("clean,perturbed,expected", [
        ([1] * 80 + [0] * 20, [1] * 60 + [0] * 40, 0.25),  # some clean-correct items broke
        ([1] * 40, [1] * 40, 0.0),                          # none broke
    ], ids=["partial", "none_broke"])
    def test_clean_conditioned_failure(self, clean, perturbed, expected):
        assert math.isclose(stats.clean_conditioned_failure(clean, perturbed), expected)

    def test_clean_conditioned_failure_with_no_clean_correct_items_does_not_crash(self):
        # Undefined (0/0); must return some sentinel, not raise.
        assert stats.clean_conditioned_failure([0] * 10, [0] * 10) is not None

    def test_answer_flip_rate(self):
        assert math.isclose(stats.answer_flip_rate(["A", "B", "C"], ["A", "X", "C"]), 1 / 3)

    def test_over_robustness_and_appropriate_change_rates(self):
        perturbed, new_gold, old_gold = ["19", "berlin", "19"], ["21", "berlin", "21"], ["19", "paris", "19"]
        assert math.isclose(stats.appropriate_change_rate(perturbed, new_gold), 1 / 3)
        assert math.isclose(stats.over_robustness_rate(perturbed, old_gold), 2 / 3)


class TestInvalidOrClarificationRate:

    @pytest.mark.parametrize("statuses,expected", [
        ([ParseStatus.VALID, ParseStatus.VALID, ParseStatus.CLARIFICATION,
          ParseStatus.REFUSAL, ParseStatus.UNPARSEABLE], 3 / 5),
        ([ParseStatus.VALID] * 10, 0.0),
        ([ParseStatus.REFUSAL, ParseStatus.CLARIFICATION, ParseStatus.UNPARSEABLE], 1.0),
        (["valid", "clarification", "refusal"], 2 / 3),  # (str, Enum): plain strings work too
    ], ids=["mixed", "all_valid", "all_failure", "plain_strings"])
    def test_rate_over_a_status_sequence(self, statuses, expected):
        assert math.isclose(stats.invalid_or_clarification_rate(statuses), expected)

    def test_empty_sequence_does_not_crash(self):
        try:
            result = stats.invalid_or_clarification_rate([])
            assert math.isnan(result) or result == 0.0
        except (ZeroDivisionError, ValueError):
            pass  # either degrades gracefully or raises; both are acceptable


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals and cell summaries.
# ---------------------------------------------------------------------------

class TestBootstrapConfidenceInterval:

    def test_interval_brackets_its_own_point_estimate(self):
        clean, perturbed = [1] * 70 + [0] * 30, [1] * 50 + [0] * 50
        interval = stats.bootstrap_confidence_interval_paired(clean, perturbed, "delta", resamples=500)
        assert interval.low <= interval.estimate <= interval.high

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=10)
    def test_interval_brackets_estimate_across_random_paired_data(self, seed):
        rng = __import__("random").Random(seed)
        clean = [rng.randint(0, 1) for _ in range(100)]
        perturbed = [rng.randint(0, 1) for _ in range(100)]
        interval = stats.bootstrap_confidence_interval_paired(clean, perturbed, "delta", resamples=300)
        assert interval.low <= interval.estimate <= interval.high

    def test_identical_clean_and_perturbed_gives_a_zero_delta_interval_bracketing_zero(self):
        outcomes = [1, 0, 1, 0, 1, 0] * 20
        interval = stats.bootstrap_confidence_interval_paired(outcomes, outcomes, "delta", resamples=300)
        assert math.isclose(interval.estimate, 0.0, abs_tol=1e-9)
        assert interval.low <= 0.0 <= interval.high


class TestCellSummary:

    def test_default_resamples_constant(self):
        assert stats.DEFAULT_BOOTSTRAP_RESAMPLES == 10000

    @pytest.mark.parametrize("clean,perturbed,expected_n", [([], [], 0), ([1], [0], 1)],
                             ids=["n_equals_zero", "n_equals_one"])
    def test_insufficient_n_degrades_gracefully(self, clean, perturbed, expected_n):
        summary = stats.summarize_cell(clean, perturbed, resamples=100)
        assert summary["n"] == expected_n
        assert summary["delta_ci_method"] == "insufficient_n"

    def test_full_cell_summary(self):
        clean, perturbed = [1] * 80 + [0] * 20, [1] * 60 + [0] * 40
        summary = stats.summarize_cell(clean, perturbed, resamples=500)
        assert summary["n"] == 100
        assert summary["broke"] == 20
        assert math.isclose(summary["delta"], 0.20, abs_tol=1e-9)
        assert summary["mcnemar_p_value"] < 0.05
