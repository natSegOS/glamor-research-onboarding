"""Pre-registered statistical machinery: golden values and contract tests.

Each golden value here is a number the design documents promise (sample-size
tables, κ gates, adjustment procedures). Breaking any test means the released
analysis no longer computes what the pre-registration says it computes.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings, strategies

from analysis.statistics import (
    BENJAMINI_HOCHBERG_FDR_Q,
    HOLM_PRIMARY_FAMILY_ALPHA,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    benjamini_hochberg_adjusted_p_values,
    bootstrap_confidence_interval_paired,
    build_paired_table,
    clean_conditioned_failure,
    discordant_rate,
    holm_adjusted_p_values,
    mcnemar_sample_size,
    mcnemar_test,
    audit_sample_size,
    paired_degradation,
    retention,
    summarize_cell,
)
from enums import McNemarTestMethod, SampleSizeMethod


class TestMcNemar:

    def test_dispatch_rule_is_count_based_and_fixed(self):
        """Exact mid-p below 25 discordants, asymptotic at/above — the rule is
        fixed in advance; drifting it would let the test be chosen after
        seeing the p-value."""
        assert mcnemar_test(10, 5).method == McNemarTestMethod.EXACT_MIDP
        assert mcnemar_test(20, 10).method == McNemarTestMethod.ASYMPTOTIC
        assert mcnemar_test(0, 0).p_value == 1.0

    def test_mid_p_is_less_conservative_than_exact_but_still_a_probability(self):
        """The mid-p correction subtracts half the point mass: p_midp < p_exact
        always, and both stay in [0, 1] (Fagerland et al. 2013)."""
        exact = mcnemar_test(9, 2, use_mid_p=False)
        mid_p = mcnemar_test(9, 2, use_mid_p=True)
        assert 0.0 <= mid_p.p_value < exact.p_value <= 1.0

    @settings(max_examples=100, deadline=None)
    @given(broke=strategies.integers(min_value=0, max_value=200),
           recovered=strategies.integers(min_value=0, max_value=200))
    def test_symmetry_and_probability_bounds_hold_everywhere(self, broke, recovered):
        """McNemar is symmetric in (broke, recovered) and always yields a
        valid probability — an asymmetry would bias the two-sided test."""
        forward = mcnemar_test(broke, recovered)
        backward = mcnemar_test(recovered, broke)
        assert forward.p_value == pytest.approx(backward.p_value)
        assert 0.0 <= forward.p_value <= 1.0


class TestSampleSizeGoldens:

    @pytest.mark.parametrize("delta,discordant,expected_simple,expected_connor", [
        (0.05, 0.20, 628, 626),
        (0.03, 0.10, 873, 870),
        (0.05, 0.10, 314, 312),
    ])
    def test_design_table_numbers_come_from_the_simple_approximation(
            self, delta, discordant, expected_simple, expected_connor):
        """The design tables use the SIMPLE planning approximation (628/873/
        314); Connor eq. (3) gives 626/870/312. The reference audit found
        these attributed to the wrong formula — this test pins both so the
        attribution can never silently swap again."""
        assert mcnemar_sample_size(delta, discordant,
                                   method=SampleSizeMethod.SIMPLE) == expected_simple
        assert mcnemar_sample_size(delta, discordant,
                                   method=SampleSizeMethod.CONNOR) == expected_connor

    @settings(max_examples=60, deadline=None)
    @given(delta=strategies.floats(min_value=0.01, max_value=0.2),
           headroom=strategies.floats(min_value=1.05, max_value=8.0))
    def test_connor_never_exceeds_the_simple_approximation(self, delta, headroom):
        """√(p_d − δ²) ≤ √p_d, so Connor ≤ simple everywhere — the direction
        the design relies on when it plans with the simple formula."""
        discordant = min(delta * headroom, 1.0)
        assert (mcnemar_sample_size(delta, discordant, method=SampleSizeMethod.CONNOR)
                <= mcnemar_sample_size(delta, discordant, method=SampleSizeMethod.SIMPLE))

    def test_invalid_effect_and_rate_combinations_are_rejected(self):
        """δ > p_d (an effect larger than the discordance carrying it) is
        impossible; accepting it would emit meaningless sample sizes."""
        for bad_delta, bad_rate in ((0.3, 0.2), (0.0, 0.2), (0.05, 1.5)):
            with pytest.raises(ValueError):
                mcnemar_sample_size(bad_delta, bad_rate)

    def test_audit_sample_sizes_match_the_locked_wald_margins(self):
        """385 (±5 pp) and 1068 (±3 pp) are locked in design/09 §9.3."""
        assert audit_sample_size(0.05) == 385
        assert audit_sample_size(0.03) == 1068


class TestPairedTableAndMetrics:

    def test_quadrants_partition_the_items_exactly(self):
        """a+b+c+d must equal n for every input — a lost pair silently biases
        every downstream metric."""
        table = build_paired_table([1, 1, 0, 0, 1], [1, 0, 1, 0, 0])
        assert (table.both_correct, table.broke, table.recovered, table.both_wrong) == (
            1, 2, 1, 1)
        assert table.total == 5

    def test_mismatched_arrays_are_rejected(self):
        """Unequal-length arrays mean the pairing broke upstream; computing
        anything from them would be an unpaired analysis in disguise."""
        with pytest.raises(ValueError):
            build_paired_table([1, 0], [1])

    def test_metric_definitions_and_degenerate_guards(self):
        """Δ, CCF, retention, discordant rate — exact definitional values on a
        crafted table, and NaN (never a crash or a fake 0) on degenerate input."""
        clean, perturbed = [1, 1, 1, 0], [1, 0, 0, 0]
        assert paired_degradation(clean, perturbed) == pytest.approx(0.5)
        assert clean_conditioned_failure(clean, perturbed) == pytest.approx(2 / 3)
        assert retention(clean, perturbed) == pytest.approx(1 / 3)
        assert discordant_rate(clean, perturbed) == pytest.approx(0.5)
        assert math.isnan(clean_conditioned_failure([0, 0], [0, 1]))
        assert math.isnan(retention([0, 0], [1, 1]))


class TestBootstrapIntervals:

    def test_registered_resample_count_is_ten_thousand(self):
        """B = 10,000 is a registered quantity (design/06 §6.5)."""
        assert DEFAULT_BOOTSTRAP_RESAMPLES == 10_000

    def test_interval_brackets_the_estimate_on_ordinary_data(self):
        """A BCa interval that fails to contain its own point estimate signals
        a broken pairing or statistic plumbing."""
        clean = [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
        perturbed = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]
        interval = bootstrap_confidence_interval_paired(
            clean, perturbed, "delta", resamples=500)
        assert interval.low <= interval.estimate <= interval.high
        assert interval.method in ("BCa", "percentile")

    def test_degenerate_cells_fall_back_instead_of_crashing(self):
        """All-identical outcomes make BCa undefined; the pre-registered
        contingency is a zero-width degenerate interval, not an exception."""
        interval = bootstrap_confidence_interval_paired(
            [1, 1, 1, 1], [1, 1, 1, 1], "delta", resamples=200)
        assert interval.low == interval.estimate == interval.high

    def test_cells_below_minimum_size_are_rejected(self):
        """n < 2 cannot support an interval; silently returning one would
        fabricate precision."""
        with pytest.raises(ValueError):
            bootstrap_confidence_interval_paired([1], [0], "delta")

    def test_summarize_cell_reports_everything_a_reader_needs_to_recompute(self):
        """The per-cell block must carry raw counts + test + interval —
        design/06 §6.10's reporting standard."""
        summary = summarize_cell([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0],
                                 resamples=200)
        for required_key in ("n", "broke", "recovered", "delta", "delta_ci_low",
                             "delta_ci_high", "mcnemar_p_value", "mcnemar_method",
                             "clean_conditioned_failure", "discordant_rate"):
            assert required_key in summary


class TestMultiplicityAdjustments:

    def test_benjamini_hochberg_golden_vector(self):
        """Hand-computed BH step-up on [.01,.02,.03,.04]: every adjusted value
        is .04. A drifting implementation changes which exploratory cells the
        paper calls discoveries."""
        assert benjamini_hochberg_adjusted_p_values(
            [0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.04, 0.04, 0.04])

    def test_holm_golden_vector(self):
        """Hand-computed Holm step-down on [.01,.02,.03,.04] →
        [.04,.06,.06,.06]; controls FWER within a model's primary family."""
        assert holm_adjusted_p_values(
            [0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])

    def test_nan_entries_neither_crash_nor_consume_correction_budget(self):
        """A cell with no computable test must stay NaN in place while the
        rest are adjusted as a 2-test family — including it would make every
        real correction too harsh."""
        adjusted = benjamini_hochberg_adjusted_p_values([0.01, float("nan"), 0.04])
        assert math.isnan(adjusted[1])
        assert adjusted[0] == pytest.approx(0.02)   # 0.01 * 2/1
        assert adjusted[2] == pytest.approx(0.04)   # 0.04 * 2/2

    def test_adjustment_order_is_preserved_and_thresholds_are_locked(self):
        """Adjusted values return in input order (cells are keyed by
        position), and the locked thresholds are 0.05/0.05."""
        shuffled = [0.04, 0.01, 0.03, 0.02]
        adjusted = holm_adjusted_p_values(shuffled)
        assert adjusted[1] == min(adjusted)
        assert BENJAMINI_HOCHBERG_FDR_Q == 0.05
        assert HOLM_PRIMARY_FAMILY_ALPHA == 0.05
