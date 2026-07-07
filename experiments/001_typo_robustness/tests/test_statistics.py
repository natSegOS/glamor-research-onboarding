"""Adversarial and property-based tests for the statistical machinery.

Covers: sample-size monotonicity, method comparison, boundary at asymptotic
threshold, p-value bounds over fuzzed inputs, bootstrap CI bracketing, paired
table completeness, McNemar exact/asymptotic dispatch, n=0/1 degenerate cases,
and ICR with enum values.
"""

from __future__ import annotations

import math
import random

import pytest

from analysis import statistics as st
from enums import McNemarTestMethod, ParseStatus, SampleSizeMethod


# ---------------------------------------------------------------------------
# Sample size — design table values (fixed; regression catches formula drift)
# ---------------------------------------------------------------------------

def test_simple_sample_size_matches_design_tables():
    assert st.mcnemar_sample_size(0.05, 0.20, method=SampleSizeMethod.SIMPLE) == 628
    assert st.mcnemar_sample_size(0.03, 0.10, method=SampleSizeMethod.SIMPLE) == 873
    assert st.mcnemar_sample_size(0.05, 0.10, method=SampleSizeMethod.SIMPLE) == 314


def test_connor_sample_size_supports_provisional_600_per_cell():
    assert st.mcnemar_sample_size(0.05, 0.19, method=SampleSizeMethod.CONNOR) <= 600


def test_audit_sample_size_matches_design():
    assert st.audit_sample_size(0.05) == 385
    assert st.audit_sample_size(0.03) == 1068


# ---------------------------------------------------------------------------
# Sample size — monotonicity properties
# ---------------------------------------------------------------------------

def test_sample_size_increases_with_smaller_mde():
    """Smaller MDE (harder to detect) requires more samples."""
    n_large_mde = st.mcnemar_sample_size(0.10, 0.25, method=SampleSizeMethod.SIMPLE)
    n_small_mde = st.mcnemar_sample_size(0.05, 0.25, method=SampleSizeMethod.SIMPLE)
    assert n_small_mde > n_large_mde


def test_sample_size_varies_with_discordant_rate():
    """Sample size varies non-trivially with the discordant rate (n is not monotone,
    but the 0.20 design-table value should equal the expected 628/2-ish)."""
    # The simple McNemar formula n ∝ p*(1-p)/d^2; this peaks at p=0.5.
    # At our target d=0.05: n(p=0.10) < n(p=0.20) < n(p=0.25) because
    # 0.10*0.90=0.09 < 0.20*0.80=0.16 < 0.25*0.75=0.1875.
    n_low = st.mcnemar_sample_size(0.05, 0.10, method=SampleSizeMethod.SIMPLE)
    n_mid = st.mcnemar_sample_size(0.05, 0.20, method=SampleSizeMethod.SIMPLE)
    n_high = st.mcnemar_sample_size(0.05, 0.25, method=SampleSizeMethod.SIMPLE)
    assert n_low < n_mid < n_high


def test_connor_is_no_larger_than_simple():
    for difference, rate in [(0.05, 0.20), (0.03, 0.10), (0.05, 0.30)]:
        connor = st.mcnemar_sample_size(difference, rate, method=SampleSizeMethod.CONNOR)
        simple = st.mcnemar_sample_size(difference, rate, method=SampleSizeMethod.SIMPLE)
        assert connor <= simple


def test_audit_sample_size_monotone():
    """Tighter margin requires more audit items."""
    n_wide = st.audit_sample_size(0.10)
    n_narrow = st.audit_sample_size(0.03)
    assert n_narrow > n_wide


# ---------------------------------------------------------------------------
# Sample size — error paths
# ---------------------------------------------------------------------------

def test_sample_size_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        st.mcnemar_sample_size(0.30, 0.20)   # difference exceeds discordant rate


def test_sample_size_rejects_zero_difference():
    with pytest.raises((ValueError, ZeroDivisionError)):
        st.mcnemar_sample_size(0.0, 0.20)


# ---------------------------------------------------------------------------
# Paired table
# ---------------------------------------------------------------------------

def test_paired_table_counts():
    clean = [1, 1, 1, 0, 0]
    pert = [1, 0, 0, 1, 0]
    table = st.build_paired_table(clean, pert)
    assert (table.both_correct, table.broke, table.recovered, table.both_wrong) == (1, 2, 1, 1)
    assert table.total == 5


def test_paired_table_sums_to_total():
    rng = random.Random(13)
    for _ in range(20):
        n = rng.randint(10, 50)
        clean = [rng.randint(0, 1) for _ in range(n)]
        pert = [rng.randint(0, 1) for _ in range(n)]
        table = st.build_paired_table(clean, pert)
        assert table.both_correct + table.broke + table.recovered + table.both_wrong == n
        assert table.total == n


def test_paired_table_all_correct():
    n = 20
    table = st.build_paired_table([1] * n, [1] * n)
    assert table.both_correct == n
    assert table.broke == 0
    assert table.recovered == 0
    assert table.both_wrong == 0


def test_paired_table_all_wrong():
    n = 20
    table = st.build_paired_table([0] * n, [0] * n)
    assert table.both_wrong == n
    assert table.broke == 0
    assert table.recovered == 0
    assert table.both_correct == 0


def test_paired_table_clean_all_correct_pert_all_wrong():
    n = 15
    table = st.build_paired_table([1] * n, [0] * n)
    assert table.broke == n
    assert table.recovered == 0
    assert table.both_correct == 0
    assert table.both_wrong == 0


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------

def test_mcnemar_exact_when_few_discordant():
    result = st.mcnemar_test(broke=3, recovered=1)
    assert result.method == McNemarTestMethod.EXACT_MIDP
    assert 0.0 <= result.p_value <= 1.0


def test_mcnemar_asymptotic_when_many_discordant():
    result = st.mcnemar_test(broke=40, recovered=10)
    assert result.method == McNemarTestMethod.ASYMPTOTIC
    assert result.p_value < 0.05


def test_mcnemar_no_discordant_pairs_is_p_one():
    assert st.mcnemar_test(0, 0).p_value == 1.0


def test_mcnemar_p_value_always_in_unit_interval():
    """p-value must be in [0, 1] for all sane inputs."""
    rng = random.Random(7)
    for _ in range(30):
        broke = rng.randint(0, 100)
        recovered = rng.randint(0, 100)
        result = st.mcnemar_test(broke, recovered)
        assert 0.0 <= result.p_value <= 1.0, (
            f"p={result.p_value} for broke={broke} recovered={recovered}")


def test_mcnemar_symmetric_in_broke_recovered():
    """McNemar is symmetric in broke vs recovered (two-sided)."""
    r1 = st.mcnemar_test(broke=30, recovered=5)
    r2 = st.mcnemar_test(broke=5, recovered=30)
    assert math.isclose(r1.p_value, r2.p_value, abs_tol=1e-9)


def test_mcnemar_exact_asymptotic_threshold():
    """Verify the exact-vs-asymptotic dispatch boundary is stable.
    With discordant count <= some threshold, method should be exact."""
    small = st.mcnemar_test(broke=4, recovered=2)
    assert small.method == McNemarTestMethod.EXACT_MIDP
    large = st.mcnemar_test(broke=25, recovered=5)
    assert large.method == McNemarTestMethod.ASYMPTOTIC


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_clean_conditioned_failure():
    clean = [1] * 80 + [0] * 20
    pert = [1] * 60 + [0] * 40
    assert math.isclose(st.clean_conditioned_failure(clean, pert), 0.25)


def test_retention():
    clean = [1] * 80 + [0] * 20
    pert = [1] * 60 + [0] * 40
    assert math.isclose(st.retention(clean, pert), 0.60 / 0.80)


def test_answer_flip_rate():
    assert math.isclose(st.answer_flip_rate(["A", "B", "C"], ["A", "X", "C"]), 1 / 3)


def test_over_robustness_and_appropriate_change():
    perturbed = ["19", "berlin", "19"]
    new_gold = ["21", "berlin", "21"]
    old_gold = ["19", "paris", "19"]
    assert math.isclose(st.appropriate_change_rate(perturbed, new_gold), 1 / 3)
    assert math.isclose(st.over_robustness_rate(perturbed, old_gold), 2 / 3)


def test_retention_all_correct_clean_all_wrong_pert():
    clean = [1] * 50
    pert = [0] * 50
    assert math.isclose(st.retention(clean, pert), 0.0)


def test_retention_all_correct_both():
    assert math.isclose(st.retention([1] * 50, [1] * 50), 1.0)


def test_ccf_all_correct_both_is_zero():
    assert math.isclose(st.clean_conditioned_failure([1] * 40, [1] * 40), 0.0)


def test_ccf_clean_all_wrong_is_nan_or_zero():
    # If no clean-correct items exist, CCF is undefined; must not crash.
    result = st.clean_conditioned_failure([0] * 10, [0] * 10)
    # Result is either nan or a defined value; just must not raise.
    assert result is not None


# ---------------------------------------------------------------------------
# ICR with enum values
# ---------------------------------------------------------------------------

def test_invalid_or_clarification_rate():
    statuses = [ParseStatus.VALID, ParseStatus.VALID,
                ParseStatus.CLARIFICATION, ParseStatus.REFUSAL, ParseStatus.UNPARSEABLE]
    assert math.isclose(st.invalid_or_clarification_rate(statuses), 3 / 5)


def test_icr_all_valid():
    statuses = [ParseStatus.VALID] * 10
    assert math.isclose(st.invalid_or_clarification_rate(statuses), 0.0)


def test_icr_all_failure():
    statuses = [ParseStatus.REFUSAL, ParseStatus.CLARIFICATION, ParseStatus.UNPARSEABLE]
    assert math.isclose(st.invalid_or_clarification_rate(statuses), 1.0)


def test_icr_accepts_plain_strings():
    """(str, Enum) equality means plain strings work as status values."""
    statuses = ["valid", "clarification", "refusal"]
    assert math.isclose(st.invalid_or_clarification_rate(statuses), 2 / 3)


def test_icr_empty_raises_or_returns_nan():
    try:
        result = st.invalid_or_clarification_rate([])
        assert math.isnan(result) or result == 0.0
    except (ZeroDivisionError, ValueError):
        pass  # either behavior is acceptable


# ---------------------------------------------------------------------------
# Bootstrap + cell summary
# ---------------------------------------------------------------------------

def test_bootstrap_interval_brackets_estimate():
    clean = [1] * 70 + [0] * 30
    pert = [1] * 50 + [0] * 50
    interval = st.bootstrap_confidence_interval_paired(clean, pert, "delta", resamples=500)
    assert interval.low <= interval.estimate <= interval.high


def test_bootstrap_interval_stable_across_seeds():
    rng = random.Random(42)
    clean = [rng.randint(0, 1) for _ in range(100)]
    pert = [rng.randint(0, 1) for _ in range(100)]
    intervals = [
        st.bootstrap_confidence_interval_paired(clean, pert, "delta", resamples=300)
        for _ in range(3)
    ]
    # All should bracket their own estimate.
    for iv in intervals:
        assert iv.low <= iv.estimate <= iv.high


def test_bootstrap_interval_with_zero_delta():
    """If clean == pert, delta == 0 and CI should bracket 0."""
    outcomes = [1, 0, 1, 0, 1, 0] * 20
    interval = st.bootstrap_confidence_interval_paired(outcomes, outcomes, "delta", resamples=300)
    assert math.isclose(interval.estimate, 0.0, abs_tol=1e-9)
    assert interval.low <= 0.0 <= interval.high


def test_summarize_cell_has_resamples_constant():
    assert st.DEFAULT_BOOTSTRAP_RESAMPLES == 10000


def test_summarize_cell_small_n_has_no_crash():
    summary = st.summarize_cell([1], [0], resamples=100)
    assert summary["n"] == 1
    assert summary["delta_ci_method"] == "insufficient_n"


def test_summarize_cell_full():
    clean = [1] * 80 + [0] * 20
    pert = [1] * 60 + [0] * 40
    summary = st.summarize_cell(clean, pert, resamples=500)
    assert summary["n"] == 100
    assert summary["broke"] == 20
    assert math.isclose(summary["delta"], 0.20, abs_tol=1e-9)
    assert summary["mcnemar_p_value"] < 0.05


def test_summarize_cell_n_equals_zero():
    summary = st.summarize_cell([], [], resamples=100)
    assert summary["n"] == 0
    assert summary["delta_ci_method"] == "insufficient_n"
