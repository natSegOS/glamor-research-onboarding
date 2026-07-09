"""Statistical modeling and results aggregation (src/analysis/models.py,
src/analysis/results.py): mixed-effects logistic regression and mediation
result structures and error paths, matched-pair joining, cell summarization,
the audit-exclusion gate (Part 7), and VALID-only sensitivity (Part 4).

Model-fitting is exercised with minimal synthetic data — not to validate
statsmodels itself, but to verify this codebase's wrapper calls it correctly
and packages the result, regardless of whether the fit actually converges.
"""

from __future__ import annotations

import math

import pytest

from analysis.models import (
    MixedEffectsLogisticResult,
    MediationResult,
    fit_crossed_mixed_effects_logistic,
    compute_mediation_proportion,
)
from analysis.results import MatchedPair, join_matched_pairs, summarize_all_cells
from analysis.audit import ItemAuditOutcome
from enums import ConvergenceMethod, ParseStatus, SemanticClass


def _make_dataframe(n: int = 40, seed: int = 0):
    """Minimal synthetic DataFrame for smoke-testing the model wrappers.
    Skipped when pandas/numpy are unavailable."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        pytest.skip("pandas/numpy not installed")

    rng = np.random.default_rng(seed)
    item_ids = [f"item_{i:03d}" for i in range(n // 2)]
    model_ids = ["model_a", "model_b"]

    rows = []
    for _ in range(2):               # clean + perturbed pass per item
        for item_id in item_ids:
            is_perturbed = int(rng.integers(0, 2))
            tir = 1.0 + rng.normal(0.1 * is_perturbed, 0.05)
            rows.append({
                "task_id": item_id,
                "model_revision": rng.choice(model_ids),
                "is_perturbed": is_perturbed,
                "is_correct": int(rng.integers(0, 2)),
                "token_inflation_ratio": float(tir),
                "edit_budget_k": int(rng.choice([1, 2, 4])),
            })
    return pd.DataFrame(rows)


def _statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


_SKIP_NO_STATSMODELS = pytest.mark.skipif(not _statsmodels_available(), reason="statsmodels not installed")


# ---------------------------------------------------------------------------
# Result dataclass structure.
# ---------------------------------------------------------------------------

class TestResultDataclasses:

    def test_mixed_effects_result_carries_its_fields(self):
        result = MixedEffectsLogisticResult(
            converged=True, method=ConvergenceMethod.LAPLACE, log_likelihood=-42.3,
            n_observations=100, n_items=50, n_models=3,
            fixed_effects={"is_perturbed": {"coef": -0.5, "or": 0.6, "p": 0.02}},
            random_effects_variance={"item_intercept": 0.1}, model_summary="summary text")
        assert result.converged is True
        assert result.method == ConvergenceMethod.LAPLACE
        assert result.n_observations == 100
        assert "is_perturbed" in result.fixed_effects

    def test_mediation_result_carries_its_fields(self):
        result = MediationResult(
            total_effect=-0.15, direct_effect=-0.08, indirect_effect=-0.07,
            proportion_mediated=0.47, treatment_on_mediator_coef=0.12,
            mediator_on_outcome_coef=-0.58, n_observations=300, bootstrap_ci_proportion=(-0.1, 0.9))
        assert result.proportion_mediated == pytest.approx(0.47)
        assert result.n_observations == 300
        assert result.bootstrap_ci_proportion is not None


# ---------------------------------------------------------------------------
# Error paths — non-DataFrame input and missing required columns.
# ---------------------------------------------------------------------------

class TestModelFittingErrorPaths:

    def test_mixed_effects_rejects_a_non_dataframe(self):
        with pytest.raises((TypeError, ImportError)):
            fit_crossed_mixed_effects_logistic({"is_correct": [1, 0]})

    def test_mediation_rejects_a_non_dataframe(self):
        with pytest.raises((TypeError, ImportError)):
            compute_mediation_proportion([1, 0, 1])

    @_SKIP_NO_STATSMODELS
    def test_mixed_effects_rejects_missing_columns(self):
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({"is_correct": [1, 0, 1], "is_perturbed": [0, 1, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            fit_crossed_mixed_effects_logistic(data)

    @_SKIP_NO_STATSMODELS
    def test_mediation_rejects_missing_columns(self):
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({"is_correct": [1, 0], "is_perturbed": [0, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_mediation_proportion(data)


# ---------------------------------------------------------------------------
# Smoke tests — interface contracts that hold regardless of convergence.
# ---------------------------------------------------------------------------

class TestMixedEffectsSmokeTest:

    @_SKIP_NO_STATSMODELS
    def test_returns_a_result_object_with_the_expected_observation_count(self):
        data = _make_dataframe()
        result = fit_crossed_mixed_effects_logistic(data)
        assert isinstance(result, MixedEffectsLogisticResult)
        assert result.n_observations == len(data)
        assert result.method in set(ConvergenceMethod)

    @_SKIP_NO_STATSMODELS
    def test_n_items_and_n_models_match_the_data(self):
        data = _make_dataframe()
        result = fit_crossed_mixed_effects_logistic(data)
        assert result.n_items == data["task_id"].nunique()
        assert result.n_models == data["model_revision"].nunique()


class TestMediationSmokeTest:

    @_SKIP_NO_STATSMODELS
    def test_returns_a_result_object_with_the_expected_observation_count(self):
        data = _make_dataframe()
        result = compute_mediation_proportion(data)
        assert isinstance(result, MediationResult)
        assert result.n_observations == len(data)

    @_SKIP_NO_STATSMODELS
    def test_proportion_mediated_is_finite_when_present(self):
        result = compute_mediation_proportion(_make_dataframe())
        if result.proportion_mediated is not None:
            assert math.isfinite(result.proportion_mediated)

    @_SKIP_NO_STATSMODELS
    def test_direct_plus_indirect_effect_equals_total_effect(self):
        result = compute_mediation_proportion(_make_dataframe())
        assert math.isclose(result.direct_effect + result.indirect_effect,
                            result.total_effect, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# join_matched_pairs.
# ---------------------------------------------------------------------------

def _clean_and_perturbed_rows(task_id="t1", perturbed_correct=0,
                              perturbed_parse_status=ParseStatus.UNPARSEABLE):
    shared = {
        "model_revision": "m1", "task_id": task_id, "task_family": "gsm_symbolic_synthetic",
        "r_semantic_class": str(SemanticClass.A), "r_operation": "substitute",
        "r_selection_policy": "keyboard_neighbor", "r_scope": "anywhere", "r_edit_budget": 1,
    }
    clean_row = {**shared, "is_clean": True, "is_correct": 1,
                "parsed_answer": "42", "parse_status": str(ParseStatus.VALID)}
    perturbed_row = {**shared, "is_clean": False, "is_correct": perturbed_correct,
                     "parsed_answer": None, "parse_status": str(perturbed_parse_status)}
    return clean_row, perturbed_row


class TestJoinMatchedPairs:

    def test_joins_a_clean_and_perturbed_row_into_one_pair(self):
        clean_row, perturbed_row = _clean_and_perturbed_rows()
        pairs = join_matched_pairs([clean_row, perturbed_row])
        assert len(pairs) == 1
        assert pairs[0].clean_is_correct == 1
        assert pairs[0].perturbed_is_correct == 0

    def test_a_perturbed_row_with_no_clean_partner_is_skipped(self):
        _, orphan_perturbed_row = _clean_and_perturbed_rows(task_id="t_orphan")
        assert join_matched_pairs([orphan_perturbed_row]) == []


# ---------------------------------------------------------------------------
# summarize_all_cells: basic aggregation, the audit-exclusion gate (Part 7),
# and VALID-only sensitivity (Part 4).
# ---------------------------------------------------------------------------

def _make_pair(task_id, clean_correct, pert_correct, parse_status=ParseStatus.VALID,
              model_revision="model_a", task_family="gsm_symbolic_synthetic"):
    cell_key = (model_revision, task_family,
               str(SemanticClass.A), "substitute", "keyboard_neighbor", "anywhere", 1)
    return MatchedPair(
        model_revision=model_revision, task_id=task_id, task_family=task_family,
        clean_is_correct=clean_correct, perturbed_is_correct=pert_correct,
        clean_answer=str(clean_correct),
        perturbed_answer=str(pert_correct) if parse_status == ParseStatus.VALID else None,
        perturbed_parse_status=parse_status, cell_key=cell_key)


def _make_audit_outcome(task_id, excluded: bool, regime=SemanticClass.A):
    return ItemAuditOutcome(
        item_id=task_id, regime_label=regime, majority_intent_preserved=not excluded,
        majority_gold_unchanged=True, was_adjudicated=False, rating_count=3,
        excluded_from_primary=excluded)


class TestSummarizeAllCellsBasic:

    def test_delta_and_n_for_a_uniform_cell(self):
        pairs = [_make_pair(f"t{i}", 1, 0) for i in range(10)]
        summary = summarize_all_cells(pairs, resamples=100)[0]
        assert summary["n"] == 10
        assert math.isclose(summary["delta"], 1.0, abs_tol=1e-9)

    def test_n_audit_excluded_is_zero_with_no_audit_gate(self):
        pairs = [_make_pair(f"t{i}", 1, 0) for i in range(5)]
        assert summarize_all_cells(pairs, resamples=100)[0]["n_audit_excluded"] == 0


class TestAuditExclusionGate:

    def test_flagged_items_are_excluded_and_counted(self):
        pairs = [_make_pair(f"t{i}", 1, 0) for i in range(6)]
        audit_outcomes = {"t0": _make_audit_outcome("t0", excluded=True),
                          "t1": _make_audit_outcome("t1", excluded=True)}
        summary = summarize_all_cells(pairs, resamples=100, audit_outcomes=audit_outcomes)[0]
        assert summary["n_audit_excluded"] == 2
        assert summary["n"] == 4

    def test_no_gate_and_a_gate_with_nothing_excluded_agree(self):
        pairs = [_make_pair(f"t{i}", 1, 0) for i in range(5)]
        without_gate = summarize_all_cells(pairs, resamples=100, audit_outcomes=None)[0]
        with_gate_none_excluded = summarize_all_cells(
            pairs, resamples=100, audit_outcomes={"t0": _make_audit_outcome("t0", excluded=False)})[0]
        assert without_gate["n"] == with_gate_none_excluded["n"]

    def test_every_item_excluded_yields_an_empty_cell(self):
        pairs = [_make_pair("t0", 1, 0)]
        audit_outcomes = {"t0": _make_audit_outcome("t0", excluded=True)}
        summary = summarize_all_cells(pairs, resamples=100, audit_outcomes=audit_outcomes)[0]
        assert summary["n_audit_excluded"] == 1
        assert summary["n"] == 0

    def test_confirmed_items_are_kept_only_excluded_ones_drop(self):
        pairs = [_make_pair(f"t{i}", 1, 0) for i in range(4)]
        audit_outcomes = {"t0": _make_audit_outcome("t0", excluded=True),
                          "t1": _make_audit_outcome("t1", excluded=False)}
        summary = summarize_all_cells(pairs, resamples=100, audit_outcomes=audit_outcomes)[0]
        assert summary["n_audit_excluded"] == 1
        assert summary["n"] == 3


class TestValidOnlySensitivity:

    def test_delta_valid_only_present_with_enough_valid_pairs(self):
        pairs = [_make_pair(f"t{i}", 1, 0, parse_status=ParseStatus.VALID) for i in range(10)]
        summary = summarize_all_cells(pairs, resamples=100)[0]
        assert summary["delta_valid_only"] is not None
        assert math.isclose(summary["delta_valid_only"], 1.0, abs_tol=1e-9)

    def test_delta_valid_only_is_computed_on_only_the_valid_subset(self):
        valid_pairs = [_make_pair(f"tv{i}", 1, 0, parse_status=ParseStatus.VALID) for i in range(6)]
        unparseable_pairs = [
            _make_pair(f"tu{i}", 1, 0, parse_status=ParseStatus.UNPARSEABLE) for i in range(4)]
        summary = summarize_all_cells(valid_pairs + unparseable_pairs, resamples=100)[0]
        assert math.isclose(summary["delta_valid_only"], 1.0, abs_tol=1e-9)  # 6 valid pairs, all broke
        assert summary["n"] == 10  # all-in count includes the unparseable pairs

    def test_none_when_fewer_than_two_valid_pairs(self):
        pairs = [_make_pair("t0", 1, 0, parse_status=ParseStatus.UNPARSEABLE)]
        summary = summarize_all_cells(pairs, resamples=100)[0]
        assert summary["delta_valid_only"] is None
        assert summary["mcnemar_p_valid_only"] is None

    def test_coincides_with_all_in_delta_when_every_pair_is_valid(self):
        pairs = [_make_pair(f"t{i}", 1, 0, parse_status=ParseStatus.VALID) for i in range(20)]
        summary = summarize_all_cells(pairs, resamples=200)[0]
        assert math.isclose(summary["delta"], summary["delta_valid_only"], abs_tol=1e-9)

    def test_mcnemar_p_valid_only_is_in_the_unit_interval(self):
        pairs = [
            _make_pair("t0", 1, 0, parse_status=ParseStatus.VALID),
            _make_pair("t1", 1, 1, parse_status=ParseStatus.VALID),
            _make_pair("t2", 0, 0, parse_status=ParseStatus.VALID),
        ]
        summary = summarize_all_cells(pairs, resamples=100)[0]
        assert summary["mcnemar_p_valid_only"] is not None
        assert 0.0 <= summary["mcnemar_p_valid_only"] <= 1.0
