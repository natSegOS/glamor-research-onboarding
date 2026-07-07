"""Tests for the mixed-effects logistic regression and mediation analysis.

Covers: result dataclass structure, error paths (missing columns, wrong types),
graceful ImportError when statsmodels is absent, and interface contracts that
hold regardless of convergence (return type, field presence, n_observations).

Fitting is tested with minimal synthetic data — not to validate statsmodels
but to verify that our wrapper calls it correctly and packages the result.
"""

from __future__ import annotations

import pytest

from analysis.models import (
    MixedEffectsLogisticResult,
    MediationResult,
    fit_crossed_mixed_effects_logistic,
    compute_mediation_proportion,
)
from enums import ConvergenceMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataframe(n: int = 40, seed: int = 0):
    """Minimal synthetic DataFrame for smoke-testing the model wrappers.

    Requires pandas; skipped when pandas is unavailable.
    """
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


_SKIP_NO_STATSMODELS = pytest.mark.skipif(
    not _statsmodels_available(),
    reason="statsmodels not installed")


# ---------------------------------------------------------------------------
# Result dataclass structure
# ---------------------------------------------------------------------------

def test_mixed_effects_result_has_required_fields():
    result = MixedEffectsLogisticResult(
        converged=True,
        method=ConvergenceMethod.LAPLACE,
        log_likelihood=-42.3,
        n_observations=100,
        n_items=50,
        n_models=3,
        fixed_effects={"is_perturbed": {"coef": -0.5, "or": 0.6, "p": 0.02}},
        random_effects_variance={"item_intercept": 0.1},
        model_summary="summary text",
    )
    assert result.converged is True
    assert result.method == ConvergenceMethod.LAPLACE
    assert result.n_observations == 100
    assert "is_perturbed" in result.fixed_effects


def test_mediation_result_has_required_fields():
    result = MediationResult(
        total_effect=-0.15,
        direct_effect=-0.08,
        indirect_effect=-0.07,
        proportion_mediated=0.47,
        treatment_on_mediator_coef=0.12,
        mediator_on_outcome_coef=-0.58,
        n_observations=300,
        bootstrap_ci_proportion=(-0.1, 0.9),
    )
    assert result.proportion_mediated == pytest.approx(0.47)
    assert result.n_observations == 300
    assert result.bootstrap_ci_proportion is not None


# ---------------------------------------------------------------------------
# Error paths — missing columns and wrong input type
# ---------------------------------------------------------------------------

def test_mixed_effects_rejects_non_dataframe():
    with pytest.raises((TypeError, ImportError)):
        fit_crossed_mixed_effects_logistic({"is_correct": [1, 0]})


def test_mediation_rejects_non_dataframe():
    with pytest.raises((TypeError, ImportError)):
        compute_mediation_proportion([1, 0, 1])


@_SKIP_NO_STATSMODELS
def test_mixed_effects_rejects_missing_columns():
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    data = pd.DataFrame({"is_correct": [1, 0, 1], "is_perturbed": [0, 1, 1]})
    with pytest.raises(ValueError, match="missing required columns"):
        fit_crossed_mixed_effects_logistic(data)


@_SKIP_NO_STATSMODELS
def test_mediation_rejects_missing_columns():
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    data = pd.DataFrame({"is_correct": [1, 0], "is_perturbed": [0, 1]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_mediation_proportion(data)


# ---------------------------------------------------------------------------
# Smoke tests — interface contracts (not convergence quality)
# ---------------------------------------------------------------------------

@_SKIP_NO_STATSMODELS
def test_mixed_effects_returns_result_object():
    data = _make_dataframe()
    result = fit_crossed_mixed_effects_logistic(data)
    assert isinstance(result, MixedEffectsLogisticResult)
    assert result.n_observations == len(data)
    assert result.method in set(ConvergenceMethod)


@_SKIP_NO_STATSMODELS
def test_mixed_effects_n_items_and_models():
    data = _make_dataframe()
    result = fit_crossed_mixed_effects_logistic(data)
    assert result.n_items == data["task_id"].nunique()
    assert result.n_models == data["model_revision"].nunique()


@_SKIP_NO_STATSMODELS
def test_mediation_returns_result_object():
    data = _make_dataframe()
    result = compute_mediation_proportion(data)
    assert isinstance(result, MediationResult)
    assert result.n_observations == len(data)


@_SKIP_NO_STATSMODELS
def test_mediation_proportion_is_finite_or_none():
    data = _make_dataframe()
    result = compute_mediation_proportion(data)
    if result.proportion_mediated is not None:
        import math
        assert math.isfinite(result.proportion_mediated)


@_SKIP_NO_STATSMODELS
def test_mediation_decomposition_sums_to_total():
    """direct + indirect should equal total (up to float tolerance)."""
    import math
    data = _make_dataframe()
    result = compute_mediation_proportion(data)
    assert math.isclose(
        result.direct_effect + result.indirect_effect,
        result.total_effect,
        abs_tol=1e-9,
    )
