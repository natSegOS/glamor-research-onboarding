"""Mixed-effects logistic regression and mediation analysis (design/06 §6.6 and §6.8).

Provenance
----------
Mixed-effects logistic:
  Baayen, Davidson & Bates (2008) — crossed random effects specification.
  Barr, Levy, Scheepers & Tily (2013) — maximal random-effects motivation.
  Jaeger (2008) — categorical regression with random effects.

Mediation — Method B, product-of-coefficients:
  Imai, Keele & Tingley (2010) "A General Approach to Causal Mediation Analysis",
  Psychological Methods 15(4):309–334 — the canonical reference for the
  proportion-mediated decomposition used here.

Both require statsmodels (not in the minimal requirements). The module degrades
gracefully: if statsmodels is absent, all fitting functions raise ImportError
with a helpful message rather than crashing at import time.

See ``ConvergenceMethod`` for the fallback ladder this actually implements
and how it differs from design/06 §6.6's pre-registered structural
contingency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from enums import ConvergenceMethod


# Machine-readable reasons why proportion_mediated is withheld.
PROPORTION_WITHHELD_TOTAL_NUMERICALLY_ZERO = "total_effect_numerically_zero"
PROPORTION_WITHHELD_TOTAL_CI_INCLUDES_ZERO = "total_effect_ci_includes_zero"
PROPORTION_WITHHELD_NO_BOOTSTRAP = "bootstrap_ci_unavailable"

PRIMARY_MEDIATOR_COLUMN = "token_inflation_excess"
SUPPLEMENTARY_MEDIATOR_COLUMN = "subword_count_change"

_MEDIATION_MAX_ITERATIONS = 200
_BOOTSTRAP_MAX_ITERATIONS = 100
_BOOTSTRAP_RESAMPLES = 500
_BOOTSTRAP_SEED = 1729
_MINIMUM_OBSERVATIONS_FOR_BOOTSTRAP = 50
_MINIMUM_BOOTSTRAP_SUCCESSES = 10
_TOTAL_EFFECT_NUMERICALLY_ZERO = 1e-12
_CONFIDENCE_PERCENTILES = (2.5, 97.5)


def _with_token_inflation_excess(data):
    """Derive the mediator ``token_inflation_excess`` = token_inflation_ratio − 1.

    The ratio is definitionally 1.0 on clean rows, so τ ≡ 1 + is_perturbed·(τ−1)
    and a separate design/06 §6.6 ``perturbed:token_inflation_tau`` interaction
    column would be perfectly collinear. In the excess coding (0 for clean) the
    mediator slope IS that H1 interaction.
    """
    return data.assign(
        token_inflation_excess=data["token_inflation_ratio"] - 1.0)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MixedEffectsLogisticResult:
    """Summary of a fitted crossed mixed-effects logistic model."""
    converged: bool
    method: ConvergenceMethod
    log_likelihood: Optional[float]
    n_observations: int
    n_items: int
    n_models: int
    fixed_effects: dict           # {parameter_name: {"coef": float, "or": float, "p": float}}
    random_effects_variance: dict # {"item_intercept": float, "model_intercept": float, ...}
    model_summary: Optional[str]  # statsmodels model.summary() text, for the paper supplement


@dataclass
class MediationResult:
    """Summary of a mediation decomposition (Method B, product-of-coefficients).

    Primary mediator: ``token_inflation_excess`` = token_inflation_ratio − 1
    (excess subword fragmentation; exactly 0 for clean rows by definition).
    Supplementary mediator: ``subword_count_change`` (absolute subword count
    difference); reported alongside the primary for robustness.

    ``indirect_effect`` with ``bootstrap_ci_indirect`` is the primary reported
    quantity. ``proportion_mediated`` is a ratio with the total effect in the
    denominator and explodes when the total effect is near zero (VanderWeele);
    it is therefore reported only when the bootstrapped total-effect CI
    excludes zero — otherwise it is None and ``proportion_mediated_reason``
    says why.
    """
    total_effect: float
    direct_effect: float
    indirect_effect: float                      # through token_inflation_excess
    proportion_mediated: Optional[float]
    treatment_on_mediator_coef: float           # α: effect of is_perturbed on mediator
    mediator_on_outcome_coef: float             # β: effect of mediator on is_correct
    n_observations: int
    bootstrap_ci_proportion: Optional[tuple]    # (low, high) at 95%, or None if n < 50
    # Supplementary mediator (subword_count_change) — None when column absent.
    supplementary_indirect_effect: Optional[float] = None
    supplementary_proportion_mediated: Optional[float] = None
    bootstrap_ci_indirect: Optional[tuple] = None
    bootstrap_ci_total: Optional[tuple] = None
    proportion_mediated_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Mixed-effects logistic regression (design/06 §6.6)
# ---------------------------------------------------------------------------

def fit_crossed_mixed_effects_logistic(data) -> MixedEffectsLogisticResult:
    """Fit a crossed random-effects logistic regression.

    Expected columns in ``data`` (pandas DataFrame):
      - is_correct      (int 0/1)   — binary outcome
      - is_perturbed    (int 0/1)   — clean=0, perturbed=1 within-item contrast
      - task_id         (str)       — item identifier (random effect)
      - model_revision  (str)       — model identifier (random effect)
      - token_inflation_ratio (float) — mediation covariate
      - edit_budget_k   (int)       — severity (optional; included when present)
      - precision       (str)       — fp16/awq/gptq (optional fixed effect)

    Random-effects specification (Baayen, Davidson & Bates 2008):
      (1 + is_perturbed | task_id) + (1 + is_perturbed | model_revision)

    Convergence fallback ladder: statsmodels' default quasi-Newton cascade →
    Nelder-Mead → GLM approximation (see ``ConvergenceMethod``).
    """
    try:
        import statsmodels.formula.api as smf
        import numpy as np
    except ImportError as error:
        raise ImportError(
            "fit_crossed_mixed_effects_logistic requires statsmodels and pandas: "
            "pip install statsmodels pandas") from error

    if not hasattr(data, "columns"):
        raise TypeError("data must be a pandas DataFrame")

    required = {"is_correct", "is_perturbed", "task_id", "model_revision",
                "token_inflation_ratio"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    n_obs = len(data)
    n_items = data["task_id"].nunique()
    n_models = data["model_revision"].nunique()

    # token_inflation_excess carries the H1 interaction (see
    # _with_token_inflation_excess); word_length_before is the preregistered
    # word-length confound control (design/06 §6.6); subword_count_change is
    # the supplementary mediator. Single-level factors are dropped — they add
    # only a singularity.
    data = _with_token_inflation_excess(data)
    fixed_terms = ["is_perturbed", "token_inflation_excess"]
    if "word_length_before" in data.columns:
        fixed_terms.append("word_length_before")
    if "subword_count_change" in data.columns:
        fixed_terms.append("subword_count_change")
    if "edit_budget_k" in data.columns:
        fixed_terms.append("edit_budget_k")
    if "precision" in data.columns and data["precision"].nunique() > 1:
        fixed_terms.extend(["C(precision)", "is_perturbed:C(precision)"])
    if "operation" in data.columns and data["operation"].nunique() > 1:
        fixed_terms.append("C(operation)")

    fixed_formula = " + ".join(fixed_terms)
    formula = f"is_correct ~ {fixed_formula}"

    statistical_fit_failure_exceptions = _fit_failure_exceptions()

    # fit_failure_reasons accumulates across rungs so a totally-non-converged
    # result still says why each rung was skipped, not just "GLM ran".
    fit_failure_reasons: list[str] = []
    for method_name, fit_kwargs in [
        # method omitted -> statsmodels' own default cascade (bfgs, lbfgs, cg).
        (ConvergenceMethod.QUASI_NEWTON_CASCADE, {"maxiter": 200}),
        (ConvergenceMethod.NELDER_MEAD_FALLBACK, {"method": "nm", "maxiter": 400}),
    ]:
        try:
            model = smf.mixedlm(
                formula, data=data,
                groups=data["task_id"],
                re_formula="~is_perturbed",
            )
            result = model.fit(reml=False, **fit_kwargs)
            if result.converged:
                return _pack_mixed_result(
                    result, method_name, n_obs, n_items, n_models)
            fit_failure_reasons.append(f"{method_name}: fit ran but did not converge")
        except statistical_fit_failure_exceptions as error:
            fit_failure_reasons.append(f"{method_name}: {error}")
            continue

    # GLM approximation fallback: treat item and model as fixed factors.
    try:
        import statsmodels.api as sm

        glm_formula = f"is_correct ~ {fixed_formula} + C(task_id) + C(model_revision)"
        glm_model = smf.glm(
            glm_formula, data=data,
            family=sm.families.Binomial())
        glm_result = glm_model.fit()
        core_params = {
            name: {
                "coef": float(glm_result.params[name]),
                "or": float(np.exp(glm_result.params[name])),
                "p": float(glm_result.pvalues[name]),
            }
            for name in fixed_terms
            if name in glm_result.params
        }
        return MixedEffectsLogisticResult(
            converged=True,
            method=ConvergenceMethod.GLM_APPROXIMATION,
            log_likelihood=float(glm_result.llf),
            n_observations=n_obs,
            n_items=n_items,
            n_models=n_models,
            fixed_effects=core_params,
            random_effects_variance={},
            model_summary=str(glm_result.summary()),
        )
    except statistical_fit_failure_exceptions as error:
        fit_failure_reasons.append(f"{ConvergenceMethod.GLM_APPROXIMATION}: {error}")
        return MixedEffectsLogisticResult(
            converged=False,
            method=ConvergenceMethod.GLM_APPROXIMATION,
            log_likelihood=None,
            n_observations=n_obs,
            n_items=n_items,
            n_models=n_models,
            fixed_effects={},
            random_effects_variance={},
            model_summary="; ".join(fit_failure_reasons),
        )


def _pack_mixed_result(result, method_name: ConvergenceMethod,
                       n_obs: int, n_items: int, n_models: int
                       ) -> MixedEffectsLogisticResult:
    import numpy as np
    fixed_effects = {}
    for name in result.params.index:
        if "Group" in name or "Intercept" in name:
            continue
        fixed_effects[name] = {
            "coef": float(result.params[name]),
            "or": float(np.exp(result.params[name])),
            "p": float(result.pvalues[name]) if name in result.pvalues else float("nan"),
        }

    # Random-effects variance is supplementary reporting, not part of the
    # core result — a malformed or absent cov_re (AttributeError/KeyError
    # from an unexpected statsmodels result shape) degrades to an empty dict
    # rather than failing the whole fit.
    re_var = {}
    try:
        for key, value in result.cov_re.to_dict().items():
            re_var[key] = {sub_key: float(sub_val) for sub_key, sub_val in value.items()}
    except (AttributeError, KeyError, ValueError):
        pass

    return MixedEffectsLogisticResult(
        converged=bool(result.converged),
        method=method_name,
        log_likelihood=float(result.llf) if hasattr(result, "llf") else None,
        n_observations=n_obs,
        n_items=n_items,
        n_models=n_models,
        fixed_effects=fixed_effects,
        random_effects_variance=re_var,
        model_summary=str(result.summary()),
    )


# ---------------------------------------------------------------------------
# Mediation — Method B, product-of-coefficients (design/06 §6.8)
# ---------------------------------------------------------------------------

def compute_mediation_proportion(data) -> MediationResult:
    """Estimate the proportion of typo-induced accuracy loss mediated by
    token-inflation (Method B: product-of-coefficients, Imai et al. 2010).

    Expected columns in ``data`` (pandas DataFrame; Regime-A perturbed rows
    plus their matching clean rows): is_correct (0/1), is_perturbed (0/1),
    token_inflation_ratio (1.0 on clean rows), task_id; optionally
    subword_count_change for the supplementary mediator.

    Per mediator M (primary: token_inflation_excess = ratio − 1, see
    _with_token_inflation_excess):

      M ~ is_perturbed                 → α
      is_correct ~ is_perturbed + M    → β (on M), γ (on is_perturbed)
      indirect = α·β,  total = γ + α·β,  proportion = indirect / total

    The primary reported quantity is the indirect effect with its bootstrap
    CI. proportion_mediated is withheld (None, with a machine-readable
    proportion_mediated_reason) unless the bootstrapped total-effect CI
    excludes zero, because the ratio is unstable near a zero denominator.
    """
    try:
        import statsmodels.formula.api  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "compute_mediation_proportion requires statsmodels and pandas: "
            "pip install statsmodels pandas") from error

    if not hasattr(data, "columns"):
        raise TypeError("data must be a pandas DataFrame")

    required = {"is_correct", "is_perturbed", "token_inflation_ratio", "task_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    data = _with_token_inflation_excess(data)
    primary = _fit_product_of_coefficients(
        data, PRIMARY_MEDIATOR_COLUMN, _MEDIATION_MAX_ITERATIONS)

    supplementary: Optional[_MediationDecomposition] = None
    if SUPPLEMENTARY_MEDIATOR_COLUMN in data.columns:
        try:
            supplementary = _fit_product_of_coefficients(
                data, SUPPLEMENTARY_MEDIATOR_COLUMN, _MEDIATION_MAX_ITERATIONS)
        except _fit_failure_exceptions():
            pass  # supplementary is advisory; primary mediation must succeed

    intervals = (_bootstrap_mediation_intervals(data) or {}
                 if len(data) >= _MINIMUM_OBSERVATIONS_FOR_BOOTSTRAP else {})
    proportion, withheld_reason = _guarded_proportion_mediated(
        primary.indirect_effect, primary.total_effect, intervals.get("total"))

    supplementary_proportion = (
        supplementary.indirect_effect / supplementary.total_effect
        if supplementary is not None
        and abs(supplementary.total_effect) > _TOTAL_EFFECT_NUMERICALLY_ZERO
        else None)

    return MediationResult(
        total_effect=primary.total_effect,
        direct_effect=primary.treatment_on_outcome,
        indirect_effect=primary.indirect_effect,
        proportion_mediated=proportion,
        treatment_on_mediator_coef=primary.treatment_on_mediator,
        mediator_on_outcome_coef=primary.mediator_on_outcome,
        n_observations=len(data),
        bootstrap_ci_proportion=intervals.get("proportion"),
        supplementary_indirect_effect=(
            supplementary.indirect_effect if supplementary is not None else None),
        supplementary_proportion_mediated=supplementary_proportion,
        bootstrap_ci_indirect=intervals.get("indirect"),
        bootstrap_ci_total=intervals.get("total"),
        proportion_mediated_reason=withheld_reason,
    )


@dataclass(frozen=True)
class _MediationDecomposition:
    treatment_on_mediator: float   # α
    mediator_on_outcome: float     # β
    treatment_on_outcome: float    # γ (the direct effect)

    @property
    def indirect_effect(self) -> float:
        return self.treatment_on_mediator * self.mediator_on_outcome

    @property
    def total_effect(self) -> float:
        return self.treatment_on_outcome + self.indirect_effect


def _fit_failure_exceptions() -> tuple:
    """Exceptions statsmodels/numpy raise for a numerical fit failure (singular
    matrix, non-convergence, quasi-complete separation) — expected, and what
    the fallback ladders absorb. Anything else (TypeError, KeyError, ...) is a
    real bug in the calling code and must propagate."""
    import numpy
    return (ValueError, RuntimeError, numpy.linalg.LinAlgError)


def _fit_product_of_coefficients(
        data, mediator_column: str, max_iterations: int) -> _MediationDecomposition:
    import statsmodels.formula.api as smf

    mediator_fit = smf.mixedlm(
        f"{mediator_column} ~ is_perturbed",
        data=data, groups=data["task_id"]).fit(reml=False, maxiter=max_iterations)
    outcome_fit = smf.mixedlm(
        f"is_correct ~ is_perturbed + {mediator_column}",
        data=data, groups=data["task_id"]).fit(reml=False, maxiter=max_iterations)
    return _MediationDecomposition(
        treatment_on_mediator=float(
            mediator_fit.params.get("is_perturbed", float("nan"))),
        mediator_on_outcome=float(
            outcome_fit.params.get(mediator_column, float("nan"))),
        treatment_on_outcome=float(
            outcome_fit.params.get("is_perturbed", float("nan"))),
    )


def _guarded_proportion_mediated(
        indirect_effect: float, total_effect: float,
        total_interval: Optional[tuple]) -> tuple:
    """Return (proportion, None) when reportable, else (None, reason)."""
    if abs(total_effect) <= _TOTAL_EFFECT_NUMERICALLY_ZERO:
        return None, PROPORTION_WITHHELD_TOTAL_NUMERICALLY_ZERO
    if total_interval is None:
        return None, PROPORTION_WITHHELD_NO_BOOTSTRAP
    interval_low, interval_high = total_interval
    if interval_low <= 0.0 <= interval_high:
        return None, PROPORTION_WITHHELD_TOTAL_CI_INCLUDES_ZERO
    return indirect_effect / total_effect, None


def _bootstrap_mediation_intervals(
        data,
        n_resamples: int = _BOOTSTRAP_RESAMPLES,
        seed: int = _BOOTSTRAP_SEED) -> Optional[dict]:
    """Cluster (by-item) percentile-bootstrap CIs for the primary mediator's
    indirect effect, total effect, and proportion mediated.

    Each resampled item gets a fresh task_id so an item drawn twice counts as
    two clusters — filtering the original frame with ``isin`` would silently
    deduplicate and understate the variance.
    """
    import numpy
    import pandas

    rng = numpy.random.default_rng(seed)
    frames_by_task_id = {task_id: frame for task_id, frame in data.groupby("task_id")}
    task_ids = numpy.array(list(frames_by_task_id))

    decompositions: list[_MediationDecomposition] = []
    for _ in range(n_resamples):
        sample = pandas.concat(
            [frames_by_task_id[task_id].assign(task_id=f"{task_id}#{draw_index}")
             for draw_index, task_id in enumerate(
                 rng.choice(task_ids, size=len(task_ids), replace=True))],
            ignore_index=True)
        try:
            decompositions.append(_fit_product_of_coefficients(
                sample, PRIMARY_MEDIATOR_COLUMN, _BOOTSTRAP_MAX_ITERATIONS))
        except _fit_failure_exceptions():
            continue  # this resample failed to fit; skip it, keep resampling

    if len(decompositions) < _MINIMUM_BOOTSTRAP_SUCCESSES:
        return None

    def percentile_interval(values: list) -> Optional[tuple]:
        if len(values) < _MINIMUM_BOOTSTRAP_SUCCESSES:
            return None
        low_percentile, high_percentile = _CONFIDENCE_PERCENTILES
        return (float(numpy.percentile(values, low_percentile)),
                float(numpy.percentile(values, high_percentile)))

    return {
        "indirect": percentile_interval(
            [decomposition.indirect_effect for decomposition in decompositions]),
        "total": percentile_interval(
            [decomposition.total_effect for decomposition in decompositions]),
        "proportion": percentile_interval(
            [decomposition.indirect_effect / decomposition.total_effect
             for decomposition in decompositions
             if abs(decomposition.total_effect) > _TOTAL_EFFECT_NUMERICALLY_ZERO]),
    }
