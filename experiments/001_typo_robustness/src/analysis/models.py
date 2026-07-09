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

Convergence fallback ladder actually implemented here (see
``ConvergenceMethod`` for how this differs from design/06 §6.6's
pre-registered *structural* contingency, which this function does not
implement):
  1. Statsmodels' own default quasi-Newton cascade (bfgs -> lbfgs -> cg) over
     the maximal model.
  2. A derivative-free fallback optimizer (Nelder-Mead) over the same model.
  3. A GLM approximation that treats item/model as fixed factors (use when
     random-effects convergence fails outright with only ~5 model levels) —
     this step is what design/06 §6.6 step 3 describes.
The fallback is triggered automatically; the ``method`` field in the result
records which step succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from enums import ConvergenceMethod


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

    Primary mediator: ``token_inflation_ratio`` (subword fragmentation ratio).
    Supplementary mediator: ``subword_count_change`` (absolute subword count
    difference); reported alongside the primary for robustness.
    """
    total_effect: float
    direct_effect: float
    indirect_effect: float                      # through token_inflation_ratio
    proportion_mediated: Optional[float]
    treatment_on_mediator_coef: float           # α: effect of is_perturbed on mediator
    mediator_on_outcome_coef: float             # β: effect of mediator on is_correct
    n_observations: int
    bootstrap_ci_proportion: Optional[tuple]    # (low, high) at 95%, or None if n < 50
    # Supplementary mediator (subword_count_change) — None when column absent.
    supplementary_indirect_effect: Optional[float] = None
    supplementary_proportion_mediated: Optional[float] = None


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

    # Build formula — add optional covariates when present.
    # word_length_before: controls for the confound that longer words are both
    # more likely to be perturbed and more likely to be task-critical; declared
    # as a preregistered covariate (design/06 §6.6, Workstream 3/9).
    # subword_count_change: supplementary tokenization mediator (Workstream 9).
    fixed_terms = ["is_perturbed", "token_inflation_ratio"]
    if "word_length_before" in data.columns:
        fixed_terms.append("word_length_before")
    if "subword_count_change" in data.columns:
        fixed_terms.append("subword_count_change")
    if "edit_budget_k" in data.columns:
        fixed_terms.append("edit_budget_k")
    if "precision" in data.columns:
        fixed_terms.append("C(precision)")
    if "operation" in data.columns:
        fixed_terms.append("C(operation)")

    fixed_formula = " + ".join(fixed_terms)
    formula = f"is_correct ~ {fixed_formula}"

    # Exceptions statsmodels/numpy raise for a numerical fit failure (singular
    # matrix, non-convergence, quasi-complete separation) — expected, and
    # exactly what the fallback ladder exists to absorb. Anything else
    # (TypeError, AttributeError, KeyError, ...) is a real bug in the calling
    # code, not a fit failure, and must propagate rather than be silently
    # treated as "try the next method."
    statistical_fit_failure_exceptions = (ValueError, RuntimeError, np.linalg.LinAlgError)

    # Convergence fallback ladder — see ConvergenceMethod's docstring for what
    # each rung actually is and how it differs from design/06 §6.6's
    # pre-registered structural contingency. The outer label is our result
    # tag (ConvergenceMethod); fit_kwargs["method"] (when present) is
    # statsmodels' own solver name for that rung.
    # Failure reasons accumulate here so a totally-non-converged result still
    # tells the caller why each rung was skipped, instead of just "GLM ran".
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

    Expected columns in ``data`` (pandas DataFrame; perturbed Regime-A rows only):
      - is_correct          (int 0/1)
      - is_perturbed        (int 0/1)
      - token_inflation_ratio (float)   — the mediator τ
      - task_id             (str)       — for random effects
      - model_revision      (str)       — for random effects

    Step 1: regress mediator on treatment → coefficient α
      token_inflation_ratio ~ is_perturbed + (1|task_id)

    Step 2: regress outcome on treatment + mediator → coefficients β (mediator),
      γ (treatment)
      is_correct ~ is_perturbed + token_inflation_ratio + (1|task_id)

    Decomposition:
      indirect (mediated) effect = α × β
      total effect               = γ + α × β
      proportion mediated        = (α × β) / total_effect

    A bootstrap CI on proportion_mediated is computed when n >= 50.
    """
    try:
        import statsmodels.formula.api as smf
        import numpy as np
    except ImportError as error:
        raise ImportError(
            "compute_mediation_proportion requires statsmodels and pandas: "
            "pip install statsmodels pandas") from error

    # See fit_crossed_mixed_effects_logistic for why this tuple, not Exception.
    statistical_fit_failure_exceptions = (ValueError, RuntimeError, np.linalg.LinAlgError)

    if not hasattr(data, "columns"):
        raise TypeError("data must be a pandas DataFrame")

    required = {"is_correct", "is_perturbed", "token_inflation_ratio", "task_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    n_obs = len(data)

    # method omitted throughout this function -> statsmodels' own default
    # quasi-Newton cascade (bfgs, then lbfgs, then cg; see
    # ConvergenceMethod.QUASI_NEWTON_CASCADE).
    # Step 1: mediator model.
    mediator_model = smf.mixedlm(
        "token_inflation_ratio ~ is_perturbed",
        data=data, groups=data["task_id"])
    mediator_result = mediator_model.fit(reml=False, maxiter=200)
    alpha = float(mediator_result.params.get("is_perturbed", float("nan")))

    # Step 2: outcome model.
    outcome_model = smf.mixedlm(
        "is_correct ~ is_perturbed + token_inflation_ratio",
        data=data, groups=data["task_id"])
    outcome_result = outcome_model.fit(reml=False, maxiter=200)
    beta = float(outcome_result.params.get("token_inflation_ratio", float("nan")))
    gamma = float(outcome_result.params.get("is_perturbed", float("nan")))

    indirect = alpha * beta
    total = gamma + indirect
    proportion = (indirect / total) if abs(total) > 1e-12 else None

    # Supplementary mediator: subword_count_change (Workstream 9).
    # Reported alongside the primary; shares the same outcome model but uses a
    # different mediator column, so we re-run Step 1 on the supplementary column.
    supp_indirect: Optional[float] = None
    supp_proportion: Optional[float] = None
    if "subword_count_change" in data.columns:
        try:
            supp_med_model = smf.mixedlm(
                "subword_count_change ~ is_perturbed",
                data=data, groups=data["task_id"])
            supp_med_result = supp_med_model.fit(reml=False, maxiter=200)
            alpha_s = float(supp_med_result.params.get("is_perturbed", float("nan")))

            supp_out_model = smf.mixedlm(
                "is_correct ~ is_perturbed + subword_count_change",
                data=data, groups=data["task_id"])
            supp_out_result = supp_out_model.fit(reml=False, maxiter=200)
            beta_s = float(supp_out_result.params.get("subword_count_change", float("nan")))
            gamma_s = float(supp_out_result.params.get("is_perturbed", float("nan")))

            supp_indirect = alpha_s * beta_s
            supp_total = gamma_s + supp_indirect
            supp_proportion = (supp_indirect / supp_total) if abs(supp_total) > 1e-12 else None
        except statistical_fit_failure_exceptions:
            pass  # supplementary is advisory; primary mediation must succeed

    # Bootstrap CI on proportion mediated (n >= 50 only).
    bootstrap_ci: Optional[tuple] = None
    if n_obs >= 50 and proportion is not None:
        bootstrap_ci = _bootstrap_proportion_mediated(data, n_resamples=500, seed=1729)

    return MediationResult(
        total_effect=total,
        direct_effect=gamma,
        indirect_effect=indirect,
        proportion_mediated=proportion,
        treatment_on_mediator_coef=alpha,
        mediator_on_outcome_coef=beta,
        n_observations=n_obs,
        bootstrap_ci_proportion=bootstrap_ci,
        supplementary_indirect_effect=supp_indirect,
        supplementary_proportion_mediated=supp_proportion,
    )


def _bootstrap_proportion_mediated(data, n_resamples: int = 500, seed: int = 1729
                                    ) -> Optional[tuple]:
    """Percentile bootstrap CI for proportion_mediated (for large n only)."""
    import numpy as np
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    # See fit_crossed_mixed_effects_logistic for why this tuple, not Exception.
    statistical_fit_failure_exceptions = (ValueError, RuntimeError, np.linalg.LinAlgError)
    rng = np.random.default_rng(seed)
    proportions = []
    task_ids = data["task_id"].unique()

    for _ in range(n_resamples):
        sampled_ids = rng.choice(task_ids, size=len(task_ids), replace=True)
        sample = data[data["task_id"].isin(sampled_ids)]
        try:
            med_result = smf.mixedlm(
                "token_inflation_ratio ~ is_perturbed",
                data=sample, groups=sample["task_id"]).fit(
                reml=False, maxiter=100)
            alpha_b = float(med_result.params.get("is_perturbed", float("nan")))

            out_result = smf.mixedlm(
                "is_correct ~ is_perturbed + token_inflation_ratio",
                data=sample, groups=sample["task_id"]).fit(
                reml=False, maxiter=100)
            beta_b = float(out_result.params.get("token_inflation_ratio", float("nan")))
            gamma_b = float(out_result.params.get("is_perturbed", float("nan")))

            total_b = gamma_b + alpha_b * beta_b
            if abs(total_b) > 1e-12:
                proportions.append((alpha_b * beta_b) / total_b)
        except statistical_fit_failure_exceptions:
            continue  # this resample failed to fit; skip it, keep resampling

    if len(proportions) < 10:
        return None
    proportions_arr = np.array(proportions)
    low = float(np.percentile(proportions_arr, 2.5))
    high = float(np.percentile(proportions_arr, 97.5))
    return (low, high)
