"""Confirmatory logistic GLMM and causal mediation (design/06 §6.6 and §6.8).

Provenance
----------
Confirmatory model — a **logistic** mixed model, never a linear one:
  Jaeger (2008) — why binary accuracy demands a logit link, not a linear model.
  Baayen, Davidson & Bates (2008) — crossed random effects for items and models.
  Barr, Levy, Scheepers & Tily (2013) — maximal random effects, and the
  convergence-simplification guidance (pp. 275–276) the ladder below encodes.
The estimator is ``lme4::glmer`` (binomial, logit link, bobyqa optimizer)
reached through the rpy2 bridge — the same estimator the cited literature is
built on. When no R installation is present the ladder degrades to the
pure-Python fixed-factor logistic GLM rung with a loud ``method`` label.

Mediation — the general algorithm of Imai, Keele & Tingley (2010):
  a mixed linear model for the (continuous) mediator, a mixed **logistic**
  model (by-item intercept) for the binary outcome, and effects computed on
  the probability scale by the paper's quasi-Bayesian Monte Carlo algorithm
  (1,000 parameter draws), reported conditional on the median item. The paper
  warns (p. 316) that the linear product-of-coefficients shortcut does not
  generalize to binary outcomes, so that shortcut survives only as the
  clearly-labeled offline fallback. The by-item structure is not optional:
  on the pilot data, a pooled (no-item-effect) outcome model flips the
  mediator coefficient's sign through between-item confounding.

A linear-probability mixed model is retained ONLY as a labeled
robustness-appendix estimator (``fit_linear_probability_mixed_model``); its
coefficients are risk differences and are never exponentiated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from enums import ConvergenceMethod


# Machine-readable reasons why proportion_mediated is withheld.
PROPORTION_WITHHELD_TOTAL_NUMERICALLY_ZERO = "total_effect_numerically_zero"
PROPORTION_WITHHELD_TOTAL_CI_INCLUDES_ZERO = "total_effect_ci_includes_zero"
PROPORTION_WITHHELD_NO_BOOTSTRAP = "bootstrap_ci_unavailable"

PRIMARY_MEDIATOR_COLUMN = "token_inflation_excess"
SUPPLEMENTARY_MEDIATOR_COLUMN = "subword_count_change"

_MINIMUM_OBSERVATIONS_FOR_BOOTSTRAP = 50
_MINIMUM_BOOTSTRAP_SUCCESSES = 10
_TOTAL_EFFECT_NUMERICALLY_ZERO = 1e-12
_CONFIDENCE_PERCENTILES = (2.5, 97.5)

_GLMER_MAX_FUNCTION_EVALUATIONS = 200_000

_LINEAR_PROBABILITY_METHOD_LABEL = "linear_probability_mixed_model"


def _with_token_inflation_excess(data):
    """Derive the mediator ``token_inflation_excess`` = token_inflation_ratio − 1.

    The ratio is definitionally 1.0 on clean rows, so τ ≡ 1 + is_perturbed·(τ−1)
    and a separate ``perturbed:token_inflation_tau`` interaction column would be
    perfectly collinear. In the excess coding (0 for clean) the mediator slope
    IS the design/06 §6.6 H1 interaction.
    """
    return data.assign(
        token_inflation_excess=data["token_inflation_ratio"] - 1.0)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MixedEffectsLogisticResult:
    """Summary of the confirmatory logistic mixed model (or its GLM fallback).

    ``fixed_effects`` values carry ``coef`` (log-odds), ``or`` (exp(coef) — a
    real odds ratio, because every rung of this ladder is logistic),
    ``std_error``, and ``p`` (Wald z). ``ladder_notes`` records why each
    earlier rung was rejected, so a non-maximal ``method`` is self-explaining.
    """
    converged: bool
    method: ConvergenceMethod
    log_likelihood: Optional[float]
    n_observations: int
    n_items: int
    n_models: int
    fixed_effects: dict           # {term: {"coef", "or", "std_error", "p"}}
    random_effects_variance: dict # {label: variance}
    is_singular_fit: bool = False
    ladder_notes: list = field(default_factory=list)
    model_summary: Optional[str] = None


@dataclass
class LinearProbabilityMixedResult:
    """Robustness-appendix estimator: linear mixed model on the 0/1 outcome.

    Coefficients are risk differences (probability-scale changes). There is
    deliberately no odds-ratio field — exponentiating a linear-probability
    coefficient produces a number with no interpretation, which is exactly the
    mislabeling this dataclass exists to prevent.
    """
    converged: bool
    method: str
    n_observations: int
    fixed_effects: dict           # {term: {"coef", "p"}}
    random_effects_variance: dict
    model_summary: Optional[str] = None


@dataclass
class MediationResult:
    """Mediation decomposition on the probability scale (Imai et al. 2010).

    ``total_effect = direct_effect + indirect_effect`` exactly (telescoping
    counterfactual decomposition): with p(t, m) the predicted probability of a
    correct answer at treatment t and the mediator's predicted level under
    treatment m,
        indirect = p(1, M(1)) − p(1, M(0))     (through the mediator)
        direct   = p(1, M(0)) − p(0, M(0))     (holding the mediator at control)
        total    = p(1, M(1)) − p(0, M(0)).
    All three are percentage-point effects, directly comparable to the paired
    Δ of the per-cell analysis.

    ``proportion_mediated`` is a ratio with the total effect in the
    denominator; it explodes near a zero total (VanderWeele), so it is
    reported only when the bootstrapped total-effect CI excludes zero —
    otherwise it is None and ``proportion_mediated_reason`` says why.

    ``treatment_on_mediator_coef`` is the mediator model's α (mediator units);
    ``mediator_on_outcome_coef`` is the outcome model's β — log-odds per
    mediator unit on the quasi-Bayes path, probability per unit on the
    offline fallback path (see ``estimator``).
    """
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: Optional[float]
    treatment_on_mediator_coef: float
    mediator_on_outcome_coef: float
    n_observations: int
    bootstrap_ci_proportion: Optional[tuple]
    supplementary_indirect_effect: Optional[float] = None
    supplementary_proportion_mediated: Optional[float] = None
    bootstrap_ci_indirect: Optional[tuple] = None
    bootstrap_ci_total: Optional[tuple] = None
    proportion_mediated_reason: Optional[str] = None
    # Which estimator produced these numbers (quasi-Bayes mixed-logistic vs
    # labeled offline fallback) — set by compute_mediation_proportion.
    estimator: str = ""


# ---------------------------------------------------------------------------
# Fixed-effect terms, rendered for both formula dialects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedEffectTerm:
    """One fixed-effect term in both formula dialects: R's (character columns
    become factors automatically) and patsy's (categoricals need C())."""
    r_syntax: str
    patsy_syntax: str


def _confirmatory_fixed_effect_terms(data) -> list[FixedEffectTerm]:
    """The design/06 §6.6 fixed-effect structure, restricted to columns that
    vary in ``data`` (a single-level factor adds only a singularity)."""
    terms = [
        FixedEffectTerm("is_perturbed", "is_perturbed"),
        FixedEffectTerm("token_inflation_excess", "token_inflation_excess"),
    ]
    if "word_length_before" in data.columns:
        terms.append(FixedEffectTerm("word_length_before", "word_length_before"))
    if "subword_count_change" in data.columns:
        terms.append(FixedEffectTerm("subword_count_change", "subword_count_change"))
    if "edit_budget_k" in data.columns and data["edit_budget_k"].nunique() > 1:
        terms.append(FixedEffectTerm("edit_budget_k", "edit_budget_k"))
    if "precision" in data.columns and data["precision"].nunique() > 1:
        terms.append(FixedEffectTerm("precision", "C(precision)"))
        terms.append(FixedEffectTerm("is_perturbed:precision",
                                     "is_perturbed:C(precision)"))
    # `operation` is deliberately NOT a fixed effect: clean rows carry
    # operation "none", so an operation factor is perfectly collinear with
    # is_perturbed and would make the design matrix singular. Operation-level
    # effects are reported by the per-cell descriptive analysis instead
    # (design/03 §3.8).
    return terms


# ---------------------------------------------------------------------------
# The rpy2 → lme4 bridge
# ---------------------------------------------------------------------------

# None = not yet probed; "" = available; any other string = why unavailable.
_lme4_bridge_unavailable_reason: Optional[str] = None

_GLMER_FIT_FUNCTION_R_SOURCE = """
function(formula_text, model_frame) {
  fitted_model <- lme4::glmer(
    stats::as.formula(formula_text),
    data = model_frame,
    family = stats::binomial(),
    control = lme4::glmerControl(optimizer = "bobyqa",
                                 optCtrl = list(maxfun = %d)))
  coefficient_matrix <- summary(fitted_model)$coefficients
  variance_frame <- as.data.frame(lme4::VarCorr(fitted_model))
  variance_labels <- trimws(paste(
    variance_frame$grp,
    ifelse(is.na(variance_frame$var1), "", variance_frame$var1),
    ifelse(is.na(variance_frame$var2), "", variance_frame$var2)))
  list(
    coefficient_names = rownames(coefficient_matrix),
    estimates = unname(coefficient_matrix[, "Estimate"]),
    standard_errors = unname(coefficient_matrix[, "Std. Error"]),
    p_values = unname(coefficient_matrix[, "Pr(>|z|)"]),
    log_likelihood = as.numeric(stats::logLik(fitted_model)),
    is_converged = length(fitted_model@optinfo$conv$lme4$messages) == 0,
    is_singular = lme4::isSingular(fitted_model),
    variance_labels = variance_labels,
    variance_values = as.numeric(variance_frame$vcov),
    summary_text = paste(utils::capture.output(summary(fitted_model)),
                         collapse = "\\n")
  )
}
""" % _GLMER_MAX_FUNCTION_EVALUATIONS


def _probe_lme4_bridge() -> str:
    """'' when rpy2 + R + lme4 are all importable, else the reason they are
    not. Probed once per process."""
    global _lme4_bridge_unavailable_reason
    if _lme4_bridge_unavailable_reason is not None:
        return _lme4_bridge_unavailable_reason
    try:
        from rpy2.robjects.packages import importr
        importr("lme4")
        _lme4_bridge_unavailable_reason = ""
    except Exception as bridge_error:  # rpy2 raises several concrete types,
        # and an absent R installation surfaces as an OSError from cffi.
        _lme4_bridge_unavailable_reason = f"{type(bridge_error).__name__}: {bridge_error}"
    return _lme4_bridge_unavailable_reason


@dataclass(frozen=True)
class _GlmerFit:
    coefficient_names: list
    estimates: list
    standard_errors: list
    p_values: list
    log_likelihood: float
    is_converged: bool
    is_singular: bool
    variance_labels: list
    variance_values: list
    summary_text: str


def _fit_glmer(formula_text: str, data) -> _GlmerFit:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    glmer_fit_function = robjects.r(_GLMER_FIT_FUNCTION_R_SOURCE)
    # Convert the frame INSIDE the pandas converter but make the call OUTSIDE
    # it: with the pandas converter active for the call, rpy2 converts the
    # returned named R list to a NamedList (no .rx2), which broke every rung
    # on the first Colab run. Under the default converter the result stays a
    # ListVector.
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_model_frame = robjects.conversion.get_conversion().py2rpy(data)
    r_result = glmer_fit_function(formula_text, r_model_frame)

    def field_of(name: str):
        return list(r_result.rx2(name))

    return _GlmerFit(
        coefficient_names=[str(name) for name in field_of("coefficient_names")],
        estimates=[float(value) for value in field_of("estimates")],
        standard_errors=[float(value) for value in field_of("standard_errors")],
        p_values=[float(value) for value in field_of("p_values")],
        log_likelihood=float(field_of("log_likelihood")[0]),
        is_converged=bool(field_of("is_converged")[0]),
        is_singular=bool(field_of("is_singular")[0]),
        variance_labels=[str(label) for label in field_of("variance_labels")],
        variance_values=[float(value) for value in field_of("variance_values")],
        summary_text=str(field_of("summary_text")[0]),
    )


def _random_effect_ladder(model_count: int) -> list[tuple[ConvergenceMethod, str]]:
    """The pre-registered random-effects simplification ladder (Barr 2013).

    With a single model in the data (the pilot case) the by-model grouping
    factor has one level — lme4 refuses it, correctly — so the by-model terms
    are omitted and rungs that then render identically are deduplicated.
    """
    has_multiple_models = model_count > 1
    by_model_maximal = " + (1 + is_perturbed | model_revision)" if has_multiple_models else ""
    by_model_uncorrelated = " + (1 + is_perturbed || model_revision)" if has_multiple_models else ""
    by_model_intercept = " + (1 | model_revision)" if has_multiple_models else ""

    rungs = [
        (ConvergenceMethod.GLMER_MAXIMAL,
         "(1 + is_perturbed | task_id)" + by_model_maximal),
        (ConvergenceMethod.GLMER_NO_RANDOM_CORRELATIONS,
         "(1 + is_perturbed || task_id)" + by_model_uncorrelated),
        (ConvergenceMethod.GLMER_NO_MODEL_SLOPE,
         "(1 + is_perturbed | task_id)" + by_model_intercept),
        (ConvergenceMethod.GLMER_INTERCEPTS_ONLY,
         "(1 | task_id)" + by_model_intercept),
    ]
    deduplicated: list[tuple[ConvergenceMethod, str]] = []
    for method, random_effects in rungs:
        if all(random_effects != seen for _, seen in deduplicated):
            deduplicated.append((method, random_effects))
    return deduplicated


# ---------------------------------------------------------------------------
# The confirmatory model (design/06 §6.6)
# ---------------------------------------------------------------------------

_CONFIRMATORY_MODEL_REQUIRED_COLUMNS = frozenset(
    {"is_correct", "is_perturbed", "task_id", "model_revision",
     "token_inflation_ratio"})


def _validated_model_frame(data):
    if not hasattr(data, "columns"):
        raise TypeError("data must be a pandas DataFrame")
    missing = _CONFIRMATORY_MODEL_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    return _with_token_inflation_excess(data)


def fit_crossed_mixed_effects_logistic(data) -> MixedEffectsLogisticResult:
    """Fit the confirmatory **logistic** model down the pre-registered ladder.

    Rungs 1–4 are ``lme4::glmer`` fits with progressively simpler random
    effects (see ``_random_effect_ladder``); a rung is accepted only when it
    converges with a non-singular random-effects estimate. Rung 5 — also the
    offline fallback when no R/lme4 bridge exists — is a logistic GLM with
    item and model as fixed factors. Every rejected rung leaves a note in
    ``ladder_notes``.

    Expected columns: is_correct (0/1), is_perturbed (0/1), task_id,
    model_revision, token_inflation_ratio; optional word_length_before,
    subword_count_change, edit_budget_k, precision, operation.
    """
    data = _validated_model_frame(data)
    n_observations = len(data)
    n_items = data["task_id"].nunique()
    n_models = data["model_revision"].nunique()
    fixed_terms = _confirmatory_fixed_effect_terms(data)
    ladder_notes: list[str] = []

    bridge_unavailable_reason = _probe_lme4_bridge()
    if bridge_unavailable_reason:
        ladder_notes.append(
            f"glmer rungs unavailable (no R/lme4 bridge): {bridge_unavailable_reason}")
    else:
        r_fixed_formula = " + ".join(term.r_syntax for term in fixed_terms)
        for method, random_effects in _random_effect_ladder(n_models):
            formula_text = f"is_correct ~ {r_fixed_formula} + {random_effects}"
            try:
                glmer_fit = _fit_glmer(formula_text, data)
            except Exception as fit_error:
                ladder_notes.append(f"{method}: {fit_error}")
                continue
            if not glmer_fit.is_converged:
                ladder_notes.append(f"{method}: did not converge")
                continue
            if glmer_fit.is_singular:
                ladder_notes.append(f"{method}: singular random-effects fit")
                continue
            return _pack_glmer_result(
                glmer_fit, method, n_observations, n_items, n_models, ladder_notes)

    return _fit_fixed_effects_logistic_glm(
        data, fixed_terms, n_observations, n_items, n_models, ladder_notes)


def _pack_glmer_result(glmer_fit: _GlmerFit, method: ConvergenceMethod,
                       n_observations: int, n_items: int, n_models: int,
                       ladder_notes: list) -> MixedEffectsLogisticResult:
    import numpy

    fixed_effects = {
        name: {
            "coef": estimate,
            "or": float(numpy.exp(estimate)),
            "std_error": standard_error,
            "p": p_value,
        }
        for name, estimate, standard_error, p_value in zip(
            glmer_fit.coefficient_names, glmer_fit.estimates,
            glmer_fit.standard_errors, glmer_fit.p_values)
        if name != "(Intercept)"
    }
    random_effects_variance = dict(
        zip(glmer_fit.variance_labels, glmer_fit.variance_values))

    return MixedEffectsLogisticResult(
        converged=True,
        method=method,
        log_likelihood=glmer_fit.log_likelihood,
        n_observations=n_observations,
        n_items=n_items,
        n_models=n_models,
        fixed_effects=fixed_effects,
        random_effects_variance=random_effects_variance,
        is_singular_fit=glmer_fit.is_singular,
        ladder_notes=ladder_notes,
        model_summary=glmer_fit.summary_text,
    )


def _fit_fixed_effects_logistic_glm(data, fixed_terms, n_observations: int,
                                    n_items: int, n_models: int,
                                    ladder_notes: list) -> MixedEffectsLogisticResult:
    """Rung 5: logistic GLM with item and model as fixed factors — the
    design/06 §6.6 step-(3) contingency and the offline fallback.

    Caveat (why this is a fallback and never the estimator of record): with
    few rows per item, per-item dummies in a logit carry the classic
    incidental-parameters bias — slope magnitudes inflate several-fold on
    2-row panels — so this rung's coefficients are directionally informative
    but not magnitude-faithful. Confirmatory artifacts come from the glmer
    rungs (see tests/test_models_glmm_mediation.py for the demonstration)."""
    import numpy
    import statsmodels.api as statsmodels_api
    import statsmodels.formula.api as statsmodels_formula

    patsy_fixed_formula = " + ".join(term.patsy_syntax for term in fixed_terms)
    glm_formula = (f"is_correct ~ {patsy_fixed_formula} "
                   f"+ C(task_id) + C(model_revision)")
    try:
        glm_result = statsmodels_formula.glm(
            glm_formula, data=data,
            family=statsmodels_api.families.Binomial()).fit()
    except _fit_failure_exceptions() as glm_error:
        ladder_notes.append(
            f"{ConvergenceMethod.FIXED_EFFECTS_LOGISTIC_GLM}: {glm_error}")
        return MixedEffectsLogisticResult(
            converged=False,
            method=ConvergenceMethod.FIXED_EFFECTS_LOGISTIC_GLM,
            log_likelihood=None,
            n_observations=n_observations,
            n_items=n_items,
            n_models=n_models,
            fixed_effects={},
            random_effects_variance={},
            ladder_notes=ladder_notes,
            model_summary="; ".join(ladder_notes),
        )

    core_term_names = [term.patsy_syntax for term in fixed_terms]
    fixed_effects = {
        name: {
            "coef": float(glm_result.params[name]),
            "or": float(numpy.exp(glm_result.params[name])),
            "std_error": float(glm_result.bse[name]),
            "p": float(glm_result.pvalues[name]),
        }
        for name in core_term_names
        if name in glm_result.params
    }
    return MixedEffectsLogisticResult(
        converged=True,
        method=ConvergenceMethod.FIXED_EFFECTS_LOGISTIC_GLM,
        log_likelihood=float(glm_result.llf),
        n_observations=n_observations,
        n_items=n_items,
        n_models=n_models,
        fixed_effects=fixed_effects,
        random_effects_variance={},
        ladder_notes=ladder_notes,
        model_summary=str(glm_result.summary()),
    )


# ---------------------------------------------------------------------------
# Robustness appendix: the linear-probability mixed model
# ---------------------------------------------------------------------------

def fit_linear_probability_mixed_model(data) -> LinearProbabilityMixedResult:
    """Linear mixed model on the 0/1 outcome — the risk-difference-scale
    robustness check reported in the appendix, never the confirmatory model.

    By-item random intercept and is_perturbed slope, statsmodels ``MixedLM``.
    Coefficients are probability-scale effects; no odds ratios exist here.
    """
    import statsmodels.formula.api as statsmodels_formula

    data = _validated_model_frame(data)
    fixed_formula = " + ".join(
        term.patsy_syntax for term in _confirmatory_fixed_effect_terms(data))

    fit_attempts = [
        {"maxiter": 200},                       # statsmodels' default cascade
        {"method": "nm", "maxiter": 400},       # derivative-free retry
    ]
    last_failure = ""
    for fit_keyword_arguments in fit_attempts:
        try:
            fitted = statsmodels_formula.mixedlm(
                f"is_correct ~ {fixed_formula}", data=data,
                groups=data["task_id"], re_formula="~is_perturbed",
            ).fit(reml=False, **fit_keyword_arguments)
        except _fit_failure_exceptions() as fit_error:
            last_failure = str(fit_error)
            continue
        if fitted.converged:
            return _pack_linear_probability_result(fitted, len(data))
        last_failure = "fit ran but did not converge"

    return LinearProbabilityMixedResult(
        converged=False,
        method=_LINEAR_PROBABILITY_METHOD_LABEL,
        n_observations=len(data),
        fixed_effects={},
        random_effects_variance={},
        model_summary=last_failure,
    )


def _pack_linear_probability_result(fitted, n_observations: int) -> LinearProbabilityMixedResult:
    # Variance components appear in params under names containing "Var" or
    # "Cov"; they are not fixed effects and must not be reported as such
    # (the pre-rewrite implementation leaked "is_perturbed Var" into the
    # fixed-effects table).
    is_variance_component = lambda name: (
        "Var" in name or "Cov" in name or "Group" in name or "Intercept" in name)
    fixed_effects = {
        name: {
            "coef": float(fitted.params[name]),
            "p": float(fitted.pvalues[name]) if name in fitted.pvalues else float("nan"),
        }
        for name in fitted.params.index
        if not is_variance_component(name)
    }
    try:
        random_effects_variance = {
            outer_key: {inner_key: float(inner_value)
                        for inner_key, inner_value in inner.items()}
            for outer_key, inner in fitted.cov_re.to_dict().items()}
    except (AttributeError, KeyError, ValueError):
        random_effects_variance = {}

    return LinearProbabilityMixedResult(
        converged=True,
        method=_LINEAR_PROBABILITY_METHOD_LABEL,
        n_observations=n_observations,
        fixed_effects=fixed_effects,
        random_effects_variance=random_effects_variance,
        model_summary=str(fitted.summary()),
    )


# ---------------------------------------------------------------------------
# Mediation (design/06 §6.8, Method B)
# ---------------------------------------------------------------------------

def _fit_failure_exceptions() -> tuple:
    """Exceptions statsmodels/numpy raise for a numerical fit failure (singular
    matrix, non-convergence, quasi-complete separation) — expected, and what
    the ladders and bootstrap absorb. Anything else is a real bug in the
    calling code and must propagate."""
    import numpy
    return (ValueError, RuntimeError, numpy.linalg.LinAlgError,
            ZeroDivisionError)


# Estimator labels serialized with every mediation result. The item structure
# is load-bearing here: a pooled (no-item-effect) outcome model flips the
# mediator coefficient's sign on the pilot data (between-item confounding —
# items that tokenize long are not exchangeable with items that tokenize
# short), so both estimators below condition on the item.
MEDIATION_ESTIMATOR_QUASIBAYES = (
    "imai_quasibayes_mixed_logistic_conditional_on_median_item")
MEDIATION_ESTIMATOR_OFFLINE_FALLBACK = (
    "linear_probability_within_item_alpha_beta_offline_fallback")

_QUASIBAYES_PARAMETER_DRAWS = 1000
_QUASIBAYES_SEED = 1729

_MEDIATION_QUASIBAYES_R_SOURCE = """
function(model_frame, mediator_column, n_draws, seed) {
  set.seed(seed)
  mediator_model <- lme4::lmer(
    stats::as.formula(paste(mediator_column, "~ is_perturbed + (1 | task_id)")),
    data = model_frame, REML = FALSE)
  outcome_model <- lme4::glmer(
    stats::as.formula(paste("is_correct ~ is_perturbed +", mediator_column,
                            "+ (1 | task_id)")),
    data = model_frame, family = stats::binomial(),
    control = lme4::glmerControl(optimizer = "bobyqa",
                                 optCtrl = list(maxfun = 200000)))

  mediator_draws <- MASS::mvrnorm(n_draws, lme4::fixef(mediator_model),
                                  as.matrix(stats::vcov(mediator_model)))
  outcome_draws <- MASS::mvrnorm(n_draws, lme4::fixef(outcome_model),
                                 as.matrix(stats::vcov(outcome_model)))
  expit <- function(x) 1 / (1 + exp(-x))

  mediator_control <- mediator_draws[, "(Intercept)"]
  mediator_treated <- mediator_control + mediator_draws[, "is_perturbed"]
  probability_at <- function(treatment, mediator_level) {
    expit(outcome_draws[, "(Intercept)"]
          + outcome_draws[, "is_perturbed"] * treatment
          + outcome_draws[, mediator_column] * mediator_level)
  }
  indirect_draws <- (probability_at(1, mediator_treated)
                     - probability_at(1, mediator_control))
  direct_draws <- (probability_at(1, mediator_control)
                   - probability_at(0, mediator_control))
  total_draws <- indirect_draws + direct_draws

  interval <- function(draws) as.numeric(stats::quantile(draws, c(0.025, 0.975)))
  list(
    treatment_on_mediator = as.numeric(lme4::fixef(mediator_model)["is_perturbed"]),
    mediator_on_outcome = as.numeric(lme4::fixef(outcome_model)[mediator_column]),
    indirect_effect = mean(indirect_draws),
    direct_effect = mean(direct_draws),
    total_effect = mean(total_draws),
    ci_indirect = interval(indirect_draws),
    ci_direct = interval(direct_draws),
    ci_total = interval(total_draws),
    ci_proportion = interval((indirect_draws / total_draws)[abs(total_draws) > 1e-12]),
    outcome_converged =
      length(outcome_model@optinfo$conv$lme4$messages) == 0
  )
}
"""

# Pre-registered fallback bootstrap for the offline estimator: by-item cluster
# percentile intervals, B = 1,000 (Hesterberg's 1,000–5,000 sufficiency range;
# each resample refits two mixed models, so the B = 10,000 of the cheap
# per-cell BCa intervals is not the relevant convention here).
_FALLBACK_BOOTSTRAP_RESAMPLES = 1000


@dataclass(frozen=True)
class _MediationDecomposition:
    """One estimator's decomposition. On both estimator paths the three
    effects satisfy total = direct + indirect exactly."""
    treatment_on_mediator: float      # α (mediator units)
    mediator_on_outcome: float        # β (log-odds per unit — quasi-Bayes path;
                                      #    probability per unit — offline fallback)
    indirect_effect: float
    direct_effect: float

    @property
    def total_effect(self) -> float:
        return self.direct_effect + self.indirect_effect


def _fit_quasibayes_mediation(data, mediator_column: str) -> dict:
    """Imai et al. (2010)'s quasi-Bayesian Monte Carlo algorithm with mixed
    models: lmer mediator, glmer (logit, by-item intercept) outcome, effects
    computed on the probability scale from parameter draws, reported
    conditional on the median item (random intercept at zero)."""
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    mediation_function = robjects.r(_MEDIATION_QUASIBAYES_R_SOURCE)
    # Same conversion discipline as _fit_glmer: frame converted inside the
    # pandas converter, call made under the default converter so the returned
    # named list keeps .rx2.
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_model_frame = robjects.conversion.get_conversion().py2rpy(data)
    r_result = mediation_function(
        r_model_frame, mediator_column, _QUASIBAYES_PARAMETER_DRAWS, _QUASIBAYES_SEED)

    def field_of(name: str):
        return list(r_result.rx2(name))

    return {
        "decomposition": _MediationDecomposition(
            treatment_on_mediator=float(field_of("treatment_on_mediator")[0]),
            mediator_on_outcome=float(field_of("mediator_on_outcome")[0]),
            indirect_effect=float(field_of("indirect_effect")[0]),
            direct_effect=float(field_of("direct_effect")[0]),
        ),
        "intervals": {
            "indirect": tuple(float(v) for v in field_of("ci_indirect")),
            "total": tuple(float(v) for v in field_of("ci_total")),
            "proportion": tuple(float(v) for v in field_of("ci_proportion")),
        },
        "outcome_converged": bool(field_of("outcome_converged")[0]),
    }


def _fit_offline_fallback_decomposition(
        data, mediator_column: str) -> _MediationDecomposition:
    """Offline fallback: within-item linear α·β, with the item fixed effects
    absorbed exactly by demeaning every variable within its item (the "within"
    estimator). Uses only within-item variation — the same identification
    logic as the paired McNemar design — so between-item confounding cannot
    touch it, and the pure-numpy least squares makes a B = 1,000 cluster
    bootstrap cost seconds. A linear-probability approximation, labeled as
    such in the result; confirmatory artifacts use the quasi-Bayesian
    estimator."""
    import numpy

    modeled_columns = ["is_correct", "is_perturbed", mediator_column]
    demeaned = (data[modeled_columns]
                - data.groupby("task_id")[modeled_columns].transform("mean"))

    treatment = demeaned["is_perturbed"].to_numpy()
    mediator = demeaned[mediator_column].to_numpy()
    outcome = demeaned["is_correct"].to_numpy()

    treatment_variance = float(treatment @ treatment)
    if treatment_variance <= _TOTAL_EFFECT_NUMERICALLY_ZERO:
        raise ValueError(
            "no within-item treatment variation — every item is all-clean or "
            "all-perturbed, so the within estimator is undefined")
    treatment_on_mediator = float(treatment @ mediator) / treatment_variance

    outcome_design = numpy.column_stack([treatment, mediator])
    (direct_effect, mediator_on_outcome), *_ = numpy.linalg.lstsq(
        outcome_design, outcome, rcond=None)

    return _MediationDecomposition(
        treatment_on_mediator=treatment_on_mediator,
        mediator_on_outcome=float(mediator_on_outcome),
        indirect_effect=treatment_on_mediator * float(mediator_on_outcome),
        direct_effect=float(direct_effect),
    )


def compute_mediation_proportion(data) -> MediationResult:
    """Mediation of the perturbation effect through token inflation.

    Expected columns (Regime-A perturbed rows plus their matching clean rows):
    is_correct (0/1), is_perturbed (0/1), token_inflation_ratio (1.0 on clean
    rows), task_id; optionally subword_count_change for the supplementary
    mediator.

    Estimator selection: with an R/lme4 bridge present, the Imai et al. (2010)
    quasi-Bayesian algorithm over mixed models (see
    ``_fit_quasibayes_mediation``); otherwise the labeled offline fallback
    (within-item linear α·β with a by-item cluster bootstrap). The
    ``estimator`` field of the result says which path produced the numbers.
    """
    try:
        import statsmodels.formula.api  # noqa: F401
    except ImportError as import_error:
        raise ImportError(
            "compute_mediation_proportion requires statsmodels and pandas: "
            "pip install statsmodels pandas") from import_error

    if not hasattr(data, "columns"):
        raise TypeError("data must be a pandas DataFrame")
    required = {"is_correct", "is_perturbed", "token_inflation_ratio", "task_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    data = _with_token_inflation_excess(data)
    use_quasibayes = not _probe_lme4_bridge()

    if use_quasibayes:
        quasibayes = _fit_quasibayes_mediation(data, PRIMARY_MEDIATOR_COLUMN)
        primary = quasibayes["decomposition"]
        intervals = quasibayes["intervals"]
        estimator = MEDIATION_ESTIMATOR_QUASIBAYES
        supplementary_fit = _fit_quasibayes_supplementary(data)
    else:
        primary = _fit_offline_fallback_decomposition(data, PRIMARY_MEDIATOR_COLUMN)
        intervals = (_bootstrap_fallback_intervals(data) or {}
                     if len(data) >= _MINIMUM_OBSERVATIONS_FOR_BOOTSTRAP else {})
        estimator = MEDIATION_ESTIMATOR_OFFLINE_FALLBACK
        supplementary_fit = _fit_offline_supplementary(data)

    proportion, withheld_reason = _guarded_proportion_mediated(
        primary.indirect_effect, primary.total_effect, intervals.get("total"))

    supplementary_indirect, supplementary_proportion = supplementary_fit

    return MediationResult(
        total_effect=primary.total_effect,
        direct_effect=primary.direct_effect,
        indirect_effect=primary.indirect_effect,
        proportion_mediated=proportion,
        treatment_on_mediator_coef=primary.treatment_on_mediator,
        mediator_on_outcome_coef=primary.mediator_on_outcome,
        n_observations=len(data),
        bootstrap_ci_proportion=intervals.get("proportion"),
        supplementary_indirect_effect=supplementary_indirect,
        supplementary_proportion_mediated=supplementary_proportion,
        bootstrap_ci_indirect=intervals.get("indirect"),
        bootstrap_ci_total=intervals.get("total"),
        proportion_mediated_reason=withheld_reason,
        estimator=estimator,
    )


def _fit_quasibayes_supplementary(data) -> tuple:
    """Supplementary-mediator (subword_count_change) point estimates on the
    quasi-Bayes path; advisory, so any fit failure degrades to (None, None)."""
    if SUPPLEMENTARY_MEDIATOR_COLUMN not in data.columns:
        return None, None
    try:
        supplementary = _fit_quasibayes_mediation(
            data, SUPPLEMENTARY_MEDIATOR_COLUMN)["decomposition"]
    except Exception:  # advisory; R errors surface as several rpy2 types
        return None, None
    return _supplementary_pair(supplementary)


def _fit_offline_supplementary(data) -> tuple:
    if SUPPLEMENTARY_MEDIATOR_COLUMN not in data.columns:
        return None, None
    try:
        supplementary = _fit_offline_fallback_decomposition(
            data, SUPPLEMENTARY_MEDIATOR_COLUMN)
    except _fit_failure_exceptions():
        return None, None
    return _supplementary_pair(supplementary)


def _supplementary_pair(supplementary: _MediationDecomposition) -> tuple:
    proportion = (
        supplementary.indirect_effect / supplementary.total_effect
        if abs(supplementary.total_effect) > _TOTAL_EFFECT_NUMERICALLY_ZERO
        else None)
    return supplementary.indirect_effect, proportion


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


def _bootstrap_fallback_intervals(
        data,
        n_resamples: int = _FALLBACK_BOOTSTRAP_RESAMPLES,
        seed: int = _QUASIBAYES_SEED) -> Optional[dict]:
    """Cluster (by-item) percentile-bootstrap CIs for the offline fallback
    estimator's indirect effect, total effect, and proportion mediated.

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
            decompositions.append(_fit_offline_fallback_decomposition(
                sample, PRIMARY_MEDIATOR_COLUMN))
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
