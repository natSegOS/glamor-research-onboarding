"""Confirmatory GLMM ladder and mediation estimators: simulated-recovery tests.

These tests hold the statistical layer to KNOWN GROUND TRUTH built into
simulated data. Breaking them means the paper's headline models either
mislabel their estimator (the pre-2026-07-20 "linear model called logistic"
failure), leak variance components into fixed-effect tables, or — the failure
class that actually bit on the pilot — let between-item confounding flip the
mediation coefficient's sign.
"""

from __future__ import annotations

import math
import random

import pytest

pandas = pytest.importorskip("pandas")
pytest.importorskip("statsmodels")

from analysis.models import (
    MEDIATION_ESTIMATOR_OFFLINE_FALLBACK,
    MEDIATION_ESTIMATOR_QUASIBAYES,
    PROPORTION_WITHHELD_NO_BOOTSTRAP,
    PROPORTION_WITHHELD_TOTAL_CI_INCLUDES_ZERO,
    PROPORTION_WITHHELD_TOTAL_NUMERICALLY_ZERO,
    _fit_offline_fallback_decomposition,
    _guarded_proportion_mediated,
    _probe_lme4_bridge,
    _random_effect_ladder,
    compute_mediation_proportion,
    fit_crossed_mixed_effects_logistic,
    fit_linear_probability_mixed_model,
)
from enums import ConvergenceMethod


_TRUE_PERTURBATION_LOG_ODDS = -1.2
_SIMULATION_ITEMS = 120


def _simulated_paired_frame(seed: int = 7) -> "pandas.DataFrame":
    """Matched clean/perturbed rows for _SIMULATION_ITEMS items with a known
    negative perturbation effect on the log-odds scale and item-level
    heterogeneity — the shape the confirmatory model must recover."""
    generator = random.Random(seed)
    records = []
    for item_index in range(_SIMULATION_ITEMS):
        item_log_odds = generator.gauss(0.8, 0.9)
        inflation_excess = abs(generator.gauss(0.04, 0.02))
        for is_perturbed in (0, 1):
            log_odds = item_log_odds + _TRUE_PERTURBATION_LOG_ODDS * is_perturbed
            probability = 1.0 / (1.0 + math.exp(-log_odds))
            records.append({
                "is_correct": int(generator.random() < probability),
                "is_perturbed": is_perturbed,
                "task_id": f"item_{item_index:04d}",
                "model_revision": "sim-model-rev",
                "token_inflation_ratio": 1.0 + inflation_excess * is_perturbed,
                "word_length_before": generator.randint(4, 12),
            })
    return pandas.DataFrame(records)


class TestConfirmatoryLadder:

    def test_every_rung_is_logistic_and_recovers_the_simulated_effect_sign(self):
        """Whatever rung fires (glmer with R present, the fixed-effects
        logistic GLM without), the fit must be logistic: coefficient on the
        log-odds scale with the right sign and or == exp(coef). The failure
        class: a linear fit mislabeled as logistic, whose 'OR' is
        meaningless."""
        result = fit_crossed_mixed_effects_logistic(_simulated_paired_frame())
        assert result.converged
        assert result.method in set(ConvergenceMethod)
        perturbation_effect = result.fixed_effects["is_perturbed"]
        assert perturbation_effect["coef"] < 0
        # Magnitude is held to truth only on the glmer rungs. The GLM rung's
        # per-item dummies on 2-row panels carry the classic incidental-
        # parameters bias (slope inflated several-fold) — a known property of
        # the FALLBACK, which is exactly why glmer is the estimator of record.
        if result.method != ConvergenceMethod.FIXED_EFFECTS_LOGISTIC_GLM:
            assert abs(perturbation_effect["coef"] - _TRUE_PERTURBATION_LOG_ODDS) < 0.8
        assert perturbation_effect["or"] == pytest.approx(
            math.exp(perturbation_effect["coef"]))
        assert perturbation_effect["std_error"] > 0
        assert 0.0 <= perturbation_effect["p"] <= 1.0

    def test_offline_environment_lands_on_the_glm_rung_with_a_recorded_reason(self):
        """Without an R/lme4 bridge the ladder must degrade to the loudly
        labeled fixed-effects logistic GLM and say WHY in ladder_notes —
        silent degradation is how the wrong estimator ships unnoticed."""
        bridge_reason = _probe_lme4_bridge()
        result = fit_crossed_mixed_effects_logistic(_simulated_paired_frame())
        if bridge_reason:   # no R locally: the CI environment of record
            assert result.method == ConvergenceMethod.FIXED_EFFECTS_LOGISTIC_GLM
            assert any("bridge" in note for note in result.ladder_notes)
        else:               # R present (Colab): a glmer rung must have fired
            assert result.method != ConvergenceMethod.FIXED_EFFECTS_LOGISTIC_GLM

    def test_variance_components_never_appear_as_fixed_effects(self):
        """The pilot artifact leaked 'is_perturbed Var' into the fixed-effects
        table with a bogus OR; no name containing Var/Cov/Group/Intercept may
        appear in either model's fixed_effects again."""
        frame = _simulated_paired_frame()
        logistic_result = fit_crossed_mixed_effects_logistic(frame)
        linear_result = fit_linear_probability_mixed_model(frame)
        for fixed_effects in (logistic_result.fixed_effects,
                              linear_result.fixed_effects):
            for term_name in fixed_effects:
                assert not any(marker in term_name for marker in
                               ("Var", "Cov", "Group", "Intercept"))

    def test_linear_probability_appendix_never_reports_odds_ratios(self):
        """The linear model's coefficients are risk differences;
        exponentiating them was the audit's headline mislabel. The appendix
        estimator must not carry an 'or' key anywhere."""
        result = fit_linear_probability_mixed_model(_simulated_paired_frame())
        assert result.converged
        assert result.method == "linear_probability_mixed_model"
        assert all("or" not in effect for effect in result.fixed_effects.values())
        assert result.fixed_effects["is_perturbed"]["coef"] < 0

    def test_ladder_adapts_to_grouping_levels_and_deduplicates(self):
        """One model in the data cannot support by-model random effects (lme4
        would refuse the single-level factor); five models must produce the
        full four-rung Barr ladder with by-model terms."""
        single_model_ladder = _random_effect_ladder(model_count=1)
        assert all("model_revision" not in formula
                   for _method, formula in single_model_ladder)
        formulas = [formula for _method, formula in single_model_ladder]
        assert len(formulas) == len(set(formulas))

        five_model_ladder = _random_effect_ladder(model_count=5)
        assert len(five_model_ladder) == 4
        assert all("model_revision" in formula
                   for _method, formula in five_model_ladder)
        assert "||" in five_model_ladder[1][1]

    def test_missing_required_columns_are_rejected_before_fitting(self):
        """A frame missing the mediator or identifiers must be rejected at the
        boundary; statsmodels' own errors downstream are uninterpretable."""
        with pytest.raises(ValueError):
            fit_crossed_mixed_effects_logistic(
                pandas.DataFrame({"is_correct": [1, 0]}))
        with pytest.raises(TypeError):
            fit_crossed_mixed_effects_logistic([{"is_correct": 1}])


def _confounded_mediation_frame() -> "pandas.DataFrame":
    """The Simpson's-paradox construction that broke the first estimator
    rewrite: WITHIN every item, higher mediator → perturbed answer flips to
    wrong (true β < 0); BETWEEN items, high-mediator items are the EASY ones
    (both rows correct more often), so a pooled outcome model estimates
    β > 0. Recovering β < 0 here is the whole point of conditioning on the
    item."""
    generator = random.Random(11)
    records = []
    for item_index in range(90):
        # Easy items: always correct clean, big mediator, break 40% of the
        # time. Hard items: rarely correct at all, tiny mediator, never break.
        # Pooled across perturbed rows: big-mediator items score HIGHER
        # (0.6 vs 0.2) → positive pooled slope. Within items: only
        # big-mediator items ever drop → negative within slope.
        is_easy_item = item_index % 3 != 0
        inflation_excess = (0.09 if is_easy_item else 0.02) + generator.gauss(0, 0.004)
        clean_correct = 1 if is_easy_item else int(generator.random() < 0.2)
        perturbed_breaks = is_easy_item and generator.random() < 0.4
        perturbed_correct = 0 if perturbed_breaks else clean_correct
        for is_perturbed, is_correct in ((0, clean_correct), (1, perturbed_correct)):
            records.append({
                "is_correct": is_correct,
                "is_perturbed": is_perturbed,
                "task_id": f"item_{item_index:04d}",
                "model_revision": "sim-model-rev",
                "token_inflation_ratio": 1.0 + inflation_excess * is_perturbed,
            })
    return pandas.DataFrame(records)


class TestMediation:

    def test_within_item_estimator_defeats_between_item_confounding(self):
        """THE pilot failure class: a pooled outcome model flips β's sign on
        confounded data. The shipped estimator must recover the true negative
        within-item mediation (and a deliberately pooled estimate computed
        here must show the trap is real)."""
        frame = _confounded_mediation_frame()
        result = compute_mediation_proportion(frame)
        assert result.mediator_on_outcome_coef < 0
        assert result.indirect_effect < 0
        assert result.total_effect == pytest.approx(
            result.direct_effect + result.indirect_effect)

        # Demonstrate the trap: the naive pooled between-item slope is
        # positive on this data (easy items have big mediators).
        perturbed_rows = frame[frame["is_perturbed"] == 1]
        mediator = perturbed_rows["token_inflation_ratio"] - 1.0
        outcome = perturbed_rows["is_correct"]
        pooled_covariance = ((mediator - mediator.mean())
                             * (outcome - outcome.mean())).sum()
        assert pooled_covariance > 0

    def test_estimator_label_matches_the_active_path(self):
        """Every serialized mediation result must say which estimator made it;
        an unlabeled number is unreviewable."""
        result = compute_mediation_proportion(_confounded_mediation_frame())
        expected_label = (MEDIATION_ESTIMATOR_OFFLINE_FALLBACK
                          if _probe_lme4_bridge()
                          else MEDIATION_ESTIMATOR_QUASIBAYES)
        assert result.estimator == expected_label

    def test_offline_decomposition_is_exactly_additive_and_deterministic(self):
        """total = direct + indirect must hold to machine precision, and the
        estimator must be a pure function of its inputs."""
        frame = _confounded_mediation_frame()
        first = _fit_offline_fallback_decomposition(
            frame.assign(token_inflation_excess=frame["token_inflation_ratio"] - 1.0),
            "token_inflation_excess")
        second = _fit_offline_fallback_decomposition(
            frame.assign(token_inflation_excess=frame["token_inflation_ratio"] - 1.0),
            "token_inflation_excess")
        assert first == second
        assert first.total_effect == pytest.approx(
            first.direct_effect + first.indirect_effect)

    def test_degenerate_all_clean_or_all_perturbed_data_is_rejected(self):
        """No within-item treatment variation → the within estimator is
        undefined and must say so, not emit NaNs downstream."""
        frame = _confounded_mediation_frame()
        all_perturbed = frame[frame["is_perturbed"] == 1].assign(
            token_inflation_excess=lambda rows: rows["token_inflation_ratio"] - 1.0)
        with pytest.raises(ValueError):
            _fit_offline_fallback_decomposition(all_perturbed, "token_inflation_excess")

    @pytest.mark.parametrize("indirect,total,interval,expected_proportion,expected_reason", [
        (0.0, 0.0, None, None, PROPORTION_WITHHELD_TOTAL_NUMERICALLY_ZERO),
        (-0.02, -0.04, None, None, PROPORTION_WITHHELD_NO_BOOTSTRAP),
        (-0.02, -0.04, (-0.09, 0.01), None, PROPORTION_WITHHELD_TOTAL_CI_INCLUDES_ZERO),
        (-0.02, -0.04, (-0.09, -0.01), 0.5, None),
    ], ids=["zero-total", "no-bootstrap", "ci-includes-zero", "reportable"])
    def test_proportion_mediated_guard_truth_table(
            self, indirect, total, interval, expected_proportion, expected_reason):
        """The pre-registered rule for when the unstable ratio may be
        reported. Loosening any row lets a near-zero denominator manufacture
        absurd headline percentages (the v1 pilot's '4.85 proportion')."""
        proportion, reason = _guarded_proportion_mediated(indirect, total, interval)
        assert reason == expected_reason
        if expected_proportion is None:
            assert proportion is None
        else:
            assert proportion == pytest.approx(expected_proportion)

    def test_missing_columns_are_rejected(self):
        """Same boundary-validation contract as the confirmatory model."""
        with pytest.raises(ValueError):
            compute_mediation_proportion(pandas.DataFrame({"is_correct": [1]}))


class TestModelFrameConstruction:

    def test_clean_rows_are_coded_at_the_definitional_inflation_ratio(self):
        """The v1 pilot coded clean rows' token_inflation_ratio as 0.0 instead
        of the definitional 1.0, manufacturing a spurious ~30x jump in the
        treatment→mediator path that dominated the mediation estimate. The
        model-frame builder must code clean rows at exactly 1.0 and must emit
        the confirmatory fixed-effect columns the GLMM conditions on."""
        import importlib.util
        from pathlib import Path

        tool_path = (Path(__file__).resolve().parent.parent
                     / "tools" / "run_analysis.py")
        specification = importlib.util.spec_from_file_location(
            "run_analysis_tool", tool_path)
        run_analysis_tool = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(run_analysis_tool)

        rows = [
            {"is_clean": True, "is_correct": 1, "task_id": "t1",
             "model_revision": "rev", "quantization_method": "fp16"},
            {"is_clean": False, "is_correct": 0, "task_id": "t1",
             "model_revision": "rev", "quantization_method": "fp16",
             "token_inflation_ratio": 1.08, "subword_count_change": 2,
             "word_length_before": 6, "r_edit_budget": 2,
             "r_semantic_class": "A", "r_selection_policy": "keyboard_neighbor",
             "r_operation": "substitute"},
        ]
        frame = run_analysis_tool._build_model_dataframe(rows)
        clean_row = frame[frame["is_perturbed"] == 0].iloc[0]
        perturbed_row = frame[frame["is_perturbed"] == 1].iloc[0]
        assert clean_row["token_inflation_ratio"] == 1.0
        assert perturbed_row["token_inflation_ratio"] == pytest.approx(1.08)
        for confirmatory_column in ("edit_budget_k", "precision",
                                    "r_selection_policy"):
            assert confirmatory_column in frame.columns
        assert perturbed_row["edit_budget_k"] == 2
