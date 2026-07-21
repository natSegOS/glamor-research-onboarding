"""Analyze generation rows into the per-cell table, statistical models, and figures.

Reads one or more JSONL generation files (produced by tools/run_generation.py),
joins matched clean/perturbed pairs, writes the per-cell summary CSV the paper's
tables are built from, runs the pre-registered mixed-effects and mediation models,
and renders the figures (design/08 §8.7).

Usage:

    python tools/run_analysis.py \\
        --generations results/main/main_generations.jsonl \\
        --output-directory analysis/main

The statistical models (design/06 §6.6 and §6.8) require statsmodels and pandas.
If those are absent the cell table and figures are still produced; the model
outputs are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import results as result_analysis
from enums import SemanticClass
from pipeline.runner import load_generation_rows

# Minimum Regime A row count below which the mediation model is not fit
# (design/06 §6.8 — too few rows to estimate the mediator/outcome coefficients
# meaningfully).
_MINIMUM_ROWS_FOR_MEDIATION_MODEL = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, nargs="+", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="the experiment config the rows came from; supplies the primary "
             "edit budgets for the Stage-1 gates (defaults otherwise)")
    return parser.parse_args()


# An unperturbed prompt inflates nothing, so the clean-row token-inflation
# ratio is definitionally 1.0. Coding it 0.0 (the pre-pilot-rerun bug)
# manufactured a spurious ~1.0 treatment-on-mediator jump that dominated the
# mediation estimate.
_CLEAN_TOKEN_INFLATION_RATIO = 1.0


def _build_model_dataframe(rows):
    """Convert raw generation rows to a pandas DataFrame for the mixed-effects
    models. token_inflation_ratio has no r_ prefix — the runner writes it as an
    extra_field, not a perturbation-state-vector entry.

    ``edit_budget_k`` / ``precision`` / ``operation`` are the design/06 §6.6
    fixed-effect columns; models.py only includes each when it actually varies
    (a single-model pilot contributes no precision term, correctly)."""
    import pandas as pd  # type: ignore[import]

    records = []
    for row in rows:
        is_clean = row.get("is_clean", False)
        is_correct = row.get("is_correct")
        task_id = row.get("task_id")
        model_revision = row.get("model_revision")
        if is_correct is None or not task_id or not model_revision:
            continue
        records.append({
            "is_correct": int(is_correct),
            "is_perturbed": 0 if is_clean else 1,
            "task_id": task_id,
            "model_revision": model_revision,
            "token_inflation_ratio": _CLEAN_TOKEN_INFLATION_RATIO if is_clean else float(
                row.get("token_inflation_ratio") or _CLEAN_TOKEN_INFLATION_RATIO),
            "subword_count_change": 0.0 if is_clean else float(
                row.get("subword_count_change", 0.0) or 0.0),
            "word_length_before": int(row.get("word_length_before", 0) or 0),
            "edit_budget_k": int(row.get("r_edit_budget", 0) or 0),
            "precision": str(row.get("quantization_method", "")),
            "operation": str(row.get("r_operation", "")),
            "r_semantic_class": row.get("r_semantic_class", SemanticClass.CLEAN),
            "r_selection_policy": row.get("r_selection_policy", ""),
            "r_edit_budget": row.get("r_edit_budget", 0),
            "task_family": row.get("task_family", ""),
            "extraction_tier": row.get("extraction_tier", ""),
        })
    return pd.DataFrame(records)


def _with_clean_partner_rows(data, perturbed_mask):
    """The rows selected by ``perturbed_mask`` plus the matching clean row of
    every selected item. Clean rows are tagged SemanticClass.CLEAN, never A,
    so selecting on the perturbation predicate alone would drop every clean
    row, leave is_perturbed constant, and make the mediator model singular."""
    selected_task_ids = data.loc[perturbed_mask, "task_id"]
    return data[perturbed_mask
                | ((data["is_perturbed"] == 0)
                   & data["task_id"].isin(selected_task_ids))]


# H1b (design/06 §6.8): filler insertion inflates the token count WITHOUT
# corrupting any word, while keyboard typos inflate by fragmenting words. If
# the mediation mechanism is fragmentation-specific rather than length-
# specific, the mediated share should be ≈ 0 for fillers and substantial for
# keyboard typos. These per-policy fits operationalize that contrast.
_H1B_CONTRAST_POLICIES = ("keyboard_neighbor", "filler_word")


def _run_statistical_models(rows, output_directory: Path) -> None:
    """Fit the confirmatory logistic GLMM, its linear-probability robustness
    appendix, and the mediation models (design/06 §6.6 and §6.8)."""
    # These imports are runtime-conditional (statsmodels + pandas are optional).
    # Pyright reportMissingImports warnings here are expected — the packages are
    # not in the dev-env requirements but ARE required for the confirmatory run.
    try:
        from analysis.models import (  # type: ignore[import]
            fit_crossed_mixed_effects_logistic,
            fit_linear_probability_mixed_model,
            compute_mediation_proportion,
        )
    except ImportError as import_error:
        print(f"  Statistical models skipped: {import_error}", file=sys.stderr)
        return

    try:
        data = _build_model_dataframe(rows)
    except ImportError as import_error:
        print(f"  Statistical models skipped (pandas unavailable): {import_error}",
              file=sys.stderr)
        return

    if data.empty:
        print("  Statistical models skipped: no scoreable rows.", file=sys.stderr)
        return

    # Confirmatory logistic GLMM (design/06 §6.6): all perturbed + clean rows.
    print("  Fitting confirmatory logistic GLMM (lme4 ladder)...")
    try:
        mixed_result = fit_crossed_mixed_effects_logistic(data)
        mixed_path = output_directory / "mixed_effects_logistic.json"
        mixed_path.write_text(
            json.dumps({
                "converged": mixed_result.converged,
                "method": mixed_result.method,
                "log_likelihood": mixed_result.log_likelihood,
                "n_observations": mixed_result.n_observations,
                "n_items": mixed_result.n_items,
                "n_models": mixed_result.n_models,
                "fixed_effects": mixed_result.fixed_effects,
                "random_effects_variance": mixed_result.random_effects_variance,
                "is_singular_fit": mixed_result.is_singular_fit,
                "ladder_notes": mixed_result.ladder_notes,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"  Confirmatory GLMM: {mixed_path}  "
              f"(method={mixed_result.method}, converged={mixed_result.converged})")
    except Exception as error:
        print(f"  Confirmatory GLMM failed: {error}", file=sys.stderr)

    # Robustness appendix: the linear-probability mixed model (risk-difference
    # scale; no odds ratios — see LinearProbabilityMixedResult).
    print("  Fitting linear-probability mixed model (appendix)...")
    try:
        linear_result = fit_linear_probability_mixed_model(data)
        linear_path = output_directory / "linear_probability_mixed_model.json"
        linear_path.write_text(
            json.dumps({
                "converged": linear_result.converged,
                "method": linear_result.method,
                "scale_note": "coefficients are risk differences "
                              "(probability-scale); not odds ratios",
                "n_observations": linear_result.n_observations,
                "fixed_effects": linear_result.fixed_effects,
                "random_effects_variance": linear_result.random_effects_variance,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"  Linear-probability appendix: {linear_path}")
    except Exception as error:
        print(f"  Linear-probability model failed: {error}", file=sys.stderr)

    # Mediation (design/06 §6.8): Regime A perturbed rows + clean partners.
    is_regime_a = data["r_semantic_class"] == SemanticClass.A
    regime_a_data = _with_clean_partner_rows(data, is_regime_a)

    # One fit per task family: families can degrade in opposite directions
    # (pilot: GSM8K positive, GSM-Symbolic ~zero/negative), and pooling them
    # cancels the total effect and destabilizes the proportion. The pooled fit
    # is kept as a labeled supplementary, not the headline.
    mediation_fits = {
        f"task_family:{task_family}": family_data
        for task_family, family_data in regime_a_data.groupby("task_family")
    }
    mediation_fits["pooled_all_families_supplementary"] = regime_a_data
    for contrast_policy in _H1B_CONTRAST_POLICIES:
        mediation_fits[f"h1b_policy:{contrast_policy}"] = _with_clean_partner_rows(
            data, is_regime_a & (data["r_selection_policy"] == contrast_policy))

    mediation_report = {}
    for fit_label, fit_data in mediation_fits.items():
        if len(fit_data) < _MINIMUM_ROWS_FOR_MEDIATION_MODEL:
            mediation_report[fit_label] = {
                "skipped": f"{len(fit_data)} rows "
                           f"(need >= {_MINIMUM_ROWS_FOR_MEDIATION_MODEL})"}
            continue
        print(f"  Fitting mediation model (Regime A, {fit_label})...")
        try:
            mediation_report[fit_label] = _mediation_result_as_dict(
                compute_mediation_proportion(fit_data))
        except Exception as error:
            print(f"  Mediation model failed ({fit_label}): {error}", file=sys.stderr)
            mediation_report[fit_label] = {"failed": str(error)}

    mediation_path = output_directory / "mediation_proportion.json"
    mediation_path.write_text(
        json.dumps(mediation_report, indent=2), encoding="utf-8")
    print(f"  Mediation results: {mediation_path}")


def _mediation_result_as_dict(result) -> dict:
    as_list = lambda interval: list(interval) if interval else None
    return {
        "estimator": result.estimator,
        "total_effect": result.total_effect,
        "direct_effect": result.direct_effect,
        "indirect_effect": result.indirect_effect,
        "bootstrap_ci_indirect": as_list(result.bootstrap_ci_indirect),
        "bootstrap_ci_total": as_list(result.bootstrap_ci_total),
        "proportion_mediated": result.proportion_mediated,
        "proportion_mediated_reason": result.proportion_mediated_reason,
        "bootstrap_ci_proportion": as_list(result.bootstrap_ci_proportion),
        "treatment_on_mediator_coef": result.treatment_on_mediator_coef,
        "mediator_on_outcome_coef": result.mediator_on_outcome_coef,
        "supplementary_indirect_effect": result.supplementary_indirect_effect,
        "supplementary_proportion_mediated": result.supplementary_proportion_mediated,
        "n_observations": result.n_observations,
    }


_HOLM_PRIMARY_ADJUSTMENT_LABEL = "holm_within_model_primary"
_BH_EXPLORATORY_ADJUSTMENT_LABEL = "bh_fdr_exploratory"


def _attach_multiplicity_adjustments(
        cell_summaries, primary_edit_budget_reasoning: int,
        primary_edit_budget_mcq: int) -> None:
    """Attach the design/06 §6.7 corrections in place: Holm within each
    model's pre-registered primary family; Benjamini–Hochberg FDR across
    everything else (the exploratory grid)."""
    from analysis.gates import PRIMARY_CONDITION_FILTERS
    from analysis.statistics import (
        BENJAMINI_HOCHBERG_FDR_Q,
        HOLM_PRIMARY_FAMILY_ALPHA,
        benjamini_hochberg_adjusted_p_values,
        holm_adjusted_p_values,
    )
    from enums import REASONING_FAMILIES

    def is_primary_cell(summary: dict) -> bool:
        primary_budget = (
            primary_edit_budget_reasoning
            if summary.get("task_family") in REASONING_FAMILIES
            else primary_edit_budget_mcq)
        return (all(summary.get(field) == value
                    for field, value in PRIMARY_CONDITION_FILTERS.items())
                and summary.get("r_edit_budget") == primary_budget)

    primary_indices = [index for index, summary in enumerate(cell_summaries)
                       if is_primary_cell(summary)]
    exploratory_indices = [index for index in range(len(cell_summaries))
                           if index not in set(primary_indices)]

    def attach(index: int, label: str, adjusted_p, threshold: float) -> None:
        cell_summaries[index]["p_value_adjustment"] = label
        cell_summaries[index]["mcnemar_p_value_adjusted"] = adjusted_p
        cell_summaries[index]["is_significant_after_adjustment"] = bool(
            adjusted_p == adjusted_p and adjusted_p < threshold)

    primary_indices_by_model: dict = {}
    for index in primary_indices:
        primary_indices_by_model.setdefault(
            cell_summaries[index].get("model_id"), []).append(index)
    for model_indices in primary_indices_by_model.values():
        adjusted_values = holm_adjusted_p_values(
            [cell_summaries[index].get("mcnemar_p_value") for index in model_indices])
        for index, adjusted_p in zip(model_indices, adjusted_values):
            attach(index, _HOLM_PRIMARY_ADJUSTMENT_LABEL, adjusted_p,
                   HOLM_PRIMARY_FAMILY_ALPHA)

    bh_adjusted_values = benjamini_hochberg_adjusted_p_values(
        [cell_summaries[index].get("mcnemar_p_value")
         for index in exploratory_indices])
    for index, adjusted_p in zip(exploratory_indices, bh_adjusted_values):
        attach(index, _BH_EXPLORATORY_ADJUSTMENT_LABEL, adjusted_p,
               BENJAMINI_HOCHBERG_FDR_Q)


def main():
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    from analysis.gates import compute_stage_gates
    from pipeline.experiment import ExperimentConfiguration

    budget_source = (ExperimentConfiguration.from_yaml(arguments.config)
                     if arguments.config else ExperimentConfiguration)

    rows = load_generation_rows(arguments.generations)
    matched_pairs = result_analysis.join_matched_pairs(rows)
    cell_summaries = result_analysis.summarize_all_cells(matched_pairs)
    _attach_multiplicity_adjustments(
        cell_summaries,
        budget_source.primary_edit_budget_reasoning,
        budget_source.primary_edit_budget_mcq)

    cell_table_path = result_analysis.write_cell_table(
        cell_summaries, arguments.output_directory / "cell_table.csv")

    gates = compute_stage_gates(
        rows,
        budget_source.primary_edit_budget_reasoning,
        budget_source.primary_edit_budget_mcq)
    gates_path = arguments.output_directory / "gates.json"
    gates_path.write_text(json.dumps(gates, indent=2, default=str), encoding="utf-8")
    print(f"Stage-1 gates:    {gates_path}")
    for family, block in gates["per_task_family"].items():
        print(f"  {family}: A0={block['clean_accuracy']}, "
              f"p_d={block.get('discordant_rate')} "
              f"({block.get('discordant_rate_bucket', 'no primary rows')})")
    print(f"  reasoning format compliance: {gates['reasoning_format_compliance']} "
          f"(target >= {gates['reasoning_format_compliance_target']})")

    fragmentation_contrast = result_analysis.summarize_fragmentation_contrast(rows)
    if fragmentation_contrast:
        contrast_path = arguments.output_directory / "method_a_fragmentation_contrast.json"
        contrast_path.write_text(
            json.dumps(fragmentation_contrast, indent=2, default=str), encoding="utf-8")
        print(f"Method A contrast: {contrast_path} ({len(fragmentation_contrast)} groups)")
    else:
        print("Method A contrast skipped: no fragmentation_matched rows.", file=sys.stderr)

    _run_statistical_models(rows, arguments.output_directory)

    figure_paths = [
        result_analysis.figure_clean_conditioned_failure_vs_edit_budget(
            cell_summaries, arguments.output_directory / "figure_ccf_vs_edit_budget.png"),
    ]

    print(f"rows analyzed:    {len(rows)}")
    print(f"matched pairs:    {len(matched_pairs)}")
    print(f"reporting cells:  {len(cell_summaries)}")
    print(f"cell table:       {cell_table_path}")
    for figure_path in figure_paths:
        status = figure_path if figure_path else "(skipped: matplotlib unavailable or no data)"
        print(f"figure:           {status}")


if __name__ == "__main__":
    main()
