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
from pipeline.runner import load_generation_rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, nargs="+", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def _build_model_dataframe(rows):
    """Convert raw generation rows to a pandas DataFrame for the mixed-effects models.

    Clean rows contribute is_perturbed=0 and token_inflation_ratio=0.0.
    Perturbed rows contribute is_perturbed=1 and token_inflation_ratio from the
    r_token_inflation_ratio field written by the runner.
    """
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
            "token_inflation_ratio": 0.0 if is_clean else float(
                row.get("r_token_inflation_ratio", 0.0) or 0.0),
            "r_semantic_class": row.get("r_semantic_class", "clean"),
            "r_edit_budget": row.get("r_edit_budget", 0),
            "task_family": row.get("task_family", ""),
        })
    return pd.DataFrame(records)


def _run_statistical_models(rows, output_directory: Path) -> None:
    """Fit the pre-registered mixed-effects logistic and mediation models."""
    # These imports are runtime-conditional (statsmodels + pandas are optional).
    # Pyright reportMissingImports warnings here are expected — the packages are
    # not in the dev-env requirements but ARE required for the confirmatory run.
    try:
        from analysis.models import (  # type: ignore[import]
            fit_crossed_mixed_effects_logistic,
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

    # Mixed-effects logistic (design/06 §6.6): all perturbed + clean rows.
    print("  Fitting crossed mixed-effects logistic regression...")
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
            }, indent=2),
            encoding="utf-8",
        )
        print(f"  Mixed-effects result: {mixed_path}  "
              f"(method={mixed_result.method}, converged={mixed_result.converged})")
    except Exception as error:
        print(f"  Mixed-effects model failed: {error}", file=sys.stderr)

    # Mediation (design/06 §6.8): Regime A perturbed rows + their clean counterparts.
    regime_a_data = data[data["r_semantic_class"] == "A"]
    if len(regime_a_data) >= 10:
        print("  Fitting mediation model (Regime A)...")
        try:
            mediation_result = compute_mediation_proportion(regime_a_data)
            mediation_path = output_directory / "mediation_proportion.json"
            mediation_path.write_text(
                json.dumps({
                    "total_effect": mediation_result.total_effect,
                    "direct_effect": mediation_result.direct_effect,
                    "indirect_effect": mediation_result.indirect_effect,
                    "proportion_mediated": mediation_result.proportion_mediated,
                    "treatment_on_mediator_coef": mediation_result.treatment_on_mediator_coef,
                    "mediator_on_outcome_coef": mediation_result.mediator_on_outcome_coef,
                    "n_observations": mediation_result.n_observations,
                    "bootstrap_ci_proportion": (
                        list(mediation_result.bootstrap_ci_proportion)
                        if mediation_result.bootstrap_ci_proportion else None),
                }, indent=2),
                encoding="utf-8",
            )
            proportion = mediation_result.proportion_mediated
            proportion_str = (f"{proportion:.3f}" if proportion is not None
                              else "indeterminate (total effect ≈ 0)")
            print(f"  Mediation result: {mediation_path}  "
                  f"(proportion_mediated={proportion_str})")
        except Exception as error:
            print(f"  Mediation model failed: {error}", file=sys.stderr)
    else:
        print(f"  Mediation skipped: {len(regime_a_data)} Regime A rows "
              f"(need ≥ 10).", file=sys.stderr)


def main():
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    rows = load_generation_rows(arguments.generations)
    matched_pairs = result_analysis.join_matched_pairs(rows)
    cell_summaries = result_analysis.summarize_all_cells(matched_pairs)

    cell_table_path = result_analysis.write_cell_table(
        cell_summaries, arguments.output_directory / "cell_table.csv")

    _run_statistical_models(rows, arguments.output_directory)

    figure_paths = [
        result_analysis.figure_clean_conditioned_failure_vs_edit_budget(
            cell_summaries, arguments.output_directory / "figure_ccf_vs_edit_budget.png"),
        result_analysis.figure_keyboard_versus_asr_profile(
            cell_summaries, arguments.output_directory / "figure_keyboard_vs_asr.png"),
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
