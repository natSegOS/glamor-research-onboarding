"""Analyze generation rows into the per-cell table and the figures.

Reads one or more JSONL generation files (produced by tools/run_generation.py),
joins matched clean/perturbed pairs, writes the per-cell summary CSV the paper's
tables are built from, and renders the figures (design/08 §8.7).

Usage:

    python tools/run_analysis.py \\
        --generations results/main/main_generations.jsonl \\
        --output-directory analysis/main
"""

from __future__ import annotations

import argparse

from pathlib import Path

from analysis import results as result_analysis
from pipeline.runner import load_generation_rows


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True, nargs="+", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    rows = load_generation_rows(arguments.generations)
    matched_pairs = result_analysis.join_matched_pairs(rows)
    cell_summaries = result_analysis.summarize_all_cells(matched_pairs)

    cell_table_path = result_analysis.write_cell_table(
        cell_summaries, arguments.output_directory / "cell_table.csv")

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
