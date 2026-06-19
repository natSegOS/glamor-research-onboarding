"""Turning generation rows into matched pairs, per-cell tables, and figures.

Provenance
----------
This module consumes the JSONL rows written by pipeline.runner.run_shard and
produces the analysis deliverables of design/06 §6.10 and design/08:
  - matched clean/perturbed pairs joined per (model_revision, task_id);
  - one statistics.summarize_cell block per reporting cell;
  - the figures of design/08 §8.7, including the keyboard-vs-ASR degradation
    profile that the latest review motivated (the two noise sources are the
    study's two arms and deserve a direct side-by-side).

Every cell is keyed by the r_-prefixed perturbation-state fields that the runner
writes, so the analysis dimensions and the logged dimensions cannot drift apart.
Cells smaller than the interval minimum are summarized without a confidence
interval rather than dropped, so an underpowered pilot cell never crashes the
run (the n<2 guard lives in statistics.summarize_cell).
"""

from __future__ import annotations

import csv

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from enums import SelectionPolicy, INTERACTIONAL_FAILURE_STATUSES
from analysis import statistics


# The dimensions that define a reporting cell (design/06 §6.10). All are
# r_-prefixed perturbation-state fields written by the runner, plus the model
# and task family.
CELL_DIMENSION_KEYS = (
    "model_revision",
    "task_family",
    "r_semantic_class",
    "r_operation",
    "r_selection_policy",
    "r_scope",
    "r_edit_budget",
)


@dataclass
class MatchedPair:
    """One clean/perturbed matched pair for a single item under a single model
    and a single perturbation condition."""
    model_revision: str
    task_id: str
    task_family: str
    clean_is_correct: int
    perturbed_is_correct: int
    clean_answer: object
    perturbed_answer: object
    perturbed_parse_status: str
    cell_key: tuple


def _cell_key(perturbed_row: dict) -> tuple:
    return tuple(perturbed_row.get(key) for key in CELL_DIMENSION_KEYS)


def join_matched_pairs(rows: Sequence[dict]) -> list[MatchedPair]:
    """Join clean and perturbed rows into matched pairs.

    Clean rows are indexed by (model_revision, task_id). Each perturbed row is
    matched to the clean row for the SAME model and item, so the pairing is
    exact and never crosses models or items (design/06 §6.2).
    """
    clean_by_model_and_task: dict = {}
    perturbed_rows: list = []

    for row in rows:
        if row.get("is_clean"):
            clean_by_model_and_task[(row["model_revision"], row["task_id"])] = row
        else:
            perturbed_rows.append(row)

    matched_pairs: list[MatchedPair] = []
    for perturbed_row in perturbed_rows:
        clean_row = clean_by_model_and_task.get(
            (perturbed_row["model_revision"], perturbed_row["task_id"]))
        if clean_row is None:
            continue                          # no clean partner; cannot pair

        matched_pairs.append(MatchedPair(
            model_revision=perturbed_row["model_revision"],
            task_id=perturbed_row["task_id"],
            task_family=perturbed_row["task_family"],
            clean_is_correct=int(clean_row["is_correct"]),
            perturbed_is_correct=int(perturbed_row["is_correct"]),
            clean_answer=clean_row.get("parsed_answer"),
            perturbed_answer=perturbed_row.get("parsed_answer"),
            perturbed_parse_status=perturbed_row.get("parse_status", "valid"),
            cell_key=_cell_key(perturbed_row),
        ))

    return matched_pairs


def group_pairs_into_cells(matched_pairs: Sequence[MatchedPair]) -> dict:
    """Group matched pairs by their cell key."""
    cells: dict = defaultdict(list)
    for pair in matched_pairs:
        cells[pair.cell_key].append(pair)
    return cells


def summarize_all_cells(matched_pairs: Sequence[MatchedPair],
                        seed: int = 1729,
                        resamples: int = statistics.DEFAULT_BOOTSTRAP_RESAMPLES) -> list[dict]:
    """Produce one summary row per cell, sorted for stable output. Each row
    carries the cell's dimension values plus the full statistics block."""
    cells = group_pairs_into_cells(matched_pairs)

    summaries: list[dict] = []
    for cell_key in sorted(cells, key=lambda key: tuple(str(part) for part in key)):
        pairs = cells[cell_key]
        clean_correctness = [pair.clean_is_correct for pair in pairs]
        perturbed_correctness = [pair.perturbed_is_correct for pair in pairs]

        summary = dict(zip(CELL_DIMENSION_KEYS, cell_key))
        summary.update(statistics.summarize_cell(
            clean_correctness, perturbed_correctness, seed=seed, resamples=resamples))
        summary["answer_flip_rate"] = statistics.answer_flip_rate(
            [pair.clean_answer for pair in pairs],
            [pair.perturbed_answer for pair in pairs])
        summary["invalid_or_clarification_rate"] = statistics.invalid_or_clarification_rate(
            [pair.perturbed_parse_status for pair in pairs])
        summaries.append(summary)

    return summaries


def write_cell_table(cell_summaries: Sequence[dict], output_path: Path) -> Path:
    """Write the per-cell summaries to a CSV the paper's tables are built from."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cell_summaries:
        output_path.write_text("")
        return output_path

    field_names = list(cell_summaries[0].keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(cell_summaries)
    return output_path


# ---------------------------------------------------------------------------
# Figures (design/08 §8.7). matplotlib is imported lazily so the analysis core
# runs without it; each figure function returns the output path, or None if
# matplotlib is unavailable.
# ---------------------------------------------------------------------------

def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        return pyplot
    except ImportError:
        return None


def figure_clean_conditioned_failure_vs_edit_budget(
        cell_summaries: Sequence[dict], output_path: Path) -> Optional[Path]:
    """Clean-conditioned failure vs edit budget, one line per operation — the
    severity-curve figure (design/08 §8.7, Figure 2)."""
    pyplot = _import_pyplot()
    if pyplot is None:
        return None

    series_by_operation: dict = defaultdict(list)
    for summary in cell_summaries:
        edit_budget = summary.get("r_edit_budget")
        operation = summary.get("r_operation")
        ccf = summary.get("clean_conditioned_failure")
        if edit_budget is not None and operation is not None and ccf == ccf:   # not NaN
            series_by_operation[operation].append((edit_budget, ccf))

    figure, axes = pyplot.subplots(figsize=(7, 5))
    for operation in sorted(series_by_operation):
        points = sorted(series_by_operation[operation])
        axes.plot([budget for budget, _ in points], [value for _, value in points],
                  marker="o", label=str(operation))

    axes.set_xlabel("edit budget k")
    axes.set_ylabel("clean-conditioned failure rate")
    axes.set_title("Typo-induced failure vs severity, by edit operation")
    if series_by_operation:
        axes.legend(title="operation")
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    pyplot.close(figure)
    return output_path


def figure_keyboard_versus_asr_profile(
        cell_summaries: Sequence[dict], output_path: Path) -> Optional[Path]:
    """Side-by-side degradation profile for the two noise sources — keyboard
    typos versus ASR transcription errors — the figure the latest review
    motivated. Bars are mean paired degradation per model, grouped by source.
    """
    pyplot = _import_pyplot()
    if pyplot is None:
        return None

    keyboard_policies = {SelectionPolicy.KEYBOARD_NEIGHBOR, SelectionPolicy.INFORMATIVE_WORD, SelectionPolicy.UNIFORM}
    asr_policies = {SelectionPolicy.ASR_CLEAN, SelectionPolicy.ASR_NOISY}

    degradation_by_model: dict = defaultdict(lambda: {"keyboard": [], "asr": []})
    for summary in cell_summaries:
        policy = summary.get("r_selection_policy")
        model = summary.get("model_revision")
        delta = summary.get("delta")
        if model is None or delta is None:
            continue
        if policy in keyboard_policies:
            degradation_by_model[model]["keyboard"].append(delta)
        elif policy in asr_policies:
            degradation_by_model[model]["asr"].append(delta)

    models = sorted(degradation_by_model)
    if not models:
        return None

    def mean_or_zero(values):
        return sum(values) / len(values) if values else 0.0

    keyboard_means = [mean_or_zero(degradation_by_model[m]["keyboard"]) for m in models]
    asr_means = [mean_or_zero(degradation_by_model[m]["asr"]) for m in models]

    figure, axes = pyplot.subplots(figsize=(8, 5))
    bar_positions = range(len(models))
    bar_width = 0.38
    axes.bar([p - bar_width / 2 for p in bar_positions], keyboard_means,
             bar_width, label="keyboard typos")
    axes.bar([p + bar_width / 2 for p in bar_positions], asr_means,
             bar_width, label="ASR transcription")

    axes.set_xticks(list(bar_positions))
    axes.set_xticklabels(models, rotation=30, ha="right")
    axes.set_ylabel("mean paired degradation")
    axes.set_title("Degradation by noise source, per model")
    axes.legend()
    figure.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    pyplot.close(figure)
    return output_path

