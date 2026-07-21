"""Turning generation rows into matched pairs, per-cell tables, and figures.

Provenance
----------
This module consumes the JSONL rows written by pipeline.runner.run_shard and
produces the analysis deliverables of design/06 §6.10 and design/08:
  - matched clean/perturbed pairs joined per (model_revision, task_id);
  - one statistics.summarize_cell block per reporting cell;
  - the figures of design/08 §8.7.

Every cell is keyed by the r_-prefixed perturbation-state fields that the runner
writes, so the analysis dimensions and the logged dimensions cannot drift apart.
Cells smaller than the interval minimum are summarized without a confidence
interval rather than dropped, so an underpowered pilot cell never crashes the
run (the n<2 guard lives in statistics.summarize_cell).
"""

from __future__ import annotations

import csv
import os

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from enums import FragmentationStratum, ParseStatus, SelectionPolicy
from analysis import statistics


# The dimensions that define a reporting cell (design/06 §6.10). All are
# r_-prefixed perturbation-state fields written by the runner, plus the model
# and task family. model_id is included alongside model_revision because
# unpinned runs share the PIN_ME revision placeholder: keying on revision
# alone would merge every unpinned model into one cell.
CELL_DIMENSION_KEYS = (
    "model_id",
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

    Clean rows are indexed by (model_id, model_revision, task_id). Each
    perturbed row is matched to the clean row for the SAME model and item, so
    the pairing is exact and never crosses models or items (design/06 §6.2).
    """
    def model_and_task(row: dict) -> tuple:
        return (row.get("model_id"), row["model_revision"], row["task_id"])

    clean_by_model_and_task = {
        model_and_task(row): row for row in rows if row.get("is_clean")}

    matched_pairs: list[MatchedPair] = []
    for perturbed_row in (row for row in rows if not row.get("is_clean")):
        clean_row = clean_by_model_and_task.get(model_and_task(perturbed_row))
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
            perturbed_parse_status=perturbed_row.get("parse_status", ParseStatus.VALID),
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
                        resamples: int = statistics.DEFAULT_BOOTSTRAP_RESAMPLES,
                        audit_outcomes: Optional[dict] = None) -> list[dict]:
    """Produce one summary row per cell, sorted for stable output.

    Each row carries the cell's dimension values plus the full statistics block.

    ``audit_outcomes``, when provided, maps task_id to
    ``analysis.audit.ItemAuditOutcome``. Pairs whose item is flagged
    ``excluded_from_primary=True`` are removed from the cell before computing
    any statistics; ``n_audit_excluded`` in the output counts them.  When None
    the gate is open (no items excluded).

    VALID-only sensitivity (Part 4): ``delta_valid_only`` and
    ``mcnemar_p_valid_only`` are computed on the subset of pairs where the
    perturbed response has ParseStatus.VALID, providing a sensitivity check
    that coincides with the all-in statistics when ICR=0 (design/06 §6.10).
    """
    cells = group_pairs_into_cells(matched_pairs)

    summaries: list[dict] = []
    for cell_key in sorted(cells, key=lambda key: tuple(str(part) for part in key)):
        all_pairs = cells[cell_key]

        # Audit-exclusion gate (Part 7).
        n_audit_excluded = 0
        if audit_outcomes is not None:
            excluded_ids = {
                pair.task_id for pair in all_pairs
                if audit_outcomes.get(pair.task_id) is not None
                and audit_outcomes[pair.task_id].excluded_from_primary
            }
            n_audit_excluded = len(excluded_ids)
            pairs = [pair for pair in all_pairs if pair.task_id not in excluded_ids]
        else:
            pairs = all_pairs

        clean_correctness = [pair.clean_is_correct for pair in pairs]
        perturbed_correctness = [pair.perturbed_is_correct for pair in pairs]

        summary: dict = dict(zip(CELL_DIMENSION_KEYS, cell_key))
        summary.update(statistics.summarize_cell(
            clean_correctness, perturbed_correctness, seed=seed, resamples=resamples))
        summary["answer_flip_rate"] = (
            statistics.answer_flip_rate(
                [pair.clean_answer for pair in pairs],
                [pair.perturbed_answer for pair in pairs])
            if pairs else float("nan"))
        summary["invalid_or_clarification_rate"] = (
            statistics.invalid_or_clarification_rate(
                [pair.perturbed_parse_status for pair in pairs])
            if pairs else float("nan"))
        summary["n_audit_excluded"] = n_audit_excluded

        # VALID-only sensitivity (Part 4).
        valid_pairs = [pair for pair in pairs
                       if pair.perturbed_parse_status == ParseStatus.VALID]
        if len(valid_pairs) >= 2:
            valid_clean = [pair.clean_is_correct for pair in valid_pairs]
            valid_perturbed = [pair.perturbed_is_correct for pair in valid_pairs]
            delta_valid_only: Optional[float] = statistics.paired_degradation(
                valid_clean, valid_perturbed)
            valid_table = statistics.build_paired_table(valid_clean, valid_perturbed)
            valid_mcnemar = statistics.mcnemar_test(
                valid_table.broke, valid_table.recovered)
            mcnemar_p_valid_only: Optional[float] = valid_mcnemar.p_value
        else:
            delta_valid_only = None
            mcnemar_p_valid_only = None

        summary["delta_valid_only"] = delta_valid_only
        summary["mcnemar_p_valid_only"] = mcnemar_p_valid_only

        summaries.append(summary)

    return summaries


# The dimensions of one Method A contrast group (all Low/High pairs of one
# model x family x budget).
FRAGMENTATION_CONTRAST_KEYS = (
    "model_id", "model_revision", "task_family", "r_edit_budget")


def summarize_fragmentation_contrast(rows: Sequence[dict],
                                     seed: int = 1729,
                                     resamples: int = statistics.DEFAULT_BOOTSTRAP_RESAMPLES,
                                     ) -> list[dict]:
    """Method A: the paired Low-vs-High fragmentation contrast (design/06 §6.8).

    Pairs the two fragmentation_matched variants of each item x budget,
    restricted to items whose CLEAN generation was correct (the contrast is
    clean-conditioned by construction). The Low variant plays the "clean" arm
    and High the "perturbed" arm of the standard paired machinery, so
    ``delta`` in each summary is ΔCCF_frag = P(High wrong) − P(Low wrong): a
    positive, significant value means more fragmentation causes more failure
    with meaning, word, position, and edit count held fixed.
    """
    clean_correct_items = {
        (row.get("model_id"), row["model_revision"], row["task_id"])
        for row in rows if row.get("is_clean") and row.get("is_correct")}

    counterfactual_rows = [
        row for row in rows
        if not row.get("is_clean")
        and row.get("r_selection_policy") == SelectionPolicy.FRAGMENTATION_MATCHED
        and (row.get("model_id"), row["model_revision"], row["task_id"])
        in clean_correct_items]

    correctness_by_pair: dict = defaultdict(dict)
    for row in counterfactual_rows:
        group_key = tuple(row.get(key) for key in FRAGMENTATION_CONTRAST_KEYS)
        correctness_by_pair[(group_key, row["task_id"])][
            row.get("r_fragmentation_stratum")] = int(row["is_correct"])

    contrast_arms_by_group: dict = defaultdict(lambda: ([], []))
    for (group_key, _task_id), correctness_by_stratum in sorted(
            correctness_by_pair.items(), key=lambda entry: tuple(map(str, entry[0]))):
        if {FragmentationStratum.LOW, FragmentationStratum.HIGH} <= set(correctness_by_stratum):
            low_arm, high_arm = contrast_arms_by_group[group_key]
            low_arm.append(correctness_by_stratum[FragmentationStratum.LOW])
            high_arm.append(correctness_by_stratum[FragmentationStratum.HIGH])

    summaries = []
    for group_key in sorted(contrast_arms_by_group,
                            key=lambda key: tuple(str(part) for part in key)):
        low_arm, high_arm = contrast_arms_by_group[group_key]
        summary: dict = dict(zip(FRAGMENTATION_CONTRAST_KEYS, group_key))
        summary.update(statistics.summarize_cell(
            low_arm, high_arm, seed=seed, resamples=resamples))
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
    # matplotlib reads MPLBACKEND at import time, before matplotlib.use() below
    # can override it, so a stray inherited value (e.g. Jupyter's own) would
    # crash the import itself. Force it first, regardless of environment.
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        return pyplot
    except ImportError:
        return None


def figure_clean_conditioned_failure_vs_edit_budget(
        cell_summaries: Sequence[dict], output_path: Path) -> Optional[Path]:
    """Clean-conditioned failure vs edit budget, one line per operation: the
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
