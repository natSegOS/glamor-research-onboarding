"""Mechanized Stage-1 pilot gates (design/11 §11.2, design/06 §6.3).

Turns "did the pilot pass?" into one machine-readable block per task family:
the discordant rate at the pre-registered primary condition with its implied
per-cell N and design/06 §6.3 bucket, clean accuracy A₀, reasoning
format-compliance, truncation rate, and the p99 clean-correct output length
that freezes max_new_tokens.
"""

from __future__ import annotations

from typing import Optional, Sequence

from enums import (
    ExtractionTier, FinishReason, Operation, Scope, SelectionPolicy,
    SemanticClass, REASONING_FAMILIES,
)
from analysis import statistics


# The primary endpoint's condition cell (design/06 §6.3): Regime-A
# keyboard-neighbor substitution anywhere, at the primary edit budget.
_PRIMARY_CONDITION_FILTERS = {
    "r_semantic_class": SemanticClass.A,
    "r_operation": Operation.SUBSTITUTE,
    "r_selection_policy": SelectionPolicy.KEYBOARD_NEIGHBOR,
    "r_scope": Scope.ANYWHERE,
}

# design/06 §6.3 decision buckets for the measured discordant rate.
_UNDERPOWERED_DISCORDANT_RATE = 0.05
_CONFIRMED_N600_DISCORDANT_RATE = 0.19
_RAISED_N_DISCORDANT_RATE = 0.30

PRIMARY_MDE = 0.05  # 5 pp minimum detectable effect (design/06 §6.3)

FORMAT_COMPLIANCE_TARGET = 0.95
_CLEAN_CORRECT_LENGTH_PERCENTILE = 99


def _discordant_rate_bucket(discordant_rate: float) -> str:
    if discordant_rate < _UNDERPOWERED_DISCORDANT_RATE:
        return "underpowered_move_primary_to_higher_k"
    if discordant_rate <= _CONFIRMED_N600_DISCORDANT_RATE:
        return "n600_confirmed"
    if discordant_rate <= _RAISED_N_DISCORDANT_RATE:
        return "raise_n_or_relax_mde"
    return "above_design_range"


def _proportion(rows: Sequence[dict], predicate) -> Optional[float]:
    return sum(map(predicate, rows)) / len(rows) if rows else None


def _nearest_rank_percentile(sorted_values: Sequence[int], percentile: int) -> int:
    rank = -(-len(sorted_values) * percentile // 100)   # ceil division
    return sorted_values[max(int(rank) - 1, 0)]


def compute_stage_gates(rows: Sequence[dict],
                        primary_edit_budget_reasoning: int,
                        primary_edit_budget_mcq: int) -> dict:
    """One gates block per task family, plus the run-wide format/truncation
    gates. ``rows`` are raw generation rows (schema 1.1; rows from older
    schemas yield null for the fields they lack)."""
    families = sorted({row["task_family"] for row in rows if row.get("task_family")})

    per_family = {}
    for family in families:
        family_rows = [row for row in rows if row.get("task_family") == family]
        clean_rows = [row for row in family_rows if row.get("is_clean")]
        primary_budget = (primary_edit_budget_reasoning
                          if family in REASONING_FAMILIES else primary_edit_budget_mcq)

        clean_correct_by_key = {
            (row.get("model_id"), row["model_revision"], row["task_id"]):
                int(row["is_correct"])
            for row in clean_rows}
        primary_rows = [
            row for row in family_rows
            if not row.get("is_clean")
            and row.get("r_edit_budget") == primary_budget
            and all(row.get(field) == value
                    for field, value in _PRIMARY_CONDITION_FILTERS.items())
            and (row.get("model_id"), row["model_revision"], row["task_id"])
            in clean_correct_by_key]

        block: dict = {
            "clean_accuracy": _proportion(clean_rows, lambda row: int(row["is_correct"])),
            "primary_edit_budget": primary_budget,
            "primary_condition_pairs": len(primary_rows),
        }
        if primary_rows:
            clean_arm = [clean_correct_by_key[
                (row.get("model_id"), row["model_revision"], row["task_id"])]
                for row in primary_rows]
            perturbed_arm = [int(row["is_correct"]) for row in primary_rows]
            discordant_rate = statistics.discordant_rate(clean_arm, perturbed_arm)
            block["discordant_rate"] = discordant_rate
            block["delta"] = statistics.paired_degradation(clean_arm, perturbed_arm)
            block["discordant_rate_bucket"] = _discordant_rate_bucket(discordant_rate)
            block["implied_n_at_5pp_mde"] = (
                statistics.mcnemar_sample_size(PRIMARY_MDE, discordant_rate)
                if discordant_rate >= PRIMARY_MDE else None)
        per_family[family] = block

    reasoning_rows = [row for row in rows
                      if row.get("task_family") in REASONING_FAMILIES]
    clean_correct_lengths = sorted(
        row["output_token_count"] for row in rows
        if row.get("is_clean") and row.get("is_correct")
        and row.get("output_token_count") is not None)

    return {
        "per_task_family": per_family,
        "reasoning_format_compliance": _proportion(
            reasoning_rows,
            lambda row: row.get("extraction_tier") == ExtractionTier.HASH_DELIMITED),
        "reasoning_format_compliance_target": FORMAT_COMPLIANCE_TARGET,
        "truncation_rate": _proportion(
            [row for row in rows if row.get("finish_reason") is not None],
            lambda row: row.get("finish_reason") == FinishReason.TRUNCATED),
        "p99_clean_correct_output_tokens": (
            _nearest_rank_percentile(
                clean_correct_lengths, _CLEAN_CORRECT_LENGTH_PERCENTILE)
            if clean_correct_lengths else None),
    }
