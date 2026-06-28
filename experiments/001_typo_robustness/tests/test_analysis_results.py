"""Tests for analysis.results: matched-pair joining, cell summarization,
audit-exclusion gate (Part 7), and VALID-only sensitivity (Part 4).
"""

from __future__ import annotations

import math
import pytest

from analysis.results import (
    MatchedPair,
    join_matched_pairs,
    summarize_all_cells,
)
from analysis.audit import ItemAuditOutcome
from enums import ParseStatus, SemanticClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(task_id, clean_correct, pert_correct, parse_status="valid",
               model_revision="model_a", task_family="gsm_symbolic_synthetic",
               cell_extra=None):
    cell_key = (
        model_revision, task_family,
        str(SemanticClass.A), "substitute", "keyboard_neighbor", "anywhere", 1
    )
    if cell_extra:
        cell_key = cell_extra
    return MatchedPair(
        model_revision=model_revision,
        task_id=task_id,
        task_family=task_family,
        clean_is_correct=clean_correct,
        perturbed_is_correct=pert_correct,
        clean_answer=str(clean_correct),
        perturbed_answer=str(pert_correct) if parse_status == "valid" else None,
        perturbed_parse_status=parse_status,
        cell_key=cell_key,
    )


def _make_outcome(task_id, excluded: bool, regime=SemanticClass.A):
    return ItemAuditOutcome(
        item_id=task_id,
        regime_label=regime,
        majority_intent_preserved=not excluded,
        majority_gold_unchanged=True,
        was_adjudicated=False,
        rating_count=3,
        excluded_from_primary=excluded,
    )


# ---------------------------------------------------------------------------
# join_matched_pairs
# ---------------------------------------------------------------------------

def test_join_matched_pairs_basic():
    rows = [
        {"model_revision": "m1", "task_id": "t1", "is_clean": True,
         "is_correct": 1, "parsed_answer": "42", "parse_status": "valid",
         "task_family": "gsm_symbolic_synthetic",
         "r_semantic_class": "A", "r_operation": "substitute",
         "r_selection_policy": "keyboard_neighbor", "r_scope": "anywhere",
         "r_edit_budget": 1},
        {"model_revision": "m1", "task_id": "t1", "is_clean": False,
         "is_correct": 0, "parsed_answer": None, "parse_status": "unparseable",
         "task_family": "gsm_symbolic_synthetic",
         "r_semantic_class": "A", "r_operation": "substitute",
         "r_selection_policy": "keyboard_neighbor", "r_scope": "anywhere",
         "r_edit_budget": 1},
    ]
    pairs = join_matched_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0].clean_is_correct == 1
    assert pairs[0].perturbed_is_correct == 0


def test_join_matched_pairs_unmatched_perturbed_is_skipped():
    rows = [
        {"model_revision": "m1", "task_id": "t_orphan", "is_clean": False,
         "is_correct": 0, "parsed_answer": None, "parse_status": "unparseable",
         "task_family": "gsm_symbolic_synthetic",
         "r_semantic_class": "A", "r_operation": "substitute",
         "r_selection_policy": "keyboard_neighbor", "r_scope": "anywhere",
         "r_edit_budget": 1},
    ]
    pairs = join_matched_pairs(rows)
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# summarize_all_cells — basic
# ---------------------------------------------------------------------------

def test_summarize_all_cells_basic():
    pairs = [_make_pair(f"t{i}", 1, 0) for i in range(10)]
    summaries = summarize_all_cells(pairs, resamples=100)
    assert len(summaries) == 1
    assert summaries[0]["n"] == 10
    assert math.isclose(summaries[0]["delta"], 1.0, abs_tol=1e-9)


def test_summarize_all_cells_n_audit_excluded_zero_when_no_gate():
    pairs = [_make_pair(f"t{i}", 1, 0) for i in range(5)]
    summaries = summarize_all_cells(pairs, resamples=100)
    assert summaries[0]["n_audit_excluded"] == 0


# ---------------------------------------------------------------------------
# Audit-exclusion gate (Part 7)
# ---------------------------------------------------------------------------

def test_audit_gate_excludes_flagged_items():
    pairs = [_make_pair(f"t{i}", 1, 0) for i in range(6)]
    audit_outcomes = {
        "t0": _make_outcome("t0", excluded=True),
        "t1": _make_outcome("t1", excluded=True),
    }
    summaries = summarize_all_cells(pairs, resamples=100, audit_outcomes=audit_outcomes)
    assert summaries[0]["n_audit_excluded"] == 2
    assert summaries[0]["n"] == 4


def test_audit_gate_none_leaves_all_pairs():
    pairs = [_make_pair(f"t{i}", 1, 0) for i in range(5)]
    summaries_no_gate = summarize_all_cells(pairs, resamples=100, audit_outcomes=None)
    summaries_with_gate = summarize_all_cells(
        pairs, resamples=100,
        audit_outcomes={"t0": _make_outcome("t0", excluded=False)})
    assert summaries_no_gate[0]["n"] == summaries_with_gate[0]["n"]


def test_audit_gate_all_excluded_yields_empty_cell():
    pairs = [_make_pair("t0", 1, 0)]
    audit_outcomes = {"t0": _make_outcome("t0", excluded=True)}
    summaries = summarize_all_cells(pairs, resamples=100, audit_outcomes=audit_outcomes)
    assert summaries[0]["n_audit_excluded"] == 1
    assert summaries[0]["n"] == 0


def test_audit_gate_confirmed_items_counted():
    pairs = [_make_pair(f"t{i}", 1, 0) for i in range(4)]
    audit_outcomes = {
        "t0": _make_outcome("t0", excluded=True),
        "t1": _make_outcome("t1", excluded=False),   # confirmed, kept
    }
    summaries = summarize_all_cells(pairs, resamples=100, audit_outcomes=audit_outcomes)
    assert summaries[0]["n_audit_excluded"] == 1
    assert summaries[0]["n"] == 3


# ---------------------------------------------------------------------------
# VALID-only sensitivity (Part 4)
# ---------------------------------------------------------------------------

def test_valid_only_delta_present_when_enough_valid_pairs():
    pairs = [_make_pair(f"t{i}", 1, 0, parse_status="valid") for i in range(10)]
    summaries = summarize_all_cells(pairs, resamples=100)
    assert summaries[0]["delta_valid_only"] is not None
    assert math.isclose(summaries[0]["delta_valid_only"], 1.0, abs_tol=1e-9)


def test_valid_only_excludes_unparseable_pairs():
    """VALID-only delta computed on only the VALID-status subset."""
    # 6 valid pairs (all clean=1, pert=0 → delta=1), 4 unparseable (pert=0 but UNPARSEABLE)
    valid_pairs = [_make_pair(f"tv{i}", 1, 0, parse_status="valid") for i in range(6)]
    # Unparseable pairs: clean=1, pert=0; but VALID-only should only use the 6 above
    unp_pairs = [_make_pair(f"tu{i}", 1, 0, parse_status="unparseable") for i in range(4)]
    summaries = summarize_all_cells(valid_pairs + unp_pairs, resamples=100)
    assert summaries[0]["delta_valid_only"] is not None
    # VALID-only uses 6 pairs, all broke → delta_valid_only = 1.0
    assert math.isclose(summaries[0]["delta_valid_only"], 1.0, abs_tol=1e-9)
    # All-in uses 10 pairs (including unparseable, which score pert_is_correct=0)
    assert summaries[0]["n"] == 10


def test_valid_only_none_when_fewer_than_two_valid_pairs():
    pairs = [_make_pair("t0", 1, 0, parse_status="unparseable")]
    summaries = summarize_all_cells(pairs, resamples=100)
    assert summaries[0]["delta_valid_only"] is None
    assert summaries[0]["mcnemar_p_valid_only"] is None


def test_valid_only_coincides_with_allin_when_icr_is_zero():
    """When all parse statuses are VALID, delta_valid_only == delta."""
    pairs = [_make_pair(f"t{i}", 1, 0, parse_status="valid") for i in range(20)]
    summaries = summarize_all_cells(pairs, resamples=200)
    assert math.isclose(
        summaries[0]["delta"],
        summaries[0]["delta_valid_only"],
        abs_tol=1e-9,
    )


def test_valid_only_mcnemar_p_present():
    pairs = [
        _make_pair("t0", 1, 0, parse_status="valid"),
        _make_pair("t1", 1, 1, parse_status="valid"),
        _make_pair("t2", 0, 0, parse_status="valid"),
    ]
    summaries = summarize_all_cells(pairs, resamples=100)
    assert summaries[0]["mcnemar_p_valid_only"] is not None
    assert 0.0 <= summaries[0]["mcnemar_p_valid_only"] <= 1.0
