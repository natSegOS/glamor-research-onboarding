"""Build a self-contained HTML results report from one or more generation JSONL files.

Two views per condition cell:
  GLOBAL — summary statistics (CCF, Δ + BCa CI, McNemar p, discordant rate, 2×2
            counts) plus a fragmentation-strata split (Low vs High) derived
            from per-item tokenization data.
  LOCAL  — per-item drill-down: clean-vs-perturbed inline diff, edit script,
            tokenization fields, raw model output, parsed / expected answer,
            correctness, and parse status.

Each global cell is a collapsible Bootstrap card. The local drill-down starts
collapsed — click "Show items" to expand.

Usage:

    python tools/build_report.py \\
        --generations results/pilot/pilot_generations.jsonl \\
        --output results/pilot/report.html

    # merge shards from multiple files:
    python tools/build_report.py \\
        --generations results/main/main_generations.jsonl \\
                      results/main/main_generations_rerun.jsonl \\
        --output results/main/report.html
"""

from __future__ import annotations

import argparse
import difflib
import html as html_module
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.results import (
    CELL_DIMENSION_KEYS,
    join_matched_pairs,
    summarize_all_cells,
    group_pairs_into_cells,
)
from analysis import statistics
from enums import FragmentationStratum, ParseStatus, SemanticClass
from pipeline.runner import load_generation_rows


# ---------------------------------------------------------------------------
# Inline diff (same helper as preview_perturbations.py)
# ---------------------------------------------------------------------------

def _inline_diff_html(original: str, perturbed: str) -> str:
    matcher = difflib.SequenceMatcher(None, original, perturbed, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(html_module.escape(original[i1:i2]))
        elif tag == "replace":
            parts.append(
                f'<del class="diff-del">{html_module.escape(original[i1:i2])}</del>'
                f'<ins class="diff-ins">{html_module.escape(perturbed[j1:j2])}</ins>')
        elif tag == "delete":
            parts.append(
                f'<del class="diff-del">{html_module.escape(original[i1:i2])}</del>')
        elif tag == "insert":
            parts.append(
                f'<ins class="diff-ins">{html_module.escape(perturbed[j1:j2])}</ins>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _pct(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.1f}%"


def _p_value(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def _delta_ci(summary: dict) -> str:
    delta = summary.get("delta")
    lo = summary.get("delta_ci_low")
    hi = summary.get("delta_ci_high")
    method = summary.get("delta_ci_method", "")
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "—"
    if method == "insufficient_n":
        return f"{_pct(delta)} (n too small for CI)"
    if lo is None or hi is None:
        return _pct(delta)
    return f"{_pct(delta)} [{_pct(lo)}, {_pct(hi)}]"


def _significance_badge(p_value) -> str:
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return ""
    if p_value < 0.001:
        return '<span class="badge bg-danger ms-1">***</span>'
    if p_value < 0.01:
        return '<span class="badge bg-warning text-dark ms-1">**</span>'
    if p_value < 0.05:
        return '<span class="badge bg-secondary ms-1">*</span>'
    return '<span class="badge bg-light text-dark border ms-1">ns</span>'


# ---------------------------------------------------------------------------
# Local drill-down items table
# ---------------------------------------------------------------------------

def _items_table_html(pairs_with_data: list[dict], items_id: str) -> str:
    """Build the collapsed per-item drill-down table for one cell."""
    if not pairs_with_data:
        return ""

    rows_html_parts = []
    for row in pairs_with_data:
        clean_prompt = row.get("clean_prompt", "")
        perturbed_prompt = row.get("prompt", "")
        diff_html = _inline_diff_html(clean_prompt, perturbed_prompt) if clean_prompt else "—"

        correct_badge = (
            '<span class="badge bg-success">✓</span>'
            if row.get("perturbed_is_correct")
            else '<span class="badge bg-danger">✗</span>')
        clean_badge = (
            '<span class="badge bg-success">✓</span>'
            if row.get("clean_is_correct")
            else '<span class="badge bg-danger">✗</span>')

        parse_status = row.get("parse_status", "")
        parse_badge = ""
        if parse_status == ParseStatus.VALID:
            parse_badge = '<span class="badge bg-light text-dark border">valid</span>'
        elif parse_status:
            parse_badge = f'<span class="badge bg-warning text-dark">{html_module.escape(parse_status)}</span>'

        frag = row.get("fragmentation_stratum", "")
        frag_badge = (
            '<span class="badge bg-info text-dark">High frag</span>'
            if frag == FragmentationStratum.HIGH
            else '<span class="badge bg-light text-dark border">Low frag</span>'
            if frag == FragmentationStratum.LOW
            else "")

        tir = row.get("token_inflation_ratio")
        tir_str = f"{tir:.2f}" if tir is not None else "—"

        edited_word = row.get("edited_word", "")
        raw_output = html_module.escape(str(row.get("model_output", ""))[:200])
        parsed = html_module.escape(str(row.get("parsed_answer", "")))
        expected = html_module.escape(str(row.get("expected_answer", "")))

        rows_html_parts.append(f"""
              <tr>
                <td class="align-top text-nowrap small text-muted">
                  {html_module.escape(row.get('task_id', ''))}
                </td>
                <td class="align-top small font-monospace"
                    style="white-space: pre-wrap; max-width: 380px">{diff_html}</td>
                <td class="align-top small text-center">
                  clean {clean_badge}<br>pert {correct_badge}<br>{parse_badge}
                </td>
                <td class="align-top small">
                  {frag_badge}<br>
                  <span class="text-muted">TIR:</span> {tir_str}<br>
                  {('<span class="text-muted">word:</span> ' + html_module.escape(edited_word)) if edited_word else ''}
                </td>
                <td class="align-top small font-monospace"
                    style="white-space: pre-wrap; max-width: 240px; color: #555">
                  <em>parsed:</em> {parsed}<br>
                  <em>expected:</em> {expected}
                  {('<br><details><summary class="text-muted">raw output</summary><pre style="font-size:.75rem;white-space:pre-wrap">' + raw_output + "</pre></details>") if raw_output else ""}
                </td>
              </tr>""")

    rows_html = "\n".join(rows_html_parts)
    return f"""
        <div class="collapse" id="{items_id}">
          <div class="card-body p-0 border-top">
            <div class="table-responsive" style="max-height: 600px; overflow-y: auto">
              <table class="table table-sm table-hover mb-0">
                <thead class="table-secondary sticky-top">
                  <tr>
                    <th style="min-width:100px">task_id</th>
                    <th style="min-width:200px">prompt diff (clean → perturbed)</th>
                    <th>correct?</th>
                    <th>tokenization</th>
                    <th>answer</th>
                  </tr>
                </thead>
                <tbody>{rows_html}
                </tbody>
              </table>
            </div>
          </div>
        </div>"""


# ---------------------------------------------------------------------------
# Fragmentation strata sub-table
# ---------------------------------------------------------------------------

def _frag_strata_html(pairs_with_data: list[dict]) -> str:
    """Build a two-row Low / High fragmentation breakdown for one cell."""
    low_clean = [row["clean_is_correct"] for row in pairs_with_data
                 if row.get("fragmentation_stratum") == FragmentationStratum.LOW]
    low_perturbed = [row["perturbed_is_correct"] for row in pairs_with_data
                     if row.get("fragmentation_stratum") == FragmentationStratum.LOW]
    high_clean = [row["clean_is_correct"] for row in pairs_with_data
                  if row.get("fragmentation_stratum") == FragmentationStratum.HIGH]
    high_perturbed = [row["perturbed_is_correct"] for row in pairs_with_data
                      if row.get("fragmentation_stratum") == FragmentationStratum.HIGH]

    def _mini_summary(clean_correctness, perturbed_correctness):
        if not clean_correctness:
            return "(no data)", "—", "—"
        pair_count = len(clean_correctness)
        ccf = statistics.clean_conditioned_failure(clean_correctness, perturbed_correctness)
        delta = statistics.paired_degradation(clean_correctness, perturbed_correctness)
        return f"n={pair_count}", _pct(ccf), _pct(delta)

    low_pair_count, low_ccf, low_delta = _mini_summary(low_clean, low_perturbed)
    high_pair_count, high_ccf, high_delta = _mini_summary(high_clean, high_perturbed)

    if not low_clean and not high_clean:
        return ""

    return f"""
        <div class="px-3 py-2 border-top bg-light">
          <span class="text-muted small fw-semibold">Fragmentation strata:</span>
          <table class="table table-sm table-borderless mb-0 d-inline-table ms-2" style="width:auto">
            <thead><tr><th class="small py-0">stratum</th><th class="small py-0">n</th>
              <th class="small py-0">CCF</th><th class="small py-0">Δ</th></tr></thead>
            <tbody>
              <tr><td><span class="badge bg-light text-dark border">Low</span></td>
                <td class="small">{low_pair_count}</td>
                <td class="small">{low_ccf}</td><td class="small">{low_delta}</td></tr>
              <tr><td><span class="badge bg-info text-dark">High</span></td>
                <td class="small">{high_pair_count}</td>
                <td class="small">{high_ccf}</td><td class="small">{high_delta}</td></tr>
            </tbody>
          </table>
        </div>"""


# ---------------------------------------------------------------------------
# Cell card
# ---------------------------------------------------------------------------

_REGIME_BADGE = {
    str(SemanticClass.A): '<span class="badge bg-warning text-dark">Regime A — nonword typo</span>',
    str(SemanticClass.B): '<span class="badge bg-info text-dark">Regime B — real-word shift</span>',
    str(SemanticClass.C): '<span class="badge bg-secondary">Regime C — meaning change</span>',
    str(SemanticClass.CLEAN): '<span class="badge bg-light text-dark border">clean baseline</span>',
}


def _cell_card_html(summary: dict, pairs_with_data: list[dict], card_index: int) -> str:
    card_id = f"cell{card_index}"
    items_id = f"items{card_index}"

    regime = str(summary.get("r_semantic_class", ""))
    operation = str(summary.get("r_operation", ""))
    policy = str(summary.get("r_selection_policy", ""))
    scope = str(summary.get("r_scope", ""))
    budget = summary.get("r_edit_budget", "?")
    task = str(summary.get("task_family", ""))
    model = str(summary.get("model_revision", ""))

    badge = _REGIME_BADGE.get(regime, f'<span class="badge bg-dark">{regime}</span>')
    budget_label = f"k={budget}"

    pair_count = summary.get("n", 0)
    p_value = summary.get("mcnemar_p_value")
    significance = _significance_badge(p_value)

    # 2×2 counts
    both_correct = summary.get("both_correct", 0)
    broke = summary.get("broke", 0)
    recovered = summary.get("recovered", 0)
    both_wrong = summary.get("both_wrong", 0)

    table_2x2 = f"""<table class="table table-bordered table-sm mb-0" style="width:auto;font-size:.8rem">
          <thead class="table-light"><tr><th></th><th>pert ✓</th><th>pert ✗</th></tr></thead>
          <tbody>
            <tr><th>clean ✓</th>
              <td class="bg-success bg-opacity-10">{both_correct}</td>
              <td class="bg-danger bg-opacity-10"><strong>{broke}</strong></td></tr>
            <tr><th>clean ✗</th>
              <td class="bg-info bg-opacity-10">{recovered}</td>
              <td>{both_wrong}</td></tr>
          </tbody>
        </table>"""

    frag_html = _frag_strata_html(pairs_with_data)
    items_html = _items_table_html(pairs_with_data, items_id)
    item_count = len(pairs_with_data)

    return f"""
    <div class="card mb-3 shadow-sm">
      <div class="card-header d-flex flex-wrap align-items-center gap-2 py-2"
           style="cursor:pointer" data-bs-toggle="collapse"
           data-bs-target="#{card_id}" aria-expanded="true">
        <span class="fw-bold text-truncate" style="max-width:260px"
              title="{html_module.escape(task)} · {html_module.escape(policy)} · {budget_label}">
          {html_module.escape(task)}&nbsp;·&nbsp;{html_module.escape(policy)}&nbsp;·&nbsp;{budget_label}
        </span>
        {badge}
        <span class="text-muted small ms-auto">
          op: <code>{html_module.escape(operation)}</code> &nbsp;
          scope: <code>{html_module.escape(scope)}</code>
        </span>
        <span class="ms-2 text-muted">&#9660;</span>
      </div>

      <div class="collapse show" id="{card_id}">
        <div class="card-body d-flex flex-wrap gap-4 align-items-start py-3">

          <div>
            <div class="text-muted small mb-1">n pairs</div>
            <div class="fs-5 fw-bold">{pair_count}</div>
          </div>
          <div>
            <div class="text-muted small mb-1">CCF</div>
            <div class="fs-5 fw-bold">{_pct(summary.get("clean_conditioned_failure"))}</div>
          </div>
          <div>
            <div class="text-muted small mb-1">Δ [BCa 95% CI]</div>
            <div class="fw-bold">{_delta_ci(summary)}</div>
          </div>
          <div>
            <div class="text-muted small mb-1">McNemar p</div>
            <div class="fw-bold">{_p_value(p_value)} {significance}</div>
          </div>
          <div>
            <div class="text-muted small mb-1">discordant rate</div>
            <div class="fw-bold">{_pct(summary.get("discordant_rate"))}</div>
          </div>

          <div class="ms-auto">
            {table_2x2}
          </div>
        </div>
        {frag_html}
        <div class="card-footer bg-transparent d-flex align-items-center py-2">
          <button class="btn btn-sm btn-outline-secondary"
                  data-bs-toggle="collapse" data-bs-target="#{items_id}"
                  aria-expanded="false">
            Show / hide {item_count} items
          </button>
          <span class="ms-3 text-muted small">
            model: <code>{html_module.escape(model[:40])}</code>
          </span>
        </div>
        {items_html}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Full HTML page
# ---------------------------------------------------------------------------

def _build_page(
        keyboard_sections: list[str],
        meta: dict,
) -> str:
    keyboard_html = "\n".join(keyboard_sections) if keyboard_sections else (
        '<p class="text-muted">No keyboard-typo conditions found.</p>')

    source_files = html_module.escape(", ".join(meta.get("source_files", [])))
    n_pairs = meta.get("n_pairs", 0)
    n_rows = meta.get("n_rows", 0)
    n_models = meta.get("n_models", 0)
    n_cells = meta.get("n_cells", 0)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Results Report — GLAMOR Exp 001</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
        crossorigin="anonymous">
  <style>
    del.diff-del {{
      background: #ffc8c8; color: #7d0000;
      text-decoration: line-through; border-radius: 2px; padding: 0 1px;
    }}
    ins.diff-ins {{
      background: #c8ffc8; color: #005a00;
      text-decoration: none; border-radius: 2px; padding: 0 1px;
    }}
    body {{ font-size: 0.9rem; }}
    .sticky-top {{ top: 0; z-index: 1; }}
  </style>
</head>
<body class="bg-light">
<div class="container-fluid py-4">

  <div class="mb-4">
    <h2 class="mb-1">Results Report — GLAMOR Lab Exp 001</h2>
    <p class="text-muted small mb-1">
      Sources: <code>{source_files}</code><br>
      {n_rows} total rows &nbsp;·&nbsp; {n_pairs} matched pairs &nbsp;·&nbsp;
      {n_models} model(s) &nbsp;·&nbsp; {n_cells} condition cells
    </p>
    <p class="text-muted small">
      <span class="badge bg-warning text-dark">Regime A</span> nonword typo &nbsp;
      <span class="badge bg-info text-dark">Regime B</span> real-word shift &nbsp;
      Δ = perturbed_acc − clean_acc &nbsp;
      CCF = clean-conditioned failure rate &nbsp;
      <del class="diff-del">deleted</del> <ins class="diff-ins">inserted</ins>
    </p>
  </div>

  <!-- === KEYBOARD ARM === -->
  <h4 class="border-bottom pb-2 mb-3">Keyboard-typo arm</h4>
  {keyboard_html}

</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        integrity="sha384-YvpcrYf0tY3lHB60NNkmXc4s9bIOgUxi8T/jzmWLzEOA6DpPOHFPk+WRZ4M9wEMo"
        crossorigin="anonymous"></script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generations", nargs="+", required=True, type=Path,
        metavar="JSONL",
        help="one or more generation JSONL files (merged, deduped by row_id)")
    parser.add_argument(
        "--output", required=True, type=Path,
        help="output HTML file path (e.g. results/pilot/report.html)")
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=statistics.DEFAULT_BOOTSTRAP_RESAMPLES,
        help=f"BCa bootstrap resamples (default: {statistics.DEFAULT_BOOTSTRAP_RESAMPLES})")
    parser.add_argument(
        "--seed", type=int, default=1729,
        help="random seed for bootstrap (default: 1729)")
    args = parser.parse_args()

    # --- Load rows ---
    print(f"loading {len(args.generations)} generation file(s) ...")
    rows = load_generation_rows(args.generations)
    print(f"  {len(rows)} rows loaded")

    # Build a lookup: (model_revision, task_id) → clean_row, for the local view.
    clean_rows_by_key: dict = {}
    perturbed_rows: list = []
    for row in rows:
        if row.get("is_clean"):
            clean_rows_by_key[(row["model_revision"], row["task_id"])] = row
        else:
            perturbed_rows.append(row)

    # --- Join pairs and compute cell summaries ---
    print("joining matched pairs ...")
    pairs = join_matched_pairs(rows)
    print(f"  {len(pairs)} matched pairs")

    cell_summaries = summarize_all_cells(
        pairs, seed=args.seed, resamples=args.bootstrap_resamples)
    print(f"  {len(cell_summaries)} condition cells")

    # --- Build per-item data for the local drill-down ---
    # Index perturbed rows by (model_revision, task_id, cell_key components)
    # so we can associate each pair with its perturbed row data.
    perturbed_by_key: dict = {}
    for row in perturbed_rows:
        key = (row["model_revision"], row["task_id"],
               tuple(row.get(dimension_key) for dimension_key in CELL_DIMENSION_KEYS))
        perturbed_by_key[key] = row

    # Group pairs by cell_key, augmented with per-row data.
    cells_by_key = group_pairs_into_cells(pairs)
    pairs_with_data_by_cell: dict = {}
    for cell_key, cell_pairs in cells_by_key.items():
        augmented = []
        for pair in cell_pairs:
            lookup_key = (pair.model_revision, pair.task_id, cell_key)
            pert_row = perturbed_by_key.get(lookup_key, {})
            clean_row = clean_rows_by_key.get((pair.model_revision, pair.task_id), {})
            augmented.append({
                "task_id": pair.task_id,
                "clean_is_correct": pair.clean_is_correct,
                "perturbed_is_correct": pair.perturbed_is_correct,
                "clean_prompt": clean_row.get("prompt", ""),
                "prompt": pert_row.get("prompt", ""),
                "fragmentation_stratum": pert_row.get("fragmentation_stratum"),
                "token_inflation_ratio": pert_row.get("token_inflation_ratio"),
                "edited_word": pert_row.get("edited_word", ""),
                "model_output": pert_row.get("model_output", ""),
                "parsed_answer": pert_row.get("parsed_answer"),
                "expected_answer": pert_row.get("expected_answer"),
                "parse_status": pert_row.get("parse_status", ""),
            })
        pairs_with_data_by_cell[cell_key] = augmented

    # --- Build HTML ---
    print("building HTML ...")

    keyboard_cards: list[str] = []
    unique_models = {summary.get("model_revision") for summary in cell_summaries}

    # Sort: by model, then task, then regime, then budget.
    def _sort_key(summary):
        return (
            str(summary.get("model_revision", "")),
            str(summary.get("task_family", "")),
            str(summary.get("r_semantic_class", "")),
            str(summary.get("r_selection_policy", "")),
            str(summary.get("r_scope", "")),
            int(summary.get("r_edit_budget") or 0),
        )

    for card_index, summary in enumerate(sorted(cell_summaries, key=_sort_key)):
        cell_key = tuple(summary.get(dimension_key) for dimension_key in CELL_DIMENSION_KEYS)
        pairs_with_data = pairs_with_data_by_cell.get(cell_key, [])
        card_html = _cell_card_html(summary, pairs_with_data, card_index)
        keyboard_cards.append(card_html)

    page_html = _build_page(
        keyboard_sections=keyboard_cards,
        meta={
            "source_files": [str(path) for path in args.generations],
            "n_rows": len(rows),
            "n_pairs": len(pairs),
            "n_models": len(unique_models),
            "n_cells": len(cell_summaries),
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page_html, encoding="utf-8")
    size_kb = args.output.stat().st_size // 1024
    print(f"\ndone → {args.output}  ({size_kb} KB)")
    print("open in any browser — no server required")


if __name__ == "__main__":
    main()
