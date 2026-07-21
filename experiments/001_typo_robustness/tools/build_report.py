"""Build a self-contained, tabbed HTML dashboard from generation JSONL files.

One file, five tabs, zero external dependencies (no CDN — opens offline):

  Overview   — headline stat tiles, the Stage-1 gate readout per task family
  Results    — filterable per-cell table + a delta-vs-severity chart
  Mediation  — per-family indirect-effect forest plot + the Method A contrast
  Items      — filterable per-item drill-down with exact clean→perturbed diffs
  Run        — shard throughput, exclusions breakdown, provenance

All statistics are recomputed from the rows via the same analysis code the
paper uses (analysis.results / analysis.gates); the mediation JSON is embedded
from --analysis-directory when present (it needs statsmodels to recompute).

Usage:

    python tools/build_report.py \\
        --generations results/pilot/pilot_generations.jsonl \\
        --output results/pilot/report.html \\
        --config configs/pilot.yaml \\
        --analysis-directory analysis/pilot
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis import statistics
from analysis.gates import compute_stage_gates
from analysis.results import (
    CELL_DIMENSION_KEYS,
    join_matched_pairs,
    summarize_all_cells,
    summarize_fragmentation_contrast,
)
from pipeline.experiment import ExperimentConfiguration
from pipeline.runner import load_generation_rows

_PAYLOAD_MARKER = "__PAYLOAD_JSON__"
_GENERATED_AT_MARKER = "__GENERATED_AT__"
_RUN_LABEL_MARKER = "__RUN_LABEL__"

# Runner convention: generation files are named <run_id>_generations.jsonl, so
# stripping the suffix from the first --generations stem recovers the run id.
_GENERATIONS_FILE_STEM_SUFFIX = "_generations"

# Fields carried per perturbed item into the drill-down, as a positional array
# (object keys repeated 6,000+ times would double the payload). Order must
# match ITEM_FIELDS in the JavaScript below.
_ITEM_FIELDS = (
    "cell_index", "task_id", "clean_ok", "perturbed_ok", "parse_status",
    "extraction_tier", "prefix_length", "suffix_length", "replacement",
    "model_output", "parsed_answer", "token_inflation_ratio",
    "subword_count_change", "fragmentation_stratum", "edited_word",
    "finish_reason",
)

_MAXIMUM_EMBEDDED_OUTPUT_CHARACTERS = 4000


def _sanitize(value):
    """Make a structure JSON-serializable: NaN/inf → None, tuples → lists."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _sanitize(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(entry) for entry in value]
    return value


def _replacement_span(clean: str, perturbed: str) -> tuple[int, int, str]:
    """The minimal (prefix_length, suffix_length, replacement) triple such that
    perturbed == clean[:p] + replacement + clean[len(clean)-s:]. Lets the page
    reconstruct exact diffs client-side without shipping both full strings."""
    shorter = min(len(clean), len(perturbed))
    prefix_length = 0
    while prefix_length < shorter and clean[prefix_length] == perturbed[prefix_length]:
        prefix_length += 1
    suffix_length = 0
    while (suffix_length < shorter - prefix_length
           and clean[len(clean) - 1 - suffix_length]
           == perturbed[len(perturbed) - 1 - suffix_length]):
        suffix_length += 1
    return (prefix_length, suffix_length,
            perturbed[prefix_length:len(perturbed) - suffix_length])


def _clip_output(text: str) -> str:
    if len(text) <= _MAXIMUM_EMBEDDED_OUTPUT_CHARACTERS:
        return text
    return text[:_MAXIMUM_EMBEDDED_OUTPUT_CHARACTERS] + " …[clipped]"


def _cell_key_of(row: dict) -> tuple:
    return tuple(row.get(key) for key in CELL_DIMENSION_KEYS)


def _items_payload(rows: list[dict], cell_index_by_key: dict) -> tuple[dict, list]:
    """(clean_store, perturbed_items): the clean prompt/answer once per item,
    and one positional record per perturbed row (see _ITEM_FIELDS)."""
    clean_store: dict[str, list] = {}
    for row in rows:
        if row.get("is_clean"):
            clean_store[row["task_id"]] = [
                row.get("task_family", ""),
                row.get("prompt") or row.get("clean_prompt", ""),
                _clip_output(str(row.get("model_output", ""))),
                str(row.get("parsed_answer", "")),
                str(row.get("expected_answer", "")),
                int(row.get("is_correct", 0)),
            ]

    perturbed_items: list[list] = []
    for row in rows:
        if row.get("is_clean"):
            continue
        cell_index = cell_index_by_key.get(_cell_key_of(row))
        clean_entry = clean_store.get(row["task_id"])
        if cell_index is None or clean_entry is None:
            continue
        prefix_length, suffix_length, replacement = _replacement_span(
            clean_entry[1], row.get("prompt") or row.get("perturbed_prompt", ""))
        perturbed_items.append([
            cell_index,
            row["task_id"],
            clean_entry[5],
            int(row.get("is_correct", 0)),
            str(row.get("parse_status", "")),
            str(row.get("extraction_tier", "")),
            prefix_length,
            suffix_length,
            replacement,
            _clip_output(str(row.get("model_output", ""))),
            str(row.get("parsed_answer", "")),
            row.get("token_inflation_ratio"),
            row.get("subword_count_change"),
            str(row.get("fragmentation_stratum", "")),
            str(row.get("edited_word", "")),
            str(row.get("finish_reason", "")),
        ])
    return clean_store, perturbed_items


_QUOTED_SEGMENT = re.compile(r"'[^']*'")


def _exclusions_summary(generation_paths: list[Path]) -> list[dict]:
    """Aggregate sibling *_exclusions.jsonl files by condition x budget x
    normalized reason. Deduped on (task, condition, budget, reason): resumed
    runs re-append the sidecar, so raw line counts double on every no-op rerun.
    Reasons are normalized by collapsing quoted words ('Python' → '…') so one
    failure mode is one row, not one row per word."""
    unique_records: set[tuple] = set()
    for generations_path in generation_paths:
        for sidecar in Path(generations_path).parent.glob("*_exclusions.jsonl"):
            for line in sidecar.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                unique_records.add((
                    record.get("task_id", ""),
                    record.get("condition_name", ""),
                    record.get("edit_budget", 0),
                    _QUOTED_SEGMENT.sub("'…'", str(record.get("failure_reason", "")))[:120],
                ))
    counts: dict[tuple, int] = {}
    for _task_id, condition, budget, reason in unique_records:
        key = (condition, budget, reason)
        counts[key] = counts.get(key, 0) + 1
    return [{"condition": condition, "budget": budget, "reason": reason, "count": count}
            for (condition, budget, reason), count in
            sorted(counts.items(), key=lambda entry: -entry[1])]


def _manifest_statistics(generation_paths: list[Path]) -> dict:
    for generations_path in generation_paths:
        for manifest_path in Path(generations_path).parent.glob("*_manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("shard_statistics"):
                return manifest["shard_statistics"]
    return {}


def _analysis_json(analysis_directory: Path | None, file_name: str):
    if analysis_directory is None:
        return None
    path = Path(analysis_directory) / file_name
    return json.loads(path.read_text()) if path.exists() else None


def build_payload(rows: list[dict], generation_paths: list[Path],
                  configuration: ExperimentConfiguration,
                  analysis_directory: Path | None,
                  seed: int, resamples: int) -> dict:
    pairs = join_matched_pairs(rows)
    cell_summaries = summarize_all_cells(pairs, seed=seed, resamples=resamples)
    cell_index_by_key = {
        tuple(summary.get(key) for key in CELL_DIMENSION_KEYS): index
        for index, summary in enumerate(cell_summaries)}
    clean_store, perturbed_items = _items_payload(rows, cell_index_by_key)

    models = sorted({str(row.get("model_id", "")) for row in rows})
    commits = sorted({str(row.get("git_commit", "")) for row in rows})
    revisions = sorted({str(row.get("model_revision", "")) for row in rows})

    return _sanitize({
        "meta": {
            "sources": [str(path) for path in generation_paths],
            "row_count": len(rows),
            "pair_count": len(pairs),
            "models": models,
            "revisions": revisions,
            "commits": commits,
        },
        "gates": compute_stage_gates(
            rows,
            configuration.primary_edit_budget_reasoning,
            configuration.primary_edit_budget_mcq),
        "cells": cell_summaries,
        "cell_dimension_keys": list(CELL_DIMENSION_KEYS),
        "mediation": _analysis_json(analysis_directory, "mediation_proportion.json"),
        "mixed_model": _analysis_json(analysis_directory, "mixed_effects_logistic.json"),
        "method_a": summarize_fragmentation_contrast(rows, seed=seed, resamples=resamples),
        "clean_store": clean_store,
        "items": perturbed_items,
        "item_fields": list(_ITEM_FIELDS),
        "exclusions": _exclusions_summary(generation_paths),
        "shard_statistics": _manifest_statistics(generation_paths),
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="experiment config (supplies the primary edit budgets for gates)")
    parser.add_argument(
        "--analysis-directory", type=Path, default=None,
        help="run_analysis output directory; embeds mediation + mixed-model JSON")
    parser.add_argument("--bootstrap-resamples", type=int,
                        default=statistics.DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=1729)
    arguments = parser.parse_args()

    print(f"loading {len(arguments.generations)} generation file(s) ...")
    rows = load_generation_rows(arguments.generations)
    print(f"  {len(rows)} rows loaded")

    configuration = (ExperimentConfiguration.from_yaml(arguments.config)
                     if arguments.config else ExperimentConfiguration)

    print("computing statistics ...")
    payload = build_payload(
        rows, arguments.generations, configuration,
        arguments.analysis_directory, arguments.seed, arguments.bootstrap_resamples)

    print("building HTML ...")
    run_label = arguments.generations[0].stem.removesuffix(_GENERATIONS_FILE_STEM_SUFFIX)
    page = (_PAGE_TEMPLATE
            .replace(_RUN_LABEL_MARKER, run_label)
            .replace(_GENERATED_AT_MARKER, time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
            .replace(_PAYLOAD_MARKER, json.dumps(payload, separators=(",", ":"))
                     .replace("</", "<\\/")))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(page, encoding="utf-8")
    print(f"\ndone → {arguments.output}  ({arguments.output.stat().st_size // 1024} KB)")
    print("open in any browser — no server or network required")


# ---------------------------------------------------------------------------
# The page. Plain string (not an f-string) so CSS/JS braces need no escaping;
# data is injected via the two markers above. Palette: the validated reference
# instance from the dataviz method (categorical order fixed; light and dark
# columns are separately validated steps, not an automatic flip).
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 001 — Typo Robustness — __RUN_LABEL__ report</title>
<style>
:root {
  --surface: #fcfcfb; --plane: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100; --s4:#008300;
  --s5:#4a3aa7; --s6:#e34948; --s7:#e87ba4; --s8:#eb6834;
  --seq-250:#86b6ef; --seq-550:#1c5cab;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --good-text:#006300;
  --diff-del-bg:#ffd7d7; --diff-del-ink:#7d0000;
  --diff-ins-bg:#cdf2cd; --diff-ins-ink:#005a00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface:#1a1a19; --plane:#0d0d0d;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --border: rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#199e70; --s3:#c98500; --s4:#008300;
    --s5:#9085e9; --s6:#e66767; --s7:#d55181; --s8:#d95926;
    /* ordinal Low/High pair: keep the deep step so the two dots stay
       distinguishable on the dark surface (2:1 floor per the ordinal rule) */
    --seq-250:#86b6ef; --seq-550:#1c5cab;
    --good-text:#0ca30c;
    --diff-del-bg:#4e1f1f; --diff-del-ink:#ffb3b3;
    --diff-ins-bg:#1c3d1c; --diff-ins-ink:#a9e8a9;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
header { padding: 18px 24px 0; }
header h1 { font-size: 19px; margin: 0 0 2px; }
header .sub { color: var(--muted); font-size: 12px; }
nav.tabs {
  display: flex; gap: 2px; padding: 12px 24px 0; border-bottom: 1px solid var(--grid);
  position: sticky; top: 0; background: var(--plane); z-index: 20;
}
nav.tabs button {
  border: none; background: none; color: var(--ink-2); font: inherit;
  padding: 8px 14px; cursor: pointer; border-bottom: 2px solid transparent;
}
nav.tabs button.active { color: var(--ink); border-bottom-color: var(--s1); font-weight: 600; }
main { padding: 18px 24px 60px; max-width: 1400px; }
section.tab { display: none; } section.tab.active { display: block; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }
.tile {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; min-width: 150px; flex: 0 1 auto;
}
.tile .label { color: var(--muted); font-size: 12px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
.tile .status { font-size: 12px; font-weight: 600; margin-top: 2px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-bottom: 18px; overflow-x: auto;
}
.card h2 { font-size: 14px; margin: 0 0 4px; }
.card .hint { color: var(--muted); font-size: 12px; margin: 0 0 12px; }
table.data { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data th {
  text-align: left; color: var(--muted); font-weight: 500; font-size: 12px;
  border-bottom: 1px solid var(--baseline); padding: 5px 10px 5px 0; white-space: nowrap;
  cursor: pointer; user-select: none;
}
table.data th.nosort { cursor: default; }
table.data td {
  border-bottom: 1px solid var(--grid); padding: 5px 10px 5px 0; vertical-align: top;
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
table.data tr.clickable { cursor: pointer; }
table.data tr.clickable:hover td { background: color-mix(in srgb, var(--s1) 7%, transparent); }
.filters {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 14px;
}
.filters label { color: var(--muted); font-size: 12px; }
.filters select, .filters input[type=search] {
  font: inherit; font-size: 13px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--baseline); border-radius: 6px; padding: 4px 8px;
}
.filters .toggle { display: inline-flex; gap: 5px; align-items: center; font-size: 13px; color: var(--ink-2); }
.badge {
  display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 9px;
  border: 1px solid var(--border); color: var(--ink-2); background: var(--plane);
}
.badge.ok { color: var(--good-text); border-color: var(--good); }
.badge.bad { color: var(--critical); border-color: var(--critical); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 6px 0 2px; font-size: 12px; color: var(--ink-2); }
.legend .key { display: inline-flex; align-items: center; gap: 6px; }
.legend .swatch-line { width: 16px; height: 2px; border-radius: 1px; display: inline-block; }
.legend .swatch-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
#tooltip {
  position: fixed; pointer-events: none; z-index: 50; display: none;
  background: var(--surface); border: 1px solid var(--baseline); border-radius: 6px;
  padding: 7px 10px; font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.18); max-width: 340px;
}
#tooltip .t-title { color: var(--muted); margin-bottom: 3px; }
#tooltip .t-row { display: flex; align-items: center; gap: 6px; }
#tooltip .t-key { width: 12px; height: 2px; display: inline-block; border-radius: 1px; }
#tooltip .t-value { font-weight: 600; font-variant-numeric: tabular-nums; }
#tooltip .t-name { color: var(--ink-2); }
svg text { fill: var(--muted); font-size: 11px; font-family: inherit; }
svg .axis { stroke: var(--baseline); stroke-width: 1; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
svg .zero { stroke: var(--baseline); stroke-width: 1; }
svg text.dlabel { fill: var(--ink-2); font-weight: 600; }
.diff { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
        white-space: pre-wrap; word-break: break-word; color: var(--ink-2); }
.diff del { background: var(--diff-del-bg); color: var(--diff-del-ink); border-radius: 2px; text-decoration: line-through; }
.diff ins { background: var(--diff-ins-bg); color: var(--diff-ins-ink); border-radius: 2px; text-decoration: none; }
.item {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px;
}
.item .head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12px; }
.item .head .tid { font-weight: 600; color: var(--ink); }
.item details { margin-top: 6px; }
.item details summary { cursor: pointer; color: var(--muted); font-size: 12px; }
.item pre {
  background: var(--plane); border: 1px solid var(--grid); border-radius: 6px;
  padding: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-word;
  max-height: 320px; overflow-y: auto;
}
.mini-note { color: var(--muted); font-size: 12px; }
.count-note { color: var(--muted); font-size: 12px; margin: 8px 0; }
button.more {
  font: inherit; font-size: 13px; padding: 6px 14px; border-radius: 6px;
  border: 1px solid var(--baseline); background: var(--surface); color: var(--ink); cursor: pointer;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 1000px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div id="tooltip" role="status"></div>
<header>
  <h1>Experiment 001 — Typo Robustness — __RUN_LABEL__ report</h1>
  <div class="sub" id="header-sub"></div>
</header>
<nav class="tabs" id="tabs"></nav>
<main id="main"></main>

<script type="application/json" id="payload">__PAYLOAD_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("payload").textContent);
const GENERATED_AT = "__GENERATED_AT__";

/* ---------- shared constants ---------- */
const F = {};  // item field name -> positional index
DATA.item_fields.forEach((name, index) => F[name] = index);
const SERIES_VARS = ["--s1","--s2","--s3","--s4","--s5","--s6","--s7","--s8"];
const OUTCOMES = {
  broke:      { label: "broke (clean ✓ → pert ✗)", test: it => it[F.clean_ok] === 1 && it[F.perturbed_ok] === 0 },
  recovered:  { label: "recovered (clean ✗ → pert ✓)", test: it => it[F.clean_ok] === 0 && it[F.perturbed_ok] === 1 },
  both_ok:    { label: "both correct", test: it => it[F.clean_ok] === 1 && it[F.perturbed_ok] === 1 },
  both_wrong: { label: "both wrong", test: it => it[F.clean_ok] === 0 && it[F.perturbed_ok] === 0 },
};
const ITEMS_PAGE_SIZE = 60;

/* ---------- tiny DOM + format helpers ---------- */
function el(tag, attrs, ...children) {
  const node = tag.startsWith("svg:")
    ? document.createElementNS("http://www.w3.org/2000/svg", tag.slice(4))
    : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}
const pct = value => value == null ? "—" : (value * 100).toFixed(1) + "%";
const pp  = value => value == null ? "—" : (value >= 0 ? "+" : "") + (value * 100).toFixed(1) + " pp";
const num = (value, digits) => value == null ? "—" : Number(value).toFixed(digits == null ? 3 : digits);
const pValue = value => value == null ? "—" : value < 0.001 ? "< 0.001" : Number(value).toFixed(3);
const stars = value => value == null ? "" : value < 0.001 ? " ***" : value < 0.01 ? " **" : value < 0.05 ? " *" : "";
function seriesColor(index) { return `var(${SERIES_VARS[index % SERIES_VARS.length]})`; }

function conditionLabel(cell) {
  const parts = [cell.r_selection_policy, cell.r_operation];
  if (cell.r_scope && cell.r_scope !== "anywhere" && cell.r_scope !== "none") parts.push(cell.r_scope);
  return parts.join(" · ");
}
/* Fixed slot per condition entity — assigned once from the full cell list so
   filtering never repaints survivors. */
const CONDITION_SLOTS = (() => {
  const slots = new Map();
  for (const cell of DATA.cells) {
    const label = conditionLabel(cell);
    if (!slots.has(label)) slots.set(label, slots.size);
  }
  return slots;
})();

/* ---------- tooltip ---------- */
const tooltip = document.getElementById("tooltip");
function showTooltip(clientX, clientY, titleText, rows) {
  tooltip.replaceChildren(el("div", { class: "t-title", text: titleText }));
  for (const row of rows) {
    tooltip.append(el("div", { class: "t-row" },
      el("span", { class: "t-key", style: `background:${row.color || "transparent"}` }),
      el("span", { class: "t-value", text: row.value }),
      el("span", { class: "t-name", text: row.name })));
  }
  tooltip.style.display = "block";
  const pad = 14, box = tooltip.getBoundingClientRect();
  tooltip.style.left = Math.min(clientX + pad, innerWidth - box.width - 8) + "px";
  tooltip.style.top  = Math.min(clientY + pad, innerHeight - box.height - 8) + "px";
}
function hideTooltip() { tooltip.style.display = "none"; }

/* ---------- select helper ---------- */
function makeSelect(labelText, options, onchange, allLabel) {
  const select = el("select", { onchange: event => onchange(event.target.value) },
    el("option", { value: "", text: allLabel || ("all " + labelText) }));
  for (const option of options) select.append(el("option", { value: option, text: option }));
  return el("span", {}, el("label", { text: labelText + " " }), select);
}
const uniqueSorted = values => [...new Set(values)].filter(v => v !== "" && v != null).sort();

/* =====================================================================
   TAB: Overview
   ===================================================================== */
function tile(label, value, note, statusText, statusColor) {
  const box = el("div", { class: "tile" },
    el("div", { class: "label", text: label }),
    el("div", { class: "value", text: value }));
  if (note) box.append(el("div", { class: "note", text: note }));
  if (statusText) box.append(el("div", { class: "status", style: `color:${statusColor}`, text: statusText }));
  return box;
}

function accuracyBar(value) {
  const width = 120, height = 14, filled = Math.round(width * (value || 0));
  const svg = el("svg:svg", { width, height, role: "img",
                              "aria-label": `clean accuracy ${pct(value)}` });
  svg.append(el("svg:rect", { x: 0, y: 3, width, height: 8, rx: 4, fill: "var(--grid)" }));
  if (filled > 0)
    svg.append(el("svg:rect", { x: 0, y: 3, width: Math.max(filled, 4), height: 8, rx: 4, fill: "var(--s1)" }));
  return svg;
}

function renderOverview(container) {
  const gates = DATA.gates, meta = DATA.meta;
  const compliance = gates.reasoning_format_compliance;
  const complianceTarget = gates.reasoning_format_compliance_target;
  const compliancePasses = compliance != null && compliance >= complianceTarget;
  const shardStats = Object.values(DATA.shard_statistics || {});
  const totalRowsPerHour = shardStats.length
    ? Math.round(shardStats.reduce((sum, s) => sum + s.rows, 0)
                 / shardStats.reduce((sum, s) => sum + s.wall_seconds, 0) * 3600)
    : null;

  container.append(el("div", { class: "tiles" },
    tile("generation rows", meta.row_count.toLocaleString(), meta.pair_count.toLocaleString() + " matched pairs"),
    tile("model", (meta.models[0] || "—").split("/").pop(),
         "rev " + (meta.revisions[0] || "").slice(0, 9) + " · code " + (meta.commits[0] || "").slice(0, 7)),
    totalRowsPerHour != null
      ? tile("throughput", totalRowsPerHour.toLocaleString() + " rows/h", "measured, whole run")
      : null,
    tile("format compliance", compliance == null ? "—" : pct(compliance),
         "target ≥ " + pct(complianceTarget),
         compliancePasses ? "✓ gate passed" : "✗ gate FAILED",
         compliancePasses ? "var(--good-text)" : "var(--critical)"),
    tile("truncation", gates.truncation_rate == null ? "—" : pct(gates.truncation_rate),
         "finish_reason = length"),
    tile("p99 clean-correct length", gates.p99_clean_correct_output_tokens == null
         ? "—" : gates.p99_clean_correct_output_tokens + " tok",
         "max_new_tokens freeze input")));

  const gatesTable = el("table", { class: "data" },
    el("thead", {}, el("tr", {},
      ...["task family", "clean accuracy A₀", "", "primary k", "pairs", "p_d", "δ at primary k",
          "implied N (5pp MDE)", "design/06 §6.3 bucket"]
        .map(head => el("th", { class: "nosort", text: head })))),
    el("tbody", {}, ...Object.entries(gates.per_task_family).map(([family, block]) => {
      const bucket = block.discordant_rate_bucket || "no primary rows";
      const bucketOk = bucket === "n600_confirmed";
      return el("tr", {},
        el("td", { text: family }),
        el("td", { text: pct(block.clean_accuracy) }),
        el("td", {}, accuracyBar(block.clean_accuracy)),
        el("td", { text: "k=" + block.primary_edit_budget }),
        el("td", { text: block.primary_condition_pairs }),
        el("td", { text: block.discordant_rate == null ? "—" : num(block.discordant_rate, 2) }),
        el("td", { text: block.delta == null ? "—" : pp(block.delta) }),
        el("td", { text: block.implied_n_at_5pp_mde == null ? "—" : String(block.implied_n_at_5pp_mde) }),
        el("td", {}, el("span", { class: "badge " + (bucketOk ? "ok" : "bad"), text: bucket })));
    })));
  container.append(el("div", { class: "card" },
    el("h2", { text: "Stage-1 gates per task family" }),
    el("p", { class: "hint", text: "p_d = discordant-pair rate at the pre-registered primary condition "
      + "(Regime A keyboard substitution, anywhere). Implied N via Connor (1987) at 5 pp MDE / 80% power." }),
    gatesTable));

  if (DATA.mixed_model && DATA.mixed_model.fixed_effects) {
    container.append(el("div", { class: "card" },
      el("h2", { text: "Mixed-effects logistic (design/06 §6.6)" }),
      el("p", { class: "hint", text: `converged: ${DATA.mixed_model.converged} · method: ${DATA.mixed_model.method}`
        + ` · n=${DATA.mixed_model.n_observations}` }),
      el("table", { class: "data" },
        el("thead", {}, el("tr", {}, ...["term", "coef", "odds ratio", "p"].map(
          head => el("th", { class: "nosort", text: head })))),
        el("tbody", {}, ...Object.entries(DATA.mixed_model.fixed_effects).map(([term, effect]) =>
          el("tr", {},
            el("td", { text: term }),
            el("td", { text: num(effect.coef) }),
            el("td", { text: num(effect.or) }),
            el("td", { text: pValue(effect.p) + stars(effect.p) })))))));
  }
}

/* =====================================================================
   TAB: Results — filterable cells + delta-vs-severity chart
   ===================================================================== */
const resultsState = {
  // Default to the first family so the severity chart opens as clean lines
  // (one point per k per condition) rather than an all-families dot cloud.
  family: uniqueSorted(DATA.cells.map(cell => cell.task_family))[0] || "",
  regime: "", condition: "", budget: "", significantOnly: false,
  sortKey: "delta", sortDescending: true,
};

function filteredCells() {
  return DATA.cells.filter(cell =>
    (!resultsState.family || cell.task_family === resultsState.family)
    && (!resultsState.regime || String(cell.r_semantic_class) === resultsState.regime)
    && (!resultsState.condition || conditionLabel(cell) === resultsState.condition)
    && (!resultsState.budget || String(cell.r_edit_budget) === resultsState.budget)
    && (!resultsState.significantOnly
        || (cell.mcnemar_p_value != null && cell.mcnemar_p_value < 0.05)));
}

function deltaChart(cells) {
  /* Δ (pp) vs edit budget k; one line per condition entity; crosshair tooltip. */
  const bySeries = new Map();
  for (const cell of cells) {
    const label = conditionLabel(cell);
    if (!bySeries.has(label)) bySeries.set(label, []);
    bySeries.get(label).push({ k: Number(cell.r_edit_budget), delta: cell.delta * 100 });
  }
  for (const points of bySeries.values()) points.sort((a, b) => a.k - b.k);

  const width = 780, height = 300, margin = { top: 14, right: 24, bottom: 34, left: 46 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const budgets = uniqueSorted(cells.map(cell => Number(cell.r_edit_budget))).map(Number).sort((a, b) => a - b);
  const deltas = cells.map(cell => cell.delta * 100);
  if (!budgets.length) return el("p", { class: "mini-note", text: "No cells match the current filters." });
  let yMin = Math.min(0, ...deltas), yMax = Math.max(0, ...deltas);
  const ySpan = (yMax - yMin) || 1; yMin -= ySpan * .08; yMax += ySpan * .08;
  const xOf = k => budgets.length === 1 ? plotWidth / 2
    : (budgets.indexOf(k) / (budgets.length - 1)) * plotWidth;
  const yOf = value => plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;

  const svg = el("svg:svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": "paired degradation versus edit budget" });
  const plot = el("svg:g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(plot);

  const tickStep = Math.max(1, Math.ceil((yMax - yMin) / 6));
  for (let tick = Math.ceil(yMin / tickStep) * tickStep; tick <= yMax; tick += tickStep) {
    plot.append(el("svg:line", { class: tick === 0 ? "zero" : "gridline",
                                 x1: 0, x2: plotWidth, y1: yOf(tick), y2: yOf(tick) }));
    plot.append(el("svg:text", { x: -8, y: yOf(tick) + 4, "text-anchor": "end", text: tick + "" }));
  }
  plot.append(el("svg:text", { x: -34, y: -4, text: "Δ pp" }));
  for (const k of budgets) {
    plot.append(el("svg:text", { x: xOf(k), y: plotHeight + 18, "text-anchor": "middle", text: "k=" + k }));
  }
  plot.append(el("svg:line", { class: "axis", x1: 0, x2: plotWidth, y1: plotHeight, y2: plotHeight }));

  for (const [label, points] of bySeries) {
    const color = seriesColor(CONDITION_SLOTS.get(label) || 0);
    // A line only makes sense when the series has one point per k — with
    // several task families in view the same condition has several points per
    // budget, so draw markers only rather than a misleading zigzag.
    const oneValuePerBudget = new Set(points.map(point => point.k)).size === points.length;
    if (points.length > 1 && oneValuePerBudget) {
      const path = points.map((point, index) =>
        (index ? "L" : "M") + xOf(point.k) + " " + yOf(point.delta)).join(" ");
      plot.append(el("svg:path", { d: path, fill: "none", stroke: color,
                                   "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    }
    for (const point of points) {
      plot.append(el("svg:circle", { cx: xOf(point.k), cy: yOf(point.delta), r: 6,
                                     fill: "var(--surface)" }));
      plot.append(el("svg:circle", { cx: xOf(point.k), cy: yOf(point.delta), r: 4, fill: color }));
    }
  }

  const crosshair = el("svg:line", { class: "gridline", y1: 0, y2: plotHeight, visibility: "hidden" });
  plot.append(crosshair);
  const hitLayer = el("svg:rect", {
    x: 0, y: 0, width: plotWidth, height: plotHeight, fill: "transparent",
    onpointermove: event => {
      const box = svg.getBoundingClientRect();
      const scale = width / box.width;
      const pointerX = (event.clientX - box.left) * scale - margin.left;
      let nearest = budgets[0];
      for (const k of budgets) if (Math.abs(xOf(k) - pointerX) < Math.abs(xOf(nearest) - pointerX)) nearest = k;
      crosshair.setAttribute("x1", xOf(nearest));
      crosshair.setAttribute("x2", xOf(nearest));
      crosshair.setAttribute("visibility", "visible");
      const rows = [...bySeries.entries()]
        .map(([label, points]) => ({ label, point: points.find(point => point.k === nearest) }))
        .filter(entry => entry.point)
        .sort((a, b) => b.point.delta - a.point.delta)
        .map(entry => ({ color: seriesColor(CONDITION_SLOTS.get(entry.label) || 0),
                         value: entry.point.delta.toFixed(1) + " pp", name: entry.label }));
      showTooltip(event.clientX, event.clientY, "edit budget k=" + nearest, rows);
    },
    onpointerleave: () => { crosshair.setAttribute("visibility", "hidden"); hideTooltip(); },
  });
  plot.append(hitLayer);

  const legend = el("div", { class: "legend" }, ...[...bySeries.keys()].map(label =>
    el("span", { class: "key" },
      el("span", { class: "swatch-line",
                   style: `background:${seriesColor(CONDITION_SLOTS.get(label) || 0)}` }),
      el("span", { text: label }))));
  return el("div", {}, legend, svg);
}

function deltaCell(cell) {
  /* small diverging bar: damage right (red), improvement left (blue) */
  const width = 110, height = 12, center = width / 2, scale = center / 0.35;
  const magnitude = Math.min(Math.abs(cell.delta || 0), 0.35) * scale;
  const svg = el("svg:svg", { width, height });
  svg.append(el("svg:line", { class: "zero", x1: center, x2: center, y1: 0, y2: height }));
  if (magnitude > 0.5) {
    const damage = (cell.delta || 0) > 0;
    svg.append(el("svg:rect", {
      x: damage ? center : center - magnitude, y: 2,
      width: Math.max(magnitude, 3), height: 8, rx: 3,
      fill: damage ? "var(--s6)" : "var(--s1)" }));
  }
  return svg;
}

function renderResults(container) {
  container.replaceChildren();
  const filters = el("div", { class: "filters" },
    makeSelect("family", uniqueSorted(DATA.cells.map(cell => cell.task_family)),
               value => { resultsState.family = value; renderResults(container); }),
    makeSelect("regime", uniqueSorted(DATA.cells.map(cell => String(cell.r_semantic_class))),
               value => { resultsState.regime = value; renderResults(container); }),
    makeSelect("condition", [...CONDITION_SLOTS.keys()],
               value => { resultsState.condition = value; renderResults(container); }),
    makeSelect("k", uniqueSorted(DATA.cells.map(cell => String(cell.r_edit_budget))),
               value => { resultsState.budget = value; renderResults(container); }),
    el("label", { class: "toggle" },
      el("input", { type: "checkbox",
                    onchange: event => { resultsState.significantOnly = event.target.checked;
                                         renderResults(container); } }),
      "significant only (p < .05)"));
  [...filters.querySelectorAll("select")].forEach((select, index) => {
    select.value = [resultsState.family, resultsState.regime,
                    resultsState.condition, resultsState.budget][index];
  });
  filters.querySelector("input[type=checkbox]").checked = resultsState.significantOnly;
  container.append(filters);

  const cells = filteredCells();

  container.append(el("div", { class: "card" },
    el("h2", { text: "Paired degradation Δ vs severity" }),
    el("p", { class: "hint", text: "Δ = clean accuracy − perturbed accuracy, in percentage points. "
      + "Above zero = the perturbation hurt. Filter to one family to compare conditions." }),
    deltaChart(cells)));

  const columns = [
    ["task_family", "family"], ["r_semantic_class", "regime"], ["__condition", "condition"],
    ["r_edit_budget", "k"], ["n", "n"], ["delta", "Δ"], ["__deltabar", ""],
    ["__ci", "95% CI"], ["clean_conditioned_failure", "CCF"],
    ["discordant_rate", "p_d"], ["mcnemar_p_value", "McNemar p"], ["broke", "broke"],
    ["recovered", "recov."],
  ];
  const sortValue = (cell, key) =>
    key === "__condition" ? conditionLabel(cell)
    : key === "__deltabar" || key === "__ci" ? cell.delta
    : cell[key];
  const sorted = [...cells].sort((a, b) => {
    const left = sortValue(a, resultsState.sortKey), right = sortValue(b, resultsState.sortKey);
    const comparison = (left == null) - (right == null)
      || (typeof left === "string" ? left.localeCompare(right) : left - right);
    return resultsState.sortDescending ? -comparison : comparison;
  });

  const head = el("tr", {}, ...columns.map(([key, label]) =>
    el("th", {
      text: label + (resultsState.sortKey === key ? (resultsState.sortDescending ? " ↓" : " ↑") : ""),
      onclick: () => {
        resultsState.sortDescending = resultsState.sortKey === key ? !resultsState.sortDescending : true;
        resultsState.sortKey = key;
        renderResults(container);
      } })));
  const body = el("tbody", {}, ...sorted.map(cell => el("tr", {
      class: "clickable",
      title: "click to inspect these items",
      onclick: () => {
        itemsState.family = cell.task_family;
        itemsState.condition = conditionLabel(cell);
        itemsState.budget = String(cell.r_edit_budget);
        itemsState.outcome = ""; itemsState.search = ""; itemsState.visible = ITEMS_PAGE_SIZE;
        activateTab("items");
      } },
    el("td", { text: cell.task_family }),
    el("td", { text: String(cell.r_semantic_class) }),
    el("td", { text: conditionLabel(cell) }),
    el("td", { text: String(cell.r_edit_budget) }),
    el("td", { text: String(cell.n) }),
    el("td", { text: pp(cell.delta) }),
    el("td", {}, deltaCell(cell)),
    el("td", { text: cell.delta_ci_method === "insufficient_n" ? "n too small"
        : `[${pp(cell.delta_ci_low)}, ${pp(cell.delta_ci_high)}]` }),
    el("td", { text: pct(cell.clean_conditioned_failure) }),
    el("td", { text: num(cell.discordant_rate, 2) }),
    el("td", { text: pValue(cell.mcnemar_p_value) + stars(cell.mcnemar_p_value) }),
    el("td", { text: String(cell.broke) }),
    el("td", { text: String(cell.recovered) }))));

  container.append(el("div", { class: "card" },
    el("h2", { text: `Condition cells (${cells.length})` }),
    el("p", { class: "hint", text: "Click a column to sort; click a row to open its items in the drill-down. "
      + "* p<.05, ** p<.01, *** p<.001 (McNemar mid-p exact)." }),
    el("table", { class: "data" }, el("thead", {}, head), body)));
}

/* =====================================================================
   TAB: Mediation — forest plot + Method A dumbbells
   ===================================================================== */
function forestPlot(entries) {
  /* rows: {name, value, low, high} — indirect effect with bootstrap CI */
  const width = 720, rowHeight = 34, margin = { top: 8, right: 178, bottom: 30, left: 210 };
  const height = margin.top + entries.length * rowHeight + margin.bottom;
  const plotWidth = width - margin.left - margin.right;
  const magnitudes = entries.flatMap(entry => [entry.low, entry.high, entry.value])
    .filter(value => value != null).map(Math.abs);
  const bound = Math.max(0.01, ...magnitudes) * 1.15;
  const xOf = value => ((value + bound) / (2 * bound)) * plotWidth;

  const svg = el("svg:svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": "indirect effect forest plot" });
  const plot = el("svg:g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(plot);
  plot.append(el("svg:line", { class: "zero", x1: xOf(0), x2: xOf(0), y1: 0,
                               y2: entries.length * rowHeight }));
  plot.append(el("svg:text", { x: xOf(0), y: entries.length * rowHeight + 16,
                               "text-anchor": "middle", text: "0" }));
  plot.append(el("svg:text", { x: 0, y: entries.length * rowHeight + 16, text: "← fragmentation hurts" }));
  plot.append(el("svg:text", { x: plotWidth, y: entries.length * rowHeight + 16,
                               "text-anchor": "end", text: "helps →" }));

  entries.forEach((entry, index) => {
    const y = index * rowHeight + rowHeight / 2;
    plot.append(el("svg:text", { x: -10, y: y + 4, "text-anchor": "end", text: entry.name }));
    if (entry.low != null && entry.high != null) {
      plot.append(el("svg:line", { x1: xOf(entry.low), x2: xOf(entry.high), y1: y, y2: y,
                                   stroke: "var(--baseline)", "stroke-width": 2 }));
    }
    const excludesZero = entry.low != null && entry.high != null
      && (entry.low > 0 || entry.high < 0);
    plot.append(el("svg:circle", { cx: xOf(entry.value), cy: y, r: 6, fill: "var(--surface)" }));
    plot.append(el("svg:circle", { cx: xOf(entry.value), cy: y, r: 4.5, fill: "var(--s1)" }));
    plot.append(el("svg:text", {
      class: "dlabel", x: plotWidth + 10, y: y + 4,
      text: num(entry.value) + (excludesZero ? "  (CI excludes 0)" : "") }));
    const hit = el("svg:rect", { x: -margin.left, y: index * rowHeight, width, height: rowHeight,
      fill: "transparent",
      onpointermove: event => showTooltip(event.clientX, event.clientY, entry.name, [
        { color: "var(--s1)", value: num(entry.value), name: "indirect effect (α·β)" },
        { value: `[${num(entry.low)}, ${num(entry.high)}]`, name: "bootstrap 95% CI" },
      ]),
      onpointerleave: hideTooltip });
    plot.append(hit);
  });
  return svg;
}

function methodADumbbells(groups) {
  const width = 720, rowHeight = 34, margin = { top: 8, right: 150, bottom: 30, left: 210 };
  const height = margin.top + groups.length * rowHeight + margin.bottom;
  const plotWidth = width - margin.left - margin.right;
  const xOf = value => value * plotWidth;
  const svg = el("svg:svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": "Method A paired accuracy, low versus high fragmentation" });
  const plot = el("svg:g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(plot);
  for (const tick of [0, 0.5, 1]) {
    plot.append(el("svg:line", { class: "gridline", x1: xOf(tick), x2: xOf(tick), y1: 0,
                                 y2: groups.length * rowHeight }));
    plot.append(el("svg:text", { x: xOf(tick), y: groups.length * rowHeight + 16,
                                 "text-anchor": "middle", text: pct(tick) }));
  }
  groups.forEach((group, index) => {
    const y = index * rowHeight + rowHeight / 2;
    const lowAccuracy = group.clean_accuracy, highAccuracy = group.perturbed_accuracy;
    plot.append(el("svg:text", { x: -10, y: y + 4, "text-anchor": "end",
                                 text: `${group.task_family} k=${group.r_edit_budget}` }));
    plot.append(el("svg:line", { x1: xOf(lowAccuracy), x2: xOf(highAccuracy), y1: y, y2: y,
                                 stroke: "var(--baseline)", "stroke-width": 2 }));
    for (const [accuracy, fill] of [[lowAccuracy, "var(--seq-250)"], [highAccuracy, "var(--seq-550)"]]) {
      plot.append(el("svg:circle", { cx: xOf(accuracy), cy: y, r: 6.5, fill: "var(--surface)" }));
      plot.append(el("svg:circle", { cx: xOf(accuracy), cy: y, r: 5, fill }));
    }
    plot.append(el("svg:text", { class: "dlabel", x: plotWidth + 10, y: y + 4,
                                 text: `n=${group.n}, Δ=${pp(group.delta)}` }));
    plot.append(el("svg:rect", { x: -margin.left, y: index * rowHeight, width, height: rowHeight,
      fill: "transparent",
      onpointermove: event => showTooltip(event.clientX, event.clientY,
        `${group.task_family} · k=${group.r_edit_budget} · ${group.n} pairs`, [
          { color: "var(--seq-250)", value: pct(lowAccuracy), name: "Low-fragmentation accuracy" },
          { color: "var(--seq-550)", value: pct(highAccuracy), name: "High-fragmentation accuracy" },
          { value: `${group.broke} / ${group.recovered}`, name: "broke / recovered" },
          { value: pValue(group.mcnemar_p_value), name: "McNemar p" },
        ]),
      onpointerleave: hideTooltip }));
  });
  return el("div", {},
    el("div", { class: "legend" },
      el("span", { class: "key" }, el("span", { class: "swatch-dot", style: "background:var(--seq-250)" }),
        el("span", { text: "Low fragmentation (same word, same k)" })),
      el("span", { class: "key" }, el("span", { class: "swatch-dot", style: "background:var(--seq-550)" }),
        el("span", { text: "High fragmentation" }))),
    svg);
}

/* Fit keys are namespaced ("task_family:gsm8k", "h1b_policy:filler_word");
   render them with a readable prefix instead of the raw namespace. */
const FIT_KEY_PREFIX_LABELS = { "task_family:": "", "h1b_policy:": "H1b policy · " };
function mediationFitLabel(name) {
  for (const [prefix, label] of Object.entries(FIT_KEY_PREFIX_LABELS)) {
    if (name.startsWith(prefix)) return label + name.slice(prefix.length);
  }
  return name;
}

function renderMediation(container) {
  const mediation = DATA.mediation;
  if (mediation) {
    const entries = Object.entries(mediation)
      .filter(([, block]) => block.indirect_effect != null)
      .map(([name, block]) => ({
        name: mediationFitLabel(name),
        value: block.indirect_effect,
        low: (block.bootstrap_ci_indirect || [])[0],
        high: (block.bootstrap_ci_indirect || [])[1],
      }));
    const estimators = [...new Set(Object.values(mediation)
      .map(block => block.estimator).filter(Boolean))];
    container.append(el("div", { class: "card" },
      el("h2", { text: "Statistical mediation (Method B) — indirect effect per task family" }),
      el("p", { class: "hint", text: "Product-of-coefficients (α·β) with by-item cluster-bootstrap 95% CI. "
        + "Negative = accuracy loss mediated by subword fragmentation. The pooled row is supplementary." }),
      forestPlot(entries),
      el("table", { class: "data", style: "margin-top:10px" },
        el("thead", {}, el("tr", {}, ...["fit", "α (treat→mediator)", "β (mediator→correct)",
            "indirect", "total", "proportion mediated", "n"].map(
          head => el("th", { class: "nosort", text: head })))),
        el("tbody", {}, ...Object.entries(mediation).map(([name, block]) => el("tr", {},
          el("td", { text: mediationFitLabel(name) }),
          el("td", { text: num(block.treatment_on_mediator_coef) }),
          el("td", { text: num(block.mediator_on_outcome_coef) }),
          el("td", { text: num(block.indirect_effect) }),
          el("td", { text: num(block.total_effect) }),
          el("td", { text: block.proportion_mediated != null ? num(block.proportion_mediated)
              : "withheld (" + (block.proportion_mediated_reason || "—") + ")" }),
          el("td", { text: String(block.n_observations || "—") }))))),
      estimators.length
        ? el("p", { class: "mini-note", style: "margin-top:8px",
                    text: "estimator: " + estimators.join(" · ") })
        : null));
  } else {
    container.append(el("div", { class: "card" },
      el("h2", { text: "Method B mediation" }),
      el("p", { class: "hint", text: "Not embedded — rerun with --analysis-directory pointing at run_analysis output." })));
  }

  if (DATA.method_a && DATA.method_a.length) {
    container.append(el("div", { class: "card" },
      el("h2", { text: "Fragmentation-matched counterfactual (Method A)" }),
      el("p", { class: "hint", text: "Same word, same edit count, same position; only the tokenization "
        + "consequence differs. Restricted to items whose clean answer was correct. Left dot = accuracy on "
        + "the Low-fragmentation variant, right/darker = High. A leftward High dot means fragmentation hurt." }),
      methodADumbbells(DATA.method_a)));
  }
}

/* =====================================================================
   TAB: Items — drill-down
   ===================================================================== */
const itemsState = { family: "", condition: "", budget: "", outcome: "", stratum: "",
                     search: "", visible: ITEMS_PAGE_SIZE };

function itemMatches(item) {
  const cell = DATA.cells[item[F.cell_index]];
  const search = itemsState.search.toLowerCase();
  return (!itemsState.family || cell.task_family === itemsState.family)
    && (!itemsState.condition || conditionLabel(cell) === itemsState.condition)
    && (!itemsState.budget || String(cell.r_edit_budget) === itemsState.budget)
    && (!itemsState.outcome || OUTCOMES[itemsState.outcome].test(item))
    && (!itemsState.stratum || item[F.fragmentation_stratum] === itemsState.stratum)
    && (!search
        || item[F.task_id].toLowerCase().includes(search)
        || (DATA.clean_store[item[F.task_id]] || ["", ""])[1].toLowerCase().includes(search));
}

const FINE_DIFF_MAXIMUM_WORDS = 300;

function wordDiff(cleanText, perturbedText) {
  /* Word-level LCS so a multi-edit span renders as several small del/ins
     pairs instead of one struck-out block. Returns [kind, text] tokens,
     kind ∈ {"=", "-", "+"}; falls back to one coarse pair on huge spans. */
  const cleanWords = cleanText.split(/(\s+)/), perturbedWords = perturbedText.split(/(\s+)/);
  if (cleanWords.length > FINE_DIFF_MAXIMUM_WORDS || perturbedWords.length > FINE_DIFF_MAXIMUM_WORDS) {
    return [["-", cleanText], ["+", perturbedText]];
  }
  const rows = cleanWords.length + 1, columns = perturbedWords.length + 1;
  const longest = Array.from({ length: rows }, () => new Int32Array(columns));
  for (let i = cleanWords.length - 1; i >= 0; i--) {
    for (let j = perturbedWords.length - 1; j >= 0; j--) {
      longest[i][j] = cleanWords[i] === perturbedWords[j]
        ? longest[i + 1][j + 1] + 1
        : Math.max(longest[i + 1][j], longest[i][j + 1]);
    }
  }
  const tokens = [];
  const push = (kind, text) => {
    if (!text) return;
    const last = tokens[tokens.length - 1];
    if (last && last[0] === kind) last[1] += text; else tokens.push([kind, text]);
  };
  let i = 0, j = 0;
  while (i < cleanWords.length && j < perturbedWords.length) {
    if (cleanWords[i] === perturbedWords[j]) { push("=", cleanWords[i]); i++; j++; }
    else if (longest[i + 1][j] >= longest[i][j + 1]) { push("-", cleanWords[i]); i++; }
    else { push("+", perturbedWords[j]); j++; }
  }
  while (i < cleanWords.length) push("-", cleanWords[i++]);
  while (j < perturbedWords.length) push("+", perturbedWords[j++]);
  return tokens;
}

function diffFragment(cleanPrompt, item) {
  const prefixLength = item[F.prefix_length], suffixLength = item[F.suffix_length];
  const cleanMiddle = cleanPrompt.slice(prefixLength, cleanPrompt.length - suffixLength);
  const contextBefore = cleanPrompt.slice(Math.max(0, prefixLength - 90), prefixLength);
  const contextAfter = cleanPrompt.slice(cleanPrompt.length - suffixLength,
                                         cleanPrompt.length - suffixLength + 90);
  const fragment = el("div", { class: "diff" });
  if (prefixLength > 90) fragment.append("…");
  fragment.append(contextBefore);
  for (const [kind, text] of wordDiff(cleanMiddle, item[F.replacement])) {
    if (kind === "=") fragment.append(text);
    else fragment.append(el(kind === "-" ? "del" : "ins", { text }));
  }
  fragment.append(contextAfter);
  if (suffixLength > 90) fragment.append("…");
  return fragment;
}

function itemCard(item) {
  const cell = DATA.cells[item[F.cell_index]];
  const clean = DATA.clean_store[item[F.task_id]] || ["", "", "", "", "", 0];
  const outcomeBadge = (ok, label) =>
    el("span", { class: "badge " + (ok ? "ok" : "bad"), text: label + (ok ? " ✓" : " ✗") });
  const card = el("div", { class: "item" },
    el("div", { class: "head" },
      el("span", { class: "tid", text: item[F.task_id] }),
      el("span", { class: "badge", text: conditionLabel(cell) + " · k=" + cell.r_edit_budget }),
      outcomeBadge(item[F.clean_ok], "clean"),
      outcomeBadge(item[F.perturbed_ok], "perturbed"),
      item[F.parse_status] && item[F.parse_status] !== "valid"
        ? el("span", { class: "badge bad", text: item[F.parse_status] }) : null,
      el("span", { class: "badge", text: item[F.extraction_tier] }),
      item[F.fragmentation_stratum]
        ? el("span", { class: "badge", text: item[F.fragmentation_stratum] + " frag" }) : null,
      item[F.token_inflation_ratio] != null
        ? el("span", { class: "mini-note", text: "τ " + num(item[F.token_inflation_ratio], 3) }) : null,
      item[F.edited_word]
        ? el("span", { class: "mini-note", text: "word: " + item[F.edited_word] }) : null),
    diffFragment(clean[1], item),
    el("div", { class: "mini-note", style: "margin-top:4px",
                text: `parsed: ${item[F.parsed_answer] || "—"} · expected: ${clean[4] || "—"}`
                  + ` · clean parsed: ${clean[3] || "—"}` }));
  const details = el("details", {},
    el("summary", { text: "model outputs (perturbed / clean)" }));
  details.addEventListener("toggle", () => {
    if (details.open && details.childElementCount === 1) {
      details.append(el("pre", { text: item[F.model_output] || "(empty)" }),
                     el("pre", { text: clean[2] || "(empty)" }));
    }
  });
  card.append(details);
  return card;
}

function renderItems(container) {
  container.replaceChildren();
  const rerender = () => renderItems(container);
  const filters = el("div", { class: "filters" },
    makeSelect("family",
      uniqueSorted(DATA.cells.map(cell => cell.task_family)),
      value => { itemsState.family = value; itemsState.visible = ITEMS_PAGE_SIZE; rerender(); }),
    makeSelect("condition", [...CONDITION_SLOTS.keys()],
      value => { itemsState.condition = value; itemsState.visible = ITEMS_PAGE_SIZE; rerender(); }),
    makeSelect("k", uniqueSorted(DATA.cells.map(cell => String(cell.r_edit_budget))),
      value => { itemsState.budget = value; itemsState.visible = ITEMS_PAGE_SIZE; rerender(); }),
    (() => {
      const select = el("select", { onchange: event => {
        itemsState.outcome = event.target.value; itemsState.visible = ITEMS_PAGE_SIZE; rerender(); } },
        el("option", { value: "", text: "all outcomes" }),
        ...Object.entries(OUTCOMES).map(([key, outcome]) =>
          el("option", { value: key, text: outcome.label })));
      return el("span", {}, el("label", { text: "outcome " }), select);
    })(),
    makeSelect("stratum", ["Low", "High"],
      value => { itemsState.stratum = value; itemsState.visible = ITEMS_PAGE_SIZE; rerender(); },
      "any stratum"),
    el("input", { type: "search", placeholder: "search task id / prompt text",
      oninput: event => { itemsState.search = event.target.value;
                          itemsState.visible = ITEMS_PAGE_SIZE; rerender(); } }));
  const selects = [...filters.querySelectorAll("select")];
  [itemsState.family, itemsState.condition, itemsState.budget,
   itemsState.outcome, itemsState.stratum].forEach((value, index) => selects[index].value = value);
  filters.querySelector("input[type=search]").value = itemsState.search;
  container.append(filters);

  const matches = DATA.items.filter(itemMatches);
  container.append(el("div", { class: "count-note",
    text: `${matches.length.toLocaleString()} of ${DATA.items.length.toLocaleString()} perturbed rows match` }));
  for (const item of matches.slice(0, itemsState.visible)) container.append(itemCard(item));
  if (matches.length > itemsState.visible) {
    container.append(el("button", { class: "more",
      text: `show ${Math.min(ITEMS_PAGE_SIZE, matches.length - itemsState.visible)} more`,
      onclick: () => { itemsState.visible += ITEMS_PAGE_SIZE; rerender(); } }));
  }
}

/* =====================================================================
   TAB: Run — throughput, exclusions, provenance
   ===================================================================== */
function renderRun(container) {
  const shardEntries = Object.entries(DATA.shard_statistics || {});
  if (shardEntries.length) {
    container.append(el("div", { class: "tiles" }, ...shardEntries.map(([shardId, stats]) =>
      tile(shardId, stats.output_tokens_per_second.toLocaleString() + " tok/s",
           `${stats.rows.toLocaleString()} rows · ${Math.round(stats.wall_seconds)}s wall · `
           + `${Math.round(stats.rows_per_hour).toLocaleString()} rows/h`))));
  }

  const byCondition = new Map();
  for (const record of DATA.exclusions) {
    const key = record.condition + " · k=" + record.budget;
    byCondition.set(key, (byCondition.get(key) || 0) + record.count);
  }
  const conditionEntries = [...byCondition.entries()].sort((a, b) => b[1] - a[1]);
  const maximumCount = Math.max(1, ...conditionEntries.map(entry => entry[1]));

  container.append(el("div", { class: "card" },
    el("h2", { text: "Exclusions by condition" }),
    el("p", { class: "hint", text: "Items the perturbation builder could not construct; every one is "
      + "recorded in the exclusions sidecar with its reason." }),
    el("table", { class: "data" },
      el("tbody", {}, ...conditionEntries.map(([label, count]) => {
        const barWidth = Math.max(4, Math.round(320 * count / maximumCount));
        return el("tr", {},
          el("td", { text: label }),
          el("td", {}, el("svg:svg", { width: 330, height: 12 },
            el("svg:rect", { x: 0, y: 2, width: barWidth, height: 8, rx: 4, fill: "var(--s1)" }))),
          el("td", { text: String(count) }));
      })))));

  container.append(el("div", { class: "card" },
    el("h2", { text: "Exclusion reasons" }),
    el("table", { class: "data" },
      el("thead", {}, el("tr", {}, ...["condition", "k", "reason", "count"].map(
        head => el("th", { class: "nosort", text: head })))),
      el("tbody", {}, ...DATA.exclusions.map(record => el("tr", {},
        el("td", { text: record.condition }),
        el("td", { text: String(record.budget) }),
        el("td", { style: "white-space:normal", text: record.reason }),
        el("td", { text: String(record.count) })))))));

  container.append(el("div", { class: "card" },
    el("h2", { text: "Provenance" }),
    el("table", { class: "data" }, el("tbody", {},
      el("tr", {}, el("td", { text: "sources" }),
        el("td", { style: "white-space:normal", text: DATA.meta.sources.join(", ") })),
      el("tr", {}, el("td", { text: "models" }),
        el("td", { text: DATA.meta.models.join(", ") })),
      el("tr", {}, el("td", { text: "model revisions" }),
        el("td", { text: DATA.meta.revisions.join(", ") })),
      el("tr", {}, el("td", { text: "code commits" }),
        el("td", { text: DATA.meta.commits.join(", ") })),
      el("tr", {}, el("td", { text: "report generated" }),
        el("td", { text: GENERATED_AT }))))));
}

/* ---------- tab plumbing ---------- */
const TABS = [
  ["overview", "Overview", renderOverview],
  ["results", "Results", renderResults],
  ["mediation", "Mediation", renderMediation],
  ["items", "Items", renderItems],
  ["run", "Run", renderRun],
];

const nav = document.getElementById("tabs");
const main = document.getElementById("main");
const sections = {};
for (const [id, label, render] of TABS) {
  nav.append(el("button", { id: "tab-" + id, text: label, onclick: () => activateTab(id) }));
  const section = el("section", { class: "tab", id: "section-" + id });
  sections[id] = { section, render, rendered: false };
  main.append(section);
}
function activateTab(id) {
  for (const [tabId, entry] of Object.entries(sections)) {
    entry.section.classList.toggle("active", tabId === id);
    document.getElementById("tab-" + tabId).classList.toggle("active", tabId === id);
  }
  // Items and Results re-render on every activation so cross-tab drill-down
  // state (a clicked cell row) is always reflected.
  const entry = sections[id];
  if (!entry.rendered || id === "items" || id === "results") {
    entry.section.replaceChildren();
    entry.render(entry.section);
    entry.rendered = true;
  }
}

document.getElementById("header-sub").textContent =
  `${DATA.meta.models.join(", ")} · ${DATA.meta.row_count.toLocaleString()} rows · `
  + `${DATA.cells.length} cells · generated ${GENERATED_AT}`;
activateTab("overview");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
