"""Build a self-contained HTML report from generation JSONL files.

One file, five tabs, zero external dependencies (no CDN, opens offline):

  Overview   : run identity, gate readout, per-model summary, damage heatmap
  Effects    : filterable per-cell table + a dot-and-CI severity chart
  Statistics : GLMM forest + coefficient tables, mediation, Method A
  Items      : filterable per-item drill-down with exact clean→perturbed diffs
  Run & data : per-model manifests, exclusions, provenance, config, figures

The report is the single place to inspect everything a reviewer would ask
for: every statistic the analysis produces, every manifest, every exclusion,
every generation (drill-down), and the analysis figures. Every metric label,
table header, axis label, and value carries a hover definition, driven by one
glossary so a term is defined identically everywhere it appears.

All statistics are recomputed from the rows via the same analysis code the
paper uses (analysis.results / analysis.gates); the mediation and model-fit
JSONs and figure PNGs are embedded from --analysis-directory when present.

Usage:

    python tools/build_report.py \\
        --generations results/pilot/pilot_generations.jsonl \\
        --output results/pilot/report.html \\
        --config configs/pilot.yaml \\
        --analysis-directory analysis/pilot
"""

from __future__ import annotations

import argparse
import base64
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

_ANALYSIS_JSON_FILES = {
    "mediation": "mediation_proportion.json",
    "mixed_model": "mixed_effects_logistic.json",
    "linear_model": "linear_probability_mixed_model.json",
}


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


def _clean_store_key(row: dict) -> str:
    # Keyed by model AND task: the same task_id exists once per model, with
    # different outputs. A task_id-only key silently joins every model's
    # perturbed rows to whichever model's clean row loaded last.
    return f"{row.get('model_id', '')}|{row['task_id']}"


def _items_payload(rows: list[dict], cell_index_by_key: dict) -> tuple[dict, list]:
    """(clean_store, perturbed_items): the clean prompt/answer once per
    model×item, and one positional record per perturbed row (_ITEM_FIELDS)."""
    clean_store: dict[str, list] = {}
    for row in rows:
        if row.get("is_clean"):
            clean_store[_clean_store_key(row)] = [
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
        clean_entry = clean_store.get(_clean_store_key(row))
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


def _manifests_by_model(generation_paths: list[Path]) -> dict:
    """Every sibling *_manifest.json, keyed by the model directory name. The
    full manifest is embedded (budgets, shard statistics, commit, revision):
    it is the run's provenance record and belongs in the report verbatim."""
    manifests = {}
    for generations_path in generation_paths:
        directory = Path(generations_path).parent
        for manifest_path in sorted(directory.glob("*_manifest.json")):
            manifests[directory.name] = json.loads(manifest_path.read_text())
    return manifests


def _analysis_json(analysis_directory: Path | None, file_name: str):
    if analysis_directory is None:
        return None
    path = Path(analysis_directory) / file_name
    return json.loads(path.read_text()) if path.exists() else None


def _analysis_figures(analysis_directory: Path | None) -> dict:
    """Every PNG in the analysis directory, base64-embedded so the report
    shows the exact figures the analysis wrote, not a re-computation."""
    if analysis_directory is None:
        return {}
    return {
        png.name: base64.b64encode(png.read_bytes()).decode()
        for png in sorted(Path(analysis_directory).glob("*.png"))
    }


def _model_summaries(rows: list[dict], configuration) -> list[dict]:
    """Per-model row counts and gate readouts, in first-seen order."""
    order: list[str] = []
    rows_by_model: dict[str, list[dict]] = {}
    for row in rows:
        model_id = str(row.get("model_id", ""))
        if model_id not in rows_by_model:
            order.append(model_id)
            rows_by_model[model_id] = []
        rows_by_model[model_id].append(row)

    summaries = []
    for model_id in order:
        model_rows = rows_by_model[model_id]
        gates = compute_stage_gates(
            model_rows,
            configuration.primary_edit_budget_reasoning,
            configuration.primary_edit_budget_mcq)
        summaries.append({
            "model_id": model_id,
            "revision": str(model_rows[0].get("model_revision", "")),
            "quantization": str(model_rows[0].get("quantization_method", "")),
            "row_count": len(model_rows),
            "clean_accuracy": (
                sum(row.get("is_correct", 0) for row in model_rows if row.get("is_clean"))
                / max(1, sum(1 for row in model_rows if row.get("is_clean")))),
            "compliance": gates.get("reasoning_format_compliance"),
            "truncation_rate": gates.get("truncation_rate"),
        })
    return summaries


def build_payload(rows: list[dict], generation_paths: list[Path],
                  configuration: ExperimentConfiguration,
                  analysis_directory: Path | None, config_path: Path | None,
                  seed: int, resamples: int) -> dict:
    pairs = join_matched_pairs(rows)
    cell_summaries = summarize_all_cells(pairs, seed=seed, resamples=resamples)
    cell_index_by_key = {
        tuple(summary.get(key) for key in CELL_DIMENSION_KEYS): index
        for index, summary in enumerate(cell_summaries)}
    clean_store, perturbed_items = _items_payload(rows, cell_index_by_key)

    return _sanitize({
        "meta": {
            "sources": [str(path) for path in generation_paths],
            "row_count": len(rows),
            "pair_count": len(pairs),
            "models": _model_summaries(rows, configuration),
            "commits": sorted({str(row.get("git_commit", "")) for row in rows}),
            "seed": seed,
            "bootstrap_resamples": resamples,
        },
        "gates": compute_stage_gates(
            rows,
            configuration.primary_edit_budget_reasoning,
            configuration.primary_edit_budget_mcq),
        "cells": cell_summaries,
        "cell_dimension_keys": list(CELL_DIMENSION_KEYS),
        **{payload_key: _analysis_json(analysis_directory, file_name)
           for payload_key, file_name in _ANALYSIS_JSON_FILES.items()},
        "method_a": summarize_fragmentation_contrast(rows, seed=seed, resamples=resamples),
        "figures": _analysis_figures(analysis_directory),
        "config_text": config_path.read_text() if config_path else None,
        "config_name": str(config_path) if config_path else None,
        "clean_store": clean_store,
        "items": perturbed_items,
        "item_fields": list(_ITEM_FIELDS),
        "exclusions": _exclusions_summary(generation_paths),
        "manifests": _manifests_by_model(generation_paths),
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
        help="run_analysis output directory; embeds its JSONs and figure PNGs")
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
        rows, arguments.generations, configuration, arguments.analysis_directory,
        arguments.config, arguments.seed, arguments.bootstrap_resamples)

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
    print("open in any browser, no server or network required")


# ---------------------------------------------------------------------------
# The page. Plain string (not an f-string) so CSS/JS braces need no escaping;
# data is injected via the markers above. Palette: the validated reference
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
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
  --seq-250:#86b6ef; --seq-550:#1c5cab;
  --div-neg:#2a78d6; --div-pos:#e34948; --div-mid:#f0efec;
  --good:#0ca30c; --critical:#d03b3b; --good-text:#006300;
  --diff-del-bg:#ffd7d7; --diff-del-ink:#7d0000;
  --diff-ins-bg:#cdf2cd; --diff-ins-ink:#005a00;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface:#1a1a19; --plane:#0d0d0d;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --border: rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
    --seq-250:#86b6ef; --seq-550:#1c5cab;
    --div-neg:#3987e5; --div-pos:#e66767; --div-mid:#383835;
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
.card .hint { color: var(--muted); font-size: 12px; margin: 0 0 12px; max-width: 900px; }
/* Anything carrying a hover definition: dotted underline = "hover me". */
.term { text-decoration: underline dotted var(--muted); text-underline-offset: 3px; cursor: help; }
table.data { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data th {
  text-align: left; color: var(--muted); font-weight: 500; font-size: 12px;
  border-bottom: 1px solid var(--baseline); padding: 5px 10px 5px 0; white-space: nowrap;
  cursor: pointer; user-select: none;
}
table.data th.nosort { cursor: help; }
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
.legend .key { display: inline-flex; align-items: center; gap: 6px; cursor: help; }
.legend .swatch-line { width: 16px; height: 2px; border-radius: 1px; display: inline-block; }
.legend .swatch-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
#tooltip {
  position: fixed; pointer-events: none; z-index: 50; display: none;
  background: var(--surface); border: 1px solid var(--baseline); border-radius: 6px;
  padding: 7px 10px; font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.18); max-width: 380px;
}
#tooltip .t-title { color: var(--ink); font-weight: 600; margin-bottom: 3px; }
#tooltip .t-def { color: var(--ink-2); margin-bottom: 4px; white-space: normal; }
#tooltip .t-row { display: flex; align-items: center; gap: 6px; }
#tooltip .t-key { width: 12px; height: 2px; display: inline-block; border-radius: 1px; flex: none; }
#tooltip .t-value { font-weight: 600; font-variant-numeric: tabular-nums; }
#tooltip .t-name { color: var(--ink-2); }
svg text { fill: var(--muted); font-size: 11px; font-family: inherit; }
svg .axis { stroke: var(--baseline); stroke-width: 1; }
svg .gridline { stroke: var(--grid); stroke-width: 1; }
svg .zero { stroke: var(--baseline); stroke-width: 1; }
svg text.dlabel { fill: var(--ink-2); font-weight: 600; }
svg text.hoverable { cursor: help; pointer-events: all; }
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
.item pre, .card pre.raw {
  background: var(--plane); border: 1px solid var(--grid); border-radius: 6px;
  padding: 8px; font-size: 12px; white-space: pre-wrap; word-break: break-word;
  max-height: 320px; overflow-y: auto;
}
.card pre.raw { max-height: 420px; }
.mini-note { color: var(--muted); font-size: 12px; }
.count-note { color: var(--muted); font-size: 12px; margin: 8px 0; }
button.more {
  font: inherit; font-size: 13px; padding: 6px 14px; border-radius: 6px;
  border: 1px solid var(--baseline); background: var(--surface); color: var(--ink); cursor: pointer;
}
.figure-block img { max-width: 100%; border: 1px solid var(--grid); border-radius: 6px; }
.figure-block .caption { color: var(--ink-2); font-size: 12px; margin-top: 6px; max-width: 900px; }
.heat-cell { cursor: pointer; }
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
const shortModel = id => (id || "").split("/").pop();

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
/* Fixed slot per selection policy — assigned once from the full cell list so
   filtering never repaints survivors. Color identity follows the policy (7
   policies fit the 8-slot palette); operation/scope variants of a policy
   share its hue and are separated by position and tooltip, never by a
   9th generated color. */
const POLICY_SLOTS = (() => {
  const slots = new Map();
  for (const cell of DATA.cells) {
    if (!slots.has(cell.r_selection_policy)) slots.set(cell.r_selection_policy, slots.size);
  }
  return slots;
})();
const CONDITION_SLOTS = (() => {
  const slots = new Map();
  for (const cell of DATA.cells) {
    const label = conditionLabel(cell);
    if (!slots.has(label)) slots.set(label, POLICY_SLOTS.get(cell.r_selection_policy));
  }
  return slots;
})();

/* ---------- tooltip ---------- */
const tooltip = document.getElementById("tooltip");
function showTooltipParts(clientX, clientY, titleText, definitionText, rows) {
  tooltip.replaceChildren();
  if (titleText) tooltip.append(el("div", { class: "t-title", text: titleText }));
  if (definitionText) tooltip.append(el("div", { class: "t-def", text: definitionText }));
  for (const row of rows || []) {
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
const showTooltip = (x, y, title, rows) => showTooltipParts(x, y, title, null, rows);
function hideTooltip() { tooltip.style.display = "none"; }

/* ---------- select helper ---------- */
function makeSelect(labelText, glossKey, options, onchange, allLabel) {
  const select = el("select", { onchange: event => onchange(event.target.value) },
    el("option", { value: "", text: allLabel || ("all " + labelText) }));
  for (const option of options) select.append(el("option", { value: option, text: option }));
  return el("span", {}, term(glossKey, labelText), el("label", { text: " " }), select);
}
const uniqueSorted = values => [...new Set(values)].filter(v => v !== "" && v != null).sort();
/* =====================================================================
   GLOSSARY — one definition per term, shown on hover wherever the term
   appears (tiles, table headers, cells, axes, legends). A reviewer should
   never need the design docs open to decode a label.
   ===================================================================== */
const GLOSSARY = {
  rows: { t: "Generation rows", d: "One row per model output: every (model, item, condition) combination, clean and perturbed. Each row records the full prompt, output, parse, score, and tokenization metrics." },
  matched_pair: { t: "Matched pair", d: "A perturbed row joined to the clean row of the same item under the same model. All paired statistics (Δ, McNemar, discordant rate) compare within these pairs, so item difficulty and model skill cancel out." },
  cell: { t: "Condition cell", d: "All matched pairs sharing one (model, task family, regime, selection policy, operation, scope, edit budget) combination. The unit at which paired accuracy is tabulated." },
  condition: { t: "Condition", d: "The perturbation recipe: how words are picked (selection policy), what is done to them (operation), and where (scope). Displayed as policy · operation [· scope]." },
  model: { t: "Model", d: "The instruction-tuned model that produced the row, identified by its HuggingFace id. Weights are fetched at the pinned revision shown next to it." },
  model_revision: { t: "Model revision", d: "The HuggingFace git commit of the model weights actually used. Recorded on every row so results are reproducible even if the repo's main branch moves." },
  quantization: { t: "Precision / quantization", d: "fp16 = full half-precision weights; awq = 4-bit AWQ-quantized weights. In the T4-only roster precision is confounded with model size (the fp16 models are the small ones); the cluster fp16 trio breaks that confound." },
  git_commit: { t: "Code commit", d: "The git commit of this repository at generation time, recorded on every row. Ties each output to the exact pipeline code that produced it." },
  seed: { t: "Seed", d: "The run-level random seed. Perturbation sampling, bootstrap resampling, and item sampling all derive from it, so the entire analysis is re-runnable bit-for-bit." },
  bootstrap: { t: "Bootstrap resamples", d: "Number of resamples used for every bootstrap confidence interval in this report. Resampling is clustered by item, so items (not rows) are the unit of resampling." },

  task_family: { t: "Task family", d: "The benchmark the item comes from. GSM families are free-form arithmetic word problems scored on the final number; MMLU families are multiple-choice scored on the chosen letter." },
  family_gsm8k: { t: "gsm8k", d: "Standard GSM8K grade-school arithmetic (test split). Widely trained on, so treated as the potentially-contaminated (memorization-prone) reasoning benchmark." },
  family_gsm_symbolic_official: { t: "gsm_symbolic_official", d: "Apple GSM-Symbolic (p1 variant): templated re-instantiations of GSM-style problems with fresh numbers/names. Contamination-controlled counterpart to gsm8k." },
  family_mmlu: { t: "mmlu", d: "Standard MMLU, 4-option multiple choice. Widely trained on, so treated as the potentially-contaminated MCQ benchmark." },
  family_mmlu_pro: { t: "mmlu_pro", d: "MMLU-Pro, 10-option multiple choice with harder distractors (chance = 10%). Contamination-resistant counterpart to mmlu." },
  regime: { t: "Semantic regime", d: "How much the edit could change meaning. A = intent-preserving nonword typo (reader recovers the word instantly). B = context-recoverable real-word shift (a real but wrong word; context disambiguates). C = meaning-changing control (the edit legitimately changes the task)." },
  policy: { t: "Selection policy", d: "How the perturbation chooses its replacement characters or words. Each policy isolates a different mechanism (motor slip, acoustic confusion, token-count change without word damage, …)." },
  policy_keyboard_neighbor: { t: "keyboard_neighbor", d: "Replaces a character with a physically adjacent key (QWERTY): the canonical motor-slip typo. Primary Regime-A condition." },
  policy_informative_word: { t: "informative_word", d: "Targets the edit at a word annotated as informative for solving the item, instead of a random eligible word." },
  policy_real_word: { t: "real_word", d: "The corrupted string must itself be a dictionary word (e.g. 'from'→'form'): a spellchecker would not flag it. Regime-B condition." },
  policy_homophone: { t: "homophone", d: "Swaps a word for an exact CMU-dictionary homophone: the pure acoustic-confusion proxy, crosswalking to the HIVE voice arm." },
  policy_whitespace: { t: "whitespace", d: "Splits or merges words at a space ('with in' / 'cannot'). Changes tokenization with minimal semantic damage." },
  policy_filler_word: { t: "filler_word", d: "Inserts a discourse particle from the frozen set {uh, um, like, so}. Adds tokens without corrupting any existing word — the key contrast for the fragmentation hypothesis (H1b)." },
  policy_fragmentation_matched: { t: "fragmentation_matched", d: "Method A pairs: the same word gets two same-size edits that differ only in their tokenization consequence (Low vs High fragmentation). Isolates fragmentation with everything else held fixed." },
  operation: { t: "Operation", d: "The edit applied: substitute / delete / insert / transpose act on characters; word_substitute swaps a whole word (Regimes B and C)." },
  scope: { t: "Scope", d: "Which part of the prompt may be edited: instruction (the task wording), content (the problem body), answer_critical (words the answer depends on), or anywhere." },
  k: { t: "Edit budget k", d: "The number of atomic edits applied to the prompt — the experiment's severity dose. Damage is expected to grow with k; the dose-response is the primary causal signal." },

  n_pairs: { t: "n (pairs)", d: "Matched pairs in this cell: each is one item's clean and perturbed runs under the same model." },
  both_correct: { t: "both correct", d: "Pairs where clean and perturbed answers were both right. The perturbation made no visible difference." },
  broke: { t: "broke", d: "Pairs where the clean answer was right and the perturbed answer wrong: the perturbation's visible damage." },
  recovered: { t: "recovered", d: "Pairs where the clean answer was wrong and the perturbed answer right. Near-floor accuracy makes this common by lucky guessing, so read it together with clean accuracy." },
  both_wrong: { t: "both wrong", d: "Pairs where both arms failed. Carries no information about the perturbation." },
  clean_accuracy: { t: "Clean accuracy", d: "Accuracy on the unperturbed arm of these pairs. The model's baseline on exactly these items — also the ceiling for how much damage is even observable." },
  perturbed_accuracy: { t: "Perturbed accuracy", d: "Accuracy on the perturbed arm of these pairs." },
  delta: { t: "Δ (paired degradation)", d: "Clean accuracy minus perturbed accuracy over the matched pairs, in percentage points. Positive = the perturbation hurt; negative = perturbed did better (usually floor noise)." },
  delta_ci: { t: "95% CI on Δ", d: "Bootstrap confidence interval for Δ, resampling pairs. If it excludes 0 the cell's effect is unlikely to be resampling noise." },
  ci_method: { t: "CI method", d: "BCa = bias-corrected accelerated bootstrap (default). percentile = plain percentile bootstrap (used when BCa is degenerate, e.g. all-identical resamples). insufficient_n = too few pairs to bootstrap at all." },
  ccf: { t: "CCF (clean-conditioned failure)", d: "broke / (broke + both correct): among pairs the model solved cleanly, the fraction the perturbation broke. Immune to floor effects because it conditions on clean success." },
  retention: { t: "Retention", d: "Perturbed accuracy / clean accuracy. 1.0 = no damage; 0.8 = the model kept 80% of its clean performance." },
  p_d: { t: "p_d (discordant rate)", d: "(broke + recovered) / n: the fraction of pairs where the two arms disagree. Only discordant pairs carry information in a paired design, so p_d drives statistical power and the implied-N calculation." },
  mcnemar: { t: "McNemar p", d: "Exact mid-p McNemar test of broke vs recovered counts: is the asymmetry between damage and improvement bigger than coin-flip noise? The correct paired test; stars: * p<.05, ** p<.01, *** p<.001." },

  compliance: { t: "Format compliance", d: "Share of reasoning generations that produced an extractable final answer in the requested '#### N' format. Gate: ≥ 0.95. Below the gate, accuracy differences could be format artifacts rather than reasoning failures." },
  truncation: { t: "Truncation rate", d: "Share of generations cut off by the max_new_tokens budget (finish_reason = length). Truncated rows score as wrong but are never counted as refusals; high truncation depresses accuracy artificially." },
  p99_tokens: { t: "p99 clean-correct length", d: "99th-percentile output length (tokens) among clean, correct answers. The evidence base for freezing max_new_tokens: the budget should comfortably exceed this number." },
  primary_condition: { t: "Primary condition", d: "The pre-registered condition the gates are computed on: Regime A keyboard-neighbor substitution, scope anywhere, at the primary edit budget for the task type (k=2 reasoning, k=4 MCQ)." },
  implied_n: { t: "Implied N (5pp MDE)", d: "Items per family needed for 80% power to detect a 5-percentage-point paired effect (the minimum detectable effect, MDE), given this run's measured discordant rate, via Connor (1987). Estimated from ~120 pairs, so it carries real sampling error — hence the margin in the main-study N." },
  bucket: { t: "Sample-size bucket", d: "design/06 §6.3 decision rule from the implied N: n600_confirmed = the planned N covers this family; raise_n_or_relax_mde = it does not — either raise N or accept a larger minimum detectable effect. (Bucket names are keyed to the planning value N=600; the main run uses N=720, design/00 §0.5.)" },

  glmm: { t: "Mixed-effects logistic model", d: "One regression over every row: is_correct ~ perturbation and tokenization terms, with random intercepts for item and model (lme4 glmer via rpy2). Pools evidence across all cells while respecting that rows cluster within items and models." },
  coef: { t: "Coefficient (log-odds)", d: "The term's effect on the log-odds of a correct answer, holding the other terms fixed. exp(coef) is the odds ratio." },
  or: { t: "Odds ratio (OR)", d: "Multiplicative effect on the odds of answering correctly. OR 0.90 per edit = each additional typo multiplies the odds of success by 0.90. 1.0 = no effect." },
  std_error: { t: "Standard error", d: "Uncertainty of the coefficient estimate; the 95% CI is roughly coef ± 1.96 × SE." },
  p_value: { t: "p-value", d: "Probability of an effect at least this large if the true effect were zero. Not corrected for multiple comparisons unless stated." },
  term_is_perturbed: { t: "is_perturbed", d: "Binary: the row is perturbed at all (any condition, any k) vs clean. With edit_budget_k also in the model, this is the 'intercept' of damage at zero extra edits — the dose term below carries the real signal." },
  term_token_inflation_excess: { t: "token_inflation_excess", d: "τ − 1: how much longer the prompt tokenizes relative to its clean version. The mediator, entered directly in the outcome model." },
  term_word_length_before: { t: "word_length_before", d: "Character length of the edited word before editing. Covariate: long words survive edits better (more redundancy)." },
  term_subword_count_change: { t: "subword_count_change", d: "How many more subword tokens the edited word occupies after the edit (word-level fragmentation)." },
  term_edit_budget_k: { t: "edit_budget_k", d: "The severity dose (number of edits) as a linear term: the per-edit effect. This is the primary dose-response estimate." },
  term_precision: { t: "precision fp16", d: "fp16 vs awq model group. CAUTION: in the T4 roster precision is perfectly confounded with model size (fp16 = the 1–3B models), so this coefficient is a size effect wearing a precision label, not a quantization effect." },
  term_interaction: { t: "is_perturbed × precision", d: "Whether perturbation damage differs between the fp16 and awq model groups. Null here = no evidence quantization changes typo sensitivity (size caveat as for the main precision term)." },
  random_effects: { t: "Random-effects variance", d: "Variance of the per-item and per-model random intercepts, on the log-odds scale. Large item variance = items differ hugely in difficulty — exactly why the design is paired within item." },
  singular: { t: "Singular fit", d: "A random-effects structure too rich for the data (some variance estimated at exactly zero). Singular fits are discarded and the ladder falls back to a simpler structure." },
  ladder: { t: "Model ladder", d: "Pre-registered fallback sequence of random-effects structures, tried richest-first; the first non-singular, converged fit is reported. Entries listed here were tried and rejected." },
  linear_model: { t: "Linear probability model", d: "The same regression on the probability scale (statsmodels MixedLM): coefficients are risk differences (percentage-point changes), not odds ratios. Robustness companion to the logistic fit." },
  risk_difference: { t: "Risk difference", d: "Change in the probability of a correct answer, in absolute terms (−0.012 = −1.2 percentage points), holding other terms fixed." },

  mediation: { t: "Statistical mediation (Method B)", d: "Decomposes the perturbation's total effect on accuracy into the part flowing through tokenization fragmentation (indirect, α×β) and the rest (direct), via the Imai et al. quasi-Bayesian algorithm on mixed logistic models, conditional on the median item." },
  alpha: { t: "α (treatment → mediator)", d: "Effect of perturbation on the mediator (token inflation): how much the typo fragments the tokenization." },
  beta: { t: "β (mediator → outcome)", d: "Effect of the mediator on the log-odds of a correct answer, holding treatment fixed: how much fragmentation itself hurts." },
  indirect: { t: "Indirect effect", d: "The accuracy change (probability scale) attributable to the fragmentation path: treatment moves the mediator (α), the mediator moves accuracy (β). Negative = fragmentation causes damage. The experiment's central quantity." },
  direct: { t: "Direct effect", d: "The accuracy change not flowing through fragmentation — semantic damage, attention disruption, everything else the typo does." },
  total: { t: "Total effect", d: "Indirect + direct: the overall accuracy change from the perturbation in this fit." },
  prop_mediated: { t: "Proportion mediated", d: "indirect / total. Withheld when the total effect's CI includes zero: dividing by a possibly-zero total makes the ratio numerically meaningless (its bootstrap CI explodes), so reporting it would be noise." },
  boot_ci: { t: "Bootstrap 95% CI", d: "Percentile interval over item-clustered bootstrap resamples (seeded). 'Excludes 0' = the sign of the effect is stable under resampling." },
  estimator: { t: "Estimator", d: "imai_quasibayes_mixed_logistic_conditional_on_median_item: Imai-style quasi-Bayesian mediation on mixed-effects logistic fits, with effects evaluated at the median item's random intercept." },
  supplementary_indirect: { t: "Supplementary indirect", d: "The same indirect effect under the simpler pre-registered supplementary estimator; a robustness check, not the primary number." },
  h1b: { t: "H1b policy fits", d: "The dissociation test: keyboard typos (which fragment words) should show a negative fragmentation path, filler-word insertion (which adds tokens without corrupting words) should not. Opposite signs = the mechanism is fragmentation specifically, not token count generally." },
  pooled: { t: "Pooled (supplementary)", d: "All families in one fit. Supplementary: the pre-registered analyses are per family, since pooling mixes reasoning and MCQ scoring regimes." },
  n_observations: { t: "n (observations)", d: "Rows entering this fit (clean + perturbed, after exclusions)." },

  method_a: { t: "Fragmentation-matched counterfactual (Method A)", d: "The design-based twin of the statistical mediation: the same word in the same item gets two same-size edits, one chosen to keep the tokenization intact (Low) and one to shatter it (High). Any accuracy gap is attributable to fragmentation alone. Restricted to items whose clean run was correct." },
  stratum: { t: "Fragmentation stratum", d: "Low = the edit left the word's subword count unchanged (or smaller). High = the edit increased the word's subword count. Same word, same edit size — only the tokenization consequence differs." },

  parse_status: { t: "Parse status", d: "valid = an answer was extracted. unparseable = no answer found. clarification = the model asked a question instead. refusal = the model declined. The latter three are interactional failures and always score 0." },
  extraction_tier: { t: "Extraction tier", d: "Which extraction rule found the answer, in priority order — hash_delimited ('#### N'), last_number_fallback (any final number), mcq_explicit_marker ('answer is X'), mcq_line_leading (letter starting a line), mcq_standalone_sentence (letter in last sentence), unparseable. Recorded so a reviewer can audit which surface pattern each score came from." },
  finish_reason: { t: "Finish reason", d: "Why generation stopped: 'stop' = the model ended naturally; 'length' = it hit the max_new_tokens budget (truncated — scored wrong, never counted as a refusal)." },
  tau: { t: "τ (token inflation ratio)", d: "Tokens(perturbed prompt) / Tokens(clean prompt), computed with the model's own tokenizer. τ > 1 = the perturbation fragmented the text into more pieces. The mediator of the fragmentation hypothesis." },
  subword_change: { t: "Δ subwords", d: "Tokens(edited word) − Tokens(original word), with a leading space (words tokenize differently mid-sentence). The word-level fragmentation measure behind the Low/High strata." },
  edited_word: { t: "Edited word", d: "The word the perturbation targeted." },

  shard: { t: "Shard", d: "One (run, task-type) generation batch: reasoning and multiple-choice items run as separate shards because they use different token budgets." },
  tok_per_s: { t: "Output tokens / s", d: "Generated tokens per wall-clock second for the shard — the GPU throughput actually achieved." },
  rows_per_h: { t: "Rows / hour", d: "Completed generations per hour of wall clock, the number that calibrates run-time forecasts (main study = 25× rehearsal per model)." },
  wall_seconds: { t: "Wall seconds", d: "Elapsed real time for the shard, including scheduling and I/O." },
  exclusions: { t: "Exclusions", d: "Items a condition could not be constructed for (e.g. no eligible word to corrupt under the policy's constraints). Every exclusion is recorded with its reason in the sidecar; exclusions happen before generation, so they cost no GPU time and cannot bias scored results — but systematic exclusion patterns could narrow what a condition's estimate generalizes to, which is why they are tabulated here." },
  manifest: { t: "Shard manifest", d: "The runner's provenance record per model: which shards completed, row counts, throughput, token budgets, code commit, and model revision. The resume mechanism reads it to skip completed work; budgets are recorded so a resume under different budgets is refused rather than silently mixed." },
  config: { t: "Experiment config", d: "The exact YAML that fully specifies this run: datasets and item counts, the perturbation grid, token budgets, primary edit budgets, and seed. Embedded verbatim." },
};

/* term(key, text): a span that shows its glossary entry on hover/focus. */
function glossEntry(key) { return GLOSSARY[key] || null; }
function attachGloss(node, key, extraRows) {
  const entry = glossEntry(key);
  if (!entry) return node;
  const show = event => showTooltipParts(
    event.clientX, event.clientY, entry.t, entry.d,
    typeof extraRows === "function" ? extraRows() : extraRows);
  node.addEventListener("pointermove", show);
  node.addEventListener("pointerleave", hideTooltip);
  return node;
}
function term(key, text) {
  const entry = glossEntry(key);
  if (!entry) return el("span", { text });
  return attachGloss(el("span", { class: "term", text: text != null ? text : entry.t }), key);
}
/* svgTerm: hoverable SVG text (axis labels, series labels). */
function svgTerm(attrs, key, extraRows) {
  return attachGloss(el("svg:text", { ...attrs, class: ((attrs.class || "") + " hoverable").trim() }), key, extraRows);
}

/* =====================================================================
   dataTable — every table goes through this builder so headers AND value
   cells uniformly carry hover definitions.
   columns: { key, label, gloss, sort(row) -> comparable | null,
              text(row) -> string  or  cell(row) -> node,
              tip(row) -> [{value, name, color?}]  (extra tooltip rows) }
   ===================================================================== */
function dataTable(columns, rowsData, options) {
  const state = (options && options.sortState) || { key: null, descending: true };
  const container = el("div", {});
  const render = () => {
    const sortColumn = columns.find(column => column.key === state.key);
    const ordered = sortColumn && sortColumn.sort
      ? [...rowsData].sort((a, b) => {
          const left = sortColumn.sort(a), right = sortColumn.sort(b);
          const comparison = (left == null) - (right == null)
            || (typeof left === "string" ? left.localeCompare(right) : left - right);
          return state.descending ? -comparison : comparison;
        })
      : rowsData;
    const head = el("tr", {}, ...columns.map(column => {
      const th = el("th", {
        class: column.sort ? "" : "nosort",
        text: column.label + (state.key === column.key ? (state.descending ? " ↓" : " ↑") : ""),
      });
      attachGloss(th, column.gloss);
      if (column.sort) th.addEventListener("click", () => {
        state.descending = state.key === column.key ? !state.descending : true;
        state.key = column.key;
        render();
      });
      return th;
    }));
    const body = el("tbody", {}, ...ordered.map(row => {
      const tr = el("tr", options && options.rowAttrs ? options.rowAttrs(row) : {});
      for (const column of columns) {
        const td = column.cell
          ? el("td", {}, column.cell(row))
          : el("td", { text: column.text(row) });
        attachGloss(td, column.gloss, () => [
          ...(column.text ? [{ value: column.text(row), name: "this cell" }] : []),
          ...(column.tip ? column.tip(row) : []),
        ]);
        tr.append(td);
      }
      return tr;
    }));
    container.replaceChildren(el("table", { class: "data" }, el("thead", {}, head), body));
  };
  render();
  return container;
}
/* =====================================================================
   TAB: Overview
   ===================================================================== */
function tile(glossKey, label, value, note, statusText, statusColor) {
  const box = el("div", { class: "tile" },
    el("div", { class: "label" }, term(glossKey, label)),
    el("div", { class: "value", text: value }));
  if (note) box.append(el("div", { class: "note", text: note }));
  if (statusText) box.append(el("div", { class: "status", style: `color:${statusColor}`, text: statusText }));
  return box;
}

/* Net-damage heatmap: model × family, diverging color. The single at-a-glance
   answer to "who gets hurt where", pooled over all conditions and budgets. */
function damageHeatmap() {
  const byModelFamily = new Map();
  for (const cell of DATA.cells) {
    const key = cell.model_id + "|" + cell.task_family;
    const entry = byModelFamily.get(key) || { n: 0, broke: 0, recovered: 0 };
    entry.n += cell.n; entry.broke += cell.broke; entry.recovered += cell.recovered;
    byModelFamily.set(key, entry);
  }
  const models = DATA.meta.models.map(model => model.model_id);
  const families = uniqueSorted(DATA.cells.map(cell => cell.task_family));
  const maximumMagnitude = Math.max(0.02, ...[...byModelFamily.values()]
    .map(entry => Math.abs((entry.broke - entry.recovered) / Math.max(1, entry.n))));

  const table = el("table", { class: "data" });
  const head = el("tr", {}, el("th", { class: "nosort", text: "" }),
    ...families.map(family => attachGloss(
      el("th", { class: "nosort term", text: family }), "family_" + family)));
  const body = el("tbody", {}, ...models.map(modelId => el("tr", {},
    attachGloss(el("td", { class: "term", text: shortModel(modelId) }), "model"),
    ...families.map(family => {
      const entry = byModelFamily.get(modelId + "|" + family);
      if (!entry) return el("td", { text: "—" });
      const net = (entry.broke - entry.recovered) / Math.max(1, entry.n);
      const mix = Math.round(Math.abs(net) / maximumMagnitude * 85);
      const fill = `color-mix(in srgb, var(${net >= 0 ? "--div-pos" : "--div-neg"}) ${mix}%, var(--div-mid))`;
      const td = el("td", { class: "heat-cell",
        style: `background:${fill}; text-align:center; padding:6px 10px;`,
        text: pp(net),
        onclick: () => {
          itemsState.model = modelId; itemsState.family = family;
          itemsState.condition = ""; itemsState.budget = ""; itemsState.outcome = "";
          itemsState.search = ""; itemsState.visible = ITEMS_PAGE_SIZE;
          activateTab("items");
        } });
      td.addEventListener("pointermove", event => showTooltipParts(
        event.clientX, event.clientY, `${shortModel(modelId)} · ${family}`,
        "Net damage = (broke − recovered) / pairs, pooled over every condition and edit budget. Click to open these items.",
        [
          { value: String(entry.n), name: "matched pairs" },
          { color: "var(--div-pos)", value: String(entry.broke), name: "broke (clean ✓ → perturbed ✗)" },
          { color: "var(--div-neg)", value: String(entry.recovered), name: "recovered (clean ✗ → perturbed ✓)" },
          { value: pp(net), name: "net damage rate" },
        ]));
      td.addEventListener("pointerleave", hideTooltip);
      return td;
    }))));
  table.append(el("thead", {}, head), body);
  return table;
}

function renderOverview(container) {
  const gates = DATA.gates, meta = DATA.meta;
  const compliance = gates.reasoning_format_compliance;
  const complianceTarget = gates.reasoning_format_compliance_target;
  const compliancePasses = compliance != null && compliance >= complianceTarget;

  container.append(el("div", { class: "tiles" },
    tile("rows", "generation rows", meta.row_count.toLocaleString(),
         meta.pair_count.toLocaleString() + " matched pairs"),
    tile("model", "models", String(meta.models.length),
         meta.models.map(model => shortModel(model.model_id)).join(", ")),
    tile("compliance", "format compliance", compliance == null ? "—" : pct(compliance),
         "target ≥ " + pct(complianceTarget),
         compliancePasses ? "✓ gate passed" : "✗ gate FAILED",
         compliancePasses ? "var(--good-text)" : "var(--critical)"),
    tile("truncation", "truncation", gates.truncation_rate == null ? "—" : pct(gates.truncation_rate),
         "finish_reason = length"),
    tile("p99_tokens", "p99 clean-correct length", gates.p99_clean_correct_output_tokens == null
         ? "—" : gates.p99_clean_correct_output_tokens + " tok",
         "max_new_tokens freeze input")));

  container.append(el("div", { class: "card" },
    el("h2", {}, term("model", "Models in this run")),
    el("p", { class: "hint", text: "Per-model identity and health. Compliance and truncation are computed per model over its own rows; the gate itself is evaluated on the pooled run." }),
    dataTable([
      { key: "model", label: "model", gloss: "model", sort: row => row.model_id,
        text: row => shortModel(row.model_id), tip: row => [{ value: row.model_id, name: "full id" }] },
      { key: "revision", label: "revision", gloss: "model_revision", sort: null,
        text: row => row.revision.slice(0, 10) , tip: row => [{ value: row.revision, name: "full revision" }] },
      { key: "quant", label: "precision", gloss: "quantization", sort: row => row.quantization,
        text: row => row.quantization },
      { key: "rows", label: "rows", gloss: "rows", sort: row => row.row_count,
        text: row => row.row_count.toLocaleString() },
      { key: "clean", label: "clean accuracy", gloss: "clean_accuracy", sort: row => row.clean_accuracy,
        text: row => pct(row.clean_accuracy) },
      { key: "compliance", label: "compliance", gloss: "compliance", sort: row => row.compliance,
        text: row => pct(row.compliance) },
      { key: "trunc", label: "truncation", gloss: "truncation", sort: row => row.truncation_rate,
        text: row => pct(row.truncation_rate) },
    ], meta.models)));

  container.append(el("div", { class: "card" },
    el("h2", { text: "Stage-1 gates per task family" }),
    el("p", { class: "hint", text: "Computed on the pre-registered primary condition (keyboard-neighbor substitution, scope anywhere, primary k per task type), pooled over models. These numbers decide whether the design is powered before the main study is frozen." }),
    dataTable([
      { key: "family", label: "task family", gloss: "task_family", sort: row => row.family,
        cell: row => term("family_" + row.family, row.family) },
      { key: "clean", label: "clean accuracy A₀", gloss: "clean_accuracy",
        sort: row => row.block.clean_accuracy, text: row => pct(row.block.clean_accuracy) },
      { key: "k", label: "primary k", gloss: "primary_condition",
        sort: row => row.block.primary_edit_budget, text: row => "k=" + row.block.primary_edit_budget },
      { key: "pairs", label: "pairs", gloss: "n_pairs",
        sort: row => row.block.primary_condition_pairs, text: row => String(row.block.primary_condition_pairs) },
      { key: "pd", label: "p_d", gloss: "p_d",
        sort: row => row.block.discordant_rate, text: row => num(row.block.discordant_rate, 3) },
      { key: "delta", label: "δ at primary k", gloss: "delta",
        sort: row => row.block.delta, text: row => pp(row.block.delta) },
      { key: "impliedN", label: "implied N (5pp MDE)", gloss: "implied_n",
        sort: row => row.block.implied_n_at_5pp_mde,
        text: row => row.block.implied_n_at_5pp_mde == null ? "—" : String(row.block.implied_n_at_5pp_mde) },
      { key: "bucket", label: "bucket", gloss: "bucket", sort: row => row.block.discordant_rate_bucket,
        cell: row => {
          const bucket = row.block.discordant_rate_bucket || "no primary rows";
          return el("span", { class: "badge " + (bucket === "n600_confirmed" ? "ok" : "bad"), text: bucket });
        } },
    ], Object.entries(gates.per_task_family).map(([family, block]) => ({ family, block })))));

  container.append(el("div", { class: "card" },
    el("h2", { text: "Net damage by model × family" }),
    el("p", { class: "hint", text: "(broke − recovered) / pairs, pooled over every condition and edit budget. Red = the perturbations hurt, blue = perturbed did better (near-floor cells recover by guessing — check clean accuracy before reading blue as a real improvement). Click any cell to inspect its items." }),
    damageHeatmap()));
}

/* =====================================================================
   TAB: Effects — filterable cells + dot-and-CI severity chart
   ===================================================================== */
const effectsState = {
  model: (DATA.meta.models[0] || {}).model_id || "",
  family: uniqueSorted(DATA.cells.map(cell => cell.task_family))[0] || "",
  regime: "", condition: "", budget: "", significantOnly: false,
  sort: { key: "delta", descending: true },
};

function filteredCells() {
  return DATA.cells.filter(cell =>
    (!effectsState.model || cell.model_id === effectsState.model)
    && (!effectsState.family || cell.task_family === effectsState.family)
    && (!effectsState.regime || String(cell.r_semantic_class) === effectsState.regime)
    && (!effectsState.condition || conditionLabel(cell) === effectsState.condition)
    && (!effectsState.budget || String(cell.r_edit_budget) === effectsState.budget)
    && (!effectsState.significantOnly
        || (cell.mcnemar_p_value != null && cell.mcnemar_p_value < 0.05)));
}

/* Δ vs edit budget as dodged points with bootstrap-CI whiskers. Deliberately
   NO connecting lines: each point is an independent cell estimate at a
   discrete dose, not a sampled continuous curve — a line would draw data
   that was never measured. The dose-response lives in the GLMM's
   edit_budget_k term; this chart shows the raw cell estimates behind it. */
function deltaDotChart(cells) {
  if (!cells.length) return el("p", { class: "mini-note", text: "No cells match the current filters." });
  const budgets = [...new Set(cells.map(cell => Number(cell.r_edit_budget)))].sort((a, b) => a - b);
  const seriesLabels = [...new Set(cells.map(conditionLabel))];
  const width = 820, height = 320, margin = { top: 16, right: 24, bottom: 40, left: 52 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const values = cells.flatMap(cell =>
    [cell.delta, cell.delta_ci_low, cell.delta_ci_high].filter(v => v != null).map(v => v * 100));
  let yMin = Math.min(0, ...values), yMax = Math.max(0, ...values);
  const ySpan = (yMax - yMin) || 1; yMin -= ySpan * .06; yMax += ySpan * .06;
  const bandWidth = plotWidth / budgets.length;
  const dodgeStep = Math.min(16, (bandWidth - 24) / Math.max(1, seriesLabels.length));
  const xOf = (k, label) => budgets.indexOf(k) * bandWidth + bandWidth / 2
    + (seriesLabels.indexOf(label) - (seriesLabels.length - 1) / 2) * dodgeStep;
  const yOf = value => plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;

  const svg = el("svg:svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": "paired degradation per cell versus edit budget" });
  const plot = el("svg:g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(plot);

  const tickStep = Math.max(1, Math.ceil((yMax - yMin) / 6));
  for (let tick = Math.ceil(yMin / tickStep) * tickStep; tick <= yMax; tick += tickStep) {
    plot.append(el("svg:line", { class: tick === 0 ? "zero" : "gridline",
                                 x1: 0, x2: plotWidth, y1: yOf(tick), y2: yOf(tick) }));
    plot.append(el("svg:text", { x: -8, y: yOf(tick) + 4, "text-anchor": "end", text: tick + "" }));
  }
  plot.append(svgTerm({ x: -40, y: -6, text: "Δ pp" }, "delta"));
  for (const k of budgets) {
    plot.append(svgTerm({ x: budgets.indexOf(k) * bandWidth + bandWidth / 2, y: plotHeight + 18,
                          "text-anchor": "middle", text: "k=" + k }, "k"));
  }
  plot.append(el("svg:line", { class: "axis", x1: 0, x2: plotWidth, y1: plotHeight, y2: plotHeight }));

  for (const cell of cells) {
    const label = conditionLabel(cell);
    const color = seriesColor(CONDITION_SLOTS.get(label) || 0);
    const x = xOf(Number(cell.r_edit_budget), label);
    if (cell.delta_ci_low != null && cell.delta_ci_high != null
        && cell.delta_ci_method !== "insufficient_n") {
      plot.append(el("svg:line", { x1: x, x2: x,
        y1: yOf(cell.delta_ci_low * 100), y2: yOf(cell.delta_ci_high * 100),
        stroke: color, "stroke-width": 2, opacity: 0.55 }));
    }
    plot.append(el("svg:circle", { cx: x, cy: yOf(cell.delta * 100), r: 6, fill: "var(--surface)" }));
    plot.append(el("svg:circle", { cx: x, cy: yOf(cell.delta * 100), r: 4, fill: color }));
    const hit = el("svg:circle", { cx: x, cy: yOf(cell.delta * 100), r: 13, fill: "transparent",
      onpointermove: event => showTooltipParts(event.clientX, event.clientY,
        `${label} · k=${cell.r_edit_budget}`,
        `${shortModel(cell.model_id)} · ${cell.task_family} · regime ${cell.r_semantic_class}`, [
          { color, value: pp(cell.delta), name: "Δ (clean − perturbed accuracy)" },
          { value: cell.delta_ci_method === "insufficient_n" ? "n too small"
              : `[${pp(cell.delta_ci_low)}, ${pp(cell.delta_ci_high)}]`,
            name: "bootstrap 95% CI (" + (cell.delta_ci_method || "—") + ")" },
          { value: `${cell.n} pairs · ${cell.broke} broke · ${cell.recovered} recovered`, name: "counts" },
          { value: pValue(cell.mcnemar_p_value) + stars(cell.mcnemar_p_value), name: "McNemar p" },
        ]),
      onpointerleave: hideTooltip });
    plot.append(hit);
  }

  const visiblePolicies = [...new Set(cells.map(cell => cell.r_selection_policy))];
  const legend = el("div", { class: "legend" }, ...visiblePolicies.map(policy =>
    attachGloss(el("span", { class: "key" },
      el("span", { class: "swatch-line",
                   style: `background:${seriesColor(POLICY_SLOTS.get(policy) || 0)}` }),
      el("span", { text: policy })), "policy_" + policy)));
  return el("div", {}, legend, svg);
}

function deltaBar(cell) {
  const width = 110, height = 12, center = width / 2, scale = center / 0.35;
  const magnitude = Math.min(Math.abs(cell.delta || 0), 0.35) * scale;
  const svg = el("svg:svg", { width, height });
  svg.append(el("svg:line", { class: "zero", x1: center, x2: center, y1: 0, y2: height }));
  if (magnitude > 0.5) {
    const damage = (cell.delta || 0) > 0;
    svg.append(el("svg:rect", {
      x: damage ? center : center - magnitude, y: 2,
      width: Math.max(magnitude, 3), height: 8, rx: 3,
      fill: damage ? "var(--div-pos)" : "var(--div-neg)" }));
  }
  return svg;
}

function renderEffects(container) {
  container.replaceChildren();
  const rerender = () => renderEffects(container);
  const filters = el("div", { class: "filters" },
    makeSelect("model", "model", DATA.meta.models.map(model => model.model_id),
               value => { effectsState.model = value; rerender(); }),
    makeSelect("family", "task_family", uniqueSorted(DATA.cells.map(cell => cell.task_family)),
               value => { effectsState.family = value; rerender(); }),
    makeSelect("regime", "regime", uniqueSorted(DATA.cells.map(cell => String(cell.r_semantic_class))),
               value => { effectsState.regime = value; rerender(); }),
    makeSelect("condition", "condition", [...CONDITION_SLOTS.keys()],
               value => { effectsState.condition = value; rerender(); }),
    makeSelect("k", "k", uniqueSorted(DATA.cells.map(cell => String(cell.r_edit_budget))),
               value => { effectsState.budget = value; rerender(); }),
    el("label", { class: "toggle" },
      el("input", { type: "checkbox",
                    onchange: event => { effectsState.significantOnly = event.target.checked; rerender(); } }),
      "significant only (p < .05)"));
  [...filters.querySelectorAll("select")].forEach((select, index) => {
    select.value = [effectsState.model, effectsState.family, effectsState.regime,
                    effectsState.condition, effectsState.budget][index];
  });
  filters.querySelector("input[type=checkbox]").checked = effectsState.significantOnly;
  container.append(filters);

  const cells = filteredCells();

  container.append(el("div", { class: "card" },
    el("h2", { text: "Paired degradation Δ vs edit budget" }),
    el("p", { class: "hint", text: "Each mark is one condition cell: the point is Δ, the whisker its bootstrap 95% CI. Points are not connected — each is an independent estimate at a discrete dose; the dose-response slope is estimated by the GLMM (Statistics tab), not drawn here. Hover any point for its full numbers." }),
    deltaDotChart(cells)));

  container.append(el("div", { class: "card" },
    el("h2", {}, term("cell", `Condition cells (${cells.length})`)),
    el("p", { class: "hint", text: "Every cell statistic the analysis computes, one row per cell. Click a column header to sort, hover anything for its definition, click a row to open its items in the drill-down." }),
    dataTable([
      { key: "family", label: "family", gloss: "task_family", sort: row => row.task_family,
        cell: row => term("family_" + row.task_family, row.task_family) },
      { key: "regime", label: "regime", gloss: "regime", sort: row => String(row.r_semantic_class),
        text: row => String(row.r_semantic_class) },
      { key: "condition", label: "condition", gloss: "condition", sort: row => conditionLabel(row),
        cell: row => term("policy_" + row.r_selection_policy, conditionLabel(row)) },
      { key: "k", label: "k", gloss: "k", sort: row => row.r_edit_budget, text: row => String(row.r_edit_budget) },
      { key: "n", label: "n", gloss: "n_pairs", sort: row => row.n, text: row => String(row.n) },
      { key: "bc", label: "✓✓", gloss: "both_correct", sort: row => row.both_correct, text: row => String(row.both_correct) },
      { key: "broke", label: "broke", gloss: "broke", sort: row => row.broke, text: row => String(row.broke) },
      { key: "rec", label: "recov.", gloss: "recovered", sort: row => row.recovered, text: row => String(row.recovered) },
      { key: "bw", label: "✗✗", gloss: "both_wrong", sort: row => row.both_wrong, text: row => String(row.both_wrong) },
      { key: "cleanacc", label: "clean acc", gloss: "clean_accuracy", sort: row => row.clean_accuracy,
        text: row => pct(row.clean_accuracy) },
      { key: "pertacc", label: "pert acc", gloss: "perturbed_accuracy", sort: row => row.perturbed_accuracy,
        text: row => pct(row.perturbed_accuracy) },
      { key: "delta", label: "Δ", gloss: "delta", sort: row => row.delta, text: row => pp(row.delta) },
      { key: "deltabar", label: "", gloss: "delta", sort: row => row.delta, cell: deltaBar },
      { key: "ci", label: "95% CI", gloss: "delta_ci", sort: row => row.delta,
        text: row => row.delta_ci_method === "insufficient_n" ? "n too small"
          : `[${pp(row.delta_ci_low)}, ${pp(row.delta_ci_high)}]`,
        tip: row => [{ value: row.delta_ci_method || "—", name: "CI method" }] },
      { key: "ccf", label: "CCF", gloss: "ccf", sort: row => row.clean_conditioned_failure,
        text: row => pct(row.clean_conditioned_failure) },
      { key: "ret", label: "retention", gloss: "retention", sort: row => row.retention,
        text: row => num(row.retention, 2) },
      { key: "pd", label: "p_d", gloss: "p_d", sort: row => row.discordant_rate,
        text: row => num(row.discordant_rate, 2) },
      { key: "p", label: "McNemar p", gloss: "mcnemar", sort: row => row.mcnemar_p_value,
        text: row => pValue(row.mcnemar_p_value) + stars(row.mcnemar_p_value) },
    ], cells, {
      sortState: effectsState.sort,
      rowAttrs: row => ({
        class: "clickable",
        onclick: () => {
          itemsState.model = row.model_id;
          itemsState.family = row.task_family;
          itemsState.condition = conditionLabel(row);
          itemsState.budget = String(row.r_edit_budget);
          itemsState.outcome = ""; itemsState.search = ""; itemsState.visible = ITEMS_PAGE_SIZE;
          activateTab("items");
        } }),
    })));
}

/* =====================================================================
   TAB: Statistics — GLMM, linear model, mediation, Method A
   ===================================================================== */
function orForestPlot(fixedEffects) {
  const entries = Object.entries(fixedEffects).map(([name, effect]) => ({
    name,
    or: effect.or,
    low: effect.std_error != null ? Math.exp(effect.coef - 1.96 * effect.std_error) : null,
    high: effect.std_error != null ? Math.exp(effect.coef + 1.96 * effect.std_error) : null,
    p: effect.p,
  }));
  const width = 760, rowHeight = 34, margin = { top: 8, right: 170, bottom: 34, left: 200 };
  const height = margin.top + entries.length * rowHeight + margin.bottom;
  const plotWidth = width - margin.left - margin.right;
  const magnitudes = entries.flatMap(entry => [entry.or, entry.low, entry.high])
    .filter(value => value != null && value > 0).map(Math.log);
  const bound = Math.max(0.2, ...magnitudes.map(Math.abs)) * 1.1;
  const xOf = or => ((Math.log(or) + bound) / (2 * bound)) * plotWidth;

  const svg = el("svg:svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": "odds-ratio forest plot" });
  const plot = el("svg:g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(plot);
  plot.append(el("svg:line", { class: "zero", x1: xOf(1), x2: xOf(1), y1: 0,
                               y2: entries.length * rowHeight }));
  plot.append(svgTerm({ x: xOf(1), y: entries.length * rowHeight + 16,
                        "text-anchor": "middle", text: "OR = 1 (no effect)" }, "or"));
  plot.append(el("svg:text", { x: 0, y: entries.length * rowHeight + 16, text: "← hurts accuracy" }));
  plot.append(el("svg:text", { x: plotWidth, y: entries.length * rowHeight + 16,
                               "text-anchor": "end", text: "helps →" }));
  entries.forEach((entry, index) => {
    const y = index * rowHeight + rowHeight / 2;
    plot.append(svgTerm({ x: -10, y: y + 4, "text-anchor": "end", text: entry.name },
                        "term_" + entry.name.replace("precisionfp16", "precision")
                                          .replace("is_perturbed:precisionfp16", "interaction")));
    if (entry.low != null && entry.high != null) {
      plot.append(el("svg:line", { x1: xOf(Math.max(entry.low, 0.001)), x2: xOf(entry.high),
                                   y1: y, y2: y, stroke: "var(--baseline)", "stroke-width": 2 }));
    }
    plot.append(el("svg:circle", { cx: xOf(entry.or), cy: y, r: 6, fill: "var(--surface)" }));
    plot.append(el("svg:circle", { cx: xOf(entry.or), cy: y, r: 4.5, fill: "var(--s1)" }));
    plot.append(el("svg:text", { class: "dlabel", x: plotWidth + 10, y: y + 4,
                                 text: num(entry.or, 3) + stars(entry.p) }));
    plot.append(el("svg:rect", { x: -margin.left, y: index * rowHeight, width, height: rowHeight,
      fill: "transparent",
      onpointermove: event => showTooltipParts(event.clientX, event.clientY, entry.name,
        (glossEntry("term_" + entry.name.replace("precisionfp16", "precision")
                                        .replace("is_perturbed:precisionfp16", "interaction")) || {}).d, [
          { color: "var(--s1)", value: num(entry.or, 3), name: "odds ratio" },
          { value: entry.low == null ? "—" : `[${num(entry.low, 3)}, ${num(entry.high, 3)}]`,
            name: "95% CI (coef ± 1.96 SE, exponentiated)" },
          { value: pValue(entry.p) + stars(entry.p), name: "p-value" },
        ]),
      onpointerleave: hideTooltip }));
  });
  return svg;
}

function coefficientTable(fixedEffects, isOddsScale) {
  return dataTable([
    { key: "termname", label: "term", gloss: "glmm", sort: row => row.name,
      cell: row => term("term_" + row.name
          .replace("precisionfp16", "precision")
          .replace("is_perturbed:precisionfp16", "interaction")
          .replace(/^C\(precision\)\[T\.fp16\]$/, "precision")
          .replace(/^is_perturbed:C\(precision\)\[T\.fp16\]$/, "interaction"),
        row.name) },
    { key: "coefv", label: isOddsScale ? "coef (log-odds)" : "coef (risk difference)",
      gloss: isOddsScale ? "coef" : "risk_difference",
      sort: row => row.effect.coef, text: row => num(row.effect.coef, 4) },
    ...(isOddsScale ? [
      { key: "orv", label: "odds ratio", gloss: "or", sort: row => row.effect.or,
        text: row => num(row.effect.or, 3) },
      { key: "sev", label: "SE", gloss: "std_error", sort: row => row.effect.std_error,
        text: row => num(row.effect.std_error, 3) },
      { key: "civ", label: "95% CI (OR)", gloss: "boot_ci", sort: null,
        text: row => row.effect.std_error == null ? "—"
          : `[${num(Math.exp(row.effect.coef - 1.96 * row.effect.std_error), 3)}, `
            + `${num(Math.exp(row.effect.coef + 1.96 * row.effect.std_error), 3)}]` },
    ] : []),
    { key: "pv", label: "p", gloss: "p_value", sort: row => row.effect.p,
      text: row => pValue(row.effect.p) + stars(row.effect.p) },
  ], Object.entries(fixedEffects).map(([name, effect]) => ({ name, effect })));
}

function mediationForest(entries) {
  const width = 760, rowHeight = 34, margin = { top: 8, right: 190, bottom: 34, left: 230 };
  const height = margin.top + entries.length * rowHeight + margin.bottom;
  const plotWidth = width - margin.left - margin.right;
  const magnitudes = entries.flatMap(entry => [entry.low, entry.high, entry.value])
    .filter(value => value != null).map(Math.abs);
  const bound = Math.max(0.01, ...magnitudes) * 1.15;
  const xOf = value => ((value + bound) / (2 * bound)) * plotWidth;

  const svg = el("svg:svg", { width: "100%", viewBox: `0 0 ${width} ${height}`, role: "img",
                              "aria-label": "indirect-effect forest plot" });
  const plot = el("svg:g", { transform: `translate(${margin.left},${margin.top})` });
  svg.append(plot);
  plot.append(el("svg:line", { class: "zero", x1: xOf(0), x2: xOf(0), y1: 0,
                               y2: entries.length * rowHeight }));
  plot.append(svgTerm({ x: xOf(0), y: entries.length * rowHeight + 16,
                        "text-anchor": "middle", text: "0 (no mediation)" }, "indirect"));
  plot.append(el("svg:text", { x: 0, y: entries.length * rowHeight + 16, text: "← fragmentation hurts" }));
  plot.append(el("svg:text", { x: plotWidth, y: entries.length * rowHeight + 16,
                               "text-anchor": "end", text: "helps →" }));
  entries.forEach((entry, index) => {
    const y = index * rowHeight + rowHeight / 2;
    plot.append(svgTerm({ x: -10, y: y + 4, "text-anchor": "end", text: entry.name }, entry.gloss));
    if (entry.low != null && entry.high != null) {
      plot.append(el("svg:line", { x1: xOf(entry.low), x2: xOf(entry.high), y1: y, y2: y,
                                   stroke: "var(--baseline)", "stroke-width": 2 }));
    }
    const excludesZero = entry.low != null && entry.high != null && (entry.low > 0 || entry.high < 0);
    plot.append(el("svg:circle", { cx: xOf(entry.value), cy: y, r: 6, fill: "var(--surface)" }));
    plot.append(el("svg:circle", { cx: xOf(entry.value), cy: y, r: 4.5, fill: "var(--s1)" }));
    plot.append(el("svg:text", { class: "dlabel", x: plotWidth + 10, y: y + 4,
      text: num(entry.value) + (excludesZero ? "  (CI excludes 0)" : "") }));
    plot.append(el("svg:rect", { x: -margin.left, y: index * rowHeight, width, height: rowHeight,
      fill: "transparent",
      onpointermove: event => showTooltipParts(event.clientX, event.clientY, entry.name,
        (glossEntry(entry.gloss) || {}).d, [
          { color: "var(--s1)", value: num(entry.value), name: "indirect effect (α·β)" },
          { value: `[${num(entry.low)}, ${num(entry.high)}]`, name: "bootstrap 95% CI" },
        ]),
      onpointerleave: hideTooltip }));
  });
  return svg;
}

const FIT_LABELS = [
  ["task_family:", "", "mediation"],
  ["h1b_policy:", "H1b · ", "h1b"],
  ["pooled_all_families_supplementary", "pooled (supplementary)", "pooled"],
];
function fitDisplay(name) {
  for (const [prefix, label, gloss] of FIT_LABELS) {
    if (name === prefix) return { label, gloss };
    if (name.startsWith(prefix) && prefix.endsWith(":")) return { label: label + name.slice(prefix.length), gloss };
  }
  return { label: name, gloss: "mediation" };
}

function methodADumbbells(groups) {
  const width = 860, rowHeight = 34, margin = { top: 8, right: 160, bottom: 30, left: 320 };
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
    plot.append(svgTerm({ x: -10, y: y + 4, "text-anchor": "end",
      text: `${shortModel(group.model_id)} · ${group.task_family.replace("_official", "")}`
        + ` k=${group.r_edit_budget}` }, "method_a"));
    plot.append(el("svg:line", { x1: xOf(group.clean_accuracy), x2: xOf(group.perturbed_accuracy),
                                 y1: y, y2: y, stroke: "var(--baseline)", "stroke-width": 2 }));
    for (const [accuracy, fill] of [[group.clean_accuracy, "var(--seq-250)"],
                                    [group.perturbed_accuracy, "var(--seq-550)"]]) {
      plot.append(el("svg:circle", { cx: xOf(accuracy), cy: y, r: 6.5, fill: "var(--surface)" }));
      plot.append(el("svg:circle", { cx: xOf(accuracy), cy: y, r: 5, fill }));
    }
    plot.append(el("svg:text", { class: "dlabel", x: plotWidth + 10, y: y + 4,
                                 text: `n=${group.n}, Δ=${pp(group.delta)}` }));
    plot.append(el("svg:rect", { x: -margin.left, y: index * rowHeight, width, height: rowHeight,
      fill: "transparent",
      onpointermove: event => showTooltipParts(event.clientX, event.clientY,
        `${shortModel(group.model_id)} · ${group.task_family} · k=${group.r_edit_budget}`,
        (glossEntry("stratum") || {}).d, [
          { color: "var(--seq-250)", value: pct(group.clean_accuracy), name: "Low-fragmentation accuracy" },
          { color: "var(--seq-550)", value: pct(group.perturbed_accuracy), name: "High-fragmentation accuracy" },
          { value: `${group.n} pairs · ${group.broke} broke · ${group.recovered} recovered`, name: "counts" },
          { value: pValue(group.mcnemar_p_value), name: "McNemar p" },
        ]),
      onpointerleave: hideTooltip }));
  });
  return el("div", {},
    el("div", { class: "legend" },
      attachGloss(el("span", { class: "key" },
        el("span", { class: "swatch-dot", style: "background:var(--seq-250)" }),
        el("span", { text: "Low fragmentation (same word, same k)" })), "stratum"),
      attachGloss(el("span", { class: "key" },
        el("span", { class: "swatch-dot", style: "background:var(--seq-550)" }),
        el("span", { text: "High fragmentation" })), "stratum")),
    svg);
}

function renderStatistics(container) {
  const mixed = DATA.mixed_model;
  if (mixed && mixed.fixed_effects) {
    container.append(el("div", { class: "card" },
      el("h2", {}, term("glmm", "Mixed-effects logistic regression (primary fit)")),
      el("p", { class: "hint", text: `Converged: ${mixed.converged} · method: ${mixed.method} · `
        + `n = ${mixed.n_observations} rows over ${mixed.n_items} items × ${mixed.n_models} models · `
        + `log-likelihood ${num(mixed.log_likelihood, 1)}. `
        + "Point = odds ratio, whisker = 95% CI; left of the reference line = the term reduces the odds of a correct answer." }),
      orForestPlot(mixed.fixed_effects),
      coefficientTable(mixed.fixed_effects, true),
      el("div", { style: "margin-top:12px" },
        term("random_effects", "Random-effects variance"), " — ",
        ...Object.entries(mixed.random_effects_variance || {}).flatMap(([name, variance], index) =>
          [index ? " · " : "", el("span", { class: "mini-note", text: `${name}: ${num(variance, 3)}` })])),
      mixed.ladder_notes && mixed.ladder_notes.length
        ? el("p", { class: "mini-note" }, term("ladder", "ladder"),
            el("span", { text: ": " + mixed.ladder_notes.join(" → ") + " → accepted " + mixed.method
              + (mixed.is_singular_fit ? " (SINGULAR)" : " (non-singular)") }))
        : null));
  }

  const linear = DATA.linear_model;
  if (linear && linear.fixed_effects) {
    container.append(el("div", { class: "card" },
      el("h2", {}, term("linear_model", "Linear probability mixed model (robustness)")),
      el("p", { class: "hint", text: linear.scale_note + ` · converged: ${linear.converged} · n = ${linear.n_observations}` }),
      coefficientTable(linear.fixed_effects, false)));
  }

  const mediation = DATA.mediation;
  if (mediation) {
    const entries = Object.entries(mediation)
      .filter(([, block]) => block.indirect_effect != null)
      .map(([name, block]) => ({
        name: fitDisplay(name).label, gloss: fitDisplay(name).gloss,
        value: block.indirect_effect,
        low: (block.bootstrap_ci_indirect || [])[0],
        high: (block.bootstrap_ci_indirect || [])[1],
      }));
    container.append(el("div", { class: "card" },
      el("h2", {}, term("mediation", "Statistical mediation (Method B) — indirect effect")),
      el("p", { class: "hint", text: "How much of the perturbation's damage flows through tokenization fragmentation. Negative = fragmentation causes accuracy loss. The H1b rows are the dissociation test: keyboard typos fragment words (expect negative), filler words add tokens without fragmenting (expect non-negative)." }),
      mediationForest(entries),
      dataTable([
        { key: "fit", label: "fit", gloss: "mediation", sort: row => row.name,
          cell: row => term(fitDisplay(row.name).gloss, fitDisplay(row.name).label) },
        { key: "alpha", label: "α", gloss: "alpha", sort: row => row.block.treatment_on_mediator_coef,
          text: row => num(row.block.treatment_on_mediator_coef) },
        { key: "beta", label: "β", gloss: "beta", sort: row => row.block.mediator_on_outcome_coef,
          text: row => num(row.block.mediator_on_outcome_coef) },
        { key: "ind", label: "indirect", gloss: "indirect", sort: row => row.block.indirect_effect,
          text: row => num(row.block.indirect_effect) },
        { key: "indci", label: "indirect 95% CI", gloss: "boot_ci", sort: null,
          text: row => `[${num((row.block.bootstrap_ci_indirect || [])[0])}, `
            + `${num((row.block.bootstrap_ci_indirect || [])[1])}]` },
        { key: "dir", label: "direct", gloss: "direct", sort: row => row.block.direct_effect,
          text: row => num(row.block.direct_effect) },
        { key: "tot", label: "total", gloss: "total", sort: row => row.block.total_effect,
          text: row => num(row.block.total_effect) },
        { key: "totci", label: "total 95% CI", gloss: "boot_ci", sort: null,
          text: row => `[${num((row.block.bootstrap_ci_total || [])[0])}, `
            + `${num((row.block.bootstrap_ci_total || [])[1])}]` },
        { key: "prop", label: "prop. mediated", gloss: "prop_mediated", sort: null,
          text: row => row.block.proportion_mediated != null ? num(row.block.proportion_mediated)
            : "withheld (" + (row.block.proportion_mediated_reason || "—") + ")" },
        { key: "supp", label: "suppl. indirect", gloss: "supplementary_indirect",
          sort: row => row.block.supplementary_indirect_effect,
          text: row => num(row.block.supplementary_indirect_effect) },
        { key: "nobs", label: "n", gloss: "n_observations", sort: row => row.block.n_observations,
          text: row => String(row.block.n_observations || "—") },
      ], Object.entries(mediation).map(([name, block]) => ({ name, block }))),
      el("p", { class: "mini-note", style: "margin-top:8px" },
        term("estimator", "estimator"),
        el("span", { text: ": " + [...new Set(Object.values(mediation)
          .map(block => block.estimator).filter(Boolean))].join(" · ") }))));
  } else {
    container.append(el("div", { class: "card" },
      el("h2", { text: "Method B mediation" }),
      el("p", { class: "hint", text: "Not embedded — rerun with --analysis-directory pointing at run_analysis output." })));
  }

  if (DATA.method_a && DATA.method_a.length) {
    container.append(el("div", { class: "card" },
      el("h2", {}, term("method_a", "Fragmentation-matched counterfactual (Method A)")),
      el("p", { class: "hint", text: "Same word, same edit count, same position; only the tokenization consequence differs. Left dot = accuracy on the Low-fragmentation variant, darker right dot = High. A leftward High dot means fragmentation hurt. Cells are small by construction (the matched-variant requirement is strict), so read this as convergent evidence beside Method B, not a standalone test." }),
      methodADumbbells(DATA.method_a),
      dataTable([
        { key: "m", label: "model", gloss: "model", sort: row => row.model_id, text: row => shortModel(row.model_id) },
        { key: "fam", label: "family", gloss: "task_family", sort: row => row.task_family, text: row => row.task_family },
        { key: "kk", label: "k", gloss: "k", sort: row => row.r_edit_budget, text: row => String(row.r_edit_budget) },
        { key: "nn", label: "n", gloss: "n_pairs", sort: row => row.n, text: row => String(row.n) },
        { key: "lowacc", label: "Low-frag acc", gloss: "stratum", sort: row => row.clean_accuracy,
          text: row => pct(row.clean_accuracy) },
        { key: "highacc", label: "High-frag acc", gloss: "stratum", sort: row => row.perturbed_accuracy,
          text: row => pct(row.perturbed_accuracy) },
        { key: "d", label: "Δ", gloss: "delta", sort: row => row.delta, text: row => pp(row.delta) },
        { key: "dci", label: "95% CI", gloss: "delta_ci", sort: null,
          text: row => row.delta_ci_method === "insufficient_n" ? "n too small"
            : `[${pp(row.delta_ci_low)}, ${pp(row.delta_ci_high)}]` },
        { key: "mp", label: "McNemar p", gloss: "mcnemar", sort: row => row.mcnemar_p_value,
          text: row => pValue(row.mcnemar_p_value) + stars(row.mcnemar_p_value) },
      ], DATA.method_a)));
  }
}
/* =====================================================================
   TAB: Items — drill-down
   ===================================================================== */
const itemsState = { model: "", family: "", condition: "", budget: "", outcome: "",
                     stratum: "", status: "", search: "", visible: ITEMS_PAGE_SIZE };

function itemCell(item) { return DATA.cells[item[F.cell_index]]; }
function itemCleanEntry(item) {
  return DATA.clean_store[itemCell(item).model_id + "|" + item[F.task_id]] || ["", "", "", "", "", 0];
}

function itemMatches(item) {
  const cell = itemCell(item);
  const search = itemsState.search.toLowerCase();
  return (!itemsState.model || cell.model_id === itemsState.model)
    && (!itemsState.family || cell.task_family === itemsState.family)
    && (!itemsState.condition || conditionLabel(cell) === itemsState.condition)
    && (!itemsState.budget || String(cell.r_edit_budget) === itemsState.budget)
    && (!itemsState.outcome || OUTCOMES[itemsState.outcome].test(item))
    && (!itemsState.stratum || item[F.fragmentation_stratum] === itemsState.stratum)
    && (!itemsState.status || item[F.parse_status] === itemsState.status)
    && (!search
        || item[F.task_id].toLowerCase().includes(search)
        || itemCleanEntry(item)[1].toLowerCase().includes(search));
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

function itemBadge(glossKey, text, statusClass) {
  return attachGloss(el("span", { class: "badge " + (statusClass || ""), text }), glossKey);
}

function itemCard(item) {
  const cell = itemCell(item);
  const clean = itemCleanEntry(item);
  const card = el("div", { class: "item" },
    el("div", { class: "head" },
      el("span", { class: "tid", text: item[F.task_id] }),
      itemBadge("model", shortModel(cell.model_id)),
      itemBadge("condition", conditionLabel(cell) + " · k=" + cell.r_edit_budget),
      itemBadge("clean_accuracy", "clean " + (item[F.clean_ok] ? "✓" : "✗"), item[F.clean_ok] ? "ok" : "bad"),
      itemBadge("perturbed_accuracy", "perturbed " + (item[F.perturbed_ok] ? "✓" : "✗"),
                item[F.perturbed_ok] ? "ok" : "bad"),
      item[F.parse_status] && item[F.parse_status] !== "valid"
        ? itemBadge("parse_status", item[F.parse_status], "bad") : null,
      itemBadge("extraction_tier", item[F.extraction_tier]),
      item[F.finish_reason] === "length" ? itemBadge("finish_reason", "truncated", "bad") : null,
      item[F.fragmentation_stratum]
        ? itemBadge("stratum", item[F.fragmentation_stratum] + " frag") : null,
      item[F.token_inflation_ratio] != null
        ? attachGloss(el("span", { class: "mini-note term", text: "τ " + num(item[F.token_inflation_ratio], 3) }), "tau") : null,
      item[F.subword_count_change] != null
        ? attachGloss(el("span", { class: "mini-note term", text: "Δsub " + item[F.subword_count_change] }), "subword_change") : null,
      item[F.edited_word]
        ? attachGloss(el("span", { class: "mini-note term", text: "word: " + item[F.edited_word] }), "edited_word") : null),
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
  const reset = fn => value => { fn(value); itemsState.visible = ITEMS_PAGE_SIZE; rerender(); };
  const filters = el("div", { class: "filters" },
    makeSelect("model", "model", DATA.meta.models.map(model => model.model_id),
      reset(value => itemsState.model = value)),
    makeSelect("family", "task_family", uniqueSorted(DATA.cells.map(cell => cell.task_family)),
      reset(value => itemsState.family = value)),
    makeSelect("condition", "condition", [...CONDITION_SLOTS.keys()],
      reset(value => itemsState.condition = value)),
    makeSelect("k", "k", uniqueSorted(DATA.cells.map(cell => String(cell.r_edit_budget))),
      reset(value => itemsState.budget = value)),
    (() => {
      const select = el("select", { onchange: event => reset(value => itemsState.outcome = value)(event.target.value) },
        el("option", { value: "", text: "all outcomes" }),
        ...Object.entries(OUTCOMES).map(([key, outcome]) =>
          el("option", { value: key, text: outcome.label })));
      return el("span", {}, term("matched_pair", "outcome"), el("label", { text: " " }), select);
    })(),
    makeSelect("stratum", "stratum", ["Low", "High"],
      reset(value => itemsState.stratum = value), "any stratum"),
    makeSelect("parse status", "parse_status", ["valid", "unparseable", "clarification", "refusal"],
      reset(value => itemsState.status = value), "any parse status"),
    el("input", { type: "search", placeholder: "search task id / prompt text",
      oninput: event => reset(value => itemsState.search = value)(event.target.value) }));
  const selects = [...filters.querySelectorAll("select")];
  [itemsState.model, itemsState.family, itemsState.condition, itemsState.budget,
   itemsState.outcome, itemsState.stratum, itemsState.status]
    .forEach((value, index) => selects[index].value = value);
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
   TAB: Run & data — manifests, throughput, exclusions, figures, config
   ===================================================================== */
const FIGURE_CAPTIONS = {
  "figure_ccf_vs_edit_budget.png":
    "Clean-conditioned failure vs edit budget, one series per operation — the analysis pipeline's Figure 2 "
    + "(design/08 §8.7), embedded exactly as run_analysis wrote it.",
};

function renderRun(container) {
  const manifestEntries = Object.entries(DATA.manifests || {});
  if (manifestEntries.length) {
    const shardRows = manifestEntries.flatMap(([modelDirectory, manifest]) =>
      Object.entries(manifest.shard_statistics || {}).map(([shardId, stats]) =>
        ({ modelDirectory, shardId, stats })));
    container.append(el("div", { class: "card" },
      el("h2", {}, term("shard", "Generation throughput per shard")),
      el("p", { class: "hint", text: "Measured over each shard's actual wall clock (includes scheduling and I/O). These numbers calibrate the ×25 main-study forecast." }),
      dataTable([
        { key: "mdl", label: "model dir", gloss: "manifest", sort: row => row.modelDirectory,
          text: row => row.modelDirectory },
        { key: "shard", label: "shard", gloss: "shard", sort: row => row.shardId, text: row => row.shardId },
        { key: "rows", label: "rows", gloss: "rows", sort: row => row.stats.rows,
          text: row => row.stats.rows.toLocaleString() },
        { key: "wall", label: "wall", gloss: "wall_seconds", sort: row => row.stats.wall_seconds,
          text: row => Math.round(row.stats.wall_seconds).toLocaleString() + " s" },
        { key: "tps", label: "tok/s", gloss: "tok_per_s", sort: row => row.stats.output_tokens_per_second,
          text: row => Math.round(row.stats.output_tokens_per_second).toLocaleString() },
        { key: "rph", label: "rows/h", gloss: "rows_per_h", sort: row => row.stats.rows_per_hour,
          text: row => Math.round(row.stats.rows_per_hour).toLocaleString() },
      ], shardRows)));

    container.append(el("div", { class: "card" },
      el("h2", {}, term("manifest", "Shard manifests (raw)")),
      el("p", { class: "hint", text: "The runner's full provenance record per model, embedded verbatim: completed shards, token budgets, revisions, commits." }),
      ...manifestEntries.map(([modelDirectory, manifest]) => {
        const details = el("details", {}, el("summary", { text: modelDirectory }));
        details.addEventListener("toggle", () => {
          if (details.open && details.childElementCount === 1) {
            details.append(el("pre", { class: "raw", text: JSON.stringify(manifest, null, 2) }));
          }
        });
        return details;
      })));
  }

  const byCondition = new Map();
  for (const record of DATA.exclusions) {
    const key = record.condition + " · k=" + record.budget;
    byCondition.set(key, (byCondition.get(key) || 0) + record.count);
  }
  const conditionEntries = [...byCondition.entries()].sort((a, b) => b[1] - a[1]);
  const maximumCount = Math.max(1, ...conditionEntries.map(entry => entry[1]));
  container.append(el("div", { class: "card" },
    el("h2", {}, term("exclusions", "Exclusions by condition")),
    el("p", { class: "hint", text: "Items a condition could not be constructed for, deduplicated across resumes. Bars share one scale; hover a row for the exact count." }),
    el("table", { class: "data" },
      el("tbody", {}, ...conditionEntries.map(([label, count]) => {
        const barWidth = Math.max(4, Math.round(320 * count / maximumCount));
        const tr = el("tr", {},
          el("td", { text: label }),
          el("td", {}, el("svg:svg", { width: 330, height: 12 },
            el("svg:rect", { x: 0, y: 2, width: barWidth, height: 8, rx: 4, fill: "var(--s1)" }))),
          el("td", { text: String(count) }));
        attachGloss(tr, "exclusions", [{ value: String(count), name: label }]);
        return tr;
      })))));
  container.append(el("div", { class: "card" },
    el("h2", {}, term("exclusions", "Exclusion reasons")),
    dataTable([
      { key: "cond", label: "condition", gloss: "condition", sort: row => row.condition, text: row => row.condition },
      { key: "kb", label: "k", gloss: "k", sort: row => row.budget, text: row => String(row.budget) },
      { key: "why", label: "reason", gloss: "exclusions", sort: row => row.reason,
        cell: row => el("span", { style: "white-space:normal", text: row.reason }) },
      { key: "cnt", label: "count", gloss: "exclusions", sort: row => row.count, text: row => String(row.count) },
    ], DATA.exclusions)));

  const figureNames = Object.keys(DATA.figures || {});
  if (figureNames.length) {
    container.append(el("div", { class: "card" },
      el("h2", { text: "Analysis figures" }),
      el("p", { class: "hint", text: "Every PNG the analysis run wrote, embedded so this file is the complete record." }),
      ...figureNames.map(name => el("div", { class: "figure-block" },
        el("img", { src: "data:image/png;base64," + DATA.figures[name], alt: name }),
        el("div", { class: "caption", text: name + (FIGURE_CAPTIONS[name] ? " — " + FIGURE_CAPTIONS[name] : "") })))));
  }

  if (DATA.config_text) {
    const details = el("details", {}, el("summary", { text: DATA.config_name || "experiment config" }),
      el("pre", { class: "raw", text: DATA.config_text }));
    container.append(el("div", { class: "card" },
      el("h2", {}, term("config", "Experiment configuration (verbatim)")),
      details));
  }

  container.append(el("div", { class: "card" },
    el("h2", { text: "Provenance" }),
    el("table", { class: "data" }, el("tbody", {},
      el("tr", {}, el("td", { text: "sources" }),
        el("td", { style: "white-space:normal", text: DATA.meta.sources.join(", ") })),
      el("tr", {}, attachGloss(el("td", { class: "term", text: "code commits" }), "git_commit"),
        el("td", { text: DATA.meta.commits.join(", ") })),
      el("tr", {}, attachGloss(el("td", { class: "term", text: "seed" }), "seed"),
        el("td", { text: String(DATA.meta.seed) })),
      el("tr", {}, attachGloss(el("td", { class: "term", text: "bootstrap resamples" }), "bootstrap"),
        el("td", { text: String(DATA.meta.bootstrap_resamples) })),
      el("tr", {}, el("td", { text: "report generated" }),
        el("td", { text: GENERATED_AT }))))));
}

/* ---------- tab plumbing ---------- */
const TABS = [
  ["overview", "Overview", renderOverview],
  ["effects", "Effects", renderEffects],
  ["statistics", "Statistics", renderStatistics],
  ["items", "Items", renderItems],
  ["run", "Run & data", renderRun],
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
  // Items and Effects re-render on every activation so cross-tab drill-down
  // state (a clicked cell row or heatmap cell) is always reflected.
  const entry = sections[id];
  if (!entry.rendered || id === "items" || id === "effects") {
    entry.section.replaceChildren();
    entry.render(entry.section);
    entry.rendered = true;
  }
}

document.getElementById("header-sub").textContent =
  `${DATA.meta.models.map(model => shortModel(model.model_id)).join(", ")} · `
  + `${DATA.meta.row_count.toLocaleString()} rows · ${DATA.cells.length} cells · generated ${GENERATED_AT}`;
activateTab("overview");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
