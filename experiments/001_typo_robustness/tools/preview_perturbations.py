"""Generate an HTML preview of every perturbation condition for PI/auditor review.

Runs entirely offline — no GPU, no HuggingFace access needed. Loads synthetic
reasoning items and the built-in demo MCQ items, applies every condition in the
config file (ASR conditions are skipped — they require pre-built audio items),
and writes a self-contained HTML file showing the original and perturbed text
side-by-side with changes highlighted.

Usage:

    python tools/preview_perturbations.py

    python tools/preview_perturbations.py \\
        --config configs/pilot.yaml \\
        --items 8 \\
        --seed 1729 \\
        --output data/audit/perturbation_preview.html

The output is a single self-contained HTML file (Bootstrap 5 CDN). Open it in
any browser; no server required.
"""

from __future__ import annotations

import argparse
import difflib
import html as html_module
import sys
from pathlib import Path

# Make src/ importable when run as `python tools/preview_perturbations.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enums import ConditionSource, SemanticClass
from pipeline.experiment import DatasetConfig, ExperimentConfiguration, load_task_items
from perturbation.engine import damerau_levenshtein_distance
from perturbation import PerturbationError
import regimes


# ---------------------------------------------------------------------------
# Inline character-level diff renderer
# ---------------------------------------------------------------------------

def _inline_diff_html(original: str, perturbed: str) -> str:
    """Return HTML with <del> / <ins> spans marking character-level changes."""
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
# Per-condition preview builder
# ---------------------------------------------------------------------------

def _build_condition_rows(
        condition,
        task_items: list,
        is_word,
        base_seed: int) -> list[dict]:
    """Try every (item, budget) pair for one condition. Returns a list of row
    dicts for successful perturbations (failures are silently skipped, matching
    the behavior of the main pipeline)."""
    rows: list[dict] = []

    for item in task_items:
        content_text = getattr(item, "content_text", None) or item.question_text
        scope_spans = getattr(item, "scope_spans", None)
        key_terms = list(getattr(item, "key_terms", []))

        for budget in condition.edit_budgets:
            item_seed = regimes.derived_seed(
                base_seed, condition.name, item.task_id, budget)
            try:
                if condition.semantic_class == SemanticClass.A:
                    perturbed_content, edits, metadata = regimes.make_regime_a_nonword_typo(
                        content_text, condition.operation, budget, item_seed, is_word,
                        selection_policy=condition.selection_policy,
                        scope=condition.scope,
                        scope_spans=scope_spans,
                        key_terms=key_terms,
                    )
                elif condition.semantic_class == SemanticClass.B:
                    perturbed_content, edits, metadata = regimes.make_regime_b_real_word_shift(
                        content_text, item_seed, is_word,
                        scope=condition.scope,
                        scope_spans=scope_spans,
                    )
                else:
                    continue
            except PerturbationError:
                continue

            dl_dist = damerau_levenshtein_distance(content_text, perturbed_content)
            changed_words = sorted({
                e.word_before for e in edits if e.word_before and e.word_after
            })

            rows.append({
                "task_id": item.task_id,
                "task_family": str(item.task_family),
                "budget": budget,
                "dl_distance": dl_dist,
                "original": content_text,
                "perturbed": perturbed_content,
                "diff_html": _inline_diff_html(content_text, perturbed_content),
                "words_changed": ", ".join(changed_words) if changed_words else "—",
                "edit_count": len(edits),
            })

    return rows


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_REGIME_BADGE = {
    "A": '<span class="badge bg-warning text-dark">Regime A — nonword typo</span>',
    "B": '<span class="badge bg-info text-dark">Regime B — real-word shift</span>',
    "C": '<span class="badge bg-secondary">Regime C — meaning change</span>',
}


def _condition_card_html(condition, rows: list[dict], index: int) -> str:
    regime = str(condition.semantic_class)
    badge = _REGIME_BADGE.get(regime, f'<span class="badge bg-dark">{regime}</span>')
    collapse_id = f"cond{index}"

    meta_items = [
        ("operation", str(condition.operation)),
        ("selection_policy", str(condition.selection_policy)),
        ("scope", str(condition.scope)),
        ("edit_budgets", ", ".join(str(b) for b in condition.edit_budgets)),
    ]
    meta_html = " &nbsp;·&nbsp; ".join(
        f'<span class="text-muted">{k}:</span> <code>{v}</code>'
        for k, v in meta_items
    )

    success_count = len(rows)

    if not rows:
        body_html = '<p class="text-muted p-3">No successful perturbations for this condition.</p>'
    else:
        table_rows = []
        for row in rows:
            budget_badge = f'<span class="badge bg-secondary rounded-pill">k={row["budget"]}</span>'
            dl_badge = f'<span class="badge bg-light text-dark border">DL={row["dl_distance"]}</span>'
            family_color = "primary" if "gsm" in row["task_family"] else "success"
            family_badge = (
                f'<span class="badge bg-{family_color} rounded-pill">'
                f'{row["task_family"].replace("_", " ")}</span>')

            table_rows.append(f"""
              <tr>
                <td class="align-top text-nowrap small text-muted">{html_module.escape(row["task_id"])}</td>
                <td class="align-top">{family_badge}</td>
                <td class="align-top">{budget_badge} {dl_badge}</td>
                <td class="align-top original-col font-monospace small"
                    style="white-space: pre-wrap; max-width:340px">{html_module.escape(row["original"])}</td>
                <td class="align-top perturbed-col font-monospace small"
                    style="white-space: pre-wrap; max-width:340px">{row["diff_html"]}</td>
                <td class="align-top small text-muted">{html_module.escape(row["words_changed"])}</td>
              </tr>""")

        rows_html = "\n".join(table_rows)
        body_html = f"""
        <div class="table-responsive">
          <table class="table table-sm table-hover align-middle mb-0">
            <thead class="table-light">
              <tr>
                <th>task_id</th>
                <th>type</th>
                <th>budget / DL</th>
                <th>original text</th>
                <th>perturbed text <small class="text-muted fw-normal">(changes highlighted)</small></th>
                <th>word(s) changed</th>
              </tr>
            </thead>
            <tbody>{rows_html}
            </tbody>
          </table>
        </div>"""

    return f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex align-items-center gap-2 py-2"
           style="cursor:pointer" data-bs-toggle="collapse"
           data-bs-target="#{collapse_id}" aria-expanded="true">
        <span class="fw-bold">{html_module.escape(condition.name)}</span>
        {badge}
        <span class="badge bg-light text-dark border ms-1">{success_count} examples</span>
        <span class="ms-auto small">{meta_html}</span>
        <span class="ms-2 text-muted">&#9660;</span>
      </div>
      <div class="collapse show" id="{collapse_id}">
        <div class="card-body p-0">{body_html}</div>
      </div>
    </div>"""


def _build_html(condition_cards: list[str], config_path: str,
                item_count: int, seed: int, skipped_conditions: list[str]) -> str:
    cards_html = "\n".join(condition_cards)

    skipped_note = ""
    if skipped_conditions:
        names = ", ".join(f"<code>{c}</code>" for c in skipped_conditions)
        skipped_note = (
            f'<div class="alert alert-secondary mb-3">'
            f'<strong>Skipped (ASR — requires pre-built audio items):</strong> {names}</div>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Perturbation Preview — GLAMOR Exp 001</title>
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
    .original-col {{ background: #fafafa; }}
    .perturbed-col {{ background: #f0fff0; }}
    body {{ font-size: 0.92rem; }}
  </style>
</head>
<body class="bg-light">
<div class="container-fluid py-4">

  <div class="mb-4">
    <h2 class="mb-1">Perturbation Preview — GLAMOR Lab Exp 001</h2>
    <p class="text-muted mb-1">
      Config: <code>{html_module.escape(config_path)}</code> &nbsp;·&nbsp;
      {item_count} items per task type &nbsp;·&nbsp;
      seed <code>{seed}</code> &nbsp;·&nbsp;
      Generated offline — no GPU or HuggingFace access required.
    </p>
    <p class="text-muted small">
      <span class="badge bg-warning text-dark">Regime A</span> nonword typo — intent preserved, answer should be unchanged.<br>
      <span class="badge bg-info text-dark">Regime B</span> real-word shift — different valid word, context recovers intent.<br>
      Changes: <del class="diff-del">deleted</del> &nbsp; <ins class="diff-ins">inserted</ins>
    </p>
  </div>

  {skipped_note}
  {cards_html}

</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
        integrity="sha384-YvpcrYf0tY3lHB60NNkmXc4s9bIOgUxi8T/jzmWLzEOA6DpPOHFPk+WRZ4M9wEMo"
        crossorigin="anonymous"></script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pilot.yaml"),
        help="experiment config YAML to preview (default: configs/pilot.yaml)")
    parser.add_argument(
        "--items", type=int, default=8,
        help="number of items per task type (default: 8)")
    parser.add_argument(
        "--seed", type=int, default=1729,
        help="random seed (default: 1729)")
    parser.add_argument(
        "--output", type=Path, default=Path("data/audit/perturbation_preview.html"),
        help="output HTML file (default: data/audit/perturbation_preview.html)")
    parser.add_argument(
        "--dictionary", type=Path, default=None,
        help="path to newline-delimited word list; defaults to the built-in demo list")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print(f"loading config:    {args.config}")
    configuration = ExperimentConfiguration.from_yaml(args.config)
    configuration.seed = args.seed
    # Force offline mode: override the config's dataset list with the synthetic
    # generator and the built-in demo MCQ, so this tool works before
    # build_task_items.py has been run (the main PI review use case).
    configuration.datasets = [
        DatasetConfig(key="gsm_symbolic_synthetic", item_count=args.items),
        DatasetConfig(key="mcq_demo"),
    ]
    configuration.asr_items_path = None

    print("loading word list ...")
    wordlist = regimes.load_wordlist(args.dictionary)
    is_word = regimes.make_is_word(wordlist)

    print(f"generating {args.items} items per task type (offline synthetic) ...")
    task_items = load_task_items(configuration)
    print(f"  {len(task_items)} total items")

    condition_cards: list[str] = []
    skipped: list[str] = []

    for index, condition in enumerate(configuration.conditions):
        if condition.source == ConditionSource.ASR:
            skipped.append(condition.name)
            print(f"  skip (ASR): {condition.name}")
            continue

        print(f"  building: {condition.name} ...")
        rows = _build_condition_rows(condition, task_items, is_word, args.seed)
        condition_cards.append(
            _condition_card_html(condition, rows, index))
        print(f"    {len(rows)} examples")

    print("rendering HTML ...")
    output_html = _build_html(
        condition_cards,
        config_path=str(args.config),
        item_count=args.items,
        seed=args.seed,
        skipped_conditions=skipped,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_html, encoding="utf-8")
    print(f"\ndone → {args.output}  ({args.output.stat().st_size // 1024} KB)")
    print("open in any browser — no server required")


if __name__ == "__main__":
    main()
