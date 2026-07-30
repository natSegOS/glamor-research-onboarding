"""Assemble the single-file HIVE re-analysis report from the analysis outputs.

Mirrors the experiment-001 rehearsal report architecture: a self-contained tabbed
dashboard (Overview / Integrity / Claims / Effects / Statistics / Items / Methods) with
an embedded JSON payload, a hover glossary on every technical term, sortable statistic
tables, and a per-item explorer covering all 4,176 items with break/fix exemplar
generations and clean↔perturbed word diffs.

Usage:
    python build_report.py --output-directory outputs --report ../report.html
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import pandas as pd

from report_charts import (
    diverging_break_fix_bars, diverging_cell_style, dot_with_interval_chart,
    dumbbell_chart, heatmap_table, sequential_cell_style,
)

BENCHMARK_ORDER = ["gsm8k", "gsm_symbolic", "gsm1k", "mmlu_pro", "truthfulqa", "humaneval"]
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K", "gsm_symbolic": "GSM-Symbolic", "gsm1k": "GSM1k",
    "mmlu_pro": "MMLU-Pro", "truthfulqa": "TruthfulQA", "humaneval": "HumanEval",
}
CONDITION_GROUPS = [
    ("Controls", ["clean_qfirst", "ctrl_option_perm"]),
    ("Voice — LLM-rewritten", ["spoken_casual", "spoken_formal", "spoken_recast",
                               "spoken_reflow", "spoken_reflow_llama", "spoken_filler_stripped"]),
    ("Voice — deterministic", ["clean_fillers", "clean_numwords", "clean_nofunc",
                               "clean_nocase", "clean_homophone"]),
    ("Keyboard", ["kbd_neighbor", "kbd_random", "kbd_swap", "kbd_repeat",
                  "kbd_fatfinger", "kbd_nospace"]),
]
CONDITION_ORDER = [condition for _, conditions in CONDITION_GROUPS for condition in conditions]
CONDITION_GROUP_OF = {
    condition: group for group, conditions in CONDITION_GROUPS for condition in conditions}
MODEL_LABELS = {
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "microsoft_phi-4": "phi-4 (14B)",
    "mistralai_Mistral-7B-Instruct-v0.3": "Mistral-7B",
}
DELTA_HEATMAP_FULL_SCALE_PP = 16.0
MODEL_HEATMAP_FULL_SCALE_PP = 12.0
ARTIFACT_CONDITIONS = ["clean_numwords", "spoken_recast", "spoken_casual"]
CLEAN_CONDITION = "clean"
PARTIAL_CELL_THRESHOLD = 0.5

VERDICT_CHIP = {
    "pass": '<span class="chip chip-pass">reproduces</span>',
    "revised": '<span class="chip chip-warn">revised</span>',
    "artifact": '<span class="chip chip-fail">artifact</span>',
    "defect": '<span class="chip chip-fail">defect</span>',
    "answered": '<span class="chip chip-info">answered</span>',
    "null": '<span class="chip chip-info">null result</span>',
}

GLOSSARY = {
    # ---- metrics & machinery ----
    "paired_delta": ["Paired delta",
        "mean(score − clean_score) over meaning-kept rows. Each perturbed generation is "
        "scored against the SAME (model, seed, item) under clean, so item selection by "
        "the guard cancels exactly. This is the primary effect measure in this report."],
    "unpaired_delta": ["Unpaired delta",
        "Accuracy of a condition's guard-surviving rows minus clean accuracy on ALL rows "
        "— the source README's method. Safe only when the guard keeps ~100% of items; "
        "otherwise it absorbs an item-selection term (see clean_numwords)."],
    "churn": ["Churn",
        "Share of perturbed rows whose 0/1 verdict differs from the same (model, seed, "
        "item) under clean: churn = break% + fix%. Measures instability; a mean accuracy "
        "delta is nearly blind to it because breaks and fixes largely cancel."],
    "break": ["Break",
        "A row where the clean run was scored correct and the perturbed run wrong "
        "(clean_score 1 → score 0), for the same model, seed, and item."],
    "fix": ["Fix",
        "A row where the clean run was scored wrong and the perturbed run right "
        "(clean_score 0 → score 1). Fixes are why mean deltas understate instability."],
    "net": ["Net damage",
        "fix% − break% in percentage points. Negative = the perturbation hurts on "
        "balance. This equals the paired accuracy delta computed over all rows."],
    "kept_rate": ["Kept rate",
        "Share of rows passing the routed meaning guard. Below ~90%, unpaired deltas "
        "become untrustworthy because the guard's item selection leaks into them."],
    "meaning_guard": ["Meaning guard",
        "Routed verdict on whether a perturbed stem still asks the same question: an LLM "
        "judge for LLM-rewritten conditions, a deterministic number/length validator for "
        "deterministic operators. The keyboard side checks numbers and length but NOT "
        "entities — 'June'→'Mune' passes."],
    "cluster_bootstrap": ["Item-clustered bootstrap",
        "All 95% CIs resample items — (benchmark, qid) clusters — with replacement, "
        "2,000 replicates. The 5 models × 5 seeds repeat the same items, so rows are not "
        "independent; resampling rows would give intervals that are far too tight."],
    "confidence_interval": ["95% CI",
        "Percentile interval from the item-clustered bootstrap. If it excludes 0, the "
        "effect survives item-to-item variation at the 5% level."],
    "mcnemar": ["McNemar test (naive)",
        "Sign test on breaks vs fixes among flipping rows, treating rows as independent "
        "— which they are not (25 runs share each item). Reported only as an optimistic "
        "floor; trust the clustered CI on net instead."],
    "noise_floor": ["Decode-noise floor",
        "How often the SAME untouched question flips verdict when re-generated under a "
        "different seed: for each (model, benchmark, item) clean-scored under ≥2 seeds, "
        "the share of seed pairs that disagree. Typo churn only demonstrates typo-caused "
        "instability to the extent it EXCEEDS this floor."],
    "excess_churn": ["Excess churn",
        "Keyboard churn on floor-measurable items minus the clean↔clean floor on those "
        "same items — the instability actually attributable to the typo rather than to "
        "sampling variance."],
    "overdispersion": ["Overdispersion index",
        "Variance of per-item break counts divided by the binomial expectation if every "
        "item shared one break probability. 1.0 = uniform coin-flip risk; 7.6 here = "
        "fragility is concentrated in specific items."],
    "fragility": ["Item fragility",
        "Per item: breaks ÷ clean-correct exposures across the six keyboard operators "
        "(up to 150 exposures: 6 operators × 5 models × 5 seeds). The probability a typo "
        "destroys this item's previously-correct answer."],
    "real_word_edit": ["Real-word edit",
        "An edited token (perturbed stem word-aligned against clean, punctuation "
        "stripped, lowercased) found in /usr/share/dict/words — e.g. 'form'→'dorm'. "
        "kbd_nospace merges words and is excluded from alignment."],
    "truncation": ["Generation-budget truncation",
        "Completions cut off at a token cap before the model finishes (they end "
        "mid-sentence). MMLU-Pro grading needs an extractable answer letter, so a "
        "truncated row scores 0 regardless of knowledge; GSM grading extracts bare "
        "numbers and partially rescues truncated rows."],
    "partial_cell": ["Partial cell",
        "A condition × benchmark cell with under half its expected rows. Only "
        "clean_qfirst is affected (25/4,100 HumanEval; 1,755/5,000 MMLU-Pro; 430/5,000 "
        "TruthfulQA): the fronting transform silently skips items it cannot parse. "
        "Daggered † in heatmaps; not interpretable as effects."],
    "contamination_control": ["Contamination control",
        "GSM-Symbolic and GSM1k are never-trained variants of GSM-style problems: an "
        "effect that reproduces there cannot be explained by test-set memorization."],
    "seed": ["Seed",
        "Run-level random seed (5 per model). Controls decoding sampling AND which ~200 "
        "items each run draws — seeds do not share an item pool, which limits cross-seed "
        "comparisons to the overlap."],
    "qid": ["Item (qid)",
        "Stable question identifier within a benchmark. The same qid links a perturbed "
        "row to its clean counterpart and is the clustering unit for every CI."],
    "exposure": ["Exposure",
        "One (model, seed, operator) generation of an item — the denominator unit for "
        "per-item fragility and churn."],
    "flip_same": ["Same",
        "The perturbed verdict equals the clean verdict for this (model, seed, item) — "
        "both right or both wrong."],
    # ---- conditions ----
    "clean_qfirst": ["clean_qfirst",
        "Control: the question sentence is moved in front of the context, wording "
        "otherwise unchanged. PARTIAL COVERAGE off-GSM (skips items it can't parse a "
        "question from) and visibly mangles HumanEval docstrings — treat only its GSM "
        "cells as meaningful."],
    "ctrl_option_perm": ["ctrl_option_perm",
        "Control (MCQ only): answer options are permuted and the gold label remapped by "
        "content. Tests position bias; ≈0 effect found."],
    "spoken_casual": ["spoken_casual",
        "LLM rewrite into casual spoken register ('so uh, okay, so like…'), content "
        "preserved. The flagship voice condition: −4 to −6 pp everywhere except "
        "MMLU-Pro, surviving the guard ~98% intact."],
    "spoken_formal": ["spoken_formal",
        "LLM rewrite into formal spoken register; numbers often become words. Mild "
        "penalty (−1 to −5 pp)."],
    "spoken_recast": ["spoken_recast",
        "LLM compresses the stem and is FREE TO REORDER clauses. The most damaging "
        "operator (−12.6 to −15.8 pp paired on GSM/HumanEval) and the meaning-riskiest "
        "(67–85% kept)."],
    "spoken_reflow": ["spoken_reflow",
        "LLM compresses the stem but must KEEP clause order. The reorder-vs-compress "
        "control for spoken_recast: much smaller penalty, so reordering—not "
        "compression—does the damage."],
    "spoken_reflow_llama": ["spoken_reflow_llama",
        "spoken_reflow but with a Llama rewriter instead of the default — a "
        "rewriter-robustness check. Similar to spoken_reflow except on HumanEval."],
    "spoken_filler_stripped": ["spoken_filler_stripped",
        "spoken_casual with the filler words then deterministically removed — isolates "
        "'spokenness' minus fillers. Penalty stays ≈ spoken_casual, so fillers are not "
        "the mechanism."],
    "clean_fillers": ["clean_fillers",
        "Deterministic: spoken-style filler words injected into the clean stem "
        "('um', 'like'). Small consistent penalty (−1.6 to −3.4 pp)."],
    "clean_numwords": ["clean_numwords",
        "Deterministic: digits written out as words (84 → eighty-four). Claimed as the "
        "suite's only gain; paired analysis shows ≈0 — the 'gain' was guard-induced item "
        "selection (validator keeps only 50–78% of items)."],
    "clean_nofunc": ["clean_nofunc",
        "Deterministic: function words (articles, auxiliaries) dropped — telegraphic "
        "style. ≈0 effect: models don't need function words."],
    "clean_nocase": ["clean_nocase",
        "Deterministic: all lowercase. Small penalty, largest on HumanEval (−4.5 pp) "
        "where identifiers carry case."],
    "clean_homophone": ["clean_homophone",
        "Deterministic: words replaced by homophones (their/there). ≈0 effect — models "
        "read through sound-alike spelling."],
    "kbd_neighbor": ["kbd_neighbor",
        "Keyboard: a letter replaced by a QWERTY-adjacent key ('form'→'forn'), one edit "
        "per ~5 eligible words, numbers never touched, seeded per item. The physically "
        "plausible typo."],
    "kbd_random": ["kbd_random",
        "Keyboard: a letter replaced by an ARBITRARY letter — the adjacency control for "
        "kbd_neighbor. Statistically indistinguishable from neighbor ⇒ keyboard "
        "geometry is irrelevant to the damage."],
    "kbd_swap": ["kbd_swap",
        "Keyboard: two adjacent characters transposed ('form'→'from' style). Mid-pack "
        "damage."],
    "kbd_repeat": ["kbd_repeat",
        "Keyboard: a character doubled ('form'→'forrm'). Mild."],
    "kbd_fatfinger": ["kbd_fatfinger",
        "Keyboard: an extra adjacent-key character inserted ('form'→'forrm'/'foirm'). "
        "Mild-to-mid."],
    "kbd_nospace": ["kbd_nospace",
        "Keyboard: a space deleted, merging two words ('the form'→'theform'). Weakest "
        "operator; word-merge makes it unalignable for token-level analyses."],
    # ---- benchmarks ----
    "GSM8K": ["GSM8K",
        "Grade-school arithmetic word problems, free-form numeric answer graded on the "
        "final number. Widely trained on — the memorization-prone member of the GSM "
        "trio."],
    "GSM-Symbolic": ["GSM-Symbolic",
        "GSM-style problems with symbolically regenerated numbers/entities; never "
        "trained on. Contamination control #1. NOTE: no cross-seed clean overlap exists "
        "here, so the decode floor is unmeasurable on this benchmark."],
    "GSM1k": ["GSM1k",
        "Freshly written GSM-difficulty problems (Scale AI); never trained on. "
        "Contamination control #2."],
    "MMLU-Pro": ["MMLU-Pro",
        "10-option multiple choice across domains, graded on an extracted answer "
        "letter. COMPROMISED in this run: 36–84% of clean completions are truncated "
        "before an answer letter appears and score 0. Do not cite MMLU-Pro cells."],
    "TruthfulQA": ["TruthfulQA",
        "Adversarial common-misconception MCQ. Highest decode-noise floor (19.8% — above "
        "its own typo churn) and lowest spoken-guard survival (60–75%); contributes "
        "little usable signal here."],
    "HumanEval": ["HumanEval",
        "Python function completion graded by executing tests (164 items vs 200 "
        "elsewhere). Lowest decode floor (2.7%) — churn here is nearly all real typo "
        "effect."],
    # ---- models ----
    "Llama-3.1-8B": ["Llama-3.1-8B-Instruct",
        "Meta, 8B instruct. Mid-pack churn (16.1%), floor 8.8%, worst net damage "
        "(−2.5 pp)."],
    "Qwen2.5-7B": ["Qwen2.5-7B-Instruct",
        "Alibaba, 7B instruct. Churn 15.2%, floor 9.1%."],
    "Qwen3-8B": ["Qwen3-8B",
        "Alibaba, 8B hybrid-reasoning instruct. Churn 14.3% over a LOW floor (7.4%) — "
        "the largest floor-adjusted (typo-attributable) instability, 7.9 pp."],
    "phi-4 (14B)": ["phi-4 (14B)",
        "Microsoft, 14B. Most stable (churn 11.8%, floor 7.3%) and least damaged (net "
        "−0.7 pp, CI crossing 0) — but also the most truncation-censored on MMLU-Pro "
        "(75% of clean completions) and GSM (78–90% missing ####), so its levels are "
        "least trustworthy."],
    "Mistral-7B": ["Mistral-7B-Instruct-v0.3",
        "Mistral, 7B instruct. Crowned 'least stable' by raw churn (18.3%) — but its "
        "clean↔clean floor is also highest (12.7%), so its excess churn (5.4 pp) is "
        "among the LOWEST. Its GSM accuracy (16–24%) is depressed by truncation "
        "(~26–30% of rows truncated-and-zero)."],
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def term(label, key=None):
    glossary_key = key or label
    if glossary_key not in GLOSSARY:
        return label
    return f'<span class="term" data-term="{glossary_key}">{label}</span>'


def condition_term(condition):
    return term(condition)


def benchmark_term(benchmark_key):
    return term(BENCHMARK_LABELS[benchmark_key], BENCHMARK_LABELS[benchmark_key])


def model_term(model_key):
    return term(MODEL_LABELS[model_key], MODEL_LABELS[model_key])


def load_analysis_outputs(outputs):
    read_json = lambda name: json.loads((outputs / name).read_text())
    read_csv = lambda name: pd.read_csv(outputs / name)
    slim = pd.read_parquet(outputs / "slim_instances.parquet",
                           columns=["benchmark", "condition", "score", "meaning_kept"])
    kept_accuracy = slim[slim.meaning_kept].groupby(
        ["benchmark", "condition"], observed=True).score.mean().unstack("condition")
    unpaired_deltas = kept_accuracy.drop(columns=[CLEAN_CONDITION]).sub(
        kept_accuracy[CLEAN_CONDITION], axis=0) * 100
    return {
        "integrity": read_json("integrity_report.json"),
        "deltas": read_csv("accuracy_deltas.csv"),
        "operator_flips": read_csv("keyboard_operator_flips.csv"),
        "operator_flips_kept": read_csv("keyboard_operator_flips_meaning_kept.csv"),
        "operator_by_benchmark": read_csv("keyboard_operator_by_benchmark_flips.csv"),
        "model_flips": read_csv("keyboard_model_flips.csv"),
        "model_by_operator": read_csv("keyboard_model_by_operator_flips.csv"),
        "random_vs_neighbor": read_csv("random_versus_neighbor.csv"),
        "noise_floor": read_json("clean_decode_noise_floor.json"),
        "concentration": read_json("item_break_concentration.json"),
        "real_word": read_csv("real_word_edit_breaks.csv"),
        "edit_features": read_csv("edit_feature_breaks.csv"),
        "model_condition": read_csv("model_condition_guarded_deltas.csv"),
        "clean_accuracy": read_csv("clean_accuracy_by_model_benchmark.csv"),
        "verification": read_json("headline_verification.json"),
        "truncation": read_json("truncation_audit.json"),
        "determinism": read_json("decode_determinism_audit.json"),
        "unpaired_deltas": unpaired_deltas,
        "item_payload_text": (outputs / "item_payload.json").read_text(),
    }


def partial_delta_cells(deltas):
    expected = deltas.groupby("benchmark")["rows"].max()
    partial = deltas[deltas["rows"] < expected.loc[deltas["benchmark"]].values
                     * PARTIAL_CELL_THRESHOLD]
    return {
        (row["condition"], BENCHMARK_LABELS[row["benchmark"]])
        for _, row in partial.iterrows()
    }


def figure(chart_html, caption):
    return f'<figure>{chart_html}<figcaption>{caption}</figcaption></figure>'


def card_block(title, hint, *body):
    hint_html = f'<p class="hint">{hint}</p>' if hint else ""
    return f'<div class="card"><h2>{title}</h2>{hint_html}{"".join(body)}</div>'


def plain_table(headers, rows, sortable=True):
    css = "data-table sortable" if sortable else "data-table"
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (f'<div class="table-scroll"><table class="{css}">'
            f'<thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>')


def format_ci(low, high):
    return f'<span class="ci">[{low:+.2f}, {high:+.2f}]</span>'


# --------------------------------------------------------------------------- overview

def overview_tab(data):
    integrity = data["integrity"]
    concentration = data["concentration"]
    floor = data["noise_floor"]
    tiles = [
        ("Scored instances", f"{integrity['total_rows']:,}", "one JSON row per generation"),
        ("Benchmarks", "6", "incl. 2 contamination controls"),
        ("Conditions", "19 + clean", "voice · keyboard · controls"),
        ("Models", "5", "7–14B instruct"),
        ("Seeds", "5", "decode + item sampling"),
        ("Items", "4,176", "all inspectable in the Items tab"),
        ("Exemplar generations", "11,785", "break/fix cases embedded"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>'
        for label, value, note in tiles)

    numwords = data["verification"]["clean_numwords"]["computed_per_benchmark_pp"]
    truncation = data["truncation"]["mmlu_pro_clean_per_model"]
    worst_no_marker = max(cell["no_answer_marker_pct"] for cell in truncation.values())
    cards = [
        ("pass", "Keyboard instability reproduces exactly",
         f"All six operator rows and all five model rows of the source README's paired "
         f"flip tables reproduce to within 0.1 pp, and every operator's "
         f"{term('break')} &gt; {term('fix')} asymmetry survives the "
         f"{term('item-clustered bootstrap', 'cluster_bootstrap')}."),
        ("artifact", "The one claimed gain is a selection artifact",
         f"“{term('clean_numwords')} is the only gain (+0.5 to +8.1)” only reproduces "
         f"under the {term('unpaired method', 'unpaired_delta')}, whose numbers this "
         f"report matches digit-for-digit (+0.46 to +8.08). Its "
         f"{term('meaning guard', 'meaning_guard')} drops 22–50% of items; "
         f"{term('paired on the survivors', 'paired_delta')} the effect is "
         f"{min(numwords.values()):+.1f} to {max(numwords.values()):+.1f} pp — zero."),
        ("revised", "Voice penalties are real but mis-stated",
         f"Paired-on-kept, {term('spoken_recast')} is −12.6 to −15.8 pp on GSM + "
         f"HumanEval (claimed −11 to −14 — an understatement), and "
         f"{term('spoken_casual')}'s “every benchmark” claim fails on "
         f"{term('MMLU-Pro')} (−1.4 pp). The −4 to −6 pp casual penalty holds on both "
         f"{term('contamination controls', 'contamination_control')}."),
        ("answered", "Open Q1: churn has a large decode-variance floor",
         f"Re-generating the SAME clean question under a different {term('seed')} "
         f"already flips {floor['overall']['clean_floor_pct']:.1f}% of verdicts vs "
         f"{floor['overall']['keyboard_churn_on_same_items_pct']:.1f}% keyboard "
         f"{term('churn')} on the same items. {term('TruthfulQA')} churn sits BELOW its "
         f"own {term('floor', 'noise_floor')}; GSM8K/HumanEval churn is 3–5× floor. "
         f"{term('Mistral-7B')}'s “least stable” crown is mostly baseline noise; "
         f"{term('Qwen3-8B')} has the largest {term('excess churn', 'excess_churn')}."),
        ("defect", "Two pipeline defects the aggregates hide",
         f"(1) {term('Token-cap truncation', 'truncation')}: up to "
         f"{worst_no_marker:.0f}% of clean {term('MMLU-Pro')} completions end "
         f"mid-sentence with no extractable answer and score 0 — MMLU-Pro is uncitable "
         f"from this run, and Mistral's GSM cells carry the same confound. (2) "
         f"{term('clean_qfirst')} is a {term('partial cell', 'partial_cell')} problem: "
         f"its −28 pp on HumanEval is 25 mangled prompts, not an effect."),
        ("null", "Open Q3: no adjacency effect, mechanism dead",
         f"{term('kbd_random')} vs {term('kbd_neighbor')} churn differs by +0.43 pp "
         f"[−0.14, +0.99] — not significant. The proposed mechanism fails directly: "
         f"neighbor and random substitutions produce "
         f"{term('real dictionary words', 'real_word_edit')} at the same rate (14.4% vs "
         f"13.4%). Real-word edits DO break ~2–3 pp more — equally for both operators."),
        ("answered", "Open Q2: churn is item-clustered, not diffuse",
         f"Per-item break counts are {concentration['overdispersion_index']:.1f}× "
         f"{term('overdispersed', 'overdispersion')}: the top decile of items carries "
         f"{concentration['share_of_breaks_in_top_decile_of_items_pct']:.0f}% of all "
         f"breaks and {concentration['share_of_items_with_zero_breaks_pct']:.0f}% of "
         f"items never break. Edit count and number-adjacency are nulls — "
         f"{term('fragility', 'fragility')} is a property of the item. Browse the "
         f"fragile tail in the Items tab."),
    ]
    cards_html = "".join(
        f'<div class="tcard"><div class="card-head">{VERDICT_CHIP[kind]}</div>'
        f'<h3>{title}</h3><p>{body}</p></div>'
        for kind, title, body in cards)
    reading_note = (
        '<p class="hint">Dotted-underlined terms carry hover definitions. Chart marks '
        'and heatmap cells carry hover explanations. Statistic tables sort on header '
        'click. The Items tab holds every one of the 4,176 items — clean stem, gold, '
        'per-condition flip counts, and embedded break/fix generations with '
        'clean↔perturbed diffs.</p>')
    return (f'<div class="tiles">{tiles_html}</div>{reading_note}'
            f'<div class="card-grid">{cards_html}</div>')


# --------------------------------------------------------------------------- integrity

def integrity_tab(data):
    integrity = data["integrity"]
    gates = [
        ("Row count matches README (550,370)", integrity["total_rows"] == 550370, ""),
        ("No duplicate (model, seed, benchmark, qid, condition) rows",
         integrity["duplicate_model_seed_benchmark_qid_condition_rows"] == 0, ""),
        ("flip field consistent with score/clean_score on all 550,370 rows",
         integrity["flip_field_mismatches"] == 0, ""),
        ("All clean rows have flip == same", integrity["clean_rows_with_non_same_flip"] == 0, ""),
        ("Clean stem text unique per (benchmark, qid) across models and seeds",
         integrity["clean_question_text_conflicts"] == 0, ""),
        ("Keyboard rows word-alignable to clean stem",
         integrity["keyboard_rows_with_alignable_edit_features"] >= 145000,
         f"{integrity['keyboard_rows_with_alignable_edit_features']:,} of "
         f"{integrity['keyboard_rows_total']:,} — the gap is kbd_nospace (merges words) "
         f"+ 25 stragglers"),
        ("Every condition × benchmark cell complete", False,
         "clean_qfirst is partial off-GSM — details below"),
    ]
    gate_rows = [
        (description,
         '<span class="chip chip-pass">pass</span>' if passed
         else '<span class="chip chip-fail">fail</span>', note)
        for description, passed, note in gates
    ]
    coverage_note = (
        f"<p>The failing gate: <strong>{term('clean_qfirst')} is silently partial "
        f"outside the GSM family</strong> — 25 of ~4,100 HumanEval rows, 1,755 of 5,000 "
        f"MMLU-Pro, 430 of 5,000 TruthfulQA (every other condition × benchmark cell is "
        f"complete; {term('ctrl_option_perm')} is MCQ-only by design). Every model × "
        f"seed cell contributes <em>some</em> rows, so this is not cluster preemption: "
        f"the question-fronting transform silently skips items it cannot parse a "
        f"question from. The surviving HumanEval prompts are visibly mangled (stems "
        f"beginning mid-string: <code>', '? You'll be given…</code>), and that 25-row "
        f"cell is the largest single delta in the suite (−28 pp) — a coverage hole "
        f"masquerading as an effect. {term('Partial cells', 'partial_cell')} are "
        f"daggered and uncolored in every heatmap. This reproduces, inside the export "
        f"itself, exactly the silent-partial-data failure mode the source README's "
        f"provenance note warns about.</p>")
    seed_note = (
        f"<p>One structural discovery: {term('seeds', 'seed')} do <em>not</em> share an "
        f"item pool. The union of clean stems is ~700 per benchmark while each cell "
        f"scores ~200, so each seed samples a different subset. Only 820 of 20,880 "
        f"(model, benchmark, item) triples are scored under all five seeds and "
        f"{term('GSM-Symbolic')} has no cross-seed overlap at all — which limits (but "
        f"does not block) the {term('decode-floor', 'noise_floor')} analysis in "
        f"Effects.</p>")

    truncation = data["truncation"]
    mmlu_rows = [
        (model_term(model), f"{cell['accuracy_pct']:.1f}%",
         f"{cell['no_answer_marker_pct']:.1f}%", f"{cell['marker_agrees_gold_pct']:.1f}%",
         cell["agrees_but_scored_0_rows"])
        for model, cell in truncation["mmlu_pro_clean_per_model"].items()
    ]
    gsm_rows = [
        (model_term(model) + " · " + benchmark_term(benchmark),
         f"{cell['accuracy_pct']:.1f}%", f"{cell['no_terminal_marker_pct']:.1f}%",
         f"{cell['no_marker_and_scored_0_pct']:.1f}%")
        for key, cell in truncation["gsm_clean_per_model_benchmark"].items()
        for model, benchmark in [key.split("::")]
    ]
    truncation_note = (
        f"<p>Clean MMLU-Pro completions routinely end mid-sentence (<code>at 10 "
        f"MP</code>, <code>F = ma</code>) before the model commits to a letter — a "
        f"token cap, not a character cap, so verbose or LaTeX-heavy reasoning is "
        f"censored hardest. {term('MMLU-Pro')} grading requires an extractable letter, "
        f"so a truncated row scores 0 regardless of knowledge; {term('phi-4 (14B)')} "
        f"reads 23% against ~70% in its model card. GSM grading is lenient (a bare "
        f"number suffices) and rescues most truncated rows — but {term('Mistral-7B')} "
        f"still has 26–30% of GSM rows truncated <em>and</em> scored 0. The "
        f"marker-agrees-gold-but-scored-0 column additionally suggests an extraction "
        f"bug in the grader worth inspecting. Fixes: re-run with an adequate budget or "
        f"a constrained answer format; at minimum re-grade with lenient extraction and "
        f"report per-cell truncation rates; export a per-row <code>finished</code> "
        f"flag.</p>")

    deltas = data["deltas"]
    kept = deltas.set_index(["condition", "benchmark"])["kept_rate"]
    kept_values = {
        condition: {
            BENCHMARK_LABELS[benchmark]: float(kept.loc[(condition, benchmark)])
            for benchmark in BENCHMARK_ORDER if (condition, benchmark) in kept.index
        }
        for condition in CONDITION_ORDER
    }
    kept_heatmap = heatmap_table(
        CONDITION_ORDER, [benchmark_term(benchmark) for benchmark in BENCHMARK_ORDER],
        {condition: {benchmark_term(benchmark): value
                     for benchmark in BENCHMARK_ORDER
                     if (value := kept_values[condition].get(BENCHMARK_LABELS[benchmark]))
                     is not None}
         for condition in CONDITION_ORDER},
        lambda value: sequential_cell_style(1 - value, 0.5),
        lambda value: f"{value:.0%}", row_group_of=CONDITION_GROUP_OF.get,
        partial_cells={(condition, benchmark_term(benchmark))
                       for condition, label in partial_delta_cells(deltas)
                       for benchmark in BENCHMARK_ORDER if BENCHMARK_LABELS[benchmark] == label},
        cell_title=lambda value, row, column:
            f"{value:.1%} of {row} rows passed the meaning guard on this benchmark",
        row_label_html=condition_term)
    guard_note = (
        f"<p>Guard coverage is the hinge of the voice corrections: wherever the "
        f"{term('kept rate', 'kept_rate')} dips below ~90%, "
        f"{term('unpaired deltas', 'unpaired_delta')} absorb an item-selection term. "
        f"Danger zones: {term('clean_numwords')} (50–78% kept), "
        f"{term('spoken_recast')} (67–85%), and everything TruthfulQA-spoken (60–75% — "
        f"adversarial phrasing rarely survives rewriting). {term('TruthfulQA')} "
        f"combines the lowest guard coverage with a {term('decode floor', 'noise_floor')} "
        f"above its churn, so it contributes almost no usable signal. The keyboard "
        f"conditions' guard checks numbers and length, not entities — a typo can "
        f"corrupt a name and still pass — so keyboard kept-rates are an upper bound on "
        f"true meaning preservation.</p>")

    determinism = data["determinism"]
    determinism_rows = [
        (model_term(model), f"{cell['multi_seed_triples']:,}",
         f"{cell['completion_text_differs_pct']:.1f}%", f"{cell['score_differs_pct']:.1f}%")
        for model, cell in determinism["per_model"].items()
    ]
    determinism_note = (
        f"<p>The export records no decoding parameters, so determinism was tested from "
        f"the data: for the same (model, benchmark, item) the clean prompt is "
        f"byte-identical across {term('seeds', 'seed')}, so greedy decoding would "
        f"reproduce the same completion every time. Measured on the "
        f"{determinism['overall']['multi_seed_triples']:,} triples scored under ≥2 "
        f"seeds, the completion <em>text</em> differs in "
        f"{determinism['overall']['completion_text_differs_pct']:.1f}% of them — "
        f"<strong>generation was sampled, not greedy</strong> — and that textual "
        f"variation flips the 0/1 score in "
        f"{determinism['overall']['score_differs_pct']:.1f}%. This is the direct "
        f"evidence behind the {term('decode-noise floor', 'noise_floor')} in Effects: "
        f"single-sample scoring under sampled decoding cannot separate a perturbation's "
        f"effect from resampling variance. Fix for the rerun: greedy decoding (isolates "
        f"the perturbation effect, at the cost of hiding realistic decode-time "
        f"instability) or, better, multiple samples per item at fixed temperature so "
        f"both the effect and the variance are estimated explicitly.</p>")
    return "".join([
        card_block("Data integrity gates",
                   "Structural checks recomputed from the raw instance file.",
                   plain_table(["Gate", "Status", "Note"], gate_rows, sortable=False),
                   coverage_note, seed_note),
        card_block("Decoding determinism audit",
                   "Same clean prompt, different seed — does the model say the same "
                   "thing twice?",
                   determinism_note,
                   plain_table(["Model", "multi-seed (model, benchmark, item) triples",
                                "completion text differs", "0/1 score flips"],
                               determinism_rows)),
        card_block("Generation-budget truncation audit",
                   "Share of CLEAN completions with no extractable answer, per model.",
                   truncation_note,
                   plain_table(["Model", "clean MMLU-Pro acc", "no answer marker",
                                "marker = gold", "marker = gold, scored 0 (rows)"],
                               mmlu_rows),
                   plain_table(["Model · benchmark", "clean acc", "no #### marker",
                                "no marker ∧ scored 0"], gsm_rows)),
        card_block("Meaning-guard accounting",
                   "Share of rows passing the routed guard (darker = more dropped; "
                   "hover any cell).",
                   kept_heatmap, guard_note),
    ])


# --------------------------------------------------------------------------- claims

def claims_tab(data):
    verification = data["verification"]
    operator_rows = [
        (condition_term(operator),
         f"{check['claimed']['break']:.2f} / {check['computed']['break']:.2f}",
         f"{check['claimed']['fix']:.2f} / {check['computed']['fix']:.2f}",
         f"{check['claimed']['net']:.2f} / {check['computed']['net']:.2f}",
         f"{check['claimed']['churn']:.2f} / {check['computed']['churn']:.2f}")
        for operator, check in verification["keyboard_operator_table"].items()
    ]
    model_rows = [
        (model_term(model),
         f"{check['claimed']['churn']:.2f} / {check['computed']['churn']:.2f}",
         f"{check['claimed']['net']:.2f} / {check['computed']['net']:.2f}")
        for model, check in verification["model_churn_table"].items()
    ]

    def claim_card(chip, claim, verdict_html):
        return (f'<div class="tcard"><div class="card-head">{VERDICT_CHIP[chip]}</div>'
                f'<h3>{claim}</h3>{verdict_html}</div>')

    casual = verification["spoken_casual"]["computed_per_benchmark_pp"]
    recast = verification["spoken_recast_gsm_and_humaneval"]["computed_pp"]
    numwords = verification["clean_numwords"]["computed_per_benchmark_pp"]
    per_benchmark_list = lambda values: ", ".join(
        f"{term(BENCHMARK_LABELS[benchmark], BENCHMARK_LABELS[benchmark])} "
        f"{delta:+.1f}" for benchmark, delta in values.items())
    claim_cards = "".join([
        claim_card("pass", "“12–17% of items change correctness while the mean moves "
                           "under 4 pp”",
                   f"<p>Reproduces: churn 12.3–17.5%, net −0.5 to −3.6 pp. But see the "
                   f"{term('decode floor', 'noise_floor')} reframing under Effects: "
                   f"~7–13 pp of that churn is sampling variance, not typo effect.</p>"),
        claim_card("artifact", "“clean+numwords is the only gain (+0.5 to +8.1)”",
                   f"<p>Matches the {term('unpaired', 'unpaired_delta')} computation "
                   f"(+0.46…+8.08) but not the {term('paired', 'paired_delta')} one: "
                   f"{per_benchmark_list(numwords)}. The gain is guard-induced item "
                   f"selection. Withdraw this claim.</p>"),
        claim_card("revised", "“spoken-casual costs 4.6–5.8 pp on every benchmark”",
                   f"<p>Paired: {per_benchmark_list(casual)}. Five of six benchmarks "
                   f"support −4 to −6; MMLU-Pro (−1.4) does not — and MMLU-Pro is "
                   f"compromised anyway. Restate as “4–6 pp on all interpretable "
                   f"benchmarks”.</p>"),
        claim_card("revised", "“spoken-recast: −11 to −14 on GSM family and HumanEval”",
                   f"<p>Paired: {per_benchmark_list(recast)} — the true effect is "
                   f"LARGER (−12.6 to −15.8). The claim understates because the guard "
                   f"drops the very items recast damages most.</p>"),
        claim_card("null", "“kbd_random is equal to or worse than kbd_neighbor on 4 of "
                           "6 benchmarks”",
                   f"<p>Recomputed: 3 of 6 by churn (two individually significant one "
                   f"way, one the other). Overall churn difference +0.43 pp [−0.14, "
                   f"+0.99]. Correct statement: no detectable adjacency advantage — "
                   f"directionally random ≥ neighbor, not established.</p>"),
        claim_card("pass", "“Churn tracks model, separately from damage”",
                   f"<p>Model table reproduces to 0.1 pp and the axes do separate — "
                   f"but the floor analysis shows the model ranking is largely a "
                   f"ranking of baseline decode noise; the typo-attributable ordering "
                   f"differs (Qwen3-8B worst, Mistral near best).</p>"),
        claim_card("revised", "“TruthfulQA is the most typo-sensitive benchmark”",
                   f"<p>Its churn (18.1%) sits BELOW its own clean↔clean floor "
                   f"(19.8%): the sensitivity is decode variance. Drop the label.</p>"),
    ])
    return "".join([
        card_block("Claim-by-claim verdicts",
                   "Every quantitative claim in the source README, recomputed from the "
                   "raw file.", f'<div class="card-grid">{claim_cards}</div>'),
        card_block("Keyboard operator table — claimed / recomputed",
                   "29,100 rows per operator; residual differences are rounding.",
                   plain_table(["Operator", "break%", "fix%", "net pp", "churn%"],
                               operator_rows)),
        card_block("Model churn table — claimed / recomputed", "",
                   plain_table(["Model", "churn%", "net pp"], model_rows)),
    ])


# --------------------------------------------------------------------------- effects

def effects_tab(data):
    deltas = data["deltas"]
    unpaired = data["unpaired_deltas"]
    guarded = deltas.set_index(["condition", "benchmark"])

    dumbbell_rows = []
    for condition in ARTIFACT_CONDITIONS:
        for benchmark in BENCHMARK_ORDER:
            if (condition, benchmark) not in guarded.index:
                continue
            row = guarded.loc[(condition, benchmark)]
            unpaired_value = float(unpaired.loc[benchmark, condition])
            paired_value = float(row["guarded_delta_pp"])
            dumbbell_rows.append({
                "label": f"{condition} · {BENCHMARK_LABELS[benchmark]}",
                "from_value": unpaired_value, "to_value": paired_value,
                "from_title": f"unpaired (README method): {unpaired_value:+.2f} pp",
                "to_title": (f"paired on kept items: {paired_value:+.2f} pp "
                             f"[{row['guarded_delta_ci_low_pp']:+.2f}, "
                             f"{row['guarded_delta_ci_high_pp']:+.2f}]"),
            })
    artifact_chart = dumbbell_chart(
        dumbbell_rows, -26, 10, [-25, -20, -15, -10, -5, 0, 5, 10],
        lambda tick: f"{tick:+d}", "unpaired (README)", "paired on kept items",
        zero_line=0.0, label_gutter=236)

    partial = partial_delta_cells(deltas)
    delta_values = {
        condition: {
            benchmark_term(benchmark): float(
                guarded.loc[(condition, benchmark), "guarded_delta_pp"])
            for benchmark in BENCHMARK_ORDER if (condition, benchmark) in guarded.index
        }
        for condition in CONDITION_ORDER
    }

    def delta_cell_title(value, condition, column):
        return (f"{condition}: paired meaning-guarded delta {value:+.2f} pp vs clean. "
                f"CI in the Statistics tab.")

    delta_heatmap = heatmap_table(
        CONDITION_ORDER, [benchmark_term(benchmark) for benchmark in BENCHMARK_ORDER],
        delta_values,
        lambda value: diverging_cell_style(value, DELTA_HEATMAP_FULL_SCALE_PP),
        lambda value: f"{value:+.1f}", row_group_of=CONDITION_GROUP_OF.get,
        partial_cells={(condition, benchmark_term(benchmark))
                       for condition, label in partial
                       for benchmark in BENCHMARK_ORDER if BENCHMARK_LABELS[benchmark] == label},
        cell_title=delta_cell_title, row_label_html=condition_term)

    model_condition = data["model_condition"].pivot(
        index="condition", columns="model", values="paired_delta")
    model_heatmap = heatmap_table(
        CONDITION_ORDER,
        [model_term(model) for model in MODEL_LABELS],
        {condition: {
            model_term(model): float(model_condition.loc[condition, model])
            for model in MODEL_LABELS if condition in model_condition.index
            and not pd.isna(model_condition.loc[condition, model])}
         for condition in CONDITION_ORDER},
        lambda value: diverging_cell_style(value, MODEL_HEATMAP_FULL_SCALE_PP),
        lambda value: f"{value:+.1f}", row_group_of=CONDITION_GROUP_OF.get,
        cell_title=lambda value, condition, column:
            f"{condition}: paired guarded delta {value:+.2f} pp, pooled over benchmarks",
        row_label_html=condition_term)

    numwords_kept = guarded.loc["clean_numwords"]["kept_rate"]
    voice_text = (
        f"<p><strong>{term('clean_numwords')} is the clearest casualty of the "
        f"{term('unpaired method', 'unpaired_delta')}.</strong> Its guard keeps only "
        f"{numwords_kept.min():.0%}–{numwords_kept.max():.0%} of items outside "
        f"TruthfulQA, and the claimed +0.5…+8.1 gain tracks the unpaired numbers almost "
        f"digit-for-digit. {term('Paired', 'paired_delta')}, the effect is −1.2…+1.3 pp: "
        f"<em>writing numbers as words does not help; dropping hard items did.</em> "
        f"{term('spoken_recast')} moves the other way — its paired penalty is larger "
        f"than claimed. {term('spoken_casual')} holds at −4 to −6 pp on five benchmarks "
        f"including both {term('contamination controls', 'contamination_control')}. The "
        f"{term('spoken_reflow')} / {term('spoken_recast')} contrast isolates "
        f"<em>clause reordering</em> as the damaging ingredient of the rewrite, and "
        f"{term('spoken_filler_stripped')} ≈ {term('spoken_casual')} rules fillers out "
        f"as the mechanism.</p>")

    operator_rows = [
        {"label": row["condition"], "break_pct": row["break_pct"], "fix_pct": row["fix_pct"],
         "churn_pct": row["churn_pct"]}
        for _, row in data["operator_flips"].sort_values(
            "churn_pct", ascending=False).iterrows()
    ]
    floor = data["noise_floor"]
    benchmark_rows = [
        {"label": BENCHMARK_LABELS[benchmark],
         "from_value": cell["clean_floor_pct"],
         "to_value": cell["keyboard_churn_on_same_items_pct"],
         "from_title": (f"clean↔clean disagreement across seeds: "
                        f"{cell['clean_floor_pct']:.1f}% ({cell['items_measured']} items)"),
         "to_title": (f"keyboard churn on the same items: "
                      f"{cell['keyboard_churn_on_same_items_pct']:.1f}%")}
        for benchmark, cell in floor["per_benchmark"].items()
    ]
    model_rows = [
        {"label": MODEL_LABELS[model],
         "from_value": cell["clean_floor_pct"],
         "to_value": cell["keyboard_churn_on_same_items_pct"],
         "from_title": f"clean↔clean floor: {cell['clean_floor_pct']:.1f}%",
         "to_title": f"keyboard churn, same items: {cell['keyboard_churn_on_same_items_pct']:.1f}%"}
        for model, cell in sorted(floor["per_model"].items(),
                                  key=lambda pair: -pair[1]["keyboard_churn_on_same_items_pct"])
    ]
    excess_rows = [
        (model_term(model), f"{cell['keyboard_churn_on_same_items_pct']:.1f}",
         f"{cell['clean_floor_pct']:.1f}",
         f"<strong>{cell['keyboard_churn_on_same_items_pct'] - cell['clean_floor_pct']:.1f}</strong>")
        for model, cell in sorted(
            floor["per_model"].items(),
            key=lambda pair: -(pair[1]["keyboard_churn_on_same_items_pct"]
                               - pair[1]["clean_floor_pct"]))
    ]
    floor_text = (
        f"<p><strong>Reframing the headline.</strong> “12–17% of items change "
        f"correctness while the mean moves under 4 pp” is true but conflates two "
        f"sources. These are single-sample runs: regenerating the SAME clean question "
        f"under another seed already flips {floor['overall']['clean_floor_pct']:.1f}% "
        f"of verdicts. The defensible claim is {term('excess churn', 'excess_churn')}: "
        f"typos add roughly 4–8 pp of flip probability over the resampling floor "
        f"(benchmark-dependent — {term('HumanEval')} 2.7→12.4, {term('GSM8K')} "
        f"4.9→17.2, {term('TruthfulQA')} 19.8→18.1, i.e. nothing). The floor-adjusted "
        f"model ranking also changes: {term('Qwen3-8B')} shows the largest "
        f"typo-attributable instability while {term('Mistral-7B')}'s near-symmetric "
        f"churn looks like baseline sampling noise — which also answers open question 4 "
        f"(“unstable yet undamaged”) without invoking decision boundaries. Caveats: "
        f"the floor is measurable on only 23% of triples, none on "
        f"{term('GSM-Symbolic')}; and cross-seed disagreement conflates decode variance "
        f"with any seed-linked harness variation. The clean-vs-clean rerun the README "
        f"proposes remains the right confirmatory experiment — these numbers predict "
        f"it will confirm a high floor.</p>")

    contrasts = data["random_vs_neighbor"]
    contrast_rows = [
        {"label": ("All benchmarks" if row["scope"] == "all_benchmarks"
                   else BENCHMARK_LABELS[row["scope"]]),
         "value": row["churn_random_minus_neighbor_pp"],
         "low": row["churn_difference_ci_low_pp"],
         "high": row["churn_difference_ci_high_pp"],
         "title": (f"churn(random) − churn(neighbor): "
                   f"{row['churn_random_minus_neighbor_pp']:+.2f} pp "
                   f"[{row['churn_difference_ci_low_pp']:+.2f}, "
                   f"{row['churn_difference_ci_high_pp']:+.2f}]")}
        for _, row in contrasts.iterrows()
    ]

    concentration = data["concentration"]
    real_word = data["real_word"]
    token_rates = real_word.groupby("condition").token_level_real_word_rate.first()
    real_word_rows = [
        {"label": (f"{row['condition']} · "
                   f"{'has real-word edit' if row['row_has_real_word_edit'] else 'non-words only'}"),
         "value": row["break_pct"], "low": row["break_ci_low_pct"],
         "high": row["break_ci_high_pct"],
         "title": (f"break rate {row['break_pct']:.1f}% "
                   f"[{row['break_ci_low_pct']:.1f}, {row['break_ci_high_pct']:.1f}] "
                   f"(n={row['clean_correct_rows']:,})")}
        for _, row in real_word.iterrows()
    ]
    anatomy_text = (
        f"<p><strong>Breaks cluster on fragile items.</strong> Among the "
        f"{concentration['items_with_at_least_10_clean_correct_exposures']:,} items "
        f"with ≥10 clean-correct {term('exposures', 'exposure')}, the top decile "
        f"carries {concentration['share_of_breaks_in_top_decile_of_items_pct']:.0f}% "
        f"of all breaks, {concentration['share_of_items_with_zero_breaks_pct']:.0f}% "
        f"never break, the median item breaks "
        f"{concentration['median_item_break_rate_pct']:.0f}% of the time, the 90th "
        f"percentile {concentration['p90_item_break_rate_pct']:.0f}% "
        f"({term('overdispersion', 'overdispersion')} "
        f"{concentration['overdispersion_index']:.1f}×). "
        f"<strong>The real-word mechanism fails, but real words do hurt:</strong> "
        f"{token_rates['kbd_neighbor']:.1%} of neighbor-edited tokens are dictionary "
        f"words vs {token_rates['kbd_random']:.1%} for random — no difference, so "
        f"lexicalization cannot explain any neighbor/random gap; yet rows containing a "
        f"{term('real-word edit', 'real_word_edit')} break 2–3 pp more for BOTH "
        f"operators. Edit-count and number-adjacency slices are flat (Statistics tab). "
        f"Inspect the fragile tail item-by-item in the Items tab.</p>")

    return "".join([
        card_block("Voice conditions — unpaired vs paired",
                   "Each gray→orange gap is pure item-selection bias from computing "
                   "deltas unpaired under an aggressive guard.",
                   voice_text,
                   figure(artifact_chart,
                          "Figure E1 — README-method (unpaired, gray) vs paired-on-kept "
                          "(orange) for the three conditions where the method changes "
                          "the story."),
                   figure(delta_heatmap,
                          "Figure E2 — Paired meaning-guarded delta vs clean, pp, 19 "
                          "conditions × 6 benchmarks (blue = gain, red = loss; † = "
                          "partial cell; hover cells). CIs in Statistics."),
                   figure(model_heatmap,
                          "Figure E3 — The same paired deltas per model, pooled over "
                          "benchmarks. The voice penalty is model-universal; keyboard "
                          "net damage concentrates in Qwen2.5/Llama; phi-4 is flattest "
                          "(but most truncation-censored).")),
        card_block("Keyboard instability against the decode-noise floor", "",
                   figure(diverging_break_fix_bars(operator_rows),
                          "Figure E4 — Paired break/fix rates per operator (29,100 rows "
                          "each), sorted by churn. Substitutions (random/neighbor) "
                          "damage most; space deletion least."),
                   figure(dumbbell_chart(benchmark_rows, 0, 22, [0, 5, 10, 15, 20], str,
                                         "clean↔clean floor", "keyboard churn, same items"),
                          "Figure E5 — Per benchmark: decode floor (gray) vs keyboard "
                          "churn on the same items (orange). GSM-Symbolic is absent — "
                          "no cross-seed clean overlap."),
                   figure(dumbbell_chart(model_rows, 0, 22, [0, 5, 10, 15, 20], str,
                                         "clean↔clean floor", "keyboard churn, same items"),
                          "Figure E6 — Per model. Mistral's chart-topping churn sits on "
                          "the highest baseline floor."),
                   plain_table(["Model", "churn% (same items)", "clean floor%",
                                "excess churn pp"], excess_rows),
                   floor_text),
        card_block("Adjacency: random vs neighbor",
                   "Paired per item within model × seed; a null result, not a reversal.",
                   figure(dot_with_interval_chart(contrast_rows, -4, 4, [-4, -2, 0, 2, 4],
                                                  lambda tick: f"{tick:+d}"),
                          "Figure E7 — Churn difference kbd_random − kbd_neighbor with "
                          "item-clustered 95% CIs.")),
        card_block("What churn is made of", "",
                   anatomy_text,
                   figure(dot_with_interval_chart(real_word_rows, 10, 22, [10, 14, 18, 22],
                                                  str, zero_line=None),
                          "Figure E8 — Break rate among clean-correct rows by whether "
                          "any edited token is a real dictionary word.")),
    ])


# --------------------------------------------------------------------------- statistics

def statistics_tab(data):
    deltas = data["deltas"].sort_values(["condition", "benchmark"])
    delta_rows = [
        (condition_term(row["condition"]), benchmark_term(row["benchmark"]),
         f"{row['rows']:,}", f"{row['kept_rate']:.1%}",
         f"{row['clean_accuracy_on_kept']:.1%}", f"{row['perturbed_accuracy_on_kept']:.1%}",
         f"{row['raw_delta_pp']:+.2f}", f"{row['guarded_delta_pp']:+.2f}",
         format_ci(row["guarded_delta_ci_low_pp"], row["guarded_delta_ci_high_pp"]))
        for _, row in deltas.iterrows()
    ]
    flip_headers = [term("scope", None) or "scope", "rows",
                    term("break%", "break"), term("fix%", "fix"),
                    term("net pp", "net"), "net 95% CI",
                    term("churn%", "churn"), "churn 95% CI",
                    term("McNemar p", "mcnemar")]

    def flip_rows(frame, label_of):
        return [
            (label_of(row), f"{row['rows']:,}",
             f"{row['break_pct']:.2f}", f"{row['fix_pct']:.2f}",
             f"{row['net_pp']:+.2f}", format_ci(row["net_ci_low_pp"], row["net_ci_high_pp"]),
             f"{row['churn_pct']:.2f}",
             format_ci(row["churn_ci_low_pct"], row["churn_ci_high_pct"]),
             f"{row['mcnemar_naive_p']:.2e}")
            for _, row in frame.iterrows()
        ]

    contrasts = data["random_vs_neighbor"]
    contrast_rows = [
        (("All benchmarks" if row["scope"] == "all_benchmarks"
          else benchmark_term(row["scope"])), f"{row['paired_rows']:,}",
         f"{row['churn_random_minus_neighbor_pp']:+.2f}",
         format_ci(row["churn_difference_ci_low_pp"], row["churn_difference_ci_high_pp"]),
         f"{row['net_random_minus_neighbor_pp']:+.2f}",
         format_ci(row["net_difference_ci_low_pp"], row["net_difference_ci_high_pp"]))
        for _, row in contrasts.iterrows()
    ]

    floor = data["noise_floor"]
    floor_rows = (
        [(model_term(model), f"{cell['clean_floor_pct']:.2f}", f"{cell['items_measured']:,}",
          f"{cell['keyboard_churn_on_same_items_pct']:.2f}",
          f"{cell['keyboard_churn_on_same_items_pct'] - cell['clean_floor_pct']:.2f}")
         for model, cell in floor["per_model"].items()] +
        [(benchmark_term(benchmark), f"{cell['clean_floor_pct']:.2f}",
          f"{cell['items_measured']:,}",
          f"{cell['keyboard_churn_on_same_items_pct']:.2f}",
          f"{cell['keyboard_churn_on_same_items_pct'] - cell['clean_floor_pct']:.2f}")
         for benchmark, cell in floor["per_benchmark"].items()])

    concentration_rows = [(key.replace("_", " "), f"{value:,.2f}" if isinstance(value, float)
                           else f"{value:,}")
                          for key, value in data["concentration"].items()]
    real_word_rows = [
        (condition_term(row["condition"]),
         f"{row['token_level_real_word_rate']:.1%}",
         "yes" if row["row_has_real_word_edit"] else "no",
         f"{row['share_of_clean_correct_rows']:.1%}", f"{row['clean_correct_rows']:,}",
         f"{row['break_pct']:.2f}",
         format_ci(row["break_ci_low_pct"], row["break_ci_high_pct"]))
        for _, row in data["real_word"].iterrows()
    ]
    edit_feature_rows = [
        (row["grouping"], row["group"], f"{row['clean_correct_rows']:,}",
         f"{row['break_pct']:.2f}",
         format_ci(row["break_ci_low_pct"], row["break_ci_high_pct"]))
        for _, row in data["edit_features"].iterrows()
    ]
    clean_accuracy_rows = [
        (model_term(row["model"]), benchmark_term(row["benchmark"]), f"{row['score']:.1f}%")
        for _, row in data["clean_accuracy"].iterrows()
    ]
    bootstrap_note = (
        f"<p class='hint'>Every CI is a {term('2,000-replicate item-clustered bootstrap', 'cluster_bootstrap')}; "
        f"{term('McNemar p-values', 'mcnemar')} treat rows as independent and are "
        f"reported only as optimistic floors. Click any header to sort.</p>")
    return "".join([
        bootstrap_note,
        card_block("Paired deltas — every condition × benchmark",
                   "The full table behind the Effects heatmap.",
                   plain_table(["condition", "benchmark", "rows", "kept",
                                "clean acc (kept)", "perturbed acc (kept)",
                                "raw Δpp", "guarded Δpp", "95% CI"], delta_rows)),
        card_block("Keyboard flips by operator — all rows",
                   "",
                   plain_table(flip_headers,
                               flip_rows(data["operator_flips"],
                                         lambda row: condition_term(row["condition"])))),
        card_block("Keyboard flips by operator — meaning-kept rows only",
                   "Sensitivity check: the guard changes nothing material.",
                   plain_table(flip_headers,
                               flip_rows(data["operator_flips_kept"],
                                         lambda row: condition_term(row["condition"])))),
        card_block("Keyboard flips by model", "",
                   plain_table(flip_headers,
                               flip_rows(data["model_flips"],
                                         lambda row: model_term(row["model"])))),
        card_block("Keyboard flips by model × operator", "",
                   plain_table(flip_headers,
                               flip_rows(data["model_by_operator"],
                                         lambda row: f"{model_term(row['model'])} · "
                                                     f"{condition_term(row['condition'])}"))),
        card_block("Keyboard flips by operator × benchmark", "",
                   plain_table(flip_headers,
                               flip_rows(data["operator_by_benchmark"],
                                         lambda row: f"{condition_term(row['condition'])} · "
                                                     f"{benchmark_term(row['benchmark'])}"))),
        card_block("Random − neighbor paired contrast", "",
                   plain_table(["scope", "paired rows", "Δchurn pp", "95% CI",
                                "Δnet pp", "95% CI"], contrast_rows)),
        card_block("Decode-noise floor vs churn",
                   f"{floor['model_benchmark_items_with_2plus_seeds']:,} of "
                   f"{floor['model_benchmark_items_total']:,} (model, benchmark, item) "
                   f"triples are floor-measurable; "
                   f"{floor['items_scored_on_all_five_seeds']} have all five seeds.",
                   plain_table(["scope", "clean floor %", "items", "churn % (same items)",
                                "excess pp"], floor_rows)),
        card_block("Item break concentration", "",
                   plain_table(["statistic", "value"], concentration_rows, sortable=False)),
        card_block("Real-word edit analysis", "",
                   plain_table(["operator", "token-level real-word rate",
                                "row has real-word edit", "share", "rows", "break %",
                                "95% CI"], real_word_rows)),
        card_block("Edit-feature slices", "",
                   plain_table(["feature", "level", "rows", "break %", "95% CI"],
                               edit_feature_rows)),
        card_block("Clean accuracy per model × benchmark",
                   "Absolute levels — read with the truncation audit in mind.",
                   plain_table(["model", "benchmark", "clean accuracy"],
                               clean_accuracy_rows)),
    ])


# --------------------------------------------------------------------------- items

def items_tab():
    return (
        '<p class="hint">Every one of the 4,176 items. Filter, sort, and click a row '
        'to open the full stem, gold answer, per-condition flip counts, and embedded '
        'break/fix generations (perturbed stem diffed against clean; completion tails '
        'clipped to 500 characters). Fragility = breaks ÷ clean-correct keyboard '
        'exposures; churn = flips ÷ all keyboard exposures.</p>'
        '<div class="filters" id="item-filters"></div>'
        '<div id="item-table"></div>')


# --------------------------------------------------------------------------- methods

def methods_tab(data):
    integrity = data["integrity"]
    definitions_rows = [
        (f'<span class="term" data-term="{key}">{title}</span>', description)
        for key, (title, description) in GLOSSARY.items()
    ]
    limitations = (
        "<p><strong>Limitations of this re-analysis.</strong> (i) CIs cluster items "
        "but not models/seeds — five models is too few to bootstrap, so cross-model "
        "generalization is descriptive. (ii) The decode-noise floor covers 23% of "
        "triples and no GSM-Symbolic; it conflates decode variance with any other "
        "seed-linked harness variation. (iii) The truncation audit detects answer "
        "markers by regex — a proxy for the actual grader. (iv) The meaning guard is "
        "inherited as-is; keyboard-side entity corruption passes it. (v) Exemplars in "
        "the Items tab are deterministic picks (first by condition/model/seed sort per "
        "bucket), not random samples. (vi) Completion tails are clipped to 500 "
        "characters; full text lives in the raw file.</p>")
    provenance = (
        f"<p><strong>Provenance.</strong> Input: "
        f"<code>experiments/002_hive/hive_all_instances.jsonl[.gz]</code> — every "
        f"pipeline script auto-detects gzip vs plain "
        f"({integrity['total_rows']:,} rows), exported from "
        f"<code>zizhao-hu/human-input-variations</code> · "
        f"<code>experiments/002_voice_variations</code>. Pipeline (all under "
        f"<code>experiments/002_hive/analysis/</code>): "
        f"<code>build_slim_dataset.py</code> → <code>run_hive_analysis.py</code> → "
        f"<code>audit_truncation.py</code> → <code>build_item_payload.py</code> → "
        f"<code>build_report.py</code>. Every table in this report is a CSV/JSON in "
        f"<code>analysis/outputs/</code>. Nothing committed.</p>")
    reproduction = (
        "<pre class='repro'>cd experiments/002_hive/analysis\n"
        "python build_slim_dataset.py  --instances ../hive_all_instances.jsonl --output-directory outputs\n"
        "python run_hive_analysis.py   --output-directory outputs\n"
        "python audit_truncation.py    --instances ../hive_all_instances.jsonl --output outputs/truncation_audit.json\n"
        "python audit_decode_determinism.py --instances ../hive_all_instances.jsonl --output outputs/decode_determinism_audit.json\n"
        "python build_item_payload.py  --instances ../hive_all_instances.jsonl --output-directory outputs\n"
        "python build_report.py        --output-directory outputs --report ../report.html</pre>")
    return "".join([
        card_block("How every number is computed", "", limitations, provenance,
                   reproduction),
        card_block("Glossary — every hover definition in one place", "",
                   plain_table(["Term", "Definition"], definitions_rows, sortable=False)),
    ])


# --------------------------------------------------------------------------- assembly

STYLESHEET = """
:root { color-scheme: light dark;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --negative:#e34948; --positive:#2a78d6;
  --chip-pass-bg:#e2f2e2; --chip-pass-ink:#006300;
  --chip-fail-bg:#fbe3e3; --chip-fail-ink:#a02020;
  --chip-warn-bg:#fdf0d8; --chip-warn-ink:#7a5300;
  --chip-info-bg:#e2ecfa; --chip-info-ink:#1c5cab;
  --diff-del-bg:#ffd7d7; --diff-del-ink:#7d0000;
  --diff-ins-bg:#cdf2cd; --diff-ins-ink:#005a00;
  --cell-ink-light:#0b0b0b; --cell-ink-dark:#ffffff;
}
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --negative:#e66767; --positive:#3987e5;
  --chip-pass-bg:#10360f; --chip-pass-ink:#7dd87d;
  --chip-fail-bg:#3d1414; --chip-fail-ink:#f09a9a;
  --chip-warn-bg:#3a2d0d; --chip-warn-ink:#ecc36b;
  --chip-info-bg:#12294a; --chip-info-ink:#8ab6ee;
  --diff-del-bg:#4e1f1f; --diff-del-ink:#ffb3b3;
  --diff-ins-bg:#1c3d1c; --diff-ins-ink:#a9e8a9;
}}
:root[data-theme="dark"] {
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --negative:#e66767; --positive:#3987e5;
  --chip-pass-bg:#10360f; --chip-pass-ink:#7dd87d;
  --chip-fail-bg:#3d1414; --chip-fail-ink:#f09a9a;
  --chip-warn-bg:#3a2d0d; --chip-warn-ink:#ecc36b;
  --chip-info-bg:#12294a; --chip-info-ink:#8ab6ee;
  --diff-del-bg:#4e1f1f; --diff-del-ink:#ffb3b3;
  --diff-ins-bg:#1c3d1c; --diff-ins-ink:#a9e8a9;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif; }
header { padding:18px 24px 0; }
header h1 { font-size:19px; margin:0 0 2px; }
header .sub { color:var(--muted); font-size:12px; }
nav.tabs { display:flex; gap:2px; padding:12px 24px 0; border-bottom:1px solid var(--grid);
  position:sticky; top:0; background:var(--page); z-index:20; overflow-x:auto; }
nav.tabs button { border:none; background:none; color:var(--ink-2); font:inherit;
  padding:8px 14px; cursor:pointer; border-bottom:2px solid transparent; white-space:nowrap; }
nav.tabs button.active { color:var(--ink); border-bottom-color:var(--series-1); font-weight:600; }
main { padding:18px 24px 60px; max-width:1400px; margin:0 auto; }
section.tab { display:none; } section.tab.active { display:block; }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:12px 16px; min-width:140px; }
.tile .label { color:var(--muted); font-size:12px; }
.tile .value { font-size:24px; font-weight:600; margin-top:2px; }
.tile .note { color:var(--ink-2); font-size:12px; margin-top:2px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:16px; margin-bottom:18px; overflow-x:auto; }
.card h2 { font-size:15px; margin:0 0 4px; }
.card .hint, p.hint { color:var(--muted); font-size:12.5px; margin:0 0 12px; max-width:960px; }
.card-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px;
  margin-bottom:18px; }
.tcard { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:14px 16px; }
.tcard h3 { font-size:14px; margin:8px 0 6px; }
.tcard p { color:var(--ink-2); font-size:13px; margin:0; }
p, li { font-size:13.5px; line-height:1.55; max-width:980px; }
.chip { border-radius:999px; padding:2px 10px; font-size:11.5px; font-weight:600; }
.chip-pass { background:var(--chip-pass-bg); color:var(--chip-pass-ink); }
.chip-fail { background:var(--chip-fail-bg); color:var(--chip-fail-ink); }
.chip-warn { background:var(--chip-warn-bg); color:var(--chip-warn-ink); }
.chip-info { background:var(--chip-info-bg); color:var(--chip-info-ink); }
.term { text-decoration:underline dotted var(--muted); text-underline-offset:3px; cursor:help; }
#tooltip { position:fixed; z-index:99; max-width:360px; background:var(--surface);
  border:1px solid var(--baseline); border-radius:8px; padding:10px 12px;
  box-shadow:0 4px 18px rgba(0,0,0,.18); pointer-events:none; }
#tooltip .tt { font-weight:600; font-size:13px; margin-bottom:4px; }
#tooltip .td { font-size:12.5px; color:var(--ink-2); }
figure { margin:16px 0; background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:14px; }
figcaption { font-size:12.5px; color:var(--ink-2); margin-top:8px; line-height:1.5; }
.table-scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:13px; margin:10px 0; }
th, td { padding:5px 10px; text-align:left; }
.data-table th { color:var(--muted); font-weight:500; font-size:12px;
  border-bottom:1px solid var(--baseline); white-space:nowrap; }
.data-table.sortable th { cursor:pointer; user-select:none; }
.data-table.sortable th:hover { color:var(--ink); }
.data-table td { border-bottom:1px solid var(--grid); font-variant-numeric:tabular-nums;
  white-space:nowrap; }
.ci { color:var(--muted); font-size:12px; }
.heatmap th { color:var(--ink-2); font-weight:600; font-size:12px; }
.heatmap-row-label { font-weight:500; font-size:12.5px; white-space:nowrap; }
.heatmap-cell { background:var(--cell-light); color:var(--cell-ink-light); text-align:right;
  font-variant-numeric:tabular-nums; font-size:12.5px; border:2px solid var(--surface);
  border-radius:4px; min-width:52px; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .heatmap-cell {
  background:var(--cell-dark); color:var(--cell-ink-dark); }}
:root[data-theme="dark"] .heatmap-cell { background:var(--cell-dark); color:var(--cell-ink-dark); }
.heatmap-missing, .heatmap-partial { background:transparent !important; color:var(--muted) !important; }
.heatmap-partial { border:2px dashed var(--grid); }
.heatmap-group td { color:var(--muted); font-size:11.5px; text-transform:uppercase;
  letter-spacing:.06em; padding-top:12px; }
.chart-label { fill:var(--ink); } .chart-muted { fill:var(--muted); }
.chart-baseline { stroke:var(--baseline); stroke-width:1; }
.chart-zeroline { stroke:var(--baseline); stroke-width:1; stroke-dasharray:3 3; }
.mark-negative { fill:var(--negative); } .mark-positive { fill:var(--positive); }
.mark-accent { fill:var(--series-2); stroke:var(--surface); stroke-width:2; }
.mark-reference { fill:var(--muted); stroke:var(--surface); stroke-width:2; }
.dumbbell-connector { stroke:var(--grid); stroke-width:2; }
.interval-whisker { stroke:var(--ink-2); stroke-width:2; stroke-linecap:round; }
.filters { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:10px 0 14px; }
.filters label { color:var(--muted); font-size:12px; }
.filters select, .filters input[type=search] { font:inherit; font-size:13px;
  background:var(--surface); color:var(--ink); border:1px solid var(--baseline);
  border-radius:6px; padding:5px 8px; }
.filters .count { color:var(--muted); font-size:12px; margin-left:auto; }
#item-table table { width:100%; }
#item-table tr.item-row { cursor:pointer; }
#item-table tr.item-row:hover td { background:color-mix(in srgb, var(--series-1) 8%, transparent); }
#item-table td.stem-preview { white-space:normal; max-width:520px; color:var(--ink-2); }
tr.detail-row > td { background:color-mix(in srgb, var(--series-1) 4%, transparent);
  white-space:normal; padding:14px 16px; }
.detail-stem { font-size:13.5px; background:var(--surface); border:1px solid var(--border);
  border-radius:8px; padding:10px 12px; margin:6px 0 10px; white-space:pre-wrap; max-width:none; }
.detail-meta { color:var(--ink-2); font-size:12.5px; margin-bottom:8px; }
.cond-table td, .cond-table th { font-size:12px; padding:3px 8px; }
.flipbar { display:inline-block; height:9px; border-radius:3px; vertical-align:middle; }
.flipbar.brk { background:var(--negative); } .flipbar.fx { background:var(--positive); }
.exemplar { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:10px 12px; margin:10px 0; }
.exemplar .ex-head { font-size:12.5px; color:var(--ink-2); margin-bottom:6px; }
.exemplar .badge { border-radius:999px; padding:1px 8px; font-size:11px; font-weight:600; }
.badge.brk { background:var(--chip-fail-bg); color:var(--chip-fail-ink); }
.badge.fx { background:var(--chip-pass-bg); color:var(--chip-pass-ink); }
.badge.guard { background:var(--chip-warn-bg); color:var(--chip-warn-ink); }
.diff { font-size:13px; line-height:1.6; white-space:pre-wrap; margin:6px 0; }
.diff del { background:var(--diff-del-bg); color:var(--diff-del-ink); text-decoration:none;
  border-radius:3px; padding:0 1px; }
.diff ins { background:var(--diff-ins-bg); color:var(--diff-ins-ink); text-decoration:none;
  border-radius:3px; padding:0 1px; }
.completions { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:10px; }
.completions h4 { font-size:12px; color:var(--muted); margin:0 0 4px; font-weight:600; }
.completions pre { font-size:11.5px; line-height:1.45; white-space:pre-wrap; margin:0;
  background:var(--page); border:1px solid var(--grid); border-radius:6px; padding:8px;
  max-height:260px; overflow-y:auto; }
.pager { display:flex; gap:8px; align-items:center; margin:12px 0; }
.pager button { font:inherit; font-size:12.5px; background:var(--surface); color:var(--ink);
  border:1px solid var(--baseline); border-radius:6px; padding:4px 10px; cursor:pointer; }
.pager button:disabled { opacity:.4; cursor:default; }
.repro { font-size:12px; background:var(--page); border:1px solid var(--grid);
  border-radius:6px; padding:10px; overflow-x:auto; }
footer { padding:0 24px 40px; color:var(--muted); font-size:12px; max-width:1400px; margin:0 auto; }
"""

JAVASCRIPT = r"""
const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);
const ITEMS_PAYLOAD = JSON.parse(document.getElementById("item-payload").textContent);
const GLOSSARY = PAYLOAD.glossary;
const MODEL_LABELS = PAYLOAD.model_labels;
const BENCHMARK_LABELS = PAYLOAD.benchmark_labels;
const CONDITION_GROUP_OF = PAYLOAD.condition_group_of;
const F = {}; ITEMS_PAYLOAD.item_fields.forEach((name, index) => F[name] = index);
const EX = {}; ITEMS_PAYLOAD.exemplar_fields.forEach((name, index) => EX[name] = index);
const ITEMS_PAGE_SIZE = 50;
const DIFF_WORD_CAP = 700;

/* ---------- tabs ---------- */
document.querySelectorAll("nav.tabs button").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("nav.tabs button").forEach(other =>
      other.classList.toggle("active", other === button));
    document.querySelectorAll("section.tab").forEach(sectionElement =>
      sectionElement.classList.toggle("active",
        sectionElement.id === "section-" + button.dataset.tab));
    if (button.dataset.tab === "items" && !itemsRendered) renderItems();
  });
});

/* ---------- glossary tooltip ---------- */
const tooltip = document.getElementById("tooltip");
document.addEventListener("mouseover", event => {
  const termElement = event.target.closest(".term");
  if (!termElement) { tooltip.hidden = true; return; }
  const entry = GLOSSARY[termElement.dataset.term];
  if (!entry) { tooltip.hidden = true; return; }
  tooltip.innerHTML = `<div class="tt">${entry[0]}</div><div class="td">${entry[1]}</div>`;
  tooltip.hidden = false;
  const rect = termElement.getBoundingClientRect();
  tooltip.style.left = Math.min(rect.left, window.innerWidth - 380) + "px";
  tooltip.style.top = (rect.bottom + 8 + 220 > window.innerHeight
    ? rect.top - tooltip.offsetHeight - 8 : rect.bottom + 8) + "px";
});
document.addEventListener("scroll", () => { tooltip.hidden = true; }, true);

/* ---------- sortable tables ---------- */
document.querySelectorAll("table.sortable").forEach(table => {
  table.querySelectorAll("thead th").forEach((header, columnIndex) => {
    header.addEventListener("click", () => {
      const body = table.querySelector("tbody");
      const rows = [...body.querySelectorAll("tr")];
      const ascending = header.dataset.sorted !== "asc";
      table.querySelectorAll("thead th").forEach(other => delete other.dataset.sorted);
      header.dataset.sorted = ascending ? "asc" : "desc";
      const numeric = text => {
        const match = text.replace(/[,%]/g, "").match(/-?\d+(\.\d+)?(e-?\d+)?/i);
        return match ? parseFloat(match[0]) : null;
      };
      rows.sort((rowA, rowB) => {
        const textA = rowA.cells[columnIndex]?.textContent.trim() ?? "";
        const textB = rowB.cells[columnIndex]?.textContent.trim() ?? "";
        const numberA = numeric(textA), numberB = numeric(textB);
        const comparison = (numberA !== null && numberB !== null)
          ? numberA - numberB : textA.localeCompare(textB);
        return ascending ? comparison : -comparison;
      });
      rows.forEach(row => body.append(row));
    });
  });
});

/* ---------- items tab ---------- */
let itemsRendered = false;
const itemState = { benchmark: "", search: "", sort: "fragility", breaksOnly: false, page: 0 };
const conditionStats = item => item[F.condition_stats];
const totalBreaks = item => conditionStats(item)
  .reduce((sum, stats) => sum + (stats ? stats[2] : 0), 0);
const cleanCorrect = item => item[F.clean_correct_by_model].reduce((a, b) => a + b, 0);
const cleanTotal = item => item[F.clean_n_by_model].reduce((a, b) => a + b, 0);
const worstCondition = item => {
  let best = null, bestRate = -1;
  ITEMS_PAYLOAD.condition_order.forEach((condition, index) => {
    const stats = conditionStats(item)[index];
    if (stats && stats[0] > 0 && stats[2] / stats[0] > bestRate) {
      bestRate = stats[2] / stats[0]; best = condition;
    }
  });
  return bestRate > 0 ? `${best} (${(100 * bestRate).toFixed(0)}%)` : "—";
};

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function wordDiff(cleanText, perturbedText) {
  const cleanWords = cleanText.split(/(\s+)/), perturbedWords = perturbedText.split(/(\s+)/);
  if (cleanWords.length > DIFF_WORD_CAP || perturbedWords.length > DIFF_WORD_CAP)
    return escapeHtml(perturbedText);
  const rows = cleanWords.length + 1, columns = perturbedWords.length + 1;
  const lcs = Array.from({ length: rows }, () => new Uint16Array(columns));
  for (let i = rows - 2; i >= 0; i--)
    for (let j = columns - 2; j >= 0; j--)
      lcs[i][j] = cleanWords[i] === perturbedWords[j]
        ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const parts = []; let i = 0, j = 0;
  while (i < cleanWords.length && j < perturbedWords.length) {
    if (cleanWords[i] === perturbedWords[j]) {
      parts.push(escapeHtml(cleanWords[i])); i++; j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      if (cleanWords[i].trim()) parts.push(`<del>${escapeHtml(cleanWords[i])}</del>`); i++;
    } else {
      parts.push(perturbedWords[j].trim()
        ? `<ins>${escapeHtml(perturbedWords[j])}</ins>` : escapeHtml(perturbedWords[j])); j++;
    }
  }
  while (i < cleanWords.length) { if (cleanWords[i].trim())
    parts.push(`<del>${escapeHtml(cleanWords[i])}</del>`); i++; }
  while (j < perturbedWords.length) { parts.push(perturbedWords[j].trim()
    ? `<ins>${escapeHtml(perturbedWords[j])}</ins>` : escapeHtml(perturbedWords[j])); j++; }
  return parts.join("");
}

function filteredItems() {
  const query = itemState.search.toLowerCase();
  let selection = ITEMS_PAYLOAD.items.filter(item =>
    (!itemState.benchmark || item[F.benchmark] === itemState.benchmark) &&
    (!itemState.breaksOnly || totalBreaks(item) > 0) &&
    (!query || item[F.qid].toLowerCase().includes(query) ||
      item[F.stem].toLowerCase().includes(query)));
  const keyOf = {
    fragility: item => -(item[F.keyboard_fragility] ?? -1),
    churn: item => -(item[F.keyboard_churn] ?? -1),
    breaks: item => -totalBreaks(item),
    clean_asc: item => cleanTotal(item) ? cleanCorrect(item) / cleanTotal(item) : 2,
    qid: item => item[F.qid],
  }[itemState.sort];
  selection.sort((itemA, itemB) => {
    const keyA = keyOf(itemA), keyB = keyOf(itemB);
    return keyA < keyB ? -1 : keyA > keyB ? 1 : 0;
  });
  return selection;
}

function renderItemFilters() {
  const container = document.getElementById("item-filters");
  const benchmarkOptions = ['<option value="">all benchmarks</option>']
    .concat(Object.keys(BENCHMARK_LABELS).map(key =>
      `<option value="${key}">${BENCHMARK_LABELS[key]}</option>`)).join("");
  container.innerHTML = `
    <label>benchmark <select id="flt-benchmark">${benchmarkOptions}</select></label>
    <label>sort <select id="flt-sort">
      <option value="fragility">keyboard fragility ↓</option>
      <option value="churn">keyboard churn ↓</option>
      <option value="breaks">total breaks ↓</option>
      <option value="clean_asc">clean accuracy ↑</option>
      <option value="qid">qid</option>
    </select></label>
    <label><input type="checkbox" id="flt-breaks"> only items with breaks</label>
    <input type="search" id="flt-search" placeholder="search qid or stem text…" size="28">
    <span class="count" id="flt-count"></span>`;
  document.getElementById("flt-benchmark").onchange = event => {
    itemState.benchmark = event.target.value; itemState.page = 0; renderItemTable(); };
  document.getElementById("flt-sort").onchange = event => {
    itemState.sort = event.target.value; itemState.page = 0; renderItemTable(); };
  document.getElementById("flt-breaks").onchange = event => {
    itemState.breaksOnly = event.target.checked; itemState.page = 0; renderItemTable(); };
  document.getElementById("flt-search").oninput = event => {
    itemState.search = event.target.value; itemState.page = 0; renderItemTable(); };
}

function conditionDetailTable(item) {
  const rows = ITEMS_PAYLOAD.condition_order.map((condition, index) => {
    const stats = conditionStats(item)[index];
    if (!stats) return "";
    const [n, kept, breaks, fixes] = stats;
    const breakWidth = n ? Math.round(60 * breaks / n) : 0;
    const fixWidth = n ? Math.round(60 * fixes / n) : 0;
    return `<tr><td><span class="term" data-term="${condition}">${condition}</span></td>
      <td>${n}</td><td>${kept}</td>
      <td>${breaks} <span class="flipbar brk" style="width:${breakWidth}px"></span></td>
      <td>${fixes} <span class="flipbar fx" style="width:${fixWidth}px"></span></td></tr>`;
  }).join("");
  return `<table class="data-table cond-table"><thead><tr>
    <th>condition</th><th>runs</th><th>kept</th><th>breaks</th><th>fixes</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

function exemplarBlock(item, exemplarIndex) {
  const exemplar = ITEMS_PAYLOAD.exemplars[exemplarIndex];
  const flip = exemplar[EX.flip];
  const badge = flip === "break" ? '<span class="badge brk">break</span>'
                                 : '<span class="badge fx">fix</span>';
  const guardBadge = exemplar[EX.meaning_kept] ? ""
    : ' <span class="badge guard">guard-dropped</span>';
  const modelLabel = MODEL_LABELS[exemplar[EX.model]] || exemplar[EX.model];
  const diffHtml = wordDiff(item[F.stem], exemplar[EX.perturbed_question]);
  const tailNote = `last ${ITEMS_PAYLOAD.completion_tail_characters} chars`;
  return `<div class="exemplar">
    <div class="ex-head">${badge}${guardBadge}
      <span class="term" data-term="${exemplar[EX.condition]}">${exemplar[EX.condition]}</span>
      · ${modelLabel} · ${exemplar[EX.seed]}</div>
    <div class="diff">${diffHtml}</div>
    <div class="completions">
      <div><h4>clean completion (${flip === "break" ? "scored ✓" : "scored ✗"}, ${tailNote})</h4>
        <pre>${escapeHtml(exemplar[EX.clean_completion_tail] || "(not captured)")}</pre></div>
      <div><h4>perturbed completion (${flip === "break" ? "scored ✗" : "scored ✓"}, ${tailNote})</h4>
        <pre>${escapeHtml(exemplar[EX.perturbed_completion_tail] || "(not captured)")}</pre></div>
    </div></div>`;
}

function itemDetail(item) {
  const models = ITEMS_PAYLOAD.model_order.map((model, index) =>
    `${MODEL_LABELS[model] || model} ${item[F.clean_correct_by_model][index]}/${item[F.clean_n_by_model][index]}`)
    .join(" · ");
  const choices = item[F.choices]
    ? "<ol type='A' style='margin:6px 0 10px'>" + item[F.choices].map(choice =>
        `<li>${escapeHtml(String(choice))}</li>`).join("") + "</ol>" : "";
  const exemplars = item[F.exemplar_indices].map(index =>
    exemplarBlock(item, index)).join("") ||
    "<p class='hint'>No flips anywhere for this item — no exemplars to show.</p>";
  return `
    <div class="detail-meta"><strong>gold:</strong> ${escapeHtml(String(item[F.gold]))}
      &nbsp;·&nbsp; <strong>clean per model:</strong> ${models}</div>
    <div class="detail-stem">${escapeHtml(item[F.stem])}</div>${choices}
    <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">
      <div>${conditionDetailTable(item)}</div>
      <div style="flex:1;min-width:340px">${exemplars}</div>
    </div>`;
}

function renderItemTable() {
  const selection = filteredItems();
  document.getElementById("flt-count").textContent =
    `${selection.length.toLocaleString()} items`;
  const start = itemState.page * ITEMS_PAGE_SIZE;
  const pageItems = selection.slice(start, start + ITEMS_PAGE_SIZE);
  const rows = pageItems.map((item, rowIndex) => {
    const fragility = item[F.keyboard_fragility];
    const churn = item[F.keyboard_churn];
    const accuracy = cleanTotal(item) ? cleanCorrect(item) + "/" + cleanTotal(item) : "—";
    return `<tr class="item-row" data-row="${rowIndex}">
      <td>${item[F.qid]}</td><td>${BENCHMARK_LABELS[item[F.benchmark]]}</td>
      <td class="stem-preview">${escapeHtml(item[F.stem].slice(0, 110))}${item[F.stem].length > 110 ? "…" : ""}</td>
      <td>${accuracy}</td>
      <td>${fragility === null ? "—" : (100 * fragility).toFixed(1) + "%"}</td>
      <td>${churn === null ? "—" : (100 * churn).toFixed(1) + "%"}</td>
      <td>${totalBreaks(item)}</td><td>${worstCondition(item)}</td></tr>`;
  }).join("");
  const pageCount = Math.max(1, Math.ceil(selection.length / ITEMS_PAGE_SIZE));
  document.getElementById("item-table").innerHTML = `
    <div class="table-scroll"><table class="data-table"><thead><tr>
      <th><span class="term" data-term="qid">qid</span></th><th>benchmark</th><th>stem</th>
      <th>clean ✓ (of 25)</th>
      <th><span class="term" data-term="fragility">fragility</span></th>
      <th><span class="term" data-term="churn">kbd churn</span></th>
      <th><span class="term" data-term="break">breaks</span></th>
      <th>worst condition</th></tr></thead>
      <tbody id="item-body">${rows}</tbody></table></div>
    <div class="pager">
      <button id="pg-prev" ${itemState.page === 0 ? "disabled" : ""}>← prev</button>
      <span class="count">page ${itemState.page + 1} / ${pageCount}</span>
      <button id="pg-next" ${itemState.page >= pageCount - 1 ? "disabled" : ""}>next →</button>
    </div>`;
  document.getElementById("pg-prev").onclick = () => { itemState.page--; renderItemTable(); };
  document.getElementById("pg-next").onclick = () => { itemState.page++; renderItemTable(); };
  document.querySelectorAll("#item-body tr.item-row").forEach(row => {
    row.addEventListener("click", () => {
      const existing = row.nextElementSibling;
      if (existing && existing.classList.contains("detail-row")) { existing.remove(); return; }
      const item = pageItems[parseInt(row.dataset.row)];
      const detail = document.createElement("tr");
      detail.className = "detail-row";
      detail.innerHTML = `<td colspan="8">${itemDetail(item)}</td>`;
      row.after(detail);
    });
  });
}

function renderItems() {
  itemsRendered = true;
  renderItemFilters();
  renderItemTable();
}
"""

TAB_DEFINITIONS = [
    ("overview", "Overview"),
    ("integrity", "Integrity & audits"),
    ("claims", "Claims check"),
    ("effects", "Effects"),
    ("statistics", "Statistics"),
    ("items", "Items"),
    ("methods", "Methods & glossary"),
]


def assemble_report(data):
    integrity = data["integrity"]
    sections = {
        "overview": overview_tab(data),
        "integrity": integrity_tab(data),
        "claims": claims_tab(data),
        "effects": effects_tab(data),
        "statistics": statistics_tab(data),
        "items": items_tab(),
        "methods": methods_tab(data),
    }
    nav = "".join(
        f'<button data-tab="{tab_id}"{" class=" + chr(34) + "active" + chr(34) if index == 0 else ""}>'
        f'{label}</button>'
        for index, (tab_id, label) in enumerate(TAB_DEFINITIONS))
    main = "".join(
        f'<section class="tab{" active" if index == 0 else ""}" id="section-{tab_id}">'
        f'{sections[tab_id]}</section>'
        for index, (tab_id, _) in enumerate(TAB_DEFINITIONS))
    small_payload = json.dumps({
        "glossary": GLOSSARY,
        "model_labels": MODEL_LABELS,
        "benchmark_labels": BENCHMARK_LABELS,
        "condition_group_of": CONDITION_GROUP_OF,
    }, separators=(",", ":")).replace("</", "<\\/")
    item_payload = data["item_payload_text"].replace("</", "<\\/")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Experiment 002 — HIVE re-analysis report</title>"
        f"<style>{STYLESHEET}</style></head><body>"
        "<header><h1>Experiment 002 — HIVE input-perturbation suite — independent "
        "re-analysis</h1>"
        f"<div class='sub'>{integrity['total_rows']:,} scored instances · 6 benchmarks "
        f"· 19 perturbations + clean · 5 models · 5 seeds · every README claim "
        f"recomputed from the raw export · hover any dotted term for its definition · "
        f"analysis date 2026-07-29 · uncommitted working analysis</div></header>"
        f"<nav class='tabs'>{nav}</nav>"
        f"<main>{main}</main>"
        "<div id='tooltip' hidden></div>"
        "<footer>Generated by experiments/002_hive/analysis/build_report.py from "
        "hive_all_instances.jsonl. Prepared in the glamor-research-onboarding repo; "
        "data provenance: zizhao-hu/human-input-variations.</footer>"
        f"<script id='payload' type='application/json'>{small_payload}</script>"
        f"<script id='item-payload' type='application/json'>{item_payload}</script>"
        f"<script>{JAVASCRIPT}</script></body></html>")


def main():
    arguments = parse_arguments()
    data = load_analysis_outputs(arguments.output_directory)
    arguments.report.write_text(assemble_report(data))
    print(f"wrote {arguments.report} ({arguments.report.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
