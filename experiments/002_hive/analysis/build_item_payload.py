"""Build the per-item inspection payload embedded in the report's Items tab.

For every (benchmark, qid) item: the clean stem, gold answer, choices, per-model clean
accuracy, per-condition flip counts (n / kept / breaks / fixes over all model × seed
runs), keyboard fragility and churn, and up to four exemplar generations — breaks drawn
from distinct condition groups plus one fix — each carrying the perturbed question, the
paired clean completion tail, and the perturbed completion tail so a reviewer can read
exactly where the reasoning diverged.

Usage:
    python build_item_payload.py --instances ../hive_all_instances.jsonl \
        --output-directory outputs
"""

from __future__ import annotations

import argparse
import gzip
import json

from pathlib import Path

import pandas as pd

CLEAN_CONDITION = "clean"
FLIP_BREAK, FLIP_FIX, FLIP_SAME = "break", "fix", "same"
CONDITION_ORDER = [
    "clean_qfirst", "ctrl_option_perm",
    "spoken_casual", "spoken_formal", "spoken_recast", "spoken_reflow",
    "spoken_reflow_llama", "spoken_filler_stripped",
    "clean_fillers", "clean_numwords", "clean_nofunc", "clean_nocase", "clean_homophone",
    "kbd_neighbor", "kbd_random", "kbd_swap", "kbd_repeat", "kbd_fatfinger", "kbd_nospace",
]
KEYBOARD_CONDITIONS = {
    "kbd_neighbor", "kbd_random", "kbd_swap", "kbd_repeat", "kbd_fatfinger", "kbd_nospace"}
LLM_VOICE_PREFIX = "spoken_"
COMPLETION_TAIL_CHARACTERS = 500
EXEMPLAR_BUCKETS = ["keyboard_break", "voice_break", "deterministic_break", "fix"]

ITEM_FIELDS = [
    "benchmark", "qid", "stem", "gold", "choices",
    "clean_correct_by_model", "clean_n_by_model",
    "condition_stats", "keyboard_fragility", "keyboard_churn", "exemplar_indices",
]
EXEMPLAR_FIELDS = [
    "condition", "model", "seed", "flip", "meaning_kept",
    "perturbed_question", "clean_completion_tail", "perturbed_completion_tail",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def stream_instances(instances_path):
    opener = gzip.open if instances_path.suffix == ".gz" else open
    with opener(instances_path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def exemplar_bucket(condition, flip):
    if flip == FLIP_FIX:
        return "fix"
    if condition in KEYBOARD_CONDITIONS:
        return "keyboard_break"
    if condition.startswith(LLM_VOICE_PREFIX):
        return "voice_break"
    return "deterministic_break"


def select_exemplar_rows(slim):
    """Up to one flip row per (item, bucket), chosen deterministically by
    (condition, model, seed) sort order."""
    flips = slim[slim["flip"] != FLIP_SAME].copy()
    flips["bucket"] = [
        exemplar_bucket(condition, flip)
        for condition, flip in zip(flips["condition"], flips["flip"])
    ]
    return (flips.sort_values(["condition", "model", "seed"])
            .drop_duplicates(["benchmark", "qid", "bucket"]))


def condition_stats_by_item(slim):
    """{(benchmark, qid): {condition: [n, kept, breaks, fixes]}}"""
    perturbed = slim[slim["condition"] != CLEAN_CONDITION]
    grouped = perturbed.groupby(["benchmark", "qid", "condition"], observed=True).agg(
        n=("flip", "size"),
        kept=("meaning_kept", "sum"),
        breaks=("flip", lambda flips: int((flips == FLIP_BREAK).sum())),
        fixes=("flip", lambda flips: int((flips == FLIP_FIX).sum())),
    )
    stats = {}
    for (benchmark, qid, condition), row in grouped.iterrows():
        stats.setdefault((benchmark, qid), {})[condition] = [
            int(row["n"]), int(row["kept"]), int(row["breaks"]), int(row["fixes"])]
    return stats


def clean_accuracy_by_item(slim, model_order):
    clean = slim[slim["condition"] == CLEAN_CONDITION]
    grouped = clean.groupby(["benchmark", "qid", "model"], observed=True)["score"].agg(
        ["sum", "count"])
    correct, totals = {}, {}
    for (benchmark, qid, model), row in grouped.iterrows():
        model_index = model_order.index(model)
        correct.setdefault((benchmark, qid), [0] * len(model_order))[model_index] = int(row["sum"])
        totals.setdefault((benchmark, qid), [0] * len(model_order))[model_index] = int(row["count"])
    return correct, totals


def keyboard_fragility_and_churn(slim):
    keyboard = slim[slim["condition"].isin(KEYBOARD_CONDITIONS)]
    grouped = keyboard.groupby(["benchmark", "qid"], observed=True).agg(
        exposures=("flip", "size"),
        clean_correct=("clean_score", "sum"),
        breaks=("flip", lambda flips: int((flips == FLIP_BREAK).sum())),
        flipping=("flip", lambda flips: int((flips != FLIP_SAME).sum())),
    )
    return {
        (benchmark, qid): (
            round(row["breaks"] / row["clean_correct"], 4) if row["clean_correct"] else None,
            round(row["flipping"] / row["exposures"], 4),
        )
        for (benchmark, qid), row in grouped.iterrows()
    }


def completion_tail(completion):
    text = completion or ""
    return text[-COMPLETION_TAIL_CHARACTERS:] if len(text) > COMPLETION_TAIL_CHARACTERS else text


def harvest_raw_text(instances_path, exemplar_keys, clean_pair_keys):
    """One stream over the raw file collecting: per-item stem/gold/choices, the
    exemplar rows' question + completion tail, and the paired clean completion tails."""
    item_header, exemplar_text, clean_pair_text = {}, {}, {}
    for row in stream_instances(instances_path):
        item_key = (row["benchmark"], row["qid"])
        row_key = (row["benchmark"], row["qid"], row["model"], row["seed"], row["condition"])
        if row["condition"] == CLEAN_CONDITION:
            if item_key not in item_header:
                item_header[item_key] = (row["question"], row["gold"], row["choices"])
            pair_key = row_key[:4]
            if pair_key in clean_pair_keys and pair_key not in clean_pair_text:
                clean_pair_text[pair_key] = completion_tail(row["completion"])
        elif row_key in exemplar_keys:
            exemplar_text[row_key] = (row["question"], completion_tail(row["completion"]))
    return item_header, exemplar_text, clean_pair_text


def main():
    arguments = parse_arguments()
    outputs = arguments.output_directory
    slim = pd.read_parquet(outputs / "slim_instances.parquet")

    model_order = sorted(slim["model"].unique())
    exemplar_rows = select_exemplar_rows(slim)
    exemplar_keys = {
        (row.benchmark, row.qid, row.model, row.seed, row.condition)
        for row in exemplar_rows.itertuples()
    }
    clean_pair_keys = {key[:4] for key in exemplar_keys}

    item_header, exemplar_text, clean_pair_text = harvest_raw_text(
        arguments.instances, exemplar_keys, clean_pair_keys)
    stats = condition_stats_by_item(slim)
    clean_correct, clean_totals = clean_accuracy_by_item(slim, model_order)
    fragility_churn = keyboard_fragility_and_churn(slim)

    exemplars, exemplar_index_of = [], {}
    for row in exemplar_rows.itertuples():
        row_key = (row.benchmark, row.qid, row.model, row.seed, row.condition)
        harvested = exemplar_text.get(row_key)
        if harvested is None:
            continue
        question, perturbed_tail = harvested
        exemplar_index_of.setdefault((row.benchmark, row.qid), []).append(len(exemplars))
        exemplars.append([
            row.condition, row.model, row.seed, row.flip, bool(row.meaning_kept),
            question, clean_pair_text.get(row_key[:4], ""), perturbed_tail,
        ])

    items = []
    for (benchmark, qid), (stem, gold, choices) in sorted(item_header.items()):
        fragility, churn = fragility_churn.get((benchmark, qid), (None, None))
        condition_stats = stats.get((benchmark, qid), {})
        items.append([
            benchmark, qid, stem, gold, choices,
            clean_correct.get((benchmark, qid), [0] * len(model_order)),
            clean_totals.get((benchmark, qid), [0] * len(model_order)),
            [condition_stats.get(condition) for condition in CONDITION_ORDER],
            fragility, churn,
            exemplar_index_of.get((benchmark, qid), []),
        ])

    payload = {
        "condition_order": CONDITION_ORDER,
        "model_order": model_order,
        "item_fields": ITEM_FIELDS,
        "items": items,
        "exemplar_fields": EXEMPLAR_FIELDS,
        "exemplars": exemplars,
        "completion_tail_characters": COMPLETION_TAIL_CHARACTERS,
    }
    output_path = outputs / "item_payload.json"
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {output_path} ({output_path.stat().st_size:,} bytes, "
          f"{len(items)} items, {len(exemplars)} exemplars)")


if __name__ == "__main__":
    main()
