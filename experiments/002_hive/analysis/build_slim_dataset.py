"""Stream the raw HIVE instance file into slim parquet tables plus an integrity report.

Pass A collects the clean question text for every (benchmark, qid) and verifies it is
unique. Pass B emits one slim record per row (no question/completion text) and, for
keyboard-perturbed rows, the edit features obtained by diffing the perturbed question
against its clean counterpart: which tokens changed, whether the result is a real
dictionary word, where in the question the edit sits, and whether it neighbors a number.

Usage:
    python build_slim_dataset.py --instances ../hive_all_instances.jsonl.gz \
        --output-directory outputs
"""

from __future__ import annotations

import argparse
import gzip
import json
import re

from collections import Counter
from pathlib import Path

import pandas as pd

KEYBOARD_CONDITIONS = {
    "kbd_neighbor", "kbd_random", "kbd_swap", "kbd_repeat", "kbd_fatfinger", "kbd_nospace",
}
CLEAN_CONDITION = "clean"
FLIP_BREAK, FLIP_FIX, FLIP_SAME = "break", "fix", "same"
WORD_PATTERN = re.compile(r"\S+")
CONTAINS_DIGIT_PATTERN = re.compile(r"\d")
SYSTEM_DICTIONARY_PATH = Path("/usr/share/dict/words")

SLIM_COLUMNS = [
    "model", "seed", "benchmark", "qid", "condition",
    "score", "clean_score", "flip", "meaning_kept",
    "question_characters", "completion_characters",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def load_lowercase_dictionary():
    return {word.strip().lower() for word in SYSTEM_DICTIONARY_PATH.read_text().splitlines()}


def stream_instances(instances_path):
    opener = gzip.open if instances_path.suffix == ".gz" else open
    with opener(instances_path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def collect_clean_questions(instances_path):
    """Clean question text per (benchmark, qid), with a conflict count if any
    (benchmark, qid) ever shows two different clean stems."""
    clean_questions, conflicts = {}, 0
    for row in stream_instances(instances_path):
        if row["condition"] != CLEAN_CONDITION:
            continue
        key = (row["benchmark"], row["qid"])
        already_seen = clean_questions.get(key)
        if already_seen is None:
            clean_questions[key] = row["question"]
        elif already_seen != row["question"]:
            conflicts += 1
    return clean_questions, conflicts


def strip_word_punctuation(token):
    return re.sub(r"^\W+|\W+$", "", token)


def diff_keyboard_edit_features(clean_question, perturbed_question, dictionary):
    """Edit features for one keyboard-perturbed question, or None when the texts
    cannot be aligned word-for-word (kbd_nospace merges words, so counts differ)."""
    clean_words = WORD_PATTERN.findall(clean_question)
    perturbed_words = WORD_PATTERN.findall(perturbed_question)
    if len(clean_words) != len(perturbed_words) or not clean_words:
        return None

    edited_positions = [
        index for index, (clean_word, perturbed_word)
        in enumerate(zip(clean_words, perturbed_words))
        if clean_word != perturbed_word
    ]
    if not edited_positions:
        return None

    edited_core_words = [
        strip_word_punctuation(perturbed_words[index]).lower() for index in edited_positions
    ]
    neighbors_a_number = any(
        CONTAINS_DIGIT_PATTERN.search(clean_words[adjacent])
        for index in edited_positions
        for adjacent in (index - 1, index + 1)
        if 0 <= adjacent < len(clean_words)
    )
    return {
        "edit_count": len(edited_positions),
        "real_word_edit_count": sum(word in dictionary for word in edited_core_words if word),
        "any_edit_neighbors_a_number": neighbors_a_number,
    }


def build_slim_tables(instances_path, clean_questions, dictionary):
    slim_records, keyboard_edit_records = [], []
    flip_mismatches, clean_flip_violations = 0, 0
    for row in stream_instances(instances_path):
        condition = row["condition"]
        score, clean_score, flip = row["score"], row["clean_score"], row["flip"]

        expected_flip = (
            FLIP_BREAK if (clean_score == 1 and score == 0)
            else FLIP_FIX if (clean_score == 0 and score == 1)
            else FLIP_SAME
        )
        flip_mismatches += flip != expected_flip
        clean_flip_violations += condition == CLEAN_CONDITION and flip != FLIP_SAME

        slim_records.append((
            row["model"], row["seed"], row["benchmark"], row["qid"], condition,
            int(score), int(clean_score), flip, bool(row["meaning_kept"]),
            len(row["question"]), len(row["completion"] or ""),
        ))

        if condition in KEYBOARD_CONDITIONS:
            clean_question = clean_questions.get((row["benchmark"], row["qid"]))
            edit_features = (
                diff_keyboard_edit_features(clean_question, row["question"], dictionary)
                if clean_question is not None else None
            )
            if edit_features is not None:
                keyboard_edit_records.append({
                    "model": row["model"], "seed": row["seed"],
                    "benchmark": row["benchmark"], "qid": row["qid"],
                    "condition": condition, "score": int(score),
                    "clean_score": int(clean_score), "flip": flip,
                    **edit_features,
                })

    slim_frame = pd.DataFrame(slim_records, columns=SLIM_COLUMNS)
    return slim_frame, pd.DataFrame(keyboard_edit_records), flip_mismatches, clean_flip_violations


def summarize_integrity(slim_frame, flip_mismatches, clean_flip_violations, clean_conflicts,
                        keyboard_edit_rows):
    duplicate_cells = int(
        slim_frame.duplicated(["model", "seed", "benchmark", "qid", "condition"]).sum()
    )
    keyboard_rows = int(slim_frame["condition"].isin(KEYBOARD_CONDITIONS).sum())
    return {
        "total_rows": int(len(slim_frame)),
        "rows_per_benchmark": Counter(slim_frame["benchmark"]),
        "rows_per_model": Counter(slim_frame["model"]),
        "rows_per_condition": Counter(slim_frame["condition"]),
        "seeds": sorted(slim_frame["seed"].unique()),
        "duplicate_model_seed_benchmark_qid_condition_rows": duplicate_cells,
        "flip_field_mismatches": int(flip_mismatches),
        "clean_rows_with_non_same_flip": int(clean_flip_violations),
        "clean_question_text_conflicts": int(clean_conflicts),
        "keyboard_rows_total": keyboard_rows,
        "keyboard_rows_with_alignable_edit_features": int(keyboard_edit_rows),
    }


def main():
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    dictionary = load_lowercase_dictionary()
    clean_questions, clean_conflicts = collect_clean_questions(arguments.instances)
    print(f"pass A: {len(clean_questions)} clean stems, {clean_conflicts} conflicts")

    slim_frame, keyboard_frame, flip_mismatches, clean_flip_violations = build_slim_tables(
        arguments.instances, clean_questions, dictionary)

    slim_frame.to_parquet(arguments.output_directory / "slim_instances.parquet", index=False)
    keyboard_frame.to_parquet(
        arguments.output_directory / "keyboard_edit_features.parquet", index=False)

    integrity = summarize_integrity(
        slim_frame, flip_mismatches, clean_flip_violations, clean_conflicts, len(keyboard_frame))
    (arguments.output_directory / "integrity_report.json").write_text(
        json.dumps(integrity, indent=2, default=dict))
    print(json.dumps({key: value for key, value in integrity.items()
                      if not isinstance(value, (dict, Counter))}, indent=2, default=str))


if __name__ == "__main__":
    main()
