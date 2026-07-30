"""Quantify the generation-budget truncation confound and write it to JSON.

MMLU-Pro is graded by extracting an answer letter; a completion truncated before the
model commits to a letter scores 0 regardless of knowledge. The GSM family is graded
leniently (numbers are extractable mid-solution), so truncation there is partially
rescued. This audit measures, per model: the share of clean MMLU-Pro completions with
no extractable answer marker, the share whose extracted letter matches gold but scored
0, and the share of clean GSM rows missing the #### terminal marker and scored 0.

Usage:
    python audit_truncation.py --instances ../hive_all_instances.jsonl.gz \
        --output outputs/truncation_audit.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re

from collections import Counter, defaultdict
from pathlib import Path

MULTIPLE_CHOICE_ANSWER_PATTERN = re.compile(r"[Aa]nswer\s*(?:is)?[:\s]*\**\s*\(?([A-J])\)?\b")
GSM_TERMINAL_MARKER = "####"
GSM_BENCHMARKS = {"gsm8k", "gsm_symbolic", "gsm1k"}
MMLU_PRO_BENCHMARK = "mmlu_pro"
CLEAN_CONDITION = "clean"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def audit(instances_path):
    mmlu_counters = defaultdict(Counter)
    gsm_counters = defaultdict(Counter)
    opener = gzip.open if instances_path.suffix == ".gz" else open
    with opener(instances_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["condition"] != CLEAN_CONDITION:
                continue
            completion = row["completion"] or ""
            if row["benchmark"] == MMLU_PRO_BENCHMARK:
                cell = mmlu_counters[row["model"]]
                cell["rows"] += 1
                cell["scored_1"] += row["score"] == 1
                marker_matches = MULTIPLE_CHOICE_ANSWER_PATTERN.findall(completion)
                if not marker_matches:
                    cell["no_answer_marker"] += 1
                elif marker_matches[-1] == row["gold"]:
                    cell["marker_agrees_gold"] += 1
                    cell["agrees_but_scored_0"] += row["score"] == 0
            elif row["benchmark"] in GSM_BENCHMARKS:
                cell = gsm_counters[(row["model"], row["benchmark"])]
                cell["rows"] += 1
                cell["scored_1"] += row["score"] == 1
                missing_marker = GSM_TERMINAL_MARKER not in completion
                cell["no_terminal_marker"] += missing_marker
                cell["no_marker_and_scored_0"] += missing_marker and row["score"] == 0

    def rate(cell, key):
        return round(cell[key] / cell["rows"] * 100, 1)

    return {
        "mmlu_pro_clean_per_model": {
            model: {
                "rows": cell["rows"],
                "accuracy_pct": rate(cell, "scored_1"),
                "no_answer_marker_pct": rate(cell, "no_answer_marker"),
                "marker_agrees_gold_pct": rate(cell, "marker_agrees_gold"),
                "agrees_but_scored_0_rows": cell["agrees_but_scored_0"],
            }
            for model, cell in sorted(mmlu_counters.items())
        },
        "gsm_clean_per_model_benchmark": {
            f"{model}::{benchmark}": {
                "rows": cell["rows"],
                "accuracy_pct": rate(cell, "scored_1"),
                "no_terminal_marker_pct": rate(cell, "no_terminal_marker"),
                "no_marker_and_scored_0_pct": rate(cell, "no_marker_and_scored_0"),
            }
            for (model, benchmark), cell in sorted(gsm_counters.items())
        },
    }


def main():
    arguments = parse_arguments()
    arguments.output.write_text(json.dumps(audit(arguments.instances), indent=2))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
