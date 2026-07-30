"""Test whether generation was deterministic (greedy) or sampled, from the data alone.

The export records no decoding parameters, but determinism is directly testable: for
the same (model, benchmark, item) the clean prompt is byte-identical across seeds, so
greedy decoding would reproduce the same completion text every time. This audit hashes
every clean completion and counts, over (model, benchmark, item) groups scored under
two or more seeds, how often the completion text differs across seeds — and how often
that textual difference flips the 0/1 score.

Usage:
    python audit_decode_determinism.py --instances ../hive_all_instances.jsonl \
        --output outputs/decode_determinism_audit.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json

from collections import defaultdict
from pathlib import Path

CLEAN_CONDITION = "clean"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def stream_instances(instances_path):
    opener = gzip.open if instances_path.suffix == ".gz" else open
    with opener(instances_path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def audit(instances_path):
    completions_by_triple = defaultdict(list)
    for row in stream_instances(instances_path):
        if row["condition"] != CLEAN_CONDITION:
            continue
        digest = hashlib.md5((row["completion"] or "").encode()).hexdigest()
        completions_by_triple[(row["model"], row["benchmark"], row["qid"])].append(
            (digest, row["score"]))

    multi_seed = {key: runs for key, runs in completions_by_triple.items() if len(runs) >= 2}
    per_model = defaultdict(lambda: {"triples": 0, "text_varies": 0, "score_varies": 0})
    for (model, _, _), runs in multi_seed.items():
        cell = per_model[model]
        cell["triples"] += 1
        cell["text_varies"] += len({digest for digest, _ in runs}) > 1
        cell["score_varies"] += len({score for _, score in runs}) > 1

    def rates(cell):
        return {
            "multi_seed_triples": cell["triples"],
            "completion_text_differs_pct": round(
                cell["text_varies"] / cell["triples"] * 100, 1),
            "score_differs_pct": round(cell["score_varies"] / cell["triples"] * 100, 1),
        }

    overall = {
        "triples": sum(cell["triples"] for cell in per_model.values()),
        "text_varies": sum(cell["text_varies"] for cell in per_model.values()),
        "score_varies": sum(cell["score_varies"] for cell in per_model.values()),
    }
    return {
        "note": "Same model, same benchmark, same item, identical clean prompt, "
                "different seed. Greedy decoding would make completion text identical.",
        "overall": rates(overall),
        "per_model": {model: rates(cell) for model, cell in sorted(per_model.items())},
    }


def main():
    arguments = parse_arguments()
    arguments.output.write_text(json.dumps(audit(arguments.instances), indent=2))
    print(arguments.output.read_text())


if __name__ == "__main__":
    main()
