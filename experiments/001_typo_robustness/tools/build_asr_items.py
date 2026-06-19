"""Pre-build the ASR perturbation set: TTS -> (noise) -> Whisper -> AsrItems.

This is a one-time pre-processing step (design/07 §7.4). It runs the ASR pipeline
over the task items, caches the audio on disk so it can be released with the
paper, and writes an AsrItems JSONL that tools/run_generation.py consumes via the
config's asr_items_path. Whisper runs deterministically (scalar temperature 0,
no cross-segment conditioning); degenerate transcriptions are flagged and, by
default, dropped (docs/PROVENANCE.md §3.1).

Usage (after `pip install -r requirements-gpu.txt`):

    python tools/build_asr_items.py \\
        --output data/perturbations/asr_items.jsonl \\
        --audio-directory data/audio \\
        --reasoning-items 600 --multiple-choice-items 600
"""

from __future__ import annotations

import argparse
import json

from dataclasses import asdict
from pathlib import Path

from asr import build_asr_items
from regimes import make_is_word
from tasks import make_demonstration_multiple_choice_items
from tasks import generate_synthetic_reasoning_items


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-directory", required=True, type=Path)
    parser.add_argument("--reasoning-items", type=int, default=600)
    parser.add_argument("--multiple-choice-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--keep-degenerate", action="store_true",
                        help="keep (still-flagged) degenerate transcriptions instead of dropping them")
    parser.add_argument("--dictionary", type=Path, default=None)
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    task_items = generate_synthetic_reasoning_items(arguments.reasoning_items, arguments.seed)
    if arguments.multiple_choice_items > 0:
        demonstration = make_demonstration_multiple_choice_items()
        repeats = (arguments.multiple_choice_items // len(demonstration)) + 1
        task_items += (demonstration * repeats)[:arguments.multiple_choice_items]

    is_word = make_is_word(
        None if arguments.dictionary is None
        else {line.strip().lower() for line in arguments.dictionary.read_text().splitlines() if line.strip()})

    asr_items = build_asr_items(
        task_items=task_items,
        audio_directory=arguments.audio_directory,
        is_word=is_word,
        keep_degenerate=arguments.keep_degenerate,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w") as output_file:
        for asr_item in asr_items:
            record = asdict(asr_item)
            record["selection_policy"] = asr_item.selection_policy
            record.pop("word_diffs", None)            # diffs are diagnostic; drop from the JSONL
            output_file.write(json.dumps(record) + "\n")

    degenerate_count = sum(1 for item in asr_items if item.is_degenerate)
    print(f"wrote {len(asr_items)} ASR items to {arguments.output}")
    print(f"  flagged degenerate (kept): {degenerate_count}")


if __name__ == "__main__":
    main()
