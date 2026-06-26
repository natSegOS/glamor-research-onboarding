"""Patch key_terms into the pre-fetched JSONL item files.

Applies the same extraction logic as the HuggingFace fetchers without
re-downloading anything. Run this once after pulling code that introduced
or updated the key_terms extraction strategy.

Usage:

    python tools/repair_item_key_terms.py
    python tools/repair_item_key_terms.py --items-directory data/items
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tasks.reasoning import extract_key_terms_from_reasoning_question
from tasks.multiple_choice import extract_key_terms_from_mcq_question


_REASONING_FILES = {"gsm_symbolic.jsonl", "gsm8k.jsonl"}
_MCQ_FILES = {"mmlu_pro.jsonl", "mmlu.jsonl"}


def _patch_reasoning_file(path: Path) -> tuple[int, int]:
    """Rewrite a reasoning JSONL file with key_terms filled in.

    Returns (total_rows, rows_updated).
    """

    lines = [l for l in path.read_text().splitlines() if l.strip()]
    updated_rows: list[str] = []
    rows_changed = 0

    for line in lines:
        record = json.loads(line)
        question = record.get("question_text", "")
        new_terms = extract_key_terms_from_reasoning_question(question)

        if record.get("key_terms") != new_terms:
            record["key_terms"] = new_terms
            rows_changed += 1

        updated_rows.append(json.dumps(record))

    path.write_text("\n".join(updated_rows) + "\n")
    return len(lines), rows_changed


def _patch_mcq_file(path: Path) -> tuple[int, int]:
    """Rewrite an MCQ JSONL file with key_terms filled in.

    Returns (total_rows, rows_updated).
    """

    lines = [l for l in path.read_text().splitlines() if l.strip()]
    updated_rows: list[str] = []
    rows_changed = 0

    for line in lines:
        record = json.loads(line)
        question = record.get("question", "")
        new_terms = extract_key_terms_from_mcq_question(question)

        if record.get("key_terms") != new_terms:
            record["key_terms"] = new_terms
            rows_changed += 1

        updated_rows.append(json.dumps(record))

    path.write_text("\n".join(updated_rows) + "\n")
    return len(lines), rows_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--items-directory",
        type=Path,
        default=Path("data/items"),
        help="directory containing the pre-fetched JSONL files (default: data/items)",
    )
    arguments = parser.parse_args()
    items_directory = arguments.items_directory

    if not items_directory.exists():
        print(f"error: {items_directory} does not exist", file=sys.stderr)
        sys.exit(1)

    for filename in sorted(_REASONING_FILES | _MCQ_FILES):
        path = items_directory / filename
        if not path.exists():
            print(f"  skipping {filename} (not found)")
            continue

        if filename in _REASONING_FILES:
            total, changed = _patch_reasoning_file(path)
        else:
            total, changed = _patch_mcq_file(path)

        print(f"  {filename}: {total} rows, {changed} updated")

    print("done.")


if __name__ == "__main__":
    main()
