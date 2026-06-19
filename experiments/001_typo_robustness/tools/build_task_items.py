"""Fetch, pin, and export the official task items for a confirmatory run.

This is a **one-time pre-processing step** that must run on a networked machine
with HuggingFace access before the main sweep. It resolves the dataset commit
SHA at fetch time (so the version is captured as of the moment this script
runs), downloads the official items, exports them to JSONL, and records full
provenance. The exported JSONL files are what the pipeline reads during
generation via the config keys ``reasoning_items_path`` and
``multiple_choice_items_path``.

Usage:

    python tools/build_task_items.py

Or with explicit options:

    python tools/build_task_items.py \\
        --reasoning-items 600 \\
        --gsm-config main \\
        --mcq-items 600 \\
        --categories math physics \\
        --seed 1729 \\
        --output-directory data/items

Outputs (in --output-directory):
    gsm_symbolic.jsonl       reasoning items (ReasoningItem schema as JSON lines)
    mmlu_pro.jsonl           multiple-choice items (MultipleChoiceItem schema)
    PROVENANCE.json          resolved revision SHAs, item counts, fetch timestamp
"""

from __future__ import annotations

import argparse
import dataclasses
import json

from datetime import datetime, timezone
from pathlib import Path

from tasks import (
    ReasoningItem,
    MultipleChoiceItem,
    load_official_gsm_symbolic,
    load_official_mmlu_pro,
)


# ---------------------------------------------------------------------------
# Revision resolution.
# ---------------------------------------------------------------------------

def resolve_dataset_revision(dataset_repo_id: str) -> str:
    """Return the current HEAD commit SHA for a HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError(
            "resolving dataset revisions requires the 'huggingface-hub' "
            "package (pip install -r requirements.txt)") from error

    sha = HfApi().dataset_info(dataset_repo_id).sha
    if sha is None:
        raise RuntimeError(
            f"HuggingFace returned None for the SHA of '{dataset_repo_id}'. "
            "The repo may be private or the dataset ID may be wrong.")
    return sha


# ---------------------------------------------------------------------------
# JSONL serialisation.  ReasoningItem and MultipleChoiceItem are dataclasses,
# but ReasoningItem carries a Callable (gold_answer_function) and a
# ReasoningTemplate that are not JSON-serialisable.  We export the fields the
# pipeline needs: the item's identity, prompt text, gold answer, and key terms.
# ---------------------------------------------------------------------------

def _reasoning_item_to_record(item: ReasoningItem) -> dict:
    """Serialise a ReasoningItem to a JSON-safe dict."""
    return {
        "task_id": item.task_id,
        "task_family": item.task_family,
        "source": item.source,
        "question_text": item.question_text,
        "instruction": item.instruction,
        "gold_answer": item.gold_answer,
        "key_terms": item.key_terms,
        # Regime C operand-swap is not supported for official items (no
        # template / answer function); leave the field null so the pipeline
        # knows not to attempt it.
        "supports_regime_c": False,
        "template_id": item.template.template_id if item.template else None,
        "parameters": item.parameters,
    }


def _multiple_choice_item_to_record(item: MultipleChoiceItem) -> dict:
    """Serialise a MultipleChoiceItem to a JSON-safe dict."""
    return {
        "task_id": item.task_id,
        "task_family": item.task_family,
        "question": item.question,
        "options": item.options,
        "answer": item.gold_letter,
        "category": item.category,
        "gold_letter_if_negated": item.gold_letter_if_negated,
        "key_terms": item.key_terms,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/items"),
        help="directory to write the JSONL files and PROVENANCE.json (default: data/items)",
    )
    parser.add_argument(
        "--gsm-config",
        default="main",
        choices=["main", "p1", "p2"],
        help="GSM-Symbolic difficulty variant: main (default), p1 (+1 clause), p2 (+2 clauses)",
    )
    parser.add_argument(
        "--reasoning-items",
        type=int,
        default=600,
        help="number of GSM-Symbolic items to export (default: 600)",
    )
    parser.add_argument(
        "--mcq-items",
        type=int,
        default=600,
        help="number of MMLU-Pro items to export (default: 600)",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="restrict MMLU-Pro to these subject categories; default: all categories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="random seed for subsampling (default: 1729)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    provenance: dict = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "seed": arguments.seed,
        "gsm_symbolic": None,
        "mmlu_pro": None,
    }

    # --- GSM-Symbolic ---

    print(f"resolving apple/GSM-Symbolic revision (config={arguments.gsm_config!r}) ...")
    gsm_revision = resolve_dataset_revision("apple/GSM-Symbolic")
    print(f"  resolved: {gsm_revision}")

    print(f"fetching {arguments.reasoning_items} reasoning items ...")
    reasoning_items = load_official_gsm_symbolic(
        configuration_name=arguments.gsm_config,
        dataset_revision=gsm_revision,
        item_count=arguments.reasoning_items,
        seed=arguments.seed,
    )

    gsm_output_path = output_directory / "gsm_symbolic.jsonl"
    with gsm_output_path.open("w") as output_file:
        for item in reasoning_items:
            output_file.write(json.dumps(_reasoning_item_to_record(item)) + "\n")

    provenance["gsm_symbolic"] = {
        "repo_id": "apple/GSM-Symbolic",
        "configuration_name": arguments.gsm_config,
        "resolved_revision_sha": gsm_revision,
        "item_count": len(reasoning_items),
        "output_file": str(gsm_output_path),
    }
    print(f"  wrote {len(reasoning_items)} items to {gsm_output_path}")

    # --- MMLU-Pro ---

    print("resolving TIGER-Lab/MMLU-Pro revision ...")
    mmlu_revision = resolve_dataset_revision("TIGER-Lab/MMLU-Pro")
    print(f"  resolved: {mmlu_revision}")

    categories_label = arguments.categories if arguments.categories else "all"
    print(f"fetching {arguments.mcq_items} MCQ items (categories: {categories_label}) ...")
    mcq_items = load_official_mmlu_pro(
        dataset_revision=mmlu_revision,
        item_count=arguments.mcq_items,
        categories=arguments.categories,
        seed=arguments.seed,
    )

    mmlu_output_path = output_directory / "mmlu_pro.jsonl"
    with mmlu_output_path.open("w") as output_file:
        for item in mcq_items:
            output_file.write(json.dumps(_multiple_choice_item_to_record(item)) + "\n")

    provenance["mmlu_pro"] = {
        "repo_id": "TIGER-Lab/MMLU-Pro",
        "resolved_revision_sha": mmlu_revision,
        "item_count": len(mcq_items),
        "categories": arguments.categories,
        "output_file": str(mmlu_output_path),
    }
    print(f"  wrote {len(mcq_items)} items to {mmlu_output_path}")

    # --- Provenance sidecar ---

    provenance_path = output_directory / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nprovenance written to {provenance_path}")
    print("\nNext step: update configs/main.yaml:")
    print(f"  reasoning_items_path: {gsm_output_path}")
    print(f"  multiple_choice_items_path: {mmlu_output_path}")


if __name__ == "__main__":
    main()

