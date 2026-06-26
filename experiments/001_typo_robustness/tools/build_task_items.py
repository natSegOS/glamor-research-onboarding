"""Fetch, pin, and export the official task items for a confirmatory run.

This is a **one-time pre-processing step** (safely re-runnable) that must run
on a networked machine with HuggingFace access before the main sweep. It
resolves the dataset commit SHA at fetch time, downloads the official items,
exports them to JSONL, and records full provenance. The exported JSONL files are
what the pipeline reads during generation via the config's ``datasets:`` list.

Re-running this script fetches fresh revisions and overwrites the JSONL files
cleanly — it is safe to re-run when you want updated dataset hashes.

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
    gsm_symbolic.jsonl   GSM-Symbolic reasoning items (primary)
    mmlu_pro.jsonl       MMLU-Pro multiple-choice items (primary)
    gsm8k.jsonl          GSM8K reasoning items (contamination contrast)
    mmlu.jsonl           MMLU multiple-choice items (contamination contrast)
    PROVENANCE.json      resolved revision SHAs, item counts, fetch timestamp
"""

from __future__ import annotations

import argparse
import json
import sys

from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from progress import ProgressBar

from tasks import (
    ReasoningItem,
    MultipleChoiceItem,
    load_official_gsm_symbolic,
    load_official_gsm8k,
    load_official_mmlu_pro,
    load_official_mmlu,
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
        help="number of reasoning items per dataset to export (default: 600)",
    )
    parser.add_argument(
        "--mcq-items",
        type=int,
        default=600,
        help="number of MCQ items per dataset to export (default: 600)",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="restrict MMLU-Pro and MMLU to these subject categories; default: all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="random seed for subsampling (default: 1729)",
    )
    return parser.parse_args()


def _write_jsonl(items, record_fn, output_path: Path, description: str = "") -> None:
    """Write items to a JSONL file, truncating any existing content."""

    with output_path.open("w") as output_file, \
         ProgressBar(total=len(items), description=description or str(output_path.name)) as progress:
        for item in items:
            output_file.write(json.dumps(record_fn(item)) + "\n")
            progress.advance()


def main() -> None:
    arguments = parse_arguments()
    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    categories_label = arguments.categories if arguments.categories else "all"
    provenance: dict = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "seed": arguments.seed,
        "gsm_symbolic": None,
        "mmlu_pro": None,
        "gsm8k": None,
        "mmlu": None,
    }

    # --- GSM-Symbolic (primary reasoning) ---

    print(f"resolving apple/GSM-Symbolic revision (config={arguments.gsm_config!r}) ...")
    gsm_sym_revision = resolve_dataset_revision("apple/GSM-Symbolic")
    print(f"  resolved: {gsm_sym_revision}")
    print(f"fetching {arguments.reasoning_items} GSM-Symbolic items ...")
    gsm_sym_items = load_official_gsm_symbolic(
        configuration_name=arguments.gsm_config,
        dataset_revision=gsm_sym_revision,
        item_count=arguments.reasoning_items,
        seed=arguments.seed,
    )
    gsm_sym_path = output_directory / "gsm_symbolic.jsonl"
    _write_jsonl(gsm_sym_items, _reasoning_item_to_record, gsm_sym_path,
                 description="writing gsm_symbolic.jsonl")
    provenance["gsm_symbolic"] = {
        "repo_id": "apple/GSM-Symbolic",
        "configuration_name": arguments.gsm_config,
        "resolved_revision_sha": gsm_sym_revision,
        "item_count": len(gsm_sym_items),
        "output_file": str(gsm_sym_path),
    }
    print(f"  wrote {len(gsm_sym_items)} items to {gsm_sym_path}")

    # --- MMLU-Pro (primary MCQ) ---

    print("resolving TIGER-Lab/MMLU-Pro revision ...")
    mmlu_pro_revision = resolve_dataset_revision("TIGER-Lab/MMLU-Pro")
    print(f"  resolved: {mmlu_pro_revision}")
    print(f"fetching {arguments.mcq_items} MMLU-Pro items (categories: {categories_label}) ...")
    mmlu_pro_items = load_official_mmlu_pro(
        dataset_revision=mmlu_pro_revision,
        item_count=arguments.mcq_items,
        seed=arguments.seed,
        categories=arguments.categories,
    )
    mmlu_pro_path = output_directory / "mmlu_pro.jsonl"
    _write_jsonl(mmlu_pro_items, _multiple_choice_item_to_record, mmlu_pro_path,
                 description="writing mmlu_pro.jsonl")
    provenance["mmlu_pro"] = {
        "repo_id": "TIGER-Lab/MMLU-Pro",
        "resolved_revision_sha": mmlu_pro_revision,
        "item_count": len(mmlu_pro_items),
        "categories": arguments.categories,
        "output_file": str(mmlu_pro_path),
    }
    print(f"  wrote {len(mmlu_pro_items)} items to {mmlu_pro_path}")

    # --- GSM8K (contamination-contrast reasoning) ---

    print("resolving openai/gsm8k revision ...")
    gsm8k_revision = resolve_dataset_revision("openai/gsm8k")
    print(f"  resolved: {gsm8k_revision}")
    print(f"fetching {arguments.reasoning_items} GSM8K items ...")
    gsm8k_items = load_official_gsm8k(
        dataset_revision=gsm8k_revision,
        item_count=arguments.reasoning_items,
        seed=arguments.seed,
    )
    gsm8k_path = output_directory / "gsm8k.jsonl"
    _write_jsonl(gsm8k_items, _reasoning_item_to_record, gsm8k_path,
                 description="writing gsm8k.jsonl")
    provenance["gsm8k"] = {
        "repo_id": "openai/gsm8k",
        "configuration_name": "main",
        "resolved_revision_sha": gsm8k_revision,
        "item_count": len(gsm8k_items),
        "output_file": str(gsm8k_path),
    }
    print(f"  wrote {len(gsm8k_items)} items to {gsm8k_path}")

    # --- MMLU (contamination-contrast MCQ) ---

    print("resolving cais/mmlu revision ...")
    mmlu_revision = resolve_dataset_revision("cais/mmlu")
    print(f"  resolved: {mmlu_revision}")
    print(f"fetching {arguments.mcq_items} MMLU items (categories: {categories_label}) ...")
    mmlu_items = load_official_mmlu(
        dataset_revision=mmlu_revision,
        item_count=arguments.mcq_items,
        seed=arguments.seed,
        categories=arguments.categories,
    )
    mmlu_path = output_directory / "mmlu.jsonl"
    _write_jsonl(mmlu_items, _multiple_choice_item_to_record, mmlu_path,
                 description="writing mmlu.jsonl")
    provenance["mmlu"] = {
        "repo_id": "cais/mmlu",
        "configuration_name": "all",
        "resolved_revision_sha": mmlu_revision,
        "item_count": len(mmlu_items),
        "categories": arguments.categories,
        "output_file": str(mmlu_path),
    }
    print(f"  wrote {len(mmlu_items)} items to {mmlu_path}")

    # --- Provenance sidecar ---

    provenance_path = output_directory / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nprovenance written to {provenance_path}")
    print("\nAdd these paths to configs/main.yaml under 'datasets:'")
    print(f"  gsm_symbolic_jsonl:  path: {gsm_sym_path}")
    print(f"  mmlu_pro_jsonl:      path: {mmlu_pro_path}")
    print(f"  gsm8k_jsonl:         path: {gsm8k_path}")
    print(f"  mmlu_jsonl:          path: {mmlu_path}")


if __name__ == "__main__":
    main()
