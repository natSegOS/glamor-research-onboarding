"""Fetch, pin, and export the official task items for a confirmatory run.

This is a one-time pre-processing step (safely re-runnable) that must run on a
networked machine with HuggingFace access before the main sweep. It resolves
each dataset's commit SHA at fetch time, downloads the official items, exports
them to JSONL, and records full provenance. The exported JSONL files are what
the pipeline reads during generation via the config's ``datasets:`` list.

Re-running this script fetches fresh revisions and overwrites the JSONL files
cleanly. It is safe to re-run when you want updated dataset hashes.

GSM-Symbolic template enrichment (--gsm-templates-dir)
-------------------------------------------------------
The ``question_annotated`` field that enables Regime C reasoning operand-swap
is not exposed in the Apple/GSM-Symbolic HuggingFace dataset.  It is available
in the companion GitHub repository (github.com/apple/ml-gsm-symbolic) under
``templates/p1/*.json``.

When ``--gsm-templates-dir`` points to a local clone of that repository, this
tool joins each fetched HF item against the template via:

    generated_data/GSM_p1.jsonl   HF question text  →  original_id
    templates/p1/{id_orig}.json   original_id        →  question_annotated

The template provides the STRUCTURE (parameter names, types, answer formula).
The HF question text provides the INSTANCE'S actual parameter values.
``extract_instance_parameters`` matches the template's format string against the
HF question text to recover the real values, then validates
``answer_function(**extracted) == gold_answer``.  Items that pass validation
get ``parameters`` set to the extracted instance values (fully Regime C capable).
Items that fail validation get ``parameters = {}`` and will be excluded
gracefully at perturbation time.

The stored ``question_annotated`` string uses the template's default values in
its ``{param,value}`` syntax (as written in the Apple repo), but the separate
``parameters`` field in the JSONL holds the validated instance values that
``load_reasoning_jsonl`` will use at run time.

Usage:

    python tools/build_task_items.py

Or with explicit options:

    python tools/build_task_items.py \\
        --reasoning-items 600 \\
        --gsm-config p1 \\
        --mcq-items 600 \\
        --seed 1729 \\
        --output-directory data/items \\
        --gsm-templates-dir /tmp/ml-gsm-symbolic

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
    extract_instance_parameters,
    serialize_parameters,
    load_official_gsm_symbolic,
    load_official_gsm8k,
    load_official_mmlu_pro,
    load_official_mmlu,
)


_DEFAULT_OUTPUT_DIRECTORY  = Path("data/items")
# "p1", not "main": the Apple template repository ships templates/p1 and
# templates/p2 only. With "main" the template-enrichment step finds nothing,
# no question_annotated fields are written, and Regime C reasoning silently
# produces zero items. p1 is what the pilot used (data/items/PROVENANCE.json).
_DEFAULT_GSM_CONFIG        = "p1"
_DEFAULT_ITEMS_PER_DATASET = 600
_DEFAULT_SAMPLING_SEED     = 1729

_GSM_SYMBOLIC_REPO_ID = "apple/GSM-Symbolic"
_MMLU_PRO_REPO_ID     = "TIGER-Lab/MMLU-Pro"
_GSM8K_REPO_ID        = "openai/gsm8k"
_MMLU_REPO_ID         = "cais/mmlu"


# ---------------------------------------------------------------------------
# Revision resolution
# ---------------------------------------------------------------------------

def resolve_dataset_revision(dataset_repo_identifier: str) -> str:
    """Return the current HEAD commit SHA for a HuggingFace dataset repo."""

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ImportError(
            "resolving dataset revisions requires the 'huggingface-hub' "
            "package (pip install -r requirements.txt)") from error

    sha = HfApi().dataset_info(dataset_repo_identifier).sha
    if sha is None:
        raise RuntimeError(
            f"HuggingFace returned None for the SHA of {dataset_repo_identifier!r}. "
            "The repo may be private or the dataset identifier may be wrong.")
    return sha


# ---------------------------------------------------------------------------
# GSM-Symbolic template enrichment
# ---------------------------------------------------------------------------

def _enrich_gsm_items_with_apple_templates(
        items: list[ReasoningItem],
        templates_dir: Path,
        gsm_config: str,
) -> tuple[int, int]:
    """Enrich each item with question_annotated and validated instance parameters.

    The HF dataset exposes the instantiated question text but not the symbolic
    template.  The Apple GitHub repo provides the template.  This function joins
    each HF item against the template repo in two stages:

    PRIMARY path (preferred, no dependency on generated_data/):
        item.id_orig  →  templates/{config}/{id_orig_key}.json
        Uses the ``original_id`` field that the HF apple/GSM-Symbolic dataset
        exposes directly in each record.  This path is robust to a missing or
        partially-downloaded generated_data/ file.

    FALLBACK path (when item.id_orig is not set):
        generated_data/GSM_{config}.jsonl  question text → original_id
        templates/{config}/{id_orig_key}.json  original_id → question_annotated
        Used when the HF record did not include original_id.

    After finding the template, uses extract_instance_parameters to match the
    template's format string against the HF question text (structure from the
    template, values from the HF question) and validates the result against
    gold_answer.  Items that pass have parameters set to validated instance
    values; items that fail keep parameters={} and will be excluded gracefully
    at Regime C perturbation time.

    Returns (joined_count, extracted_count).
    """
    templates_subdir = templates_dir / "templates" / gsm_config
    if not templates_subdir.exists():
        print(f"  [templates] templates/{gsm_config}/ not found in {templates_dir}; skipping enrichment")
        return 0, 0

    # Build id_orig → question_annotated from all template files.
    id_orig_to_question_annotated: dict[int, str] = {}
    for template_file in templates_subdir.glob("*.json"):
        template_data = json.loads(template_file.read_text())
        id_orig = template_data.get("id_orig")
        question_annotated = template_data.get("question_annotated") or ""
        if id_orig is not None and question_annotated.strip():
            id_orig_to_question_annotated[id_orig] = question_annotated

    print(f"  [templates] loaded {len(id_orig_to_question_annotated)} templates from {templates_subdir}")

    # Determine how many items have id_orig from the HF record.
    items_with_id_orig = sum(1 for item in items if item.id_orig is not None)
    print(f"  [templates] items with HF original_id: {items_with_id_orig}/{len(items)}")

    # Build question-text fallback only when some items lack id_orig.
    question_to_original_id: dict[str, int] = {}
    if items_with_id_orig < len(items):
        gen_data_path = templates_dir / "generated_data" / f"GSM_{gsm_config}.jsonl"
        if gen_data_path.exists():
            for line in gen_data_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                question_to_original_id[row["question"]] = row["original_id"]
            print(f"  [templates] fallback question-text lookup: {len(question_to_original_id)} entries")
        else:
            print(f"  [templates] generated_data not found at {gen_data_path}; "
                  f"fallback unavailable ({len(items) - items_with_id_orig} items will not be joined)")

    joined = 0
    extracted = 0
    for item in items:
        # Primary: use id_orig from the HF record.
        original_id = item.id_orig
        # Fallback: look up via question text in generated_data/.
        if original_id is None:
            original_id = question_to_original_id.get(item.question_text)
        if original_id is None:
            continue

        question_annotated = id_orig_to_question_annotated.get(original_id)
        if not question_annotated:
            continue

        item.question_annotated = question_annotated
        joined += 1

        instance_params = extract_instance_parameters(
            question_annotated=question_annotated,
            question_text=item.question_text,
            gold_answer=item.gold_answer,
        )
        if instance_params is not None:
            item.parameters = instance_params
            extracted += 1

    return joined, extracted


# ---------------------------------------------------------------------------
# JSONL serialisation
#
# ReasoningItem and MultipleChoiceItem are dataclasses, but ReasoningItem
# carries a Callable (answer_function) and a ReasoningTemplate that cannot be
# serialised to JSON. We export only the fields the pipeline needs at run time.
#
# question_annotated: the raw GSM-Symbolic annotation string (present for p1/p2
# splits). Stored so that load_reasoning_jsonl can re-parse the template at run
# time from the JSONL without a live HF connection. None for other datasets.
# ---------------------------------------------------------------------------

def _reasoning_item_to_record(item: ReasoningItem) -> dict:
    record = {
        "task_id":       item.task_id,
        "task_family":   item.task_family,
        "source":        item.source,
        "question_text": item.question_text,
        "instruction":   item.instruction,
        "gold_answer":   item.gold_answer,
        "key_terms":     item.key_terms,
        "parameters":    serialize_parameters(item.parameters),
    }
    if item.question_annotated is not None:
        record["question_annotated"] = item.question_annotated
    return record


def _multiple_choice_item_to_record(item: MultipleChoiceItem) -> dict:

    return {
        "task_id":    item.task_id,
        "task_family": item.task_family,
        "question":   item.question,
        "options":    item.options,
        "answer":     item.gold_letter,
        "category":   item.category,
        "key_terms":  item.key_terms,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=_DEFAULT_OUTPUT_DIRECTORY,
        help=f"directory to write the JSONL files and PROVENANCE.json "
             f"(default: {_DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--gsm-config",
        default=_DEFAULT_GSM_CONFIG,
        choices=["main", "p1", "p2"],
        help="GSM-Symbolic difficulty variant: p1 (+1 clause, default — has Apple "
             "templates, enabling Regime C), p2 (+2 clauses), main (no templates "
             "published: Regime C reasoning will be empty)",
    )
    parser.add_argument(
        "--reasoning-items",
        type=int,
        default=_DEFAULT_ITEMS_PER_DATASET,
        help=f"number of reasoning items per dataset to export "
             f"(default: {_DEFAULT_ITEMS_PER_DATASET})",
    )
    parser.add_argument(
        "--mcq-items",
        type=int,
        default=_DEFAULT_ITEMS_PER_DATASET,
        help=f"number of MCQ items per dataset to export "
             f"(default: {_DEFAULT_ITEMS_PER_DATASET})",
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
        default=_DEFAULT_SAMPLING_SEED,
        help=f"random seed for subsampling (default: {_DEFAULT_SAMPLING_SEED})",
    )
    parser.add_argument(
        "--gsm-templates-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="path to a local clone of github.com/apple/ml-gsm-symbolic. "
             "When supplied, each GSM-Symbolic item is enriched with its "
             "question_annotated field from the template repo, enabling Regime C "
             "reasoning operand-swap. Without this flag, question_annotated is "
             "absent and Regime C reasoning produces no items.",
    )
    return parser.parse_args()


def _write_jsonl(
        items,
        record_serialiser,
        output_path: Path,
        description: str = "",
) -> None:
    """Write items to a JSONL file, truncating any existing content."""

    with (
        output_path.open("w") as output_file,
        ProgressBar(
            total=len(items),
            description=description or output_path.name,
        ) as progress,
    ):
        for item in items:
            output_file.write(json.dumps(record_serialiser(item)) + "\n")
            progress.advance()


def main() -> None:
    arguments = parse_arguments()

    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    categories_label = arguments.categories or "all"
    provenance: dict = {
        "fetched_at":   datetime.now(timezone.utc).isoformat(),
        "seed":         arguments.seed,
        "gsm_symbolic": None,
        "mmlu_pro":     None,
        "gsm8k":        None,
        "mmlu":         None,
    }

    # --- GSM-Symbolic (primary reasoning) ---

    print(f"resolving {_GSM_SYMBOLIC_REPO_ID} revision (config={arguments.gsm_config!r}) ...")
    gsm_symbolic_revision = resolve_dataset_revision(_GSM_SYMBOLIC_REPO_ID)
    print(f"  resolved: {gsm_symbolic_revision}")
    print(f"fetching {arguments.reasoning_items} GSM-Symbolic items ...")

    gsm_symbolic_items = load_official_gsm_symbolic(
        configuration_name=arguments.gsm_config,
        dataset_revision=gsm_symbolic_revision,
        item_count=arguments.reasoning_items,
        seed=arguments.seed,
    )
    joined_count = extracted_count = 0
    if arguments.gsm_templates_dir is not None:
        print(f"enriching GSM-Symbolic items with Apple templates from {arguments.gsm_templates_dir} ...")
        joined_count, extracted_count = _enrich_gsm_items_with_apple_templates(
            gsm_symbolic_items, arguments.gsm_templates_dir, arguments.gsm_config)
        print(f"  template joined:              {joined_count}/{len(gsm_symbolic_items)} items")
        print(f"  instance params validated:    {extracted_count}/{joined_count} joined items "
              f"(Regime C capable)")
    else:
        print("  (--gsm-templates-dir not supplied; question_annotated will be absent; "
              "Regime C reasoning will produce no items)")

    gsm_symbolic_output_path = output_directory / "gsm_symbolic.jsonl"
    _write_jsonl(
        gsm_symbolic_items,
        _reasoning_item_to_record,
        gsm_symbolic_output_path,
        description="writing gsm_symbolic.jsonl",
    )
    provenance["gsm_symbolic"] = {
        "repo_id":                    _GSM_SYMBOLIC_REPO_ID,
        "configuration_name":         arguments.gsm_config,
        "resolved_revision_sha":      gsm_symbolic_revision,
        "item_count":                 len(gsm_symbolic_items),
        "apple_templates_dir":        str(arguments.gsm_templates_dir) if arguments.gsm_templates_dir else None,
        "apple_templates_joined":     joined_count if arguments.gsm_templates_dir else None,
        "apple_templates_params_validated": extracted_count if arguments.gsm_templates_dir else None,
        "output_file":                str(gsm_symbolic_output_path),
    }
    print(f"  wrote {len(gsm_symbolic_items)} items to {gsm_symbolic_output_path}")

    # --- MMLU-Pro (primary MCQ) ---

    print(f"resolving {_MMLU_PRO_REPO_ID} revision ...")
    mmlu_pro_revision = resolve_dataset_revision(_MMLU_PRO_REPO_ID)
    print(f"  resolved: {mmlu_pro_revision}")
    print(f"fetching {arguments.mcq_items} MMLU-Pro items (categories: {categories_label}) ...")

    mmlu_pro_items = load_official_mmlu_pro(
        dataset_revision=mmlu_pro_revision,
        item_count=arguments.mcq_items,
        seed=arguments.seed,
        categories=arguments.categories,
    )
    mmlu_pro_output_path = output_directory / "mmlu_pro.jsonl"
    _write_jsonl(
        mmlu_pro_items,
        _multiple_choice_item_to_record,
        mmlu_pro_output_path,
        description="writing mmlu_pro.jsonl",
    )
    provenance["mmlu_pro"] = {
        "repo_id":               _MMLU_PRO_REPO_ID,
        "resolved_revision_sha": mmlu_pro_revision,
        "item_count":            len(mmlu_pro_items),
        "categories":            arguments.categories,
        "output_file":           str(mmlu_pro_output_path),
    }
    print(f"  wrote {len(mmlu_pro_items)} items to {mmlu_pro_output_path}")

    # --- GSM8K (contamination-contrast reasoning) ---

    print(f"resolving {_GSM8K_REPO_ID} revision ...")
    gsm8k_revision = resolve_dataset_revision(_GSM8K_REPO_ID)
    print(f"  resolved: {gsm8k_revision}")
    print(f"fetching {arguments.reasoning_items} GSM8K items ...")

    gsm8k_items = load_official_gsm8k(
        dataset_revision=gsm8k_revision,
        item_count=arguments.reasoning_items,
        seed=arguments.seed,
    )
    gsm8k_output_path = output_directory / "gsm8k.jsonl"
    _write_jsonl(
        gsm8k_items,
        _reasoning_item_to_record,
        gsm8k_output_path,
        description="writing gsm8k.jsonl",
    )
    provenance["gsm8k"] = {
        "repo_id":               _GSM8K_REPO_ID,
        "configuration_name":    "main",
        "resolved_revision_sha": gsm8k_revision,
        "item_count":            len(gsm8k_items),
        "output_file":           str(gsm8k_output_path),
    }
    print(f"  wrote {len(gsm8k_items)} items to {gsm8k_output_path}")

    # --- MMLU (contamination-contrast MCQ) ---

    print(f"resolving {_MMLU_REPO_ID} revision ...")
    mmlu_revision = resolve_dataset_revision(_MMLU_REPO_ID)
    print(f"  resolved: {mmlu_revision}")
    print(f"fetching {arguments.mcq_items} MMLU items (categories: {categories_label}) ...")

    mmlu_items = load_official_mmlu(
        dataset_revision=mmlu_revision,
        item_count=arguments.mcq_items,
        seed=arguments.seed,
        categories=arguments.categories,
    )
    mmlu_output_path = output_directory / "mmlu.jsonl"
    _write_jsonl(
        mmlu_items,
        _multiple_choice_item_to_record,
        mmlu_output_path,
        description="writing mmlu.jsonl",
    )
    provenance["mmlu"] = {
        "repo_id":               _MMLU_REPO_ID,
        "configuration_name":    "all",
        "resolved_revision_sha": mmlu_revision,
        "item_count":            len(mmlu_items),
        "categories":            arguments.categories,
        "output_file":           str(mmlu_output_path),
    }
    print(f"  wrote {len(mmlu_items)} items to {mmlu_output_path}")

    # --- Provenance sidecar ---

    provenance_output_path = output_directory / "PROVENANCE.json"
    provenance_output_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nprovenance written to {provenance_output_path}")
    print("\nAdd these paths to configs/main.yaml under 'datasets:'")
    print(f"  gsm_symbolic_jsonl:  path: {gsm_symbolic_output_path}")
    print(f"  mmlu_pro_jsonl:      path: {mmlu_pro_output_path}")
    print(f"  gsm8k_jsonl:         path: {gsm8k_output_path}")
    print(f"  mmlu_jsonl:          path: {mmlu_output_path}")


if __name__ == "__main__":
    main()
