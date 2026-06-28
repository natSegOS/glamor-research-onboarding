"""Data-preparation Stage 0: annotate task items with formally-defined key terms.

This tool reads the pinned JSONL files produced by tools/build_task_items.py,
applies the formal K_P(x) key-term rule (design/02 §2.x) using a pinned spaCy
linguistic pipeline, and writes the annotated records back to the same files
(or to separate output files if --output-dir is given).

The annotated ``key_terms`` field replaces the old runtime heuristic that
previously computed key terms inside the generation runner.  After running this
tool, the experiment is a pure consumer of frozen annotations: no key-term
computation happens at runtime (design/04 §4.6).

A provenance record is written to data/items/annotation_PROVENANCE.json,
recording the exact spaCy model name, its meta.json SHA-256, the rule version,
and per-file statistics.  This record is committed alongside the JSONL files so
that a reviewer can verify the annotation is reproducible.

Usage
-----
Typical one-time invocation before a pilot or confirmatory run:

    python tools/build_annotated_dataset.py \\
        --model-name en_core_web_sm \\
        --items-dir data/items

Upgrade to a higher-accuracy model for the confirmatory run:

    python tools/build_annotated_dataset.py \\
        --model-name en_core_web_trf \\
        --items-dir data/items \\
        --force

Pass --dry-run to preview annotation statistics without writing files.

Dependency note
---------------
This tool requires spaCy and a downloaded English model.  Install with:
    pip install spacy
    python -m spacy download en_core_web_sm    # or en_core_web_md / en_core_web_trf

These are BUILD-TOOL dependencies only and are listed in requirements-build.txt,
not in requirements.txt, so the experiment and analysis layers remain installable
without spaCy on the GPU cluster nodes (where only requirements.txt is needed).
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

# Add src/ to the path so this tool can import project modules when run
# directly (e.g. ``python tools/build_annotated_dataset.py``), mirroring the
# pattern used by all other tools in this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="en_core_web_sm",
        help=(
            "spaCy model name to use for annotation, e.g. en_core_web_sm, "
            "en_core_web_md, or en_core_web_trf.  The model must be installed.  "
            "For the confirmatory run, use en_core_web_trf (highest accuracy; "
            "see data/items/annotation_PROVENANCE.json for benchmark citations). "
            "[default: en_core_web_sm]"
        ),
    )
    parser.add_argument(
        "--items-dir",
        default="data/items",
        help="Directory containing the pinned JSONL item files.  [default: data/items]",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "If given, write annotated files here rather than overwriting the "
            "input files.  Useful for validating a new model without committing "
            "changes."
        ),
    )
    parser.add_argument(
        "--provenance-path",
        default="data/items/annotation_PROVENANCE.json",
        help=(
            "Path to write the annotation provenance JSON record.  "
            "[default: data/items/annotation_PROVENANCE.json]"
        ),
    )
    parser.add_argument(
        "--question-text-field",
        default="question_text",
        help=(
            "JSON field name that holds the bare question text in each record.  "
            "Reasoning items use 'question_text'; multiple-choice items in the "
            "local schema also use 'question_text'.  Change only if the schema "
            "has diverged.  [default: question_text]"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing annotations without raising.  Required if the "
            "item files already carry key_terms from a previous annotation run.  "
            "Overwriting a pre-registered frozen dataset requires a design-doc "
            "amendment (design/10 §10.3)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print annotation statistics without writing any files.  Useful for "
            "verifying the model and rule before committing."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Run the annotation stage and return an exit code (0 = success)."""
    arguments = _parse_arguments(argv)

    items_directory = Path(arguments.items_dir)
    output_directory = Path(arguments.output_dir) if arguments.output_dir else None
    provenance_path = Path(arguments.provenance_path)

    # Locate JSONL files (skip the provenance JSON itself).
    jsonl_paths = sorted(items_directory.glob("*.jsonl"))
    if not jsonl_paths:
        print(
            f"No JSONL files found in {items_directory}.  "
            "Run tools/build_task_items.py first.",
            file=sys.stderr,
        )
        return 1

    # Import the annotation module.  This is the only place the import is done
    # so that the error message from load_linguistic_pipeline is surfaced here
    # rather than at module import time.
    try:
        from dataprep.annotate import (
            load_linguistic_pipeline,
            annotate_jsonl_file,
            build_annotation_provenance_record,
        )
    except ImportError as import_error:
        print(f"Import error: {import_error}", file=sys.stderr)
        return 1

    # Load the linguistic pipeline once (expensive) and reuse for all files.
    print(f"Loading spaCy model '{arguments.model_name}'...")
    try:
        linguistic_pipeline = load_linguistic_pipeline(arguments.model_name)
    except (ImportError, OSError) as load_error:
        print(f"Error loading spaCy model: {load_error}", file=sys.stderr)
        return 1

    print(f"  Model loaded.  Processing {len(jsonl_paths)} JSONL file(s) in {items_directory}/")
    if arguments.dry_run:
        print("  DRY RUN — no files will be written.")

    result_summaries: list[dict] = []
    total_violations = 0

    for jsonl_path in jsonl_paths:
        output_path = (
            (output_directory / jsonl_path.name)
            if output_directory
            else jsonl_path
        )

        try:
            summary = annotate_jsonl_file(
                input_path=jsonl_path,
                output_path=(Path("/dev/null") if arguments.dry_run else output_path),
                linguistic_pipeline=linguistic_pipeline,
                model_name=arguments.model_name,
                question_text_field=arguments.question_text_field,
                force=arguments.force,
            )
        except ValueError as annotation_error:
            print(f"  ERROR in {jsonl_path.name}: {annotation_error}", file=sys.stderr)
            return 1

        result_summaries.append(summary)
        total_violations += summary.get("violation_count", 0)

        print(
            f"  {jsonl_path.name}: "
            f"{summary['annotated_count']} items annotated"
            + (f", {summary['violation_count']} operand-coverage warnings"
               if summary.get("violation_count") else "")
        )

    if total_violations:
        print(
            f"\nWARNING: {total_violations} template-operand coverage violation(s) "
            "detected.  These mean that a numeric operand in a synthetic item's "
            "template was not captured by the K_P(x) rule.  Review the warnings "
            "above and consider whether the rule needs adjustment before committing "
            "the annotations as a pre-registered frozen dataset."
        )

    # Write provenance record.
    if not arguments.dry_run:
        provenance_record = build_annotation_provenance_record(
            model_name=arguments.model_name,
            input_paths=jsonl_paths,
            result_summaries=result_summaries,
        )
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(provenance_record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nProvenance record written to {provenance_path}")

    print(
        f"\nAnnotation complete.  "
        f"{sum(s['annotated_count'] for s in result_summaries)} items annotated "
        f"across {len(jsonl_paths)} file(s)."
    )
    if not arguments.dry_run:
        print(
            "Next steps:\n"
            "  1. Review data/items/annotation_PROVENANCE.json.\n"
            "  2. Spot-check a sample of annotated items.\n"
            "  3. Commit the annotated JSONL files and provenance record.\n"
            "  4. Run tools/run_generation.py --config configs/pilot.yaml\n"
            "     to verify the experiment reads the frozen annotations correctly."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
