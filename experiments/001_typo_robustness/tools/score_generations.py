"""Stage 2: score raw generation outputs to produce the research-grade dataset.

This tool reads the raw-output JSONL files written by tools/run_generation.py
and writes scored JSONL files that add four fields to each row:

  parsed_answer   — the extracted numeric or option-letter answer, or null
  is_correct      — 1 if the parsed answer matches the gold, 0 otherwise
  parse_status    — one of: valid, unparseable, clarification, refusal
  score_model     — the spaCy model used for parse-status detection

The formal four-way parse-status taxonomy (design/04 §4.5, plan §1.3) is
assigned by the linguistic classifier in src/scoring.py using spaCy dependency
structure, without any phrase lexicon.  This is the RUNTIME mechanism; the
phrase lists in tests/fixtures/ are frozen VALIDATION ORACLES only.

Dual-accounting rule (design/04 §4.5): every interactional failure
(CLARIFICATION or REFUSAL) is tallied as is_correct=0 for the accuracy
primary endpoint AND is counted separately toward the invalid-or-clarification
rate (ICR, metric M9).  Both accounting entries are in the output rows.

Three-stage pipeline context
----------------------------
  Stage 0  tools/build_annotated_dataset.py  → annotated item JSONL (CPU)
  Stage 1  tools/run_generation.py           → raw output JSONL (GPU)
  Stage 2  tools/score_generations.py        → scored JSONL (CPU)  ← this tool

The scored JSONL is the direct input for tools/run_analysis.py.

Usage
-----
Minimal:
    python tools/score_generations.py \\
        --input-path data/outputs/raw_outputs.jsonl \\
        --output-path data/outputs/scored.jsonl

Full options:
    python tools/score_generations.py \\
        --input-path data/outputs/raw_outputs.jsonl \\
        --output-path data/outputs/scored.jsonl \\
        --model-name en_core_web_sm \\
        --force \\
        --dry-run

Dependency note
---------------
Requires spaCy and a downloaded English model.  Install with:
    pip install spacy
    python -m spacy download en_core_web_sm   # or en_core_web_md / en_core_web_trf

For the confirmatory run, use en_core_web_trf (highest accuracy).  See
data/items/annotation_PROVENANCE.json for model selection rationale.
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enums import INTERACTIONAL_FAILURE_STATUSES, REASONING_FAMILIES, MCQ_FAMILIES
import scoring


SCHEMA_VERSION = "1.0"

_DEFAULT_SPACY_MODEL_NAME = "en_core_web_sm"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help=(
            "Path to a raw-output JSONL file from tools/run_generation.py.  "
            "Each row must have 'model_output', 'expected_answer', and 'task_family'."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write the scored JSONL file.  Parent directories are created if needed.",
    )
    parser.add_argument(
        "--model-name",
        default=_DEFAULT_SPACY_MODEL_NAME,
        help=(
            f"spaCy model for parse-status classification.  "
            f"[default: {_DEFAULT_SPACY_MODEL_NAME}]  "
            "Use en_core_web_trf for the confirmatory run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file without raising an error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing any output.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-row scoring
# ---------------------------------------------------------------------------

def _score_row(row: dict, linguistic_pipeline) -> dict:
    """Return the row with parsed_answer, is_correct, parse_status added.

    If the row already has a 'parse_status' field from inline scoring (the
    smoke path), it is overwritten with the formal linguistic classification.
    """
    model_output: str = row.get("model_output", "")
    gold_answer = row.get("expected_answer")
    task_family: str = row.get("task_family", "")

    if gold_answer is None:
        raise ValueError(
            f"Row {row.get('row_id', '?')!r} has no 'expected_answer' field.")

    if task_family in REASONING_FAMILIES:
        parsed_answer = scoring.extract_reasoning_answer(model_output)
    elif task_family in MCQ_FAMILIES:
        option_count = row.get("option_count", 10)
        parsed_answer = scoring.extract_multiple_choice_answer(
            model_output, option_count)
    else:
        raise ValueError(
            f"Unknown task_family {task_family!r} in row {row.get('row_id', '?')!r}.  "
            "Extend this tool if you have added a new task family.")

    parse_status = scoring.classify_parse_status_with_linguistic_pipeline(
        model_output, parsed_answer, linguistic_pipeline)

    if parse_status in INTERACTIONAL_FAILURE_STATUSES:
        is_correct = 0
    elif parsed_answer is None:
        is_correct = 0
    elif task_family in REASONING_FAMILIES:
        gold_float = float(gold_answer)
        parsed_float = float(parsed_answer)
        if gold_float.is_integer():
            is_correct = int(parsed_float == gold_float)
        else:
            is_correct = int(abs(parsed_float - gold_float) < 1e-6)
    else:
        is_correct = int(str(parsed_answer) == str(gold_answer).upper())

    scored_row = dict(row)
    scored_row["parsed_answer"] = None if parsed_answer is None else str(parsed_answer)
    scored_row["is_correct"] = is_correct
    scored_row["parse_status"] = parse_status.value
    return scored_row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    arguments = _parse_arguments(argv)

    input_path = Path(arguments.input_path)
    output_path = Path(arguments.output_path)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    if output_path.exists() and not arguments.force and not arguments.dry_run:
        print(
            f"Error: output file already exists: {output_path}  "
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        from dataprep.annotate import load_linguistic_pipeline
    except ImportError as import_error:
        print(f"Import error: {import_error}", file=sys.stderr)
        return 1

    print(f"Loading spaCy model '{arguments.model_name}'...")
    try:
        linguistic_pipeline = load_linguistic_pipeline(arguments.model_name)
    except (ImportError, OSError) as load_error:
        print(f"Error loading spaCy model: {load_error}", file=sys.stderr)
        return 1

    raw_lines = [
        line for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  Scoring {len(raw_lines)} rows from {input_path}")
    if arguments.dry_run:
        print("  DRY RUN — no output will be written.")

    status_counts: dict[str, int] = {}
    scored_rows: list[dict] = []

    for line_number, line in enumerate(raw_lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as decode_error:
            print(
                f"  Warning: skipping malformed line {line_number}: {decode_error}",
                file=sys.stderr,
            )
            continue

        try:
            scored_row = _score_row(row, linguistic_pipeline)
        except ValueError as scoring_error:
            print(
                f"  Warning: skipping row {row.get('row_id', '?')!r}: {scoring_error}",
                file=sys.stderr,
            )
            continue

        status = scored_row["parse_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        scored_rows.append(scored_row)

    print(f"\n  Parse-status breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"    {status}: {count} ({100.0 * count / len(scored_rows):.1f}%)")

    if not arguments.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in scored_rows)
            + "\n",
            encoding="utf-8",
        )
        print(f"\n  Wrote {len(scored_rows)} scored rows to {output_path}")

    print(
        f"\nScoring complete.  {len(scored_rows)} rows scored "
        f"using model '{arguments.model_name}'."
    )
    if not arguments.dry_run:
        print(
            "Next steps:\n"
            f"  python tools/run_analysis.py --scored-path {output_path}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
