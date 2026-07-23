"""Run the cross-family LLM-as-judge on a perturbation-pairs file.

Step 2 of the pre-generation validity pipeline (Workstream 8, design/09
§9.2). Reads the ``{run_id}_pairs.jsonl`` file written by
``tools/build_perturbations.py``, calls the cross-family judge (Gemma 2
9B-IT, temperature=0, cached) on each pair, and writes a summary JSONL.

Design constraints:
  - Cross-family judge: Gemma 2 (Google) judging Llama/Qwen/Mistral output,
    reducing correlation between generator and judge tendencies (see judge.py).
  - Temperature 0: greedy decoding, so (judge_revision, prompt_version, input)
    always maps to the same output.
  - Content-addressed cache: already-decided pairs are served from cache, so
    partial runs and reruns are cheap.
  - Agree/disagree flag: if the judge's regime classification differs from the
    claimed regime (e.g. "C" when the engine claimed "A"), the pair is flagged
    and written to ``--output-flagged`` with its ``task_id`` attached.
  - Fleiss' κ calibration: judge agreement with human annotations is reported
    as a calibration statistic only. The judge is never the final authority.

Flagged pairs are not removed from the generation queue: that would change
deterministic row IDs, break resume semantics, and let an LLM judge silently
veto items (design/09 §9.7 forbids this). Instead they're routed to the human
audit with priority (tools/sample_for_audit.py); items the human audit fails
are excluded at analysis time via the ``audit_outcomes`` gate in
``analysis.results.summarize_all_cells``.

Usage:

    python tools/run_judge.py \\
        --pairs data/perturbations/pilot_pairs.jsonl \\
        --judge-cache data/perturbations/judge_cache.jsonl \\
        --output-flagged data/perturbations/pilot_flagged.jsonl

To skip actual inference (dry run, mark all as unchecked):

    python tools/run_judge.py --pairs ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge import run_judge_on_sample

# Below this agreement rate (and with enough judged pairs for the rate to be
# meaningful), warn that the judge model/prompt version may need review.
_AGREEMENT_RATE_WARNING_THRESHOLD = 80.0
_MINIMUM_DECISIONS_FOR_AGREEMENT_WARNING = 10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs", required=True, type=Path,
                        help="perturbation-pairs JSONL from build_perturbations.py")
    parser.add_argument("--judge-model-revision", default="gemma2_9b_judge",
                        help="roster key (preferred; honours the pinned revision) or a raw "
                             "HuggingFace id, optionally '@<revision>' (confirmatory runs "
                             "must use a pinned roster entry)")
    parser.add_argument("--judge-cache", type=Path,
                        default=Path("data/perturbations/judge_cache.jsonl"),
                        help="append-only content-addressed judge decision cache")
    parser.add_argument("--output-flagged", type=Path,
                        default=Path("data/perturbations/flagged.jsonl"),
                        help="JSONL of pairs flagged by the judge (regime disagreement or "
                             "parse_failed), with task_id attached; routed to the human "
                             "audit (exclusion happens at analysis time, never here)")
    parser.add_argument("--sample", type=int, default=None,
                        help="only judge this many pairs (for quick pilot screening)")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip inference; write an empty flagged file and exit "
                             "(useful for testing the pipeline)")
    return parser.parse_args()


def _load_engine(model_revision: str):
    """Load the judge engine from the vLLM backend (or raise ImportError).

    ``model_revision`` is looked up in ``inference.roster`` by
    ``huggingface_identifier`` or a name alias.  If no roster entry is found
    a minimal ``ModelSpecification`` is constructed from the string directly.
    """
    try:
        from inference.engines import VllmEngine
        from inference.roster import ModelSpecification, get_model_specification
    except ImportError as error:
        raise ImportError(
            "run_judge requires the vLLM backend (GPU cluster only). "
            "Use --dry-run on CPU.") from error

    # Attempt to resolve via the roster first (honours pinned revisions).
    try:
        specification = get_model_specification(model_revision)
    except (KeyError, AttributeError):
        # Fallback: build a minimal ModelSpecification from the raw string.
        # Revision is the part after '@' if present.
        if "@" in model_revision:
            hf_id, rev = model_revision.split("@", 1)
        else:
            hf_id, rev = model_revision, None
        specification = ModelSpecification(
            roster_key=hf_id,
            huggingface_identifier=hf_id,
            revision=rev or "PIN_ME",
        )

    return VllmEngine(specification)


def main() -> None:
    arguments = parse_arguments()

    pairs_path = arguments.pairs
    if not pairs_path.exists():
        print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with pairs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if arguments.sample is not None:
        rows = rows[: arguments.sample]

    print(f"[run_judge] loaded {len(rows)} pairs from {pairs_path}")

    if arguments.dry_run:
        print("[run_judge] --dry-run: skipping inference, writing empty flagged file")
        arguments.output_flagged.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_flagged.write_text("")
        print(f"[run_judge] dry run complete. flagged file: {arguments.output_flagged}")
        return

    # Load judge engine.
    print(f"[run_judge] loading judge: {arguments.judge_model_revision}")
    try:
        engine = _load_engine(arguments.judge_model_revision)
    except ImportError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    # Judge requires the rows to have 'clean_prompt'/'prompt' keys, but the
    # pairs file from build_perturbations.py uses 'clean_text'/'perturbed_text'.
    # Translate field names here so judge.run_judge_on_sample finds them.
    translated_rows = []
    for row in rows:
        translated_rows.append({
            **row,
            "clean_prompt": row.get("clean_text", ""),
            "prompt": row.get("perturbed_text", ""),
            "r_semantic_class": row.get("claimed_regime", ""),
        })

    # Run judge (uses cache for already-decided pairs).
    arguments.judge_cache.parent.mkdir(parents=True, exist_ok=True)

    aligned_decisions = run_judge_on_sample(
        engine=engine,
        judge_revision=arguments.judge_model_revision,
        sample_rows=translated_rows,
        cache_path=arguments.judge_cache,
    )

    # Decisions are aligned 1:1 with rows (None = skipped Regime-C MCQ), so
    # each flagged decision can be written WITH its source row's identity.
    judged_pairs = [(row, decision) for row, decision in zip(rows, aligned_decisions)
                    if decision is not None]
    agreed = [decision for _row, decision in judged_pairs
              if decision.agrees_with_claimed_regime() is True]
    flagged_pairs = [(row, decision) for row, decision in judged_pairs
                     if decision.agrees_with_claimed_regime() is False or decision.parse_failed]
    unchecked = sum(decision is None for decision in aligned_decisions)

    arguments.output_flagged.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_flagged.open("w") as fh:
        for row, decision in flagged_pairs:
            fh.write(json.dumps({
                **decision.to_dict(),
                "task_id": row.get("task_id", ""),
                "condition_name": row.get("condition_name", ""),
            }) + "\n")

    flagged = [decision for _row, decision in flagged_pairs]
    agreement_rate = (len(agreed) / len(judged_pairs) * 100
                      if judged_pairs else float("nan"))
    parse_failed_count = sum(
        1 for _row, decision in judged_pairs if decision.parse_failed)

    print(
        f"\n[run_judge] Summary\n"
        f"  judged:         {len(judged_pairs):>6}\n"
        f"  agreed:         {len(agreed):>6}  ({agreement_rate:.1f}%)\n"
        f"  flagged:        {len(flagged):>6}  (regime disagree or parse failed)\n"
        f"    of which parse_failed: {parse_failed_count}\n"
        f"  skipped (C MCQ / no judge needed): {unchecked}\n"
        f"\n"
        f"  flagged pairs written to: {arguments.output_flagged}\n"
        f"  judge cache:              {arguments.judge_cache}\n"
        f"\n"
        f"  NOTE: the judge is calibrated against human annotations (κ reported\n"
        f"  in the paper appendix) but is NOT the final validity authority.\n"
        f"  Flagged pairs are excluded from the generation queue pending human review.\n"
    )

    if (agreement_rate < _AGREEMENT_RATE_WARNING_THRESHOLD
            and len(judged_pairs) > _MINIMUM_DECISIONS_FOR_AGREEMENT_WARNING):
        print(
            f"  WARNING: agreement rate {agreement_rate:.1f}% is below 80%. "
            f"Check judge model and prompt version.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
