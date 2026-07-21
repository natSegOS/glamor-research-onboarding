"""Stratified sampling and judge pre-labelling for the 200-item regime audit.

Selects 200 perturbed generation rows stratified by semantic class (A/B/C)
and task family, runs the cross-family judge on them, and writes a single JSON
file that the human-validation UI (tools/regime_audit_ui.html) loads.

The judge is run here so that the human annotator sees both the judge's label
and its rationale, which makes the annotation task faster and more consistent.
Annotators mark agree/disagree; disagreements are the audit's key output.

Usage:

    python tools/sample_for_audit.py \\
        --generations results/pilot_generations.jsonl \\
        --model gemma2_9b_judge \\
        --output data/audit/audit_sample.json

    # Skip the judge (label all items as pending human review):
    python tools/sample_for_audit.py \\
        --generations results/pilot_generations.jsonl \\
        --no-judge \\
        --output data/audit/audit_sample.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enums import SemanticClass
from pipeline.runner import load_generation_rows
from progress import ProgressBar


_DEFAULT_SAMPLE_SIZE = 200
_DEFAULT_SEED = 1729
_DEFAULT_CACHE_FILENAME = "judge_cache.jsonl"

_TARGET_REGIMES = (SemanticClass.A, SemanticClass.B, SemanticClass.C)


def _is_perturbed_row(row: dict) -> bool:
    return not row.get("is_clean", True)


def _stratified_sample(
        rows: list[dict],
        total: int,
        seed: int,
) -> list[dict]:
    """Sample ``total`` rows stratified by (r_semantic_class, task_family).

    Allocates slots proportionally to stratum size; any remainder goes to the
    largest strata first. Returns rows in a random order.
    """

    strata: dict[tuple, list[dict]] = {}
    for row in rows:
        if not _is_perturbed_row(row):
            continue
        regime = row.get("r_semantic_class", "")
        if regime not in _TARGET_REGIMES:
            continue
        key = (regime, row.get("task_family", ""))
        strata.setdefault(key, []).append(row)

    rng = random.Random(seed)
    for stratum_rows in strata.values():
        rng.shuffle(stratum_rows)

    total_available = sum(len(v) for v in strata.values())
    effective_total = min(total, total_available)

    allocation: dict[tuple, int] = {}
    for key, stratum_rows in strata.items():
        allocation[key] = max(1, round(effective_total * len(stratum_rows) / total_available))

    current_sum = sum(allocation.values())
    keys_by_size = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)
    index = 0
    while current_sum < effective_total:
        key = keys_by_size[index % len(keys_by_size)]
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            current_sum += 1
        index += 1
    while current_sum > effective_total:
        key = keys_by_size[-(index % len(keys_by_size)) - 1]
        if allocation[key] > 1:
            allocation[key] -= 1
            current_sum -= 1
        index += 1

    sampled: list[dict] = []
    for key, count in allocation.items():
        sampled.extend(strata[key][:count])

    rng.shuffle(sampled)
    return sampled


def _build_audit_item(row: dict, judge_decision: object = None) -> dict:
    """Convert a generation row (+ optional judge decision) into an audit item."""

    edit_script = row.get("edit_script", [])
    edited_word_before = ""
    edited_word_after = ""
    if edit_script:
        first_edit = edit_script[0] if isinstance(edit_script[0], dict) else {}
        edited_word_before = first_edit.get("word_before", "")
        edited_word_after = first_edit.get("word_after", "")

    item: dict = {
        "row_id": row.get("row_id", ""),
        "task_id": row.get("task_id", ""),
        "task_family": row.get("task_family", ""),
        "model_id": row.get("model_id", ""),
        "claimed_regime": row.get("r_semantic_class", ""),
        "selection_policy": row.get("r_selection_policy", ""),
        "edit_budget": row.get("r_edit_budget", ""),
        "original_text": row.get("clean_prompt", ""),
        "perturbed_text": row.get("perturbed_prompt", ""),
        "edited_word_before": edited_word_before,
        "edited_word_after": edited_word_after,
        "edit_script": edit_script,
        "token_inflation_ratio": row.get("token_inflation_ratio"),
        "fragmentation_stratum": row.get("fragmentation_stratum"),
        "judge_classification": None,
        "judge_confidence": None,
        "judge_rationale": None,
        "judge_parse_failed": False,
        "human_classification": None,
        "human_agrees_with_judge": None,
        "human_note": "",
        "human_reviewed": False,
    }

    if judge_decision is not None:
        item["judge_classification"] = getattr(judge_decision, "classification", None)
        item["judge_confidence"] = getattr(judge_decision, "confidence", None)
        item["judge_rationale"] = getattr(judge_decision, "rationale", None)
        item["judge_parse_failed"] = getattr(judge_decision, "parse_failed", False)

    return item


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--generations",
        required=True,
        nargs="+",
        type=Path,
        metavar="JSONL",
        help="one or more generation JSONL files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="JSON",
        help="output audit JSON file (loaded by regime_audit_ui.html)",
    )
    parser.add_argument(
        "--model",
        default="gemma2_9b_judge",
        help="roster key for the judge model (default: gemma2_9b_judge)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=_DEFAULT_SAMPLE_SIZE,
        help=f"number of items to sample (default: {_DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"random seed for stratified sampling (default: {_DEFAULT_SEED})",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="path to the judge decision cache JSONL (default: <output_dir>/judge_cache.jsonl)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip judge calls; produce items with null judge fields for manual-only review",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    print(f"loading generation rows from {len(arguments.generations)} file(s) ...")
    all_rows = load_generation_rows(arguments.generations)
    print(f"  loaded {len(all_rows)} rows total")

    perturbed_rows = [row for row in all_rows if _is_perturbed_row(row)]
    print(f"  {len(perturbed_rows)} perturbed rows eligible for sampling")

    sample = _stratified_sample(perturbed_rows, arguments.sample_size, arguments.seed)
    print(f"  sampled {len(sample)} rows across regimes and task families")

    regime_counts: dict[str, int] = {}
    for row in sample:
        regime = row.get("r_semantic_class", "?")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    for regime, count in sorted(regime_counts.items()):
        print(f"    regime {regime}: {count} items")

    audit_items: list[dict] = []

    if arguments.no_judge:
        print("skipping judge (--no-judge); all items will have null judge fields")
        for row in sample:
            audit_items.append(_build_audit_item(row, judge_decision=None))
    else:
        from judge import run_judge_on_sample
        from inference import VllmEngine, get_model_specification

        specification = get_model_specification(arguments.model)
        cache_path = arguments.cache or (arguments.output.parent / _DEFAULT_CACHE_FILENAME)

        print(f"building judge engine ({specification.huggingface_identifier}) ...")
        engine = VllmEngine(specification)

        print(f"running judge on {len(sample)} items (cache: {cache_path}) ...")
        with ProgressBar(total=len(sample), description="judging") as progress:
            # Aligned 1:1 with `sample` (None = skipped Regime-C MCQ). The
            # alignment contract of run_judge_on_sample is what makes this
            # zip safe; a decision list that silently dropped skipped rows
            # attached judge labels to the WRONG audit items.
            aligned_decisions = run_judge_on_sample(
                engine=engine,
                judge_revision=specification.revision,
                sample_rows=sample,
                cache_path=cache_path,
                progress_callback=progress.advance,
            )

        for row, decision in zip(sample, aligned_decisions):
            audit_items.append(_build_audit_item(row, judge_decision=decision))

        judged = [decision for decision in aligned_decisions if decision is not None]
        agreement_count = sum(
            1 for decision in judged
            if decision.agrees_with_claimed_regime() is True
        )
        disagreement_count = sum(
            1 for decision in judged
            if decision.agrees_with_claimed_regime() is False
        )
        parse_failed_count = sum(1 for decision in judged if decision.parse_failed)
        print(
            f"  judge agrees: {agreement_count}, "
            f"disagrees: {disagreement_count}, "
            f"parse failed: {parse_failed_count}, "
            f"skipped (Regime-C MCQ, structurally guaranteed): {len(sample) - len(judged)}"
        )

    output = {
        "schema_version": "1",
        "sample_size": len(audit_items),
        "seed": arguments.seed,
        "judge_model": arguments.model if not arguments.no_judge else None,
        "items": audit_items,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2))
    print(f"\naudit sample written to {arguments.output}")
    print(f"open tools/regime_audit_ui.html in a browser to begin human validation")


if __name__ == "__main__":
    main()
