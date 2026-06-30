"""Pre-generate all perturbation pairs without running a language model.

This is the first step in the pre-generation validity pipeline (Workstream 8,
design/09 §9.2).  It calls the same ``build_requests()`` function used by the
generation runner, but drives it with the ``DeterministicDummyEngine`` (no GPU
required) to enumerate all (clean, perturbed) text pairs.

The output is a JSONL file at ``--output-path`` with one record per pair.
It is the input for:
  1. tools/run_judge.py    — LLM-as-judge screening
  2. tools/regime_audit_ui.html — human annotation interface
  3. tools/run_generation.py  — actual generation (after audit gating)

By separating perturbation-pair generation from model generation:
  - Validity is assessed before ANY GPU time is spent.
  - The same pair file is reusable across all models in the study.
  - Coverage rates (how many items were successfully perturbed) are available
    before the pilot run, as methods-table inputs.

Usage:

    python tools/build_perturbations.py \\
        --config configs/pilot.yaml \\
        --dictionary data/wordlists/en_us_pinned.txt \\
        --output-path data/perturbations/pilot_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.experiment import ExperimentConfiguration, build_requests, ExclusionSidecar
from pipeline.runner import DeterministicDummyEngine
from regimes import make_is_word


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="experiment config YAML (pilot.yaml or main.yaml)")
    parser.add_argument("--dictionary", type=Path,
                        default=Path("data/wordlists/en_us_pinned.txt"),
                        help="pinned English word list for is_word predicate")
    parser.add_argument("--output-path", required=True, type=Path,
                        help="where to write the pairs JSONL")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    configuration = ExperimentConfiguration.from_yaml(arguments.config)

    # Load dictionary.
    dict_path = arguments.dictionary
    if not dict_path.exists():
        print(
            f"ERROR: dictionary not found at {dict_path}\n"
            "Run: python tools/build_dictionary.py --source scowl ...\n"
            "to build it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    words = {line.strip().lower()
             for line in dict_path.read_text().splitlines() if line.strip()}
    is_word = make_is_word(words)

    # DeterministicDummyEngine: no GPU, no generation — we only need the
    # perturbation pairs, not the model outputs.
    dummy_engine = DeterministicDummyEngine()
    dummy_tokenizer = dummy_engine  # has no tokenizer; tokenization fields will be zeros

    # Set up exclusion sidecar alongside the output file.
    output_path = Path(arguments.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path = output_path.with_name(
        output_path.stem + "_exclusions.jsonl")
    exclusion_sidecar = ExclusionSidecar(exclusion_path)

    # Build all requests (no generation).
    print(f"[build_perturbations] loading task items ...")
    from pipeline.experiment import load_task_items, load_asr_items_by_task
    task_items = load_task_items(configuration)
    asr_items_by_task = load_asr_items_by_task(configuration.asr_items_path)

    print(f"[build_perturbations] building perturbation pairs ...")
    requests = build_requests(
        task_items,
        configuration.conditions,
        is_word,
        dummy_tokenizer,
        configuration.seed,
        asr_items_by_task,
        exclusion_sidecar=exclusion_sidecar,
    )

    # Write pairs JSONL.
    perturbed_requests = [r for r in requests if not r.is_clean]
    clean_by_task = {r.task_id: r for r in requests if r.is_clean}

    print(f"[build_perturbations] writing {len(perturbed_requests)} pairs ...")
    with output_path.open("w") as fh:
        for request in perturbed_requests:
            clean = clean_by_task.get(request.task_id)
            record = {
                "task_id": request.task_id,
                "task_family": request.task_family,
                "clean_text": clean.prompt if clean else "",
                "perturbed_text": request.prompt,
                "gold_answer": str(request.gold_answer),
                "claimed_regime": request.perturbation_state_vector.get("semantic_class", ""),
                "condition_name": next(
                    (str(v) for k, v in request.perturbation_state_vector.items()
                     if k == "selection_policy"), ""),
                "edit_budget": request.perturbation_state_vector.get("edit_budget", 0),
                "edit_script": [
                    e.to_dict() if hasattr(e, "to_dict") else e
                    for e in request.edit_script
                ],
                **request.extra_fields,
            }
            fh.write(json.dumps(record) + "\n")

    print(
        f"\nPerturbation pairs written to:  {output_path}\n"
        f"  total pairs:      {len(perturbed_requests)}\n"
        f"  excluded items:   {exclusion_sidecar.count}  "
        f"(see {exclusion_path})\n"
        f"\nNext steps:\n"
        f"  1. python tools/run_judge.py --pairs {output_path} ...\n"
        f"  2. Open tools/regime_audit_ui.html for human annotation\n"
        f"  3. python tools/run_generation.py --config {arguments.config} ..."
    )


if __name__ == "__main__":
    main()
