"""Run one experiment configuration against one real model.

This is the bridge the orchestrator deliberately leaves to a script: it builds a
real inference engine from the model registry, enforces the revision-pinning
rule for confirmatory runs, and then calls run_experiment. The orchestrator
itself stays engine-agnostic (so it is testable with the dummy engine), and this
script holds the one responsibility that needs the model specifications: the
pin assertion (design/10 §10.5).

Usage (on the GPU machine, after `pip install -r requirements-gpu.txt`):

    python tools/run_generation.py \\
        --config configs/main.yaml \\
        --model llama_8b_awq \\
        --backend vllm \\
        --output-directory results/main

For the real study, also:
  - fill in the model revisions in src/inference/roster.py
    (resolve_current_revision prints each SHA), and
  - pin the dataset revisions in the loaders / swap in the official loaders.
"""

from __future__ import annotations

import argparse

from pathlib import Path

from pipeline import ExperimentConfiguration, run_experiment
from inference import build_inference_engine
from inference import assert_revisions_pinned, get_model_specification
from regimes import make_is_word


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, help="a roster key, e.g. llama_8b_awq")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "huggingface"])
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--git-commit", default="unpinned",
                        help="the code commit SHA, recorded in every row")
    parser.add_argument("--dictionary", type=Path, default=None,
                        help="a pinned English word list; defaults to the demo list")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    configuration = ExperimentConfiguration.from_yaml(arguments.config)
    specification = get_model_specification(arguments.model)

    # The pin assertion: a confirmatory run may not start against an unpinned
    # (non-reproducible) model revision.
    if configuration.is_confirmatory:
        assert_revisions_pinned([specification])

    engine = build_inference_engine(specification, backend=arguments.backend)
    is_word = make_is_word(_load_dictionary(arguments.dictionary))

    summary = run_experiment(
        configuration=configuration,
        engine=engine,
        is_word=is_word,
        tokenizer=engine.tokenizer,
        output_directory=arguments.output_directory,
        model_id=specification.huggingface_identifier,
        model_revision=specification.revision,
        quantization_method=specification.precision,
        git_commit=arguments.git_commit,
    )

    print("run complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def _load_dictionary(path):
    if path is None:
        return None
    return {line.strip().lower() for line in Path(path).read_text().splitlines() if line.strip()}


if __name__ == "__main__":
    main()
