"""Run one experiment configuration against one real model.

This is the bridge the orchestrator deliberately leaves to a script: it builds a
real inference engine from the model registry, enforces the revision-pinning
rule for confirmatory runs, and then calls run_experiment. The orchestrator
itself stays engine-agnostic (so it is testable with the dummy engine), and this
script holds the one responsibility that needs the model specifications: the
pin assertion (design/10 §10.5).

Runs on the USC GPU cluster and on Google Colab (T4) via the same vLLM path.

Usage (after `pip install -r requirements-gpu.txt`):

    python tools/run_generation.py \\
        --config configs/main.yaml \\
        --model llama_8b_awq \\
        --output-directory results/main

For the main study, also:
  - fill in model revisions in src/inference/roster.py (use
    inference.roster.resolve_current_revision), and
  - pre-fetch dataset JSONL files with tools/build_task_items.py.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from pipeline import ExperimentConfiguration, run_experiment
from inference import build_inference_engine
from inference import assert_revisions_pinned, get_model_specification
from regimes import make_is_word

# Default dictionary — built from SCOWL size-60 by tools/build_dictionary.py.
# Never falls back to the 488-word demo list in real runs.
_DEFAULT_DICTIONARY = Path(__file__).resolve().parent.parent / "data" / "wordlists" / "en_us_pinned.txt"

# spaCy model used for inline four-way parse-status classification (Workstream 5).
_SPACY_MODEL_NAME = "en_core_web_trf"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, help="a roster key, e.g. llama_8b_awq")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--git-commit", default="unpinned",
                        help="the code commit SHA, recorded in every row")
    parser.add_argument(
        "--dictionary", type=Path, default=None,
        help=f"a pinned English word list (default: {_DEFAULT_DICTIONARY})")
    parser.add_argument(
        "--no-spacy", action="store_true",
        help="disable inline spaCy scoring (falls back to structural two-way classifier)")
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="this worker's index (0-based) for parallel generation across "
             "GPUs/sessions; requires --shard-count. Each worker writes its own "
             "'..._w{index}of{count}_generations.jsonl' — start one process per "
             "GPU with the same --config/--model and a distinct --shard-index, "
             "and merge the outputs afterward (design/07 §7.7).")
    parser.add_argument(
        "--shard-count", type=int, default=None,
        help="total number of parallel workers; requires --shard-index.")
    arguments = parser.parse_args()
    if (arguments.shard_index is None) != (arguments.shard_count is None):
        parser.error("--shard-index and --shard-count must be given together")
    if arguments.shard_index is not None and not (0 <= arguments.shard_index < arguments.shard_count):
        parser.error("--shard-index must satisfy 0 <= shard_index < shard_count")
    return arguments


def _load_linguistic_pipeline(disabled: bool):
    """Load the spaCy transformer pipeline once at startup.

    Returns None when disabled or when spaCy / the model is not installed
    (a warning is printed; the run continues with the structural fallback).
    """
    if disabled:
        return None
    try:
        import spacy  # noqa: PLC0415
        return spacy.load(_SPACY_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[run_generation] WARNING: could not load spaCy model "
            f"{_SPACY_MODEL_NAME!r}: {exc}\n"
            f"  Falling back to structural two-way parse-status classifier.\n"
            f"  Install: python -m spacy download {_SPACY_MODEL_NAME}",
            file=sys.stderr,
        )
        return None


def main():
    arguments = parse_arguments()

    configuration = ExperimentConfiguration.from_yaml(arguments.config)
    specification = get_model_specification(arguments.model)

    # The pin assertion: a confirmatory run may not start against an unpinned
    # (non-reproducible) model revision.
    if configuration.is_confirmatory:
        assert_revisions_pinned([specification])

    engine = build_inference_engine(specification)

    dictionary_path = arguments.dictionary or _DEFAULT_DICTIONARY
    is_word = make_is_word(_load_dictionary(dictionary_path))

    print(f"[run_generation] loading spaCy pipeline ({_SPACY_MODEL_NAME}) ...")
    linguistic_pipeline = _load_linguistic_pipeline(arguments.no_spacy)
    if linguistic_pipeline is not None:
        print(f"[run_generation] spaCy loaded ({_SPACY_MODEL_NAME})")

    shard_partition = (
        (arguments.shard_index, arguments.shard_count)
        if arguments.shard_index is not None else None)

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
        linguistic_pipeline=linguistic_pipeline,
        shard_partition=shard_partition,
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
