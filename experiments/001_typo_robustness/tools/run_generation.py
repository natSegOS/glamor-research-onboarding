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

from dataclasses import replace
from pathlib import Path

from pipeline import (
    ExperimentConfiguration,
    build_requests,
    load_task_items,
    required_context_length,
    run_experiment,
    run_is_complete,
)
from inference import VllmEngine
from inference import (
    assert_revisions_pinned, get_model_specification, resolve_current_revision)
from regimes import make_is_word

# Default dictionary, built from SCOWL size-60 by tools/build_dictionary.py.
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
        "--fresh", action="store_true",
        help="delete this run's previous outputs (generations, exclusions, "
             "manifest) before generating. Without it the runner resumes: rows "
             "already on disk (including ones committed to the repo) are "
             "kept and skipped.")
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="this worker's index (0-based) for parallel generation across "
             "GPUs/sessions; requires --shard-count. Each worker writes its own "
             "'..._w{index}of{count}_generations.jsonl'; start one process per "
             "GPU with the same --config/--model and a distinct --shard-index, "
             "and merge the outputs afterward (design/07 §7.7).")
    parser.add_argument(
        "--shard-count", type=int, default=None,
        help="total number of parallel workers; requires --shard-index.")
    parser.add_argument(
        "--skip-if-complete", dest="skip_if_complete", action="store_true", default=True,
        help="if every shard for this run is already recorded complete in the "
             "manifest, exit before loading the tokenizer/model at all (default: on).")
    parser.add_argument(
        "--no-skip-if-complete", dest="skip_if_complete", action="store_false",
        help="always load the model, even if the manifest says this run is complete "
             "(the row-level resume in run_shard still applies).")
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


def _measure_max_model_length(configuration, specification, is_word) -> int:
    """Load the tokenizer up front and size vLLM's context window from the
    request set this run will actually submit, rather than the model's native
    (often 10x+ larger) default context.

    Rebuilds the same requests ``run_experiment`` builds internally (cheap,
    deterministic, CPU-only. See the shard_partition note in
    run_experiment's docstring for the established precedent of redoing this
    step rather than threading it through); no exclusion_sidecar is passed
    here, so this pass logs nothing and cannot double-count exclusions.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        specification.huggingface_identifier,
        revision=specification.revision if specification.revision_is_pinned else None,
    )
    task_items = load_task_items(configuration)
    requests = build_requests(
        task_items, configuration.conditions, is_word, tokenizer, configuration.seed)
    return required_context_length(
        requests, tokenizer,
        configuration.max_new_tokens_reasoning,
        configuration.max_new_tokens_multiple_choice)


# The exact output suffixes the pipeline writes (pipeline/experiment.py):
# {run_id}[_wIofN]_generations.jsonl, _manifest.json, plus the worker-0
# {run_id}_exclusions.jsonl. --fresh deletes ONLY these; a bare "{run_id}*"
# glob also matched unrelated files sharing the prefix (e.g. a user's
# pilot_results.zip next to run_id "pilot") and deleted them.
_RUN_OUTPUT_GLOB_PATTERNS = (
    "{run_id}_generations.jsonl",
    "{run_id}_manifest.json",
    "{run_id}_exclusions.jsonl",
    "{run_id}_w*of*_generations.jsonl",
    "{run_id}_w*of*_manifest.json",
)


def _delete_previous_run_outputs(output_directory: Path, run_id: str) -> list[Path]:
    """Delete every prior pipeline output of ``run_id`` (generations,
    exclusions, manifest, all workers' files) so --fresh regenerates from
    nothing. Only the known output suffixes are touched."""
    deleted = sorted(
        path
        for pattern in _RUN_OUTPUT_GLOB_PATTERNS
        for path in Path(output_directory).glob(pattern.format(run_id=run_id)))
    for path in deleted:
        path.unlink()
    return deleted


def main():
    arguments = parse_arguments()

    configuration = ExperimentConfiguration.from_yaml(arguments.config)
    specification = get_model_specification(arguments.model)

    shard_partition = (
        (arguments.shard_index, arguments.shard_count)
        if arguments.shard_index is not None else None)

    # Cheap manifest-only check: skip straight past revision resolution,
    # tokenizer, model, and spaCy loading entirely when this run already has
    # every shard written (design/07 §7.7's per-row skip in run_shard still
    # protects a partially-complete run; this is the earlier, coarser gate).
    if (not arguments.fresh and arguments.skip_if_complete
            and run_is_complete(arguments.output_directory, configuration, shard_partition)):
        print(f"[run_generation] {arguments.model!r} already complete in "
              f"{arguments.output_directory}; skipping model load")
        return

    # The pin assertion: a confirmatory run may not start against an unpinned
    # (non-reproducible) model revision.
    if configuration.is_confirmatory:
        assert_revisions_pinned([specification])
    elif not specification.revision_is_pinned:
        # Non-confirmatory runs may proceed unpinned, but stamping the resolved
        # SHA on every row keeps even the pilot reproducible.
        try:
            specification = replace(
                specification,
                revision=resolve_current_revision(specification.huggingface_identifier))
            print(f"[run_generation] resolved unpinned revision to "
                  f"{specification.revision}")
        except Exception as error:  # noqa: BLE001 (offline/no-auth is survivable here)
            print(f"[run_generation] WARNING: could not resolve current revision "
                  f"({error}); rows will carry the PIN_ME placeholder",
                  file=sys.stderr)

    if arguments.fresh:
        deleted = _delete_previous_run_outputs(
            arguments.output_directory, configuration.run_id)
        print(f"[run_generation] --fresh: deleted {len(deleted)} previous "
              f"output file(s) for run {configuration.run_id!r}")

    dictionary_path = arguments.dictionary or _DEFAULT_DICTIONARY
    is_word = make_is_word(_load_dictionary(dictionary_path))

    max_model_length = _measure_max_model_length(configuration, specification, is_word)
    print(f"[run_generation] sized max_model_len={max_model_length} from the "
          f"actual request set")

    engine = VllmEngine(specification, max_model_length=max_model_length)

    print(f"[run_generation] loading spaCy pipeline ({_SPACY_MODEL_NAME}) ...")
    linguistic_pipeline = _load_linguistic_pipeline(arguments.no_spacy)
    if linguistic_pipeline is not None:
        print(f"[run_generation] spaCy loaded ({_SPACY_MODEL_NAME})")

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
