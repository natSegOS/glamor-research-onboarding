"""Measure real generation throughput on the current GPU.

The pre-main-sweep benchmark deliverable (design/11 §11.2, design/07 §7.5):
runs a bounded sample of the configured experiment's real requests — split
between the reasoning and MCQ token budgets exactly as the real sweep is —
through the real vLLM engine, then reports output tokens/sec, rows/hour, and
the projected wall-clock hours for a full run.

    python tools/benchmark_throughput.py \\
        --config configs/pilot.yaml --model llama_1b --limit 200
"""

from __future__ import annotations

import argparse
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enums import REASONING_FAMILIES
from inference import VllmEngine, get_model_specification
from pipeline import (
    ExperimentConfiguration,
    build_requests,
    load_task_items,
    required_context_length,
)
from regimes import make_is_word

_DEFAULT_DICTIONARY = Path(__file__).resolve().parent.parent / "data" / "wordlists" / "en_us_pinned.txt"
_DEFAULT_SAMPLE_LIMIT = 200
# The worst-case free-T4 study size (design/03 §3.7): Module 1 full + Module 3
# (A/C) + Module 2 (one model) ≈ 30-40k generations.
_DEFAULT_PROJECTED_GENERATIONS = 42_000
_SECONDS_PER_HOUR = 3600


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, help="a roster key, e.g. llama_1b")
    parser.add_argument("--limit", type=int, default=_DEFAULT_SAMPLE_LIMIT,
                        help="total requests to time, split evenly between "
                             "reasoning and MCQ")
    parser.add_argument("--projected-generations", type=int,
                        default=_DEFAULT_PROJECTED_GENERATIONS,
                        help="run size to project wall-clock hours for")
    parser.add_argument("--dictionary", type=Path, default=_DEFAULT_DICTIONARY)
    return parser.parse_args()


def _timed_generation(engine, requests, max_new_tokens) -> dict:
    prompts = [engine.apply_chat_template(request.prompt) for request in requests]
    start_time = time.perf_counter()
    output_tokens = sum(
        generation.output_token_count
        for generation in engine.generate_streaming(prompts, max_new_tokens))
    wall_seconds = time.perf_counter() - start_time
    return {
        "rows": len(prompts),
        "wall_seconds": round(wall_seconds, 1),
        "output_tokens": output_tokens,
        "output_tokens_per_second": round(output_tokens / wall_seconds, 1),
        "rows_per_hour": round(len(prompts) / wall_seconds * _SECONDS_PER_HOUR, 1),
    }


def main():
    arguments = parse_arguments()

    configuration = ExperimentConfiguration.from_yaml(arguments.config)
    specification = get_model_specification(arguments.model)
    is_word = make_is_word({
        line.strip().lower()
        for line in arguments.dictionary.read_text().splitlines() if line.strip()})

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        specification.huggingface_identifier,
        revision=specification.revision if specification.revision_is_pinned else None)

    task_items = load_task_items(configuration)
    requests = build_requests(
        task_items, configuration.conditions, is_word, tokenizer, configuration.seed)

    per_family_limit = arguments.limit // 2
    reasoning_sample = [request for request in requests
                        if request.task_family in REASONING_FAMILIES][:per_family_limit]
    multiple_choice_sample = [request for request in requests
                              if request.task_family not in REASONING_FAMILIES
                              ][:arguments.limit - len(reasoning_sample)]

    engine = VllmEngine(
        specification,
        max_model_length=required_context_length(
            requests, tokenizer,
            configuration.max_new_tokens_reasoning,
            configuration.max_new_tokens_multiple_choice))

    phases = {
        "reasoning": _timed_generation(
            engine, reasoning_sample, configuration.max_new_tokens_reasoning),
        "multiple_choice": _timed_generation(
            engine, multiple_choice_sample, configuration.max_new_tokens_multiple_choice),
    }

    total_rows = sum(phase["rows"] for phase in phases.values())
    total_seconds = sum(phase["wall_seconds"] for phase in phases.values())
    combined_rows_per_hour = total_rows / total_seconds * _SECONDS_PER_HOUR

    for phase_name, phase in phases.items():
        print(f"{phase_name}:")
        for key, value in phase.items():
            print(f"  {key}: {value}")
    print(f"combined rows/hour: {combined_rows_per_hour:.1f}")
    print(f"projected hours for {arguments.projected_generations} generations: "
          f"{arguments.projected_generations / combined_rows_per_hour:.1f}")


if __name__ == "__main__":
    main()
