"""The idempotent, resumable generation runner.

Provenance
----------
The unit of work is one (model, task, condition-cell) shard. Every generation
row is keyed by a deterministic ID computed from (model revision, task id,
perturbation state vector, seed, is_clean); before generating, the runner skips
IDs already present on disk, so a killed session loses at most the in-flight
batch and shards are embarrassingly parallel across GPUs (design/07 §7.7).

Chat templates
--------------
The runner applies the engine's chat template to every prompt before
generation, when the engine exposes ``apply_chat_template``. This guarantees the
template is applied uniformly to clean and perturbed prompts alike and cannot be
forgotten by a caller (design/05 §5.7). The DeterministicDummyEngine has no chat
template, so dummy runs see the raw prompt — which is correct, since the dummy
answers by exact prompt lookup.
"""

from __future__ import annotations

import hashlib
import json
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from enums import Precision, Decoding
import scoring


SCHEMA_VERSION = "1.0"


def deterministic_row_id(model_revision: str, task_id: str,
                         perturbation_state_vector: dict, seed: int, is_clean: bool) -> str:
    """A stable 24-hex-character row ID, the hash of everything that defines a
    unique generation (design/07 §7.7). Two rows collide if and only if they
    would re-run the identical generation, which is exactly when we want to skip
    the second one."""
    payload = json.dumps(
        {
            "model_revision": model_revision,
            "task_id": task_id,
            "perturbation_state_vector": {
                key: perturbation_state_vector[key]
                for key in sorted(perturbation_state_vector)
            },
            "seed": seed,
            "is_clean": is_clean,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass
class GenerationRequest:
    """One prompt to run, carrying everything the canonical output schema needs
    (design/08 §8.4)."""
    task_id: str
    task_family: str
    prompt: str
    gold_answer: object
    is_clean: bool
    perturbation_state_vector: dict
    seed: int

    clean_prompt: str = ""
    edit_script: list = field(default_factory=list)
    extra_fields: dict = field(default_factory=dict)   # token metrics, ASR fields, etc.


class DeterministicDummyEngine:
    """A GPU-free fake engine for full pipeline tests. ``answer_function`` maps a
    prompt string to a generation string; the default echoes '#### 0'. Has no
    chat template, so the runner sends it the raw prompt — appropriate, since the
    dummy answers by exact prompt lookup."""

    revision = "dummy-engine-0"

    def __init__(self, answer_function: Optional[Callable[[str], str]] = None):
        self.answer_function = answer_function or (lambda prompt: "#### 0")

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        return [self.answer_function(prompt) for prompt in prompts]


class ShardManifest:
    """A per-run record of which shards have completed, so a resumed run skips
    finished shards entirely (design/07 §7.7)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.completed_shard_ids: set[str] = set()
        if self.path.exists():
            stored = json.loads(self.path.read_text())
            self.completed_shard_ids = set(stored.get("completed_shards", []))

    def is_shard_complete(self, shard_id: str) -> bool:
        return shard_id in self.completed_shard_ids

    def mark_shard_complete(self, shard_id: str) -> None:
        self.completed_shard_ids.add(shard_id)
        self.path.write_text(json.dumps(
            {"schema": SCHEMA_VERSION, "completed_shards": sorted(self.completed_shard_ids)},
            indent=1))


def _existing_row_ids(output_path: Path) -> set[str]:
    """The set of row IDs already written to an output file, for resume-skip."""
    if not output_path.exists():
        return set()
    row_ids: set[str] = set()
    for line in output_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row_ids.add(json.loads(line)["row_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return row_ids


def _apply_chat_template_if_available(engine, prompt: str) -> str:
    """Apply the engine's chat template if it has one; otherwise return the
    prompt unchanged (the dummy engine path)."""
    if hasattr(engine, "apply_chat_template"):
        return engine.apply_chat_template(prompt)
    return prompt


def run_shard(
        shard_id: str,
        requests: Sequence[GenerationRequest],
        engine,
        output_path: Path,
        manifest: ShardManifest,
        model_id: str,
        model_revision: str,
        quantization_method: Precision = Precision.FP16,
        max_new_tokens: int = 512,
        flush_every: int = 500,
        decoding: Decoding = Decoding.GREEDY,
        git_commit: str = "unpinned",
) -> int:
    """Run one shard idempotently and return the number of NEW rows written.

    Prompts are submitted in request order; callers group clean and perturbed
    families together and sort by length upstream so prefix caching is maximally
    effective (design/07 §7.8). The chat template is applied here, once, to every
    prompt.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest.is_shard_complete(shard_id):
        return 0

    already_written_row_ids = _existing_row_ids(output_path)

    pending: list[tuple[str, GenerationRequest]] = []
    for request in requests:
        row_id = deterministic_row_id(
            model_revision, request.task_id,
            request.perturbation_state_vector, request.seed, request.is_clean)
        if row_id not in already_written_row_ids:
            pending.append((row_id, request))

    rows_written = 0
    with output_path.open("a") as output_file:
        for batch_start in range(0, len(pending), flush_every):
            batch = pending[batch_start:batch_start + flush_every]

            templated_prompts = [
                _apply_chat_template_if_available(engine, request.prompt)
                for _row_id, request in batch
            ]

            batch_start_time = time.perf_counter()
            generations = engine.generate(templated_prompts, max_new_tokens)
            batch_elapsed_seconds = time.perf_counter() - batch_start_time

            for (row_id, request), generation in zip(batch, generations):
                score_result = scoring.score(
                    generation, request.gold_answer, request.task_family)

                row = {
                    "schema": SCHEMA_VERSION,
                    "row_id": row_id,
                    "shard_id": shard_id,
                    "git_commit": git_commit,
                    "timestamp": time.time(),
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "quantization_method": quantization_method,
                    "task_family": request.task_family,
                    "task_id": request.task_id,
                    "is_clean": request.is_clean,
                    "clean_prompt": request.clean_prompt or (request.prompt if request.is_clean else ""),
                    "perturbed_prompt": "" if request.is_clean else request.prompt,
                    "expected_answer": request.gold_answer,
                    "seed": request.seed,
                    "edit_script": [
                        edit.to_dict() if hasattr(edit, "to_dict") else edit
                        for edit in request.edit_script
                    ],
                    "decoding": decoding,
                    "max_new_tokens": max_new_tokens,
                    "model_output": generation,
                    "parsed_answer": score_result.parsed_answer,
                    "is_correct": score_result.is_correct,
                    "parse_status": score_result.parse_status,
                    "batch_latency_seconds": round(batch_elapsed_seconds, 3),
                    **{f"r_{key}": value
                       for key, value in request.perturbation_state_vector.items()},
                    **request.extra_fields,
                }
                output_file.write(json.dumps(row) + "\n")
                rows_written += 1

            output_file.flush()

    manifest.mark_shard_complete(shard_id)
    return rows_written


def load_generation_rows(paths: Sequence[Path]) -> list[dict]:
    """Load all generation rows from one or more JSONL files."""
    rows: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows

