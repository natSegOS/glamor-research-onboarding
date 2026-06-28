"""The idempotent, resumable generation runner.

The unit of work is one (model, task, condition-cell) shard. Every generation
row is keyed by a deterministic ID computed from (model revision, task id,
perturbation state vector, seed, is_clean); before generating, the runner skips
IDs already present on disk, so a killed session loses at most the in-flight
batch and shards are embarrassingly parallel across GPUs (design/07 §7.7).

The runner applies the model's chat template to every prompt before generation
when the engine exposes ``apply_chat_template``. This guarantees the template
is applied uniformly to clean and perturbed prompts alike and cannot be
forgotten by a caller (design/05 §5.7). The DeterministicDummyEngine has no
chat template, so dummy runs see the raw prompt — correct, since the dummy
answers by exact prompt lookup.
"""

from __future__ import annotations

import hashlib
import json
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from enums import Decoding, Precision
import scoring


SCHEMA_VERSION = "1.0"

_ROW_ID_HASH_HEX_LENGTH = 24
_MANIFEST_COMPLETED_SHARDS_KEY = "completed_shards"


def deterministic_row_id(
        model_revision: str,
        task_id: str,
        perturbation_state_vector: dict,
        seed: int,
        is_clean: bool,
) -> str:
    """A stable 24-hex-character row ID, the SHA-256 hash of everything that
    defines a unique generation (design/07 §7.7).

    Two rows share an ID if and only if they would produce the identical
    generation, which is exactly when the second should be skipped.
    """

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
    return hashlib.sha256(payload.encode()).hexdigest()[:_ROW_ID_HASH_HEX_LENGTH]


@dataclass
class GenerationRequest:
    """One prompt to run, carrying everything the canonical output schema needs
    (design/08 §8.4).
    """

    task_id: str
    task_family: str
    prompt: str
    gold_answer: object
    is_clean: bool
    perturbation_state_vector: dict
    seed: int

    clean_prompt: str = ""
    edit_script: list = field(default_factory=list)
    extra_fields: dict = field(default_factory=dict)


class DeterministicDummyEngine:
    """A GPU-free deterministic engine for full pipeline tests.

    ``answer_function`` maps a prompt string to a generation string; the
    default echoes ``'#### 0'``. Has no chat template, so the runner sends it
    the raw prompt — appropriate, since the dummy answers by exact prompt
    lookup.
    """

    revision = "dummy-engine-0"

    def __init__(
            self,
            answer_function: Optional[Callable[[str], str]] = None,
    ) -> None:

        self.answer_function = answer_function or (lambda prompt: "#### 0")

    def generate(
            self,
            prompts: Sequence[str],
            max_new_tokens: int,
    ) -> list[str]:

        return [self.answer_function(prompt) for prompt in prompts]


class ShardManifest:
    """A per-run record of which shards have completed.

    A resumed run loads this file and skips every shard already listed,
    so progress is never lost across session restarts (design/07 §7.7).
    """

    def __init__(self, manifest_path: Path) -> None:

        self.manifest_path = Path(manifest_path)
        self.completed_shard_ids: set[str] = set()

        if self.manifest_path.exists():
            stored_data = json.loads(self.manifest_path.read_text())
            self.completed_shard_ids = set(
                stored_data.get(_MANIFEST_COMPLETED_SHARDS_KEY, []))

    def is_shard_complete(self, shard_id: str) -> bool:
        return shard_id in self.completed_shard_ids

    def mark_shard_complete(self, shard_id: str) -> None:

        self.completed_shard_ids.add(shard_id)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    _MANIFEST_COMPLETED_SHARDS_KEY: sorted(
                        self.completed_shard_ids),
                },
                indent=1,
            )
        )


def _existing_row_ids(output_path: Path) -> set[str]:
    """Return the set of row IDs already written to ``output_path``.

    Used at resume-time to skip rows that were written before the session
    died, so we never double-write a row.
    """

    if not output_path.exists():
        return set()

    already_written_ids: set[str] = set()

    for line in output_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            already_written_ids.add(json.loads(line)["row_id"])
        except (json.JSONDecodeError, KeyError):
            continue

    return already_written_ids


def _apply_chat_template_if_available(engine: object, prompt: str) -> str:
    """Apply the engine's chat template if it exposes one; return the prompt
    unchanged for engines that do not (e.g. DeterministicDummyEngine).
    """

    if hasattr(engine, "apply_chat_template"):
        return engine.apply_chat_template(prompt)  # type: ignore[union-attr]
    return prompt


def run_shard(
        shard_id: str,
        requests: Sequence[GenerationRequest],
        engine: object,
        output_path: Path,
        manifest: ShardManifest,
        model_id: str,
        model_revision: str,
        quantization_method: Precision = Precision.FP16,
        max_new_tokens: int = 512,
        generation_batch_size: int = 500,
        decoding: Decoding = Decoding.GREEDY,
        git_commit: str = "unpinned",
        progress_callback: Optional[Callable[[int], None]] = None,
        score_inline: bool = True,
) -> int:
    """Run one shard idempotently and return the number of new rows written.

    Requests are submitted in the order given; callers group clean and
    perturbed families together and sort by prompt length so prefix caching is
    maximally effective (design/07 §7.8). The chat template is applied here,
    once, to every prompt.

    ``progress_callback``, when provided, is called with the number of rows
    just written after each batch completes.

    ``score_inline`` controls whether ``parsed_answer``, ``is_correct``, and
    ``parse_status`` are written into each row.  When True (the default) the
    structural inline classifier is used — VALID or UNPARSEABLE only.  The
    four-way taxonomy (including CLARIFICATION and REFUSAL) requires the formal
    post-stage classifier in tools/score_generations.py.  Set to False to
    produce raw rows for the post-stage tool alone.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest.is_shard_complete(shard_id):
        return 0

    already_written_row_ids = _existing_row_ids(output_path)

    pending_requests: list[tuple[str, GenerationRequest]] = []

    for request in requests:
        row_id = deterministic_row_id(
            model_revision,
            request.task_id,
            request.perturbation_state_vector,
            request.seed,
            request.is_clean,
        )
        if row_id not in already_written_row_ids:
            pending_requests.append((row_id, request))

    new_rows_written = 0

    with output_path.open("a") as output_file:
        for batch_offset in range(0, len(pending_requests), generation_batch_size):
            batch = pending_requests[batch_offset : batch_offset + generation_batch_size]

            chat_templated_prompts = [
                _apply_chat_template_if_available(engine, request.prompt)
                for _row_id, request in batch
            ]

            generation_start_time = time.perf_counter()
            generated_texts = engine.generate(  # type: ignore[union-attr]
                chat_templated_prompts, max_new_tokens)
            generation_elapsed_seconds = time.perf_counter() - generation_start_time

            for (row_id, request), generated_text in zip(batch, generated_texts):
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
                    "clean_prompt": (
                        request.clean_prompt
                        or (request.prompt if request.is_clean else "")
                    ),
                    "perturbed_prompt": "" if request.is_clean else request.prompt,
                    "expected_answer": request.gold_answer,
                    "seed": request.seed,
                    "edit_script": [
                        edit.to_dict() if hasattr(edit, "to_dict") else edit
                        for edit in request.edit_script
                    ],
                    "decoding": decoding,
                    "max_new_tokens": max_new_tokens,
                    "model_output": generated_text,
                    "generation_elapsed_seconds": round(
                        generation_elapsed_seconds, 3),
                    **{
                        f"r_{key}": value
                        for key, value in
                        request.perturbation_state_vector.items()
                    },
                    **request.extra_fields,
                }

                if score_inline:
                    score_result = scoring.score(
                        generated_text, request.gold_answer, request.task_family)
                    row["parsed_answer"] = score_result.parsed_answer
                    row["is_correct"] = score_result.is_correct
                    row["parse_status"] = score_result.parse_status
                output_file.write(json.dumps(row) + "\n")
                new_rows_written += 1

            output_file.flush()

            if progress_callback is not None:
                progress_callback(len(batch))

    manifest.mark_shard_complete(shard_id)
    return new_rows_written


def load_generation_rows(paths: Sequence[Path]) -> list[dict]:
    """Load all generation rows from one or more JSONL files."""

    rows: list[dict] = []

    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    return rows
