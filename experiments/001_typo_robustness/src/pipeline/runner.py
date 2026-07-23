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
chat template, so dummy runs see the raw prompt. That is correct: the dummy
answers by exact prompt lookup.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from enums import (
    Decoding, FinishReason, ParseStatus, Precision, ShardType,
    INTERACTIONAL_FAILURE_STATUSES, REASONING_FAMILIES)
from inference.engines import StreamedGeneration
from tasks.reasoning import REASONING_CHAT_EXEMPLAR_TURNS
import scoring


# 1.1: generation_elapsed_seconds (cumulative time-since-shard-start, a timing
# bug) replaced by request_wall_seconds; output_token_count and finish_reason
# added; per-shard throughput statistics recorded in the manifest.
SCHEMA_VERSION = "1.1"

_ROW_ID_HASH_HEX_LENGTH = 24
_MANIFEST_COMPLETED_SHARDS_KEY = "completed_shards"

# flush() only reaches the Drive FUSE cache on Colab; the cloud copy is
# uploaded lazily (often not until close), so a hard VM reclaim could lose
# every cached row. fsync forces the upload, bounding that loss to this many
# seconds of generation. Kept coarse because fsync on DriveFS blocks on the
# network.
_FSYNC_INTERVAL_SECONDS = 60.0
_MANIFEST_SHARD_STATISTICS_KEY = "shard_statistics"
_MANIFEST_GENERATION_PARAMETERS_KEY = "generation_parameters"

# DeterministicDummyEngine's revision id: a fixed, recognisable sentinel
# (never a real HuggingFace revision SHA) so a generation row's provenance
# makes it obvious the row came from the GPU-free dummy engine, not a model.
DUMMY_ENGINE_REVISION = "dummy-engine-0"


def deterministic_row_id(
        model_revision: str,
        task_id: str,
        perturbation_state_vector: dict,
        seed: int,
        is_clean: bool,
) -> str:
    """A stable 24-hex-character row ID, the SHA-256 hash of everything that
    defines a unique generation.

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
    the raw prompt, appropriate since the dummy answers by exact prompt
    lookup.
    """

    revision = DUMMY_ENGINE_REVISION

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

    def generate_streaming(
            self,
            prompts: Sequence[str],
            max_new_tokens: int,
    ) -> Iterator[StreamedGeneration]:
        """Mirrors ``VllmEngine.generate_streaming``'s ``StreamedGeneration``
        yield contract so ``run_shard`` can drive either engine identically.
        Nothing here is actually concurrent (there is no GPU to schedule), so
        this just yields every result in order, sufficient for exercising
        ``run_shard``'s incremental-write and resume behaviour in tests.
        Token counts are whitespace-word counts, good enough for tests.
        """
        for index, prompt in enumerate(prompts):
            text = self.answer_function(prompt)
            yield StreamedGeneration(
                prompt_index=index,
                text=text,
                output_token_count=len(text.split()),
                finish_reason=FinishReason.STOPPED,
                request_wall_seconds=0.0,
            )


class ShardManifest:
    """A per-run record of which shards have completed, plus each completed
    shard's throughput statistics.

    A resumed run loads this file and skips every shard already listed, so
    progress is never lost across session restarts. The statistics (wall
    seconds, output tokens, tokens/sec) are what the main-study compute
    budget is sized from (design/07 §7.5); per-row request_wall_seconds
    includes scheduler queue time and must not be summed.

    ``generation_parameters``, when given, is recorded and checked on
    resume: row_id doesn't encode the token budgets, so resuming after a
    budget change would silently skip every existing row and mix budgets in
    one file. A mismatch raises instead; a changed budget needs a fresh
    directory.
    """

    def __init__(
            self, manifest_path: Path,
            generation_parameters: Optional[dict] = None) -> None:

        self.manifest_path = Path(manifest_path)
        self.completed_shard_ids: set[str] = set()
        self.shard_statistics: dict[str, dict] = {}
        self.generation_parameters = generation_parameters

        if self.manifest_path.exists():
            stored_data = json.loads(self.manifest_path.read_text())
            # A manifest from a different schema version describes a different
            # row format; trusting its completed_shards would silently skip
            # regenerating every row (this happened: a committed 1.0 manifest
            # made a fresh clone generate nothing).
            if stored_data.get("schema") == SCHEMA_VERSION:
                stored_parameters = stored_data.get(
                    _MANIFEST_GENERATION_PARAMETERS_KEY)
                if (generation_parameters is not None
                        and stored_parameters is not None
                        and stored_parameters != generation_parameters):
                    raise ValueError(
                        f"manifest {self.manifest_path} records generation "
                        f"parameters {stored_parameters} but this run uses "
                        f"{generation_parameters}; resuming would mix "
                        f"incomparable rows in one file. Point the run at a "
                        f"fresh output directory.")
                self.completed_shard_ids = set(
                    stored_data.get(_MANIFEST_COMPLETED_SHARDS_KEY, []))
                self.shard_statistics = stored_data.get(
                    _MANIFEST_SHARD_STATISTICS_KEY, {})
            else:
                print(f"[runner] ignoring manifest {self.manifest_path} with "
                      f"stale schema {stored_data.get('schema')!r} "
                      f"(current: {SCHEMA_VERSION!r})")

    def is_shard_complete(self, shard_id: str) -> bool:
        return shard_id in self.completed_shard_ids

    def mark_shard_complete(
            self, shard_id: str, statistics: Optional[dict] = None) -> None:

        self.completed_shard_ids.add(shard_id)
        if statistics is not None:
            self.shard_statistics[shard_id] = statistics
        manifest_data = {
            "schema": SCHEMA_VERSION,
            _MANIFEST_COMPLETED_SHARDS_KEY: sorted(self.completed_shard_ids),
            _MANIFEST_SHARD_STATISTICS_KEY: self.shard_statistics,
        }
        if self.generation_parameters is not None:
            manifest_data[_MANIFEST_GENERATION_PARAMETERS_KEY] = (
                self.generation_parameters)
        self.manifest_path.write_text(json.dumps(manifest_data, indent=1))


def run_is_complete(
        output_directory: Path,
        configuration: object,
        shard_partition: Optional[tuple[int, int]] = None,
) -> bool:
    """True iff this run's manifest already records every shard complete, so
    the caller can skip loading a model entirely (tools/run_generation.py's
    --skip-if-complete): reads only the manifest file, never the tokenizer,
    engine, or requests.

    ``configuration`` needs only ``run_id``, ``max_new_tokens_reasoning``, and
    ``max_new_tokens_multiple_choice`` (an ExperimentConfiguration, or
    anything duck-typed the same way; not imported by name to avoid a
    circular import with pipeline.experiment).

    Checks both fixed shard types (ShardType.REASONING,
    ShardType.MULTIPLE_CHOICE) that run_experiment's shard_schedule always
    produces for this study's datasets (every config mixes reasoning and MCQ
    families). A shard type that ends up with zero requests is never marked
    complete by run_shard (its early-return only fires once a shard has run
    at least once), so a config that genuinely omits one shard type reports
    incomplete rather than risking a silent skip of unwritten rows.

    Raises ValueError, same as a real run would, if the manifest's recorded
    token budgets no longer match ``configuration`` (a fresh output directory
    is required after a budget change; see ShardManifest).
    """

    output_directory = Path(output_directory)
    worker_index, worker_count = shard_partition if shard_partition else (0, 1)
    worker_suffix = f"_w{worker_index}of{worker_count}" if shard_partition else ""
    run_id = configuration.run_id  # type: ignore[attr-defined]
    manifest_path = output_directory / f"{run_id}{worker_suffix}_manifest.json"
    if not manifest_path.exists():
        return False

    manifest = ShardManifest(manifest_path, generation_parameters={
        "max_new_tokens_reasoning": configuration.max_new_tokens_reasoning,  # type: ignore[attr-defined]
        "max_new_tokens_multiple_choice":
            configuration.max_new_tokens_multiple_choice,  # type: ignore[attr-defined]
    })
    return all(
        manifest.is_shard_complete(f"{run_id}{worker_suffix}_{shard_type}")
        for shard_type in (ShardType.REASONING, ShardType.MULTIPLE_CHOICE))


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


def chat_exemplar_turns_for_family(task_family) -> tuple:
    """The fixed few-shot (user, assistant) chat turns for a task family:
    the reasoning exemplars for reasoning families, none for MCQ (whose
    compliance needs no exemplars and is not gate-checked)."""

    return (REASONING_CHAT_EXEMPLAR_TURNS
            if task_family in REASONING_FAMILIES else ())


def _apply_chat_template_if_available(
        engine: object, prompt: str,
        exemplar_turns: Sequence[tuple[str, str]] = ()) -> str:
    """Apply the engine's chat template if it exposes one; return the prompt
    unchanged for engines that do not (e.g. DeterministicDummyEngine).
    """

    if hasattr(engine, "apply_chat_template"):
        return engine.apply_chat_template(  # type: ignore[union-attr]
            prompt, exemplar_turns=exemplar_turns)
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
        decoding: Decoding = Decoding.GREEDY,
        git_commit: str = "unpinned",
        progress_callback: Optional[Callable[[int], None]] = None,
        linguistic_pipeline: Optional[object] = None,
) -> int:
    """Run one shard idempotently and return the number of new rows written.

    Requests are submitted in the order given; callers group clean and
    perturbed families together and sort by prompt length so prefix caching is
    maximally effective (design/07 §7.8). The chat template is applied here,
    once, to every prompt.

    Every request already on disk (by ``row_id``) is skipped before
    generation even starts, so re-running this function (after a crash, or
    as one of several parallel workers each handed a partition of the same
    request list) never regenerates or duplicates a row.

    Rows are written and flushed to ``output_path`` as soon as each individual
    request finishes decoding (``engine.generate_streaming``), not after a
    fixed-size batch completes. A killed process therefore loses at most the
    handful of requests actually in flight at that moment, never an entire
    batch. And, since vLLM's own continuous-batching scheduler (not a
    batch-size parameter here) decides how many requests run concurrently,
    this is also never slower than batching would have been.

    ``progress_callback``, when provided, is called with ``1`` after every
    row is written.

    ``linguistic_pipeline``, when provided, is a loaded spaCy model passed from
    the run script (loaded once at startup).  When present, the full four-way
    parse-status classifier is used inline (VALID/UNPARSEABLE/CLARIFICATION/
    REFUSAL).  When absent, the structural two-way classifier is used as a
    fallback.  Scoring is always inline; there is no separate post-generation
    scoring step.

    Dual-accounting rule: CLARIFICATION and REFUSAL always force is_correct=0,
    even if the extractor finds a number elsewhere in the text (Workstream 5).
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

    if not pending_requests:
        manifest.mark_shard_complete(shard_id)
        return 0

    chat_templated_prompts = [
        _apply_chat_template_if_available(
            engine, request.prompt,
            chat_exemplar_turns_for_family(request.task_family))
        for _row_id, request in pending_requests
    ]

    new_rows_written = 0
    total_output_tokens = 0
    shard_start_time = time.perf_counter()
    last_fsync_monotonic = time.monotonic()

    with output_path.open("a") as output_file:
        for generation in engine.generate_streaming(  # type: ignore[union-attr]
                chat_templated_prompts, max_new_tokens):
            row_id, request = pending_requests[generation.prompt_index]
            generated_text = generation.text
            total_output_tokens += generation.output_token_count

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
                "output_token_count": generation.output_token_count,
                "finish_reason": generation.finish_reason,
                "request_wall_seconds": round(generation.request_wall_seconds, 3),
                **{
                    f"r_{key}": value
                    for key, value in
                    request.perturbation_state_vector.items()
                },
                **request.extra_fields,
            }

            # Inline scoring, always on (Workstream 5). Uses the full
            # four-way linguistic classifier when a spaCy pipeline is
            # provided; falls back to structural two-way otherwise.
            score_result = scoring.score(
                generated_text, request.gold_answer, request.task_family)
            if (linguistic_pipeline is not None
                    and score_result.parse_status == ParseStatus.UNPARSEABLE
                    and generation.finish_reason != FinishReason.TRUNCATED):
                # Upgrade UNPARSEABLE to CLARIFICATION/REFUSAL when the
                # linguistic classifier finds evidence. Never for truncated
                # generations: a chain of thought cut off by the token budget
                # is not an interactional failure, and classifying its dangling
                # first-person clauses as refusals inflated the M9 diagnostic
                # in the T4 rehearsal (design/04 §4.5).
                refined = scoring.classify_parse_status_with_linguistic_pipeline(
                    generated_text, score_result.parsed_answer, linguistic_pipeline)
                score_result = scoring.ScoreResult(
                    score_result.parsed_answer,
                    score_result.is_correct,
                    refined,
                    score_result.extraction_tier,
                )
            # Dual-accounting: interactional failures always score 0.
            final_is_correct = (
                0 if score_result.parse_status in INTERACTIONAL_FAILURE_STATUSES
                else score_result.is_correct)
            row["parsed_answer"] = score_result.parsed_answer
            row["is_correct"] = final_is_correct
            row["parse_status"] = score_result.parse_status
            row["extraction_tier"] = score_result.extraction_tier
            output_file.write(json.dumps(row) + "\n")
            output_file.flush()
            if time.monotonic() - last_fsync_monotonic >= _FSYNC_INTERVAL_SECONDS:
                os.fsync(output_file.fileno())
                last_fsync_monotonic = time.monotonic()
            new_rows_written += 1

            if progress_callback is not None:
                progress_callback(1)

    shard_wall_seconds = time.perf_counter() - shard_start_time
    manifest.mark_shard_complete(shard_id, statistics={
        "rows": new_rows_written,
        "wall_seconds": round(shard_wall_seconds, 1),
        "output_tokens": total_output_tokens,
        "output_tokens_per_second": round(
            total_output_tokens / shard_wall_seconds, 1) if shard_wall_seconds else 0.0,
        "rows_per_hour": round(
            new_rows_written / shard_wall_seconds * 3600, 1) if shard_wall_seconds else 0.0,
    })
    return new_rows_written


def load_generation_rows(paths: Sequence[Path]) -> list[dict]:
    """Load all generation rows from one or more JSONL files."""

    rows: list[dict] = []

    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    return rows
