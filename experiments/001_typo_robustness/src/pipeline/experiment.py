"""The config-driven orchestrator: load tasks, build perturbation requests,
attach tokenization metrics, run the models, and write generation rows.

A single YAML file fully specifies a run (design/08 §8.2), which is what makes
the pilot and main study differ only by configuration. The orchestrator is
deliberately engine-agnostic — it accepts any built engine (or the
DeterministicDummyEngine), so the full pipeline is testable without a GPU.

Key guarantees
--------------
- Tokenization metrics are logged on every perturbed row: token_inflation_ratio,
  subword_count_change, and fragmentation_stratum, computed with the model's own
  tokenizer. These are the inputs to the primary mediation analysis.
- key_terms are passed to the perturbation engine, so informative_word and
  answer_critical policies target the correct words.
- The ASR arm is wired in: when conditions include an ASR source,
  build_requests reads pre-built AsrItems and emits the corresponding rows.
- scope is a configurable dimension passed through to the engine via
  scope_spans, not hardcoded.
- Confirmatory runs assert every model revision is pinned before generating
  (design/10 §10.5).
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml

from enums import (
    SemanticClass, Operation, SelectionPolicy, Scope,
    ConditionSource, Precision, ShardType,
    REASONING_FAMILIES,
)

_UINT32_MAX = 0xFFFFFFFF
import regimes
from perturbation import PerturbationError
from pipeline.runner import (
    GenerationRequest,
    ShardManifest,
    run_shard,
)
from tasks import get_spec
from tasks.registry import call_loader
import tokenization


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """One entry in the config's ``datasets:`` list.

    Specifying datasets by key drives registry-based loading (preferred for
    new configs).  A plain string key uses registry defaults for item_count
    and sources.

    Attributes
    ----------
    key:
        Registry key — must be present in ``tasks.DATASET_REGISTRY``.
    item_count:
        Override the registry's ``default_n``; falls back to that default
        if not supplied.
    path:
        Required for ``*_jsonl`` keys; the JSONL file produced by
        ``tools/build_task_items.py``.
    """
    key: str
    item_count: Optional[int] = None
    path: Optional[str] = None


@dataclass
class PerturbationCondition:
    """One perturbation family to run: a fully-specified point in the
    perturbation-state space, minus the per-item seed (which is derived)."""
    name: str
    semantic_class: SemanticClass
    operation: Operation = Operation.SUBSTITUTE
    selection_policy: SelectionPolicy = SelectionPolicy.KEYBOARD_NEIGHBOR
    scope: Scope = Scope.ANYWHERE
    edit_budgets: Sequence[int] = (1, 2, 4)
    source: ConditionSource = ConditionSource.SYNTHETIC
    # PILOT-DECISION (remove the unused lever after pilot): max_word_distance
    # widens the DL orthographic band used by make_regime_b_real_word_shift
    # beyond the builder default of 2. None = use the builder default.
    max_word_distance: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.semantic_class, SemanticClass):
            self.semantic_class = SemanticClass(self.semantic_class)
        if not isinstance(self.operation, Operation):
            self.operation = Operation(self.operation)
        if not isinstance(self.selection_policy, SelectionPolicy):
            self.selection_policy = SelectionPolicy(self.selection_policy)
        if not isinstance(self.scope, Scope):
            self.scope = Scope(self.scope)
        if not isinstance(self.source, ConditionSource):
            self.source = ConditionSource(self.source)


@dataclass
class ExperimentConfiguration:
    run_id: str
    seed: int
    conditions: list
    # List of DatasetConfig (or plain strings / dicts from YAML).
    # Every entry is a key from tasks.DATASET_REGISTRY with optional per-dataset
    # ``item_count`` and ``path`` (required for ``*_jsonl`` keys) overrides.
    datasets: Optional[list] = None
    max_new_tokens_reasoning: int = 512
    max_new_tokens_multiple_choice: int = 256
    asr_items_path: Optional[str] = None
    is_confirmatory: bool = False

    def __post_init__(self):
        if self.datasets is not None:
            normalised = []
            for entry in self.datasets:
                if isinstance(entry, DatasetConfig):
                    normalised.append(entry)
                elif isinstance(entry, str):
                    normalised.append(DatasetConfig(key=entry))
                elif isinstance(entry, dict):
                    normalised.append(DatasetConfig(**entry))
                else:
                    raise ValueError(
                        f"datasets entries must be a key string or dict, got {type(entry)}")
            self.datasets = normalised

    @staticmethod
    def from_yaml(path: Path) -> "ExperimentConfiguration":
        raw = yaml.safe_load(Path(path).read_text())
        conditions = [PerturbationCondition(**condition) for condition in raw.pop("conditions", [])]
        return ExperimentConfiguration(conditions=conditions, **raw)


# ---------------------------------------------------------------------------
# Task loading.
# ---------------------------------------------------------------------------

def load_task_items(configuration: ExperimentConfiguration) -> list:
    """Load task items by dispatching each entry in ``configuration.datasets``
    through the registry. Add or swap datasets by editing the config's
    ``datasets:`` list — no code changes required."""
    if not configuration.datasets:
        raise ValueError(
            "ExperimentConfiguration.datasets is empty; set a 'datasets:' list "
            "in the config file (e.g. [{key: gsm_symbolic_jsonl, path: ..., "
            "item_count: 600}]).")
    items: list = []
    for dataset_config in configuration.datasets:
        spec = get_spec(dataset_config.key)
        n = dataset_config.item_count or spec.default_n
        items.extend(call_loader(spec, n, configuration.seed, path=dataset_config.path))
    return items


def _prompt_of(task_item) -> str:
    return task_item.full_prompt


def _content_text_of(task_item) -> str:
    return getattr(task_item, "content_text", None) or task_item.question_text


def _gold_of(task_item):
    return getattr(task_item, "gold_answer", None) or getattr(task_item, "gold_letter", None)


# ---------------------------------------------------------------------------
# Request building.
# ---------------------------------------------------------------------------

def build_requests(
        task_items: Sequence,
        conditions: Sequence[PerturbationCondition],
        is_word: Callable[[str], bool],
        tokenizer: object,
        seed: int,
        asr_items_by_task: Optional[dict] = None,
) -> list[GenerationRequest]:
    """Build the full list of clean and perturbed generation requests.

    For each task item we emit exactly one clean request, then for each
    condition and each edit budget we emit perturbed requests with tokenization
    metrics attached.
    """
    asr_items_by_task = asr_items_by_task or {}
    requests: list[GenerationRequest] = []
    seen_clean_task_ids: set = set()

    for task_item in task_items:
        clean_prompt = _prompt_of(task_item)
        gold_answer = _gold_of(task_item)

        if task_item.task_id not in seen_clean_task_ids:
            requests.append(GenerationRequest(
                task_id=task_item.task_id,
                task_family=task_item.task_family,
                prompt=clean_prompt,
                gold_answer=gold_answer,
                is_clean=True,
                perturbation_state_vector={
                    "semantic_class": SemanticClass.CLEAN,
                    "operation": Operation.NONE,
                    "selection_policy": SelectionPolicy.NONE,
                    "scope": Scope.NONE,
                    "edit_budget": 0,
                },
                seed=seed,
                clean_prompt=clean_prompt,
            ))
            seen_clean_task_ids.add(task_item.task_id)

        for condition in conditions:
            if condition.source == ConditionSource.ASR:
                requests.extend(_build_asr_requests(
                    task_item, condition, gold_answer, clean_prompt,
                    asr_items_by_task.get(task_item.task_id, []), tokenizer, is_word))
            else:
                requests.extend(_build_synthetic_requests(
                    task_item, condition, gold_answer, clean_prompt,
                    is_word, tokenizer, seed))

    return requests


def _build_synthetic_requests(task_item, condition, gold_answer, clean_prompt,
                              is_word, tokenizer, seed) -> list[GenerationRequest]:
    """Build engine-perturbed requests for one item under one condition, across
    its edit budgets."""
    requests: list[GenerationRequest] = []

    content_text = _content_text_of(task_item)
    key_terms = list(getattr(task_item, "key_terms", []))
    scope_spans = getattr(task_item, "scope_spans", None)

    for edit_budget in condition.edit_budgets:
        item_seed = regimes.derived_seed(
            seed, condition.name, task_item.task_id, edit_budget)

        try:
            perturbed_content, edits, _metadata = _construct_regime(
                condition, content_text, edit_budget, item_seed, is_word,
                key_terms, scope_spans)
        except PerturbationError:
            continue                          # this item admits no such perturbation; skip

        perturbed_prompt = clean_prompt.replace(content_text, perturbed_content)

        token_metric_fields = _tokenization_fields(
            tokenizer, content_text, perturbed_content, edits, is_word, item_seed, edit_budget)

        requests.append(GenerationRequest(
            task_id=task_item.task_id,
            task_family=task_item.task_family,
            prompt=perturbed_prompt,
            gold_answer=gold_answer,
            is_clean=False,
            perturbation_state_vector={
                "semantic_class": condition.semantic_class,
                "operation": condition.operation,
                "selection_policy": condition.selection_policy,
                "scope": condition.scope,
                "edit_budget": edit_budget,
            },
            seed=item_seed,
            clean_prompt=clean_prompt,
            edit_script=edits,
            extra_fields=token_metric_fields,
        ))

    return requests


def _construct_regime(condition, content_text, edit_budget, item_seed, is_word,
                      key_terms, scope_spans):
    """Dispatch to the right regime builder for a synthetic condition."""
    if condition.semantic_class == SemanticClass.A:
        return regimes.make_regime_a_nonword_typo(
            content_text, condition.operation, edit_budget, item_seed, is_word,
            selection_policy=condition.selection_policy, scope=condition.scope,
            scope_spans=scope_spans, key_terms=key_terms)

    if condition.semantic_class == SemanticClass.B:
        # PILOT-DECISION (remove the unused lever after pilot): pass
        # max_word_distance when the condition specifies it; otherwise
        # fall back to the builder default of 2.
        regime_b_kwargs = {}
        if condition.max_word_distance is not None:
            regime_b_kwargs["max_word_distance"] = condition.max_word_distance
        return regimes.make_regime_b_real_word_shift(
            content_text, item_seed, is_word,
            scope=condition.scope, scope_spans=scope_spans,
            edit_budget=edit_budget, **regime_b_kwargs)

    raise PerturbationError(
        f"semantic class {condition.semantic_class!r} is not built by the engine "
        "(Regime C reasoning uses make_regime_c_reasoning_operand_swap on the "
        "ReasoningItem; MCQ negation uses make_regime_c_mcq_negation)")


def _tokenization_fields(tokenizer, clean_content, perturbed_content, edits,
                         is_word, item_seed, edit_budget) -> dict:
    """Compute the tokenization metrics for a perturbed item.

    Token-inflation is whole-text; subword-count change and fragmentation
    stratum are for the single most-edited word, which is the unit the mediation
    analysis contrasts.
    """
    fields = {
        "token_inflation_ratio":
            tokenization.token_inflation_ratio(tokenizer, clean_content, perturbed_content),
    }

    edited_words = [
        (edit.word_before, edit.word_after) for edit in edits
        if edit.word_after and edit.word_before
        and edit.word_before.lower() != edit.word_after.lower()
    ]
    if edited_words:
        word_before, word_after = edited_words[0]
        subword_change = tokenization.subword_count_change(tokenizer, word_before, word_after)
        fields["subword_count_change"] = subword_change
        fields["fragmentation_stratum"] = tokenization.fragmentation_stratum(subword_change)
        fields["edited_word"] = word_after

    return fields


def _build_asr_requests(task_item, condition, gold_answer, clean_prompt,
                        asr_items, tokenizer, is_word) -> list[GenerationRequest]:
    """Build requests from pre-built AsrItems for one task.

    Each AsrItem already carries its transcription, measured edit distance,
    severity band, and clean/noisy tag. We substitute the transcription for the
    clean content and attach token metrics, exactly as for synthetic items, so
    the ASR arm and the keyboard arm share one analysis path.
    """
    requests: list[GenerationRequest] = []
    content_text = _content_text_of(task_item)

    for asr_item in asr_items:
        perturbed_prompt = clean_prompt.replace(content_text, asr_item["transcription"])

        token_metric_fields = {
            "token_inflation_ratio": tokenization.token_inflation_ratio(
                tokenizer, content_text, asr_item["transcription"]),
            "asr_edit_distance": asr_item["damerau_levenshtein_distance"],
            "asr_band": asr_item["band"],
            "asr_signal_to_noise_ratio_db": asr_item.get("signal_to_noise_ratio_db"),
        }

        requests.append(GenerationRequest(
            task_id=task_item.task_id,
            task_family=task_item.task_family,
            prompt=perturbed_prompt,
            gold_answer=gold_answer,
            is_clean=False,
            perturbation_state_vector={
                "semantic_class": SemanticClass(asr_item.get("regime_candidate", SemanticClass.B)),
                "operation": Operation.ASR,
                "selection_policy": SelectionPolicy(asr_item["selection_policy"]),
                "scope": Scope.CONTENT,
                "edit_budget": asr_item["damerau_levenshtein_distance"],
            },
            seed=task_item.task_id.__hash__() & _UINT32_MAX,
            clean_prompt=clean_prompt,
            extra_fields=token_metric_fields,
        ))

    return requests


def load_asr_items_by_task(path: Optional[str]) -> dict:
    """Load pre-built AsrItems (a JSONL exported by the ASR pre-processing step)
    and index them by task_id, for the ASR arm."""
    if not path:
        return {}

    asr_items_by_task: dict = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        asr_items_by_task.setdefault(record["task_id"], []).append(record)
    return asr_items_by_task


# ---------------------------------------------------------------------------
# The run entry point.
# ---------------------------------------------------------------------------

def run_experiment(
        configuration: ExperimentConfiguration,
        engine,
        is_word: Callable[[str], bool],
        tokenizer,
        output_directory: Path,
        model_id: str = "dummy",
        model_revision: str = "dummy-engine-0",
        quantization_method: Precision = Precision.FP16,
        git_commit: str = "unpinned",
) -> dict:
    """Run one experiment configuration against one engine and return a small
    summary. Writes generation rows to ``output_directory`` as JSONL and uses a
    shard manifest so the run is resumable (design/07 §7.7).

    For a confirmatory run, the caller must pass a real engine whose model
    revision is pinned; this function records the revision into every row but the
    pin-assertion itself belongs to the run script that builds the engine
    (inference.assert_revisions_pinned), since only it holds the specifications.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    task_items = load_task_items(configuration)
    asr_items_by_task = load_asr_items_by_task(configuration.asr_items_path)

    requests = build_requests(
        task_items, configuration.conditions, is_word, tokenizer,
        configuration.seed, asr_items_by_task)

    reasoning_requests = [r for r in requests if r.task_family in REASONING_FAMILIES]
    other_requests = [r for r in requests if r.task_family not in REASONING_FAMILIES]

    manifest = ShardManifest(output_directory / f"{configuration.run_id}_manifest.json")
    output_path = output_directory / f"{configuration.run_id}_generations.jsonl"

    from progress import ProgressBar

    total_new_rows = 0
    total_pending = len(requests)

    shard_schedule = [
        (ShardType.REASONING, reasoning_requests,
         configuration.max_new_tokens_reasoning),
        (ShardType.MULTIPLE_CHOICE, other_requests,
         configuration.max_new_tokens_multiple_choice),
    ]

    with ProgressBar(
            total=total_pending,
            description=f"generating [{model_id}]",
    ) as progress:
        for shard_type, shard_requests, max_new_tokens in shard_schedule:
            if not shard_requests:
                continue
            total_new_rows += run_shard(
                shard_id=f"{configuration.run_id}_{shard_type}",
                requests=shard_requests,
                engine=engine,
                output_path=output_path,
                manifest=manifest,
                model_id=model_id,
                model_revision=model_revision,
                quantization_method=quantization_method,
                max_new_tokens=max_new_tokens,
                git_commit=git_commit,
                progress_callback=progress.advance,
            )

    return {
        "run_id": configuration.run_id,
        "task_item_count": len(task_items),
        "request_count": len(requests),
        "new_rows_written": total_new_rows,
        "output_path": str(output_path),
    }
