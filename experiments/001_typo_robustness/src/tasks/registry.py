"""Pluggable dataset registry for Experiment 001.

Every dataset is registered as a ``DatasetSpec`` that carries its JSONL loader,
task type, scoring family, default item count, and HuggingFace coordinates for
the pre-fetch step (tools/build_task_items.py).

All items are loaded from pre-fetched, SHA-pinned JSONL files produced by
tools/build_task_items.py. No live HuggingFace loading occurs during a run.

To add a dataset:
  1. Pre-fetch it in tools/build_task_items.py and write a JSONL file.
  2. Add a DatasetSpec entry below pointing to the appropriate JSONL loader.
  3. Add the key to the config's ``datasets:`` list with a ``path:`` entry.

Registered datasets
-------------------
gsm_symbolic_jsonl   primary reasoning                  (confirmatory, N=600)
mmlu_pro_jsonl       primary MCQ                        (confirmatory, N=600)
gsm8k_jsonl          contamination-contrast reasoning   (paired with GSM-Symbolic)
mmlu_jsonl           contamination-contrast MCQ         (paired with MMLU-Pro)

Smoke-test keys (unit tests only, no pre-fetched data required):
  gsm_symbolic_synthetic   offline templated generator
  mcq_demo                 5-item hardcoded set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from enums import TaskFamily, REASONING_FAMILIES, MCQ_FAMILIES


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Description of one dataset / task source.

    Attributes
    ----------
    key:
        The config-level string identifier.
    loader:
        Callable that returns a list of task items.
    task_type:
        ``"reasoning"`` (numeric answer) or ``"mcq"`` (letter answer).
    task_family:
        The ``TaskFamily`` enum value; determines scorer routing and is passed
        to JSONL loaders as the default when items lack a ``task_family`` field.
    scorer_families:
        The frozenset from ``enums`` that the scorer looks up at runtime.
    default_n:
        Default item count when the config does not override it.
    role:
        ``"primary"`` (confirmatory, N≈600), ``"descriptive"`` (generalization
        probe, not held to N≈600), or ``"smoke_test"`` (offline tests only).
    hf_repo:
        HuggingFace dataset repo ID, for the pre-fetch script.
    hf_config:
        HuggingFace dataset config name (if required by the repo).
    hf_split:
        HuggingFace split to fetch (default ``"test"``).
    hf_default_revision:
        Pinned revision SHA written by build_task_items.py into PROVENANCE.json.
    """
    key: str
    loader: Optional[Callable]
    task_type: str
    task_family: TaskFamily
    scorer_families: frozenset
    default_n: int
    role: str
    hf_repo: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: str = "test"
    hf_default_revision: Optional[str] = None
    load_fn_args: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

from tasks.reasoning import (
    generate_synthetic_reasoning_items,
    load_reasoning_jsonl,
)
from tasks.multiple_choice import (
    load_multiple_choice_jsonl,
    make_demonstration_multiple_choice_items,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, DatasetSpec] = {

    # ------------------------------------------------------------------
    # Primary confirmatory datasets (N=600, JSONL from pre-fetch)
    # ------------------------------------------------------------------

    "gsm_symbolic_jsonl": DatasetSpec(
        key="gsm_symbolic_jsonl",
        loader=load_reasoning_jsonl,
        task_type="reasoning",
        task_family=TaskFamily.GSM_SYMBOLIC_OFFICIAL,
        scorer_families=REASONING_FAMILIES,
        default_n=600,
        role="primary",
        hf_repo="apple/GSM-Symbolic",
        hf_config="main",
        hf_split="test",
    ),
    "mmlu_pro_jsonl": DatasetSpec(
        key="mmlu_pro_jsonl",
        loader=load_multiple_choice_jsonl,
        task_type="mcq",
        task_family=TaskFamily.MMLU_PRO,
        scorer_families=MCQ_FAMILIES,
        default_n=600,
        role="primary",
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_split="test",
    ),

    # ------------------------------------------------------------------
    # Contamination-contrast datasets (standard benchmarks, paired with
    # primaries to show whether degradation is driven by content knowledge
    # or surface-form brittleness — design/04 §4.8)
    # ------------------------------------------------------------------

    "gsm8k_jsonl": DatasetSpec(
        key="gsm8k_jsonl",
        loader=load_reasoning_jsonl,
        task_type="reasoning",
        task_family=TaskFamily.GSM8K,
        scorer_families=REASONING_FAMILIES,
        default_n=600,
        role="contamination_contrast",
        hf_repo="openai/gsm8k",
        hf_config="main",
        hf_split="test",
    ),
    "mmlu_jsonl": DatasetSpec(
        key="mmlu_jsonl",
        loader=load_multiple_choice_jsonl,
        task_type="mcq",
        task_family=TaskFamily.MMLU,
        scorer_families=MCQ_FAMILIES,
        default_n=600,
        role="contamination_contrast",
        hf_repo="cais/mmlu",
        hf_config="all",
        hf_split="test",
    ),

    # ------------------------------------------------------------------
    # Smoke-test / offline defaults (unit tests only)
    # ------------------------------------------------------------------

    "gsm_symbolic_synthetic": DatasetSpec(
        key="gsm_symbolic_synthetic",
        loader=generate_synthetic_reasoning_items,
        task_type="reasoning",
        task_family=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
        scorer_families=REASONING_FAMILIES,
        default_n=150,
        role="smoke_test",
        hf_repo=None,
    ),
    "mcq_demo": DatasetSpec(
        key="mcq_demo",
        loader=make_demonstration_multiple_choice_items,
        task_type="mcq",
        task_family=TaskFamily.MCQ_DEMO,
        scorer_families=MCQ_FAMILIES,
        default_n=5,
        role="smoke_test",
        hf_repo=None,
    ),
}


def get_spec(key: str) -> DatasetSpec:
    """Return the ``DatasetSpec`` for ``key``, raising ``KeyError`` with a
    helpful message if the key is not registered."""
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset key {key!r} is not registered. "
            f"Available keys: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key]


def call_loader(
        spec: DatasetSpec,
        item_count: int,
        seed: int,
        path: Optional[str] = None) -> list:
    """Invoke ``spec.loader`` with the correct call signature and return up to
    ``item_count`` items.

    JSONL loaders (keys ending in ``_jsonl``) read from ``path`` and receive
    ``task_family`` from the spec so items lacking a per-record family field
    are tagged correctly. ``path`` is required for all ``_jsonl`` keys.

    Synthetic / demo loaders need no path and use positional or no arguments.
    """
    if spec.loader is None:
        raise NotImplementedError(
            f"Dataset {spec.key!r} has no loader — add one to registry.py.")

    from pathlib import Path as _Path

    # JSONL sources — load from pre-exported file; path required.
    if spec.key.endswith("_jsonl"):
        if not path:
            raise ValueError(
                f"Dataset {spec.key!r} requires a 'path' entry in the dataset "
                "config (the JSONL file produced by tools/build_task_items.py).")
        try:
            items = spec.loader(_Path(path), task_family=spec.task_family, item_count=item_count)
        except TypeError:
            items = spec.loader(_Path(path), spec.task_family)
        return items[:item_count]

    # Synthetic offline generator — positional (item_count, seed).
    if spec.key == "gsm_symbolic_synthetic":
        return spec.loader(item_count, seed)[:item_count]

    # Demo smoke-test — no arguments; repeat to fill item_count.
    if spec.key == "mcq_demo":
        base = spec.loader()
        repeats = (item_count // max(len(base), 1)) + 1
        return (base * repeats)[:item_count]

    raise ValueError(f"No call signature known for dataset key {spec.key!r}.")
