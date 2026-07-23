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
from pathlib import Path
from typing import Callable, Optional

from enums import DatasetRole, TaskFamily


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
    task_family:
        The ``TaskFamily`` enum value; determines scorer routing and is passed
        to JSONL loaders as the default when items lack a ``task_family`` field.
        The scorer dispatches on each item's ``task_family`` at runtime via
        ``enums.REASONING_FAMILIES`` and ``enums.MCQ_FAMILIES``. Those are the
        single source of truth for the reasoning/MCQ split.
    default_n:
        Default item count when the config does not override it.
    role:
        ``"primary"`` (confirmatory, N≈600), ``"contamination_contrast"``
        (standard benchmark paired with the primary to probe contamination), or
        ``"smoke_test"`` (offline tests only).
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
    task_family: TaskFamily
    default_n: int
    role: DatasetRole
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

# Each spec's key appears once here; DATASET_REGISTRY below indexes this tuple
# by spec.key, so the key can never drift out of sync with its dict entry.
_DATASET_SPECS: tuple[DatasetSpec, ...] = (

    # ------------------------------------------------------------------
    # Primary confirmatory datasets (N=600, JSONL from pre-fetch)
    # ------------------------------------------------------------------

    DatasetSpec(
        key="gsm_symbolic_jsonl",
        loader=load_reasoning_jsonl,
        task_family=TaskFamily.GSM_SYMBOLIC_OFFICIAL,
        default_n=600,
        role=DatasetRole.PRIMARY,
        hf_repo="apple/GSM-Symbolic",
        hf_config="main",
        hf_split="test",
    ),
    DatasetSpec(
        key="mmlu_pro_jsonl",
        loader=load_multiple_choice_jsonl,
        task_family=TaskFamily.MMLU_PRO,
        default_n=600,
        role=DatasetRole.PRIMARY,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_split="test",
    ),

    # ------------------------------------------------------------------
    # Contamination-contrast datasets (standard benchmarks, paired with
    # primaries to show whether degradation is driven by content knowledge
    # or surface-form brittleness, design/04 §4.8)
    # ------------------------------------------------------------------

    DatasetSpec(
        key="gsm8k_jsonl",
        loader=load_reasoning_jsonl,
        task_family=TaskFamily.GSM8K,
        default_n=600,
        role=DatasetRole.CONTAMINATION_CONTRAST,
        hf_repo="openai/gsm8k",
        hf_config="main",
        hf_split="test",
    ),
    DatasetSpec(
        key="mmlu_jsonl",
        loader=load_multiple_choice_jsonl,
        task_family=TaskFamily.MMLU,
        default_n=600,
        role=DatasetRole.CONTAMINATION_CONTRAST,
        hf_repo="cais/mmlu",
        hf_config="all",
        hf_split="test",
    ),

    # ------------------------------------------------------------------
    # Smoke-test / offline defaults (unit tests only)
    # ------------------------------------------------------------------

    DatasetSpec(
        key="gsm_symbolic_synthetic",
        loader=generate_synthetic_reasoning_items,
        task_family=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
        default_n=150,
        role=DatasetRole.SMOKE_TEST,
        hf_repo=None,
    ),
    DatasetSpec(
        key="mcq_demo",
        loader=make_demonstration_multiple_choice_items,
        task_family=TaskFamily.MCQ_DEMO,
        default_n=5,
        role=DatasetRole.SMOKE_TEST,
        hf_repo=None,
    ),
)

DATASET_REGISTRY: dict[str, DatasetSpec] = {spec.key: spec for spec in _DATASET_SPECS}


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
            f"Dataset {spec.key!r} has no loader: add one to registry.py.")

    # JSONL sources: load from pre-exported file; path required.
    if spec.key.endswith("_jsonl"):
        if not path:
            raise ValueError(
                f"Dataset {spec.key!r} requires a 'path' entry in the dataset "
                "config (the JSONL file produced by tools/build_task_items.py).")
        # load_reasoning_jsonl accepts item_count (to skip parsing rows beyond
        # what's needed); load_multiple_choice_jsonl does not, hence the retry.
        try:
            items = spec.loader(Path(path), task_family=spec.task_family, item_count=item_count)
        except TypeError:
            items = spec.loader(Path(path), spec.task_family)
        if len(items) < item_count:
            raise ValueError(
                f"Dataset {spec.key!r} pool at {path} holds only {len(items)} items "
                f"but the config requests item_count={item_count}. Re-run "
                "tools/build_task_items.py with large enough --reasoning-items/"
                "--mcq-items to rebuild the pools before this run.")
        return items[:item_count]

    # Synthetic offline generator: positional (item_count, seed).
    if spec.key == "gsm_symbolic_synthetic":
        return spec.loader(item_count, seed)[:item_count]

    # Demo smoke-test: no arguments; repeat to fill item_count.
    if spec.key == "mcq_demo":
        base = spec.loader()
        repeats = (item_count // max(len(base), 1)) + 1
        return (base * repeats)[:item_count]

    raise ValueError(f"No call signature known for dataset key {spec.key!r}.")
