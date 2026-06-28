"""Reasoning task items: GSM-Symbolic (primary) and GSM8K (contamination contrast).

Provenance
----------
GSM-Symbolic (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) is the primary
reasoning task. GSM8K (Cobbe et al., 2021, arXiv:2110.14168) is the standard
benchmark used as a contamination-contrast partner.

Items are pre-fetched from HuggingFace once by tools/build_task_items.py using
the ``load_official_*`` functions below, written to pinned JSONL files with SHA
provenance, and loaded during a run by ``load_reasoning_jsonl``. No live HF
loading occurs during a run.

``generate_synthetic_reasoning_items`` is an offline generator used for offline
unit tests and for Regime C operand-swap (which needs the template's answer
function). It is NOT used in the main study or pilot — those use JSONL.

All loaders return ``ReasoningItem`` objects; the pipeline is source-agnostic.
"""

from __future__ import annotations

import json
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from enums import TaskFamily
from lexicons import load_word_lexicon
from tasks._shared import (
    HASH_DELIMITED_ANSWER_PATTERN,
    build_full_prompt,
    build_instruction_and_content_scope_spans,
)

# Operation words used by the offline synthetic generator to record which
# mathematical-operation terms appear in the generated question text.  These
# are derived from the template vocabulary (the templates embed words like
# "buys", "gives away", "saves") rather than from a free-text heuristic, so
# they remain appropriate for the synthetic context.
#
# Note: this list is NOT used in any runtime path or JSONL loader.  Key terms
# for JSONL items are frozen at data-preparation time by the linguistic
# annotation stage (src/dataprep/annotate.py), which replaces the old
# runtime heuristic that previously called this list on raw question text.
# See design/04 §4.6 and the data-preparation provenance record.
OPERATION_WORDS: frozenset[str] = load_word_lexicon("operation_words.txt")

# Maximum number of times the synthetic generator tries to sample operands
# that satisfy the template's constraint before skipping a template.
_PARAMETER_SAMPLING_MAX_ATTEMPTS = 64

# Names for the synthetic generator; sampled to personalise template questions,
# mirroring Figure 1 of Mirzadeh et al. (2024) GSM-Symbolic.
SYNTHETIC_NAMES = (
    "Ava", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro",
    "Imani", "Jonas", "Keiko", "Liam", "Mara", "Noor", "Omar", "Priya",
)


REASONING_INSTRUCTION = (
    "Solve the following problem. Show your reasoning, then give the final "
    "numeric answer on a new line in the form '#### <number>'."
)


@dataclass(frozen=True)
class ReasoningTemplate:
    """A symbolic reasoning template: a format string with {name} and numeric
    operand placeholders, an answer function that computes the gold from the
    operands, the inclusive integer ranges each operand is sampled from, and an
    optional constraint the sampled operands must satisfy (for example, that the
    result is a positive integer)."""
    template_id: str
    question_format: str
    answer_function: Callable[..., float]
    operand_ranges: dict
    operand_constraint: Optional[Callable[..., bool]] = None


@dataclass
class ReasoningItem:
    """One reasoning task item, from either the official or synthetic source.

    ``template`` and ``parameters`` are populated only for synthetic items (the
    official items do not expose a Python answer function). Regime C's operand
    swap therefore applies only to synthetic items; the official items are used
    in Regimes A and B and as the contamination-controlled clean baseline. This
    matches design/04 §4.7, which notes the operand-swap control needs the
    template's answer function.
    """
    task_id: str
    task_family: TaskFamily
    source: TaskFamily                # GSM_SYMBOLIC_OFFICIAL | GSM_SYMBOLIC_SYNTHETIC
    question_text: str
    instruction: str
    gold_answer: int
    key_terms: list[str] = field(default_factory=list)

    template: Optional[ReasoningTemplate] = None
    parameters: dict = field(default_factory=dict)

    @property
    def full_prompt(self) -> str:
        return build_full_prompt(self.instruction, self.question_text)

    @property
    def scope_spans(self) -> dict:
        """Character spans of the instruction and content regions within the
        full prompt, for scope-restricted perturbation (design/03 §3.2)."""
        return build_instruction_and_content_scope_spans(self.instruction, self.question_text)

    @property
    def supports_regime_c_operand_swap(self) -> bool:
        """Regime C's reasoning operand swap needs the template's answer
        function, which only synthetic items carry."""
        return self.template is not None and bool(self.parameters)


# ---------------------------------------------------------------------------
# Synthetic generator (GSM-Symbolic-style; clearly labeled as synthetic).
# ---------------------------------------------------------------------------

SYNTHETIC_REASONING_TEMPLATES: tuple[ReasoningTemplate, ...] = (
    ReasoningTemplate(
        "buy_each",
        "{name} buys {a} boxes of pencils. Each box contains {b} pencils. "
        "{name} then gives away {c} pencils. How many pencils does {name} have left?",
        lambda a, b, c: a * b - c,
        {"a": (3, 12), "b": (4, 15), "c": (2, 20)},
        lambda a, b, c: a * b - c > 0,
    ),
    ReasoningTemplate(
        "save_weeks",
        "{name} saves {a} dollars every week for {b} weeks, then spends {c} dollars. "
        "How many dollars does {name} have remaining?",
        lambda a, b, c: a * b - c,
        {"a": (5, 20), "b": (3, 10), "c": (5, 40)},
        lambda a, b, c: a * b - c > 0,
    ),
    ReasoningTemplate(
        "read_pages",
        "A book has {a} pages. {name} reads {b} pages each day. "
        "After {c} days, how many pages remain unread?",
        lambda a, b, c: a - b * c,
        {"a": (80, 300), "b": (5, 25), "c": (2, 8)},
        lambda a, b, c: a - b * c > 0,
    ),
    ReasoningTemplate(
        "bake_trays",
        "{name} bakes {a} trays of muffins with {b} muffins per tray, "
        "then bakes {c} more muffins. How many muffins are there altogether?",
        lambda a, b, c: a * b + c,
        {"a": (2, 9), "b": (6, 12), "c": (1, 15)},
    ),
    ReasoningTemplate(
        "split_total",
        "{name} has {a} marbles and divides them equally among {b} friends. "
        "How many marbles does each friend receive?",
        lambda a, b: a // b,
        {"a": (12, 144), "b": (2, 12)},
        lambda a, b: a % b == 0,
    ),
    ReasoningTemplate(
        "twice_plus",
        "{name} scores {a} points in the first game and twice that in the second game. "
        "How many points does {name} score in total?",
        lambda a: a + 2 * a,
        {"a": (4, 40)},
    ),
    ReasoningTemplate(
        "fence_cost",
        "A fence section costs {a} dollars. {name} buys {b} sections and pays "
        "a {c} dollar delivery fee. What is the total cost in dollars?",
        lambda a, b, c: a * b + c,
        {"a": (8, 30), "b": (3, 15), "c": (5, 25)},
    ),
    ReasoningTemplate(
        "apples_left",
        "{name} picks {a} apples. {name} uses {b} apples for a pie and sells {c} apples. "
        "How many apples are left?",
        lambda a, b, c: a - b - c,
        {"a": (20, 90), "b": (3, 15), "c": (2, 20)},
        lambda a, b, c: a - b - c > 0,
    ),
)


def generate_synthetic_reasoning_items(
        item_count: int,
        seed: int,
        task_family: TaskFamily = TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
) -> list[ReasoningItem]:
    """Generate ``item_count`` fresh, contamination-free reasoning items from the
    synthetic templates. Deterministic given the seed. Cycles through the
    templates so the mix is balanced.

    Because the operands are known by construction, the key terms (operand
    digit strings, operation words present in the text, and the name) are known
    exactly — no parsing required.
    """
    random_generator = random.Random(seed)
    items: list[ReasoningItem] = []
    template_index = 0

    while len(items) < item_count:
        template = SYNTHETIC_REASONING_TEMPLATES[template_index % len(SYNTHETIC_REASONING_TEMPLATES)]

        sampled_parameters = None
        for _ in range(_PARAMETER_SAMPLING_MAX_ATTEMPTS):
            candidate_parameters = {
                operand: random_generator.randint(low, high)
                for operand, (low, high) in template.operand_ranges.items()
            }
            constraint_satisfied = (
                template.operand_constraint is None
                or template.operand_constraint(**candidate_parameters)
            )
            if not constraint_satisfied:
                continue
            candidate_gold = template.answer_function(**candidate_parameters)
            if candidate_gold == int(candidate_gold) and candidate_gold >= 0:
                sampled_parameters = candidate_parameters
                break

        if sampled_parameters is None:
            template_index += 1
            continue

        name = random_generator.choice(SYNTHETIC_NAMES)
        question_text = template.question_format.format(name=name, **sampled_parameters)
        gold_answer = int(template.answer_function(**sampled_parameters))

        key_terms = (
            [str(value) for value in sampled_parameters.values()]
            + [word for word in OPERATION_WORDS if f" {word}" in question_text]
            + [name]
        )

        items.append(ReasoningItem(
            task_id=f"{task_family}_{len(items):05d}",
            task_family=task_family,
            source=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
            question_text=question_text,
            instruction=REASONING_INSTRUCTION,
            gold_answer=gold_answer,
            key_terms=key_terms,
            template=template,
            parameters=sampled_parameters,
        ))
        template_index += 1

    return items


def _retag_legacy_gsm_symbolic(raw_value: str) -> TaskFamily:
    """Re-tag the legacy ``"gsm_symbolic"`` task-family string produced by early
    versions of load_official_gsm_symbolic to the current canonical value
    ``TaskFamily.GSM_SYMBOLIC_OFFICIAL``.

    This shim allows existing pinned JSONL files that pre-date the rename to
    load and score correctly without a re-fetch from HuggingFace.  All new
    exports write ``"gsm_symbolic_official"``.
    """
    if raw_value == TaskFamily.GSM_SYMBOLIC:  # "gsm_symbolic" — the old tag
        return TaskFamily.GSM_SYMBOLIC_OFFICIAL
    return TaskFamily(raw_value)


def load_reasoning_jsonl(
        path: Path,
        task_family: TaskFamily = TaskFamily.GSM_SYMBOLIC_OFFICIAL,
        item_count: Optional[int] = None,
) -> list[ReasoningItem]:
    """Load reasoning items from a JSONL file produced by tools/build_task_items.py.

    The file is a pre-fetched, SHA-pinned subsample; no network access is required
    during a run. The registry's ``call_loader`` passes ``task_family`` from the
    spec so items lacking a per-record field are tagged correctly.

    Parameters
    ----------
    path :
        Path to the JSONL file (one JSON object per line).
    task_family :
        Default ``task_family`` for items that lack a ``task_family`` field in the
        JSONL record. The per-record value takes precedence when present.
    item_count :
        If given, return at most this many items.
    """
    items: list[ReasoningItem] = []

    for line_index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)

        # Backward-compatibility shim: early versions of load_official_gsm_symbolic
        # wrote task_family = "gsm_symbolic" (the historical TaskFamily.GSM_SYMBOLIC
        # value) rather than the current "gsm_symbolic_official".  Re-tag on load so
        # existing JSONL files score correctly without requiring a re-fetch.
        resolved_task_family = _retag_legacy_gsm_symbolic(
            record.get("task_family", str(task_family)))
        resolved_source = _retag_legacy_gsm_symbolic(
            record.get("source", str(task_family)))

        items.append(ReasoningItem(
            task_id=record.get("task_id", f"{task_family}_{line_index:05d}"),
            task_family=resolved_task_family,
            source=resolved_source,
            question_text=record["question_text"],
            instruction=record.get("instruction", REASONING_INSTRUCTION),
            gold_answer=record["gold_answer"],
            key_terms=record.get("key_terms", []),
            template=None,
            parameters=record.get("parameters", {}),
        ))

    if item_count is not None:
        items = items[:item_count]

    return items


# ---------------------------------------------------------------------------
# Official HF fetchers (network; called once by tools/build_task_items.py).
# The ``datasets`` import is lazy so the offline test suite stays network-free.
# ---------------------------------------------------------------------------

def _parse_gsm_answer(answer_text: str) -> Optional[int]:
    """Extract the integer after ``#### `` in a GSM-style answer string.

    Uses the shared HASH_DELIMITED_ANSWER_PATTERN from tasks._shared so the
    gold-side and generation-side parsers both recognise the same surface forms
    (handles optional sign, optional dollar prefix, comma separators, and decimal
    points — the broader pattern is a superset of what GSM gold records contain,
    so matching behaviour is identical for valid gold strings).
    """
    match = HASH_DELIMITED_ANSWER_PATTERN.search(answer_text)
    if not match:
        return None
    try:
        return int(float(match.group(1).replace(",", "").replace("$", "")))
    except ValueError:
        return None


def _fetch_gsm_from_hf(
        hf_repo: str,
        hf_config: str,
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
        task_family: TaskFamily,
) -> list[ReasoningItem]:
    """Shared HF-fetch helper for GSM-style datasets (one question + #### answer)."""
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as error:
        raise ImportError(
            "fetching official datasets requires the 'datasets' package "
            "(pip install -r requirements.txt)") from error

    dataset = _load_dataset(hf_repo, hf_config, revision=dataset_revision, split="test")
    records = list(dataset)
    random.Random(seed).shuffle(records)

    items: list[ReasoningItem] = []

    for record_index, record in enumerate(records):
        if len(items) >= item_count:
            break
        gold = _parse_gsm_answer(record["answer"])
        if gold is None:
            continue
        items.append(ReasoningItem(
            task_id=f"{task_family}_{record_index:05d}",
            task_family=task_family,
            source=task_family,
            question_text=record["question"],
            instruction=REASONING_INSTRUCTION,
            gold_answer=gold,
            key_terms=[],  # frozen by tools/build_annotated_dataset.py before a run
            template=None,
            parameters={},
        ))
    return items


def load_official_gsm_symbolic(
        configuration_name: str,
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
) -> list[ReasoningItem]:
    """Fetch GSM-Symbolic items from HuggingFace (apple/GSM-Symbolic).

    Called once by tools/build_task_items.py; requires network access.
    ``configuration_name`` selects the difficulty variant (``"main"``, ``"p1"``,
    or ``"p2"``). Items are shuffled deterministically with ``seed`` before
    subsampling so the exported JSONL is reproducible.
    """
    return _fetch_gsm_from_hf(
        "apple/GSM-Symbolic", configuration_name, dataset_revision,
        item_count, seed, TaskFamily.GSM_SYMBOLIC_OFFICIAL)


def load_official_gsm8k(
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
) -> list[ReasoningItem]:
    """Fetch GSM8K items from HuggingFace (openai/gsm8k, config ``main``).

    Called once by tools/build_task_items.py; requires network access.
    Used as the contamination-contrast partner for GSM-Symbolic: running the same
    perturbations on familiar memorised items versus fresh items reveals whether
    degradation is driven by content knowledge or surface-form brittleness.
    """
    return _fetch_gsm_from_hf(
        "openai/gsm8k", "main", dataset_revision,
        item_count, seed, TaskFamily.GSM8K)
