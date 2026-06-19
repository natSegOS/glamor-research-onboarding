"""Reasoning task items: GSM-Symbolic.

Provenance
----------
GSM-Symbolic (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) generates reasoning
questions from symbolic templates with sampled names and numeric values under
explicit constraints. We use it because it is contamination-controlled and
because the template knows its own answer function, which lets Regime C
recompute the gold answer exactly after a numeric edit. See docs/PROVENANCE.md
§1.1 and design/04 §4.2.

Two clearly-separated sources
-----------------------------
1. OFFICIAL  ``load_official_gsm_symbolic`` loads Apple's released dataset
   (``apple/GSM-Symbolic`` on HuggingFace). This is the primary, citable source.
   IMPORTANT: Apple released the templates and a sample of 50 generated
   instances per template, but NOT their data generator, so the official data
   is capped at 50 instances per template. Use this as the headline source and
   for the contamination contrast.

2. SYNTHETIC ``generate_synthetic_reasoning_items`` is an in-house generator
   written in the SPIRIT of GSM-Symbolic, used only when a per-cell sample size
   needs more items than the official 50-per-template release provides. Every
   synthetic template's answer function is unit-tested. The paper must describe
   these as "synthetic, GSM-Symbolic-style" items, never as the Apple dataset.

The official loader returns items with the SAME interface as the synthetic
generator (``ReasoningItem``), so the rest of the pipeline is agnostic to which
source produced an item; only the ``source`` field distinguishes them.
"""

from __future__ import annotations

import json
import random
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from enums import TaskFamily


# Names and operation words for the synthetic generator. The synthetic templates
# below sample a name and numeric operands, mirroring Figure 1 of the paper.
SYNTHETIC_NAMES = (
    "Ava", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro",
    "Imani", "Jonas", "Keiko", "Liam", "Mara", "Noor", "Omar", "Priya",
)

OPERATION_WORDS = (
    "each", "more", "total", "twice", "remaining", "altogether",
    "left", "per", "half", "buys", "gives", "sells",
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
        return f"{self.instruction}\n\n{self.question_text}"

    @property
    def scope_spans(self) -> dict:
        """Character spans of the instruction and content regions within the
        full prompt, for scope-restricted perturbation (design/03 §3.2)."""
        instruction_length = len(self.instruction)
        content_start = instruction_length + len("\n\n")
        return {
            "instruction": (0, instruction_length),
            "content": (content_start, content_start + len(self.question_text)),
        }

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
        for _ in range(64):
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


# ---------------------------------------------------------------------------
# Official loader (apple/GSM-Symbolic on HuggingFace).
# ---------------------------------------------------------------------------

# The final-answer line in a GSM-Symbolic answer field is "#### <number>"
# (the dataset card states the answer format matches GSM8K).
_FINAL_ANSWER_LINE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _extract_official_gold_answer(answer_field: str) -> int:
    """Parse the integer gold answer from an official GSM-Symbolic answer field,
    whose last line is '#### <number>'."""
    matches = _FINAL_ANSWER_LINE.findall(answer_field)
    if not matches:
        raise ValueError(f"no '#### <number>' line in answer field: {answer_field!r}")
    return int(float(matches[-1].replace(",", "")))


def load_official_gsm_symbolic(
        configuration_name: str = "main",
        dataset_revision: Optional[str] = None,
        item_count: Optional[int] = None,
        seed: int = 1729,
) -> list[ReasoningItem]:
    """Load Apple's official ``apple/GSM-Symbolic`` dataset (design/04 §4.2,
    docs/PROVENANCE.md §1.1).

    Parameters
    ----------
    configuration_name : "main" | "p1" | "p2"
        GSM-Symbolic difficulty variant. p1 and p2 add 1 and 2 extra clauses
        respectively (harder); "main" is the default difficulty.
    dataset_revision :
        Pin the HuggingFace dataset revision for reproducibility. Strongly
        recommended for any confirmatory run (the maintainers have fixed
        formatting issues over time).
    item_count :
        If given, take a seeded random subsample of this many items.

    Note: the official items do not carry a Python answer function, so they do
    NOT support the Regime C operand swap (that uses synthetic items). They are
    used for Regimes A and B and as the contamination-controlled clean baseline.
    Key terms are extracted by parsing numeric tokens from the question, which is
    sufficient for the keyboard-typo policies that target content words.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "loading the official GSM-Symbolic dataset requires the 'datasets' "
            "package (pip install -r requirements.txt)") from error

    dataset = load_dataset(
        "apple/GSM-Symbolic",
        name=configuration_name,
        split="test",
        revision=dataset_revision,
    )

    rows = list(dataset)
    if item_count is not None and item_count < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:item_count]

    items: list[ReasoningItem] = []
    for row_index, row in enumerate(rows):
        question_text = row["question"]
        gold_answer = _extract_official_gold_answer(row["answer"])
        numeric_key_terms = re.findall(r"\d[\d,]*", question_text)

        items.append(ReasoningItem(
            task_id=f"gsm_symbolic_official_{configuration_name}_{row_index:05d}",
            task_family=TaskFamily.GSM_SYMBOLIC_OFFICIAL,
            source=TaskFamily.GSM_SYMBOLIC_OFFICIAL,
            question_text=question_text,
            instruction=REASONING_INSTRUCTION,
            gold_answer=gold_answer,
            key_terms=numeric_key_terms,
            template=None,
            parameters={},
        ))

    return items


def load_reasoning_jsonl(
        path: Path,
        task_family: TaskFamily = TaskFamily.GSM_SYMBOLIC,
        item_count: Optional[int] = None,
) -> list[ReasoningItem]:
    """Load reasoning items from a JSONL file exported by tools/build_task_items.py.

    This is the loader for a pre-exported, pinned subsample so a confirmatory
    run does not need live network access to HuggingFace. Version pinning and
    subsampling happen upstream when the subsample is exported.

    Parameters
    ----------
    path :
        Path to the JSONL file (one JSON object per line).
    task_family :
        Overrides the ``task_family`` field in the output items. Pass
        ``TaskFamily.GSM_SYMBOLIC_OFFICIAL`` if the file was produced by
        ``tools/build_task_items.py``.
    item_count :
        If given, return at most this many items.
    """
    items: list[ReasoningItem] = []

    for line_index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        items.append(ReasoningItem(
            task_id=record.get("task_id", f"{task_family}_{line_index:05d}"),
            task_family=TaskFamily(record.get("task_family", task_family)),
            source=TaskFamily(record.get("source", task_family)),
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

