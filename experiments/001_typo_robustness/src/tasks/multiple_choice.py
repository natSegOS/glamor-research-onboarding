"""Multiple-choice task items: MMLU-Pro.

Provenance
----------
MMLU-Pro (Wang et al., NeurIPS 2024, arXiv:2406.01574) is the multiple-choice
task. It has ten options (reducing guess-rate confounds) and is markedly more
prompt-stable than MMLU, which is exactly what a robustness study needs in its
clean baseline. See docs/PROVENANCE.md §1.2 and design/04 §4.3.

Official source (verified June 2026)
------------------------------------
HuggingFace dataset ``TIGER-Lab/MMLU-Pro``, license MIT. The test split has
12,032 examples. Each row's fields are:
    question_id   int
    question      str
    options       sequence[str]   (the option texts, in order)
    answer        str             (the correct option LETTER, e.g. "C")
    answer_index  int             (0-based index of the correct option)
    cot_content   str             (a chain-of-thought exemplar; unused here)
    category      str             (subject, used for stratified subsampling)
    src           str

We convert each row into a ``MultipleChoiceItem`` whose options are a dict from
letter to text, matching the format the rest of the pipeline expects.
"""

from __future__ import annotations

import json
import random

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from enums import TaskFamily


OPTION_LETTERS = "ABCDEFGHIJ"          # MMLU-Pro has up to ten options


MULTIPLE_CHOICE_INSTRUCTION = (
    "Answer the following multiple-choice question. Think briefly, then give "
    "your final answer on a new line as 'Answer: <letter>'."
)


@dataclass
class MultipleChoiceItem:
    """One multiple-choice item.

    ``gold_letter_if_negated`` is optional and only set for items that have been
    annotated as negation-flippable, which Regime C's MCQ negation requires. It
    is absent for ordinary items, which therefore do not enter Regime C.
    """
    task_id: str
    task_family: TaskFamily
    question: str
    options: dict                      # {"A": "...", "B": "...", ...}
    gold_letter: str
    category: str = ""
    instruction: str = MULTIPLE_CHOICE_INSTRUCTION
    gold_letter_if_negated: Optional[str] = None
    key_terms: list[str] = field(default_factory=list)

    @property
    def content_text(self) -> str:
        rendered_options = "\n".join(f"{letter}. {text}" for letter, text in self.options.items())
        return f"{self.question}\n{rendered_options}"

    @property
    def full_prompt(self) -> str:
        return f"{self.instruction}\n\n{self.content_text}"

    @property
    def scope_spans(self) -> dict:
        instruction_length = len(self.instruction)
        content_start = instruction_length + len("\n\n")
        return {
            "instruction": (0, instruction_length),
            "content": (content_start, content_start + len(self.content_text)),
        }

    @property
    def option_count(self) -> int:
        return len(self.options)


def _options_sequence_to_letter_dict(options_sequence) -> dict:
    """Convert MMLU-Pro's ordered option-text sequence into a letter->text
    dict, e.g. ["Oxygen", "Nitrogen"] -> {"A": "Oxygen", "B": "Nitrogen"}."""
    return {OPTION_LETTERS[index]: option_text
            for index, option_text in enumerate(options_sequence)}


def load_official_mmlu_pro(
        dataset_revision: Optional[str] = None,
        item_count: Optional[int] = None,
        categories: Optional[list[str]] = None,
        seed: int = 1729,
) -> list[MultipleChoiceItem]:
    """Load the official ``TIGER-Lab/MMLU-Pro`` test split (design/04 §4.3,
    docs/PROVENANCE.md §1.2).

    Parameters
    ----------
    dataset_revision :
        Pin the HuggingFace dataset revision for reproducibility (the
        maintainers have fixed option-formatting issues over time).
    item_count :
        If given, take a subject-stratified subsample of about this many items,
        allocated proportionally across ``category`` so the subsample mirrors the
        full set's subject distribution.
    categories :
        Restrict to these subjects before subsampling, if given.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "loading the official MMLU-Pro dataset requires the 'datasets' "
            "package (pip install -r requirements.txt)") from error

    dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="test",
        revision=dataset_revision,
    )

    rows = list(dataset)
    if categories is not None:
        allowed_categories = set(categories)
        rows = [row for row in rows if row["category"] in allowed_categories]

    if item_count is not None and item_count < len(rows):
        rows = _stratified_subsample_by_category(rows, item_count, seed)

    items: list[MultipleChoiceItem] = []
    for row_index, row in enumerate(rows):
        options = _options_sequence_to_letter_dict(row["options"])
        items.append(MultipleChoiceItem(
            task_id=f"mmlu_pro_{row_index:05d}",
            task_family=TaskFamily.MMLU_PRO,
            question=row["question"],
            options=options,
            gold_letter=row["answer"],
            category=row.get("category", ""),
            key_terms=[],
        ))

    return items


def _stratified_subsample_by_category(rows, item_count, seed):
    """Return about ``item_count`` rows, allocated proportionally across the
    ``category`` field so the subsample mirrors the full distribution."""
    rows_by_category: dict = defaultdict(list)
    for row in rows:
        rows_by_category[row["category"]].append(row)

    total_row_count = len(rows)
    random_generator = random.Random(seed)
    subsample: list = []

    for category in sorted(rows_by_category):
        category_rows = rows_by_category[category]
        proportional_allocation = round(item_count * len(category_rows) / total_row_count)
        take_count = min(len(category_rows), max(1, proportional_allocation))
        subsample.extend(random_generator.sample(category_rows, take_count))

    random_generator.shuffle(subsample)
    return subsample[:item_count]


def load_multiple_choice_jsonl(path: Path, task_family: TaskFamily = TaskFamily.MMLU_PRO) -> list[MultipleChoiceItem]:
    """Load multiple-choice items from a JSONL file in the local schema:
        {"question", "options": {"A": ...}, "answer", "gold_letter_if_negated"?,
         "key_terms"?}
    This is the loader for a pre-exported subsample committed to the repo, so a
    run does not need network access to HuggingFace. Subject stratification and
    license handling happen upstream when the subsample is exported."""
    items: list[MultipleChoiceItem] = []
    for line_index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        items.append(MultipleChoiceItem(
            task_id=f"{task_family}_{line_index:05d}",
            task_family=TaskFamily(record.get("task_family", task_family)),
            question=record["question"],
            options=record["options"],
            gold_letter=record["answer"],
            category=record.get("category", ""),
            gold_letter_if_negated=record.get("gold_letter_if_negated"),
            key_terms=record.get("key_terms", []),
        ))
    return items


def make_demonstration_multiple_choice_items() -> list[MultipleChoiceItem]:
    """A tiny built-in MCQ set for pipeline smoke tests ONLY. The study uses
    MMLU-Pro (design/04 §4.3). These five items require no network access."""
    raw_items = [
        ("Water is composed of hydrogen and which other element?",
         {"A": "Oxygen", "B": "Nitrogen", "C": "Carbon", "D": "Helium"}, "A", None,
         ["hydrogen", "element"]),
        ("The process by which plants make food using sunlight is called what?",
         {"A": "Respiration", "B": "Photosynthesis", "C": "Fermentation", "D": "Digestion"},
         "B", None, ["plants", "sunlight"]),
        ("A triangle with three equal sides is called what?",
         {"A": "Scalene", "B": "Isosceles", "C": "Equilateral", "D": "Obtuse"}, "C", None,
         ["triangle", "equal"]),
        ("Sound travels fastest through which medium?",
         {"A": "Vacuum", "B": "Air", "C": "Water", "D": "Steel"}, "D", None,
         ["sound", "fastest", "medium"]),
        ("Water boils at one hundred degrees on which temperature scale?",
         {"A": "Celsius", "B": "Fahrenheit", "C": "Kelvin", "D": "Rankine"}, "A", None,
         ["boils", "degrees"]),
    ]
    return [
        MultipleChoiceItem(
            task_id=f"mcq_demo_{index:05d}",
            task_family=TaskFamily.MCQ_DEMO,
            question=question,
            options=options,
            gold_letter=gold_letter,
            gold_letter_if_negated=gold_if_negated,
            key_terms=key_terms,
        )
        for index, (question, options, gold_letter, gold_if_negated, key_terms)
        in enumerate(raw_items)
    ]
