"""Multiple-choice task items: MMLU-Pro (primary) and MMLU (contamination contrast).

Provenance
----------
Items are pre-fetched from HuggingFace once by tools/build_task_items.py using
the ``load_official_*`` functions below, written to pinned JSONL files with SHA
provenance, and loaded during a run by ``load_multiple_choice_jsonl``. No live
HF loading occurs during a run.

MMLU-Pro (Wang et al., NeurIPS 2024, arXiv:2406.01574): 10-option MCQ,
  12,032 test items, subject-stratified. HF: ``TIGER-Lab/MMLU-Pro``.
MMLU (Hendrycks et al., ICLR 2021, arXiv:2009.03300): standard 4-option MCQ.
  HF: ``cais/mmlu`` (config ``all``). Used as a contamination-contrast partner.
``make_demonstration_multiple_choice_items`` is a 5-item offline set used ONLY
  for unit tests; it is not a study dataset.
"""

from __future__ import annotations

import json

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


# ---------------------------------------------------------------------------
# Official HF fetchers (network; called once by tools/build_task_items.py).
# The ``datasets`` import is lazy so the offline test suite stays network-free.
# ---------------------------------------------------------------------------

def load_official_mmlu_pro(
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
        categories: Optional[list] = None,
) -> list[MultipleChoiceItem]:
    """Fetch MMLU-Pro items from HuggingFace (TIGER-Lab/MMLU-Pro).

    Called once by tools/build_task_items.py; requires network access.
    ``categories`` optionally restricts to a list of MMLU-Pro subject strings;
    if None, all subjects are included. Items are shuffled deterministically.
    """
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as error:
        raise ImportError(
            "fetching official datasets requires the 'datasets' package "
            "(pip install -r requirements.txt)") from error

    import random as _random

    dataset = _load_dataset(
        "TIGER-Lab/MMLU-Pro", revision=dataset_revision, split="test")
    records = list(dataset)

    if categories:
        records = [r for r in records if r.get("category") in categories]

    _random.Random(seed).shuffle(records)

    items: list[MultipleChoiceItem] = []
    for i, record in enumerate(records):
        if len(items) >= item_count:
            break
        options = _options_sequence_to_letter_dict(record["options"])
        # Prefer the letter "answer" field; fall back to the integer "answer_index".
        raw_answer = record.get("answer")
        if raw_answer and isinstance(raw_answer, str) and raw_answer in options:
            gold_letter = raw_answer
        else:
            answer_index = record.get("answer_index", 0)
            gold_letter = OPTION_LETTERS[answer_index]
        items.append(MultipleChoiceItem(
            task_id=f"mmlu_pro_{i:05d}",
            task_family=TaskFamily.MMLU_PRO,
            question=record["question"],
            options=options,
            gold_letter=gold_letter,
            category=record.get("category", ""),
            key_terms=[],
        ))
    return items


def load_official_mmlu(
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
        categories: Optional[list] = None,
) -> list[MultipleChoiceItem]:
    """Fetch standard MMLU items from HuggingFace (cais/mmlu, config ``all``).

    Called once by tools/build_task_items.py; requires network access.
    Used as the contamination-contrast partner for MMLU-Pro: running the same
    perturbations on a familiar 4-option benchmark reveals whether degradation
    patterns hold across option-count and contamination exposure.
    ``categories`` optionally restricts to a list of MMLU subject strings.
    """
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as error:
        raise ImportError(
            "fetching official datasets requires the 'datasets' package "
            "(pip install -r requirements.txt)") from error

    import random as _random

    dataset = _load_dataset("cais/mmlu", "all", revision=dataset_revision, split="test")
    records = list(dataset)

    if categories:
        records = [r for r in records if r.get("subject") in categories]

    _random.Random(seed).shuffle(records)

    items: list[MultipleChoiceItem] = []
    for i, record in enumerate(records):
        if len(items) >= item_count:
            break
        # MMLU choices is a 4-item list; answer is an int 0-3.
        options = {OPTION_LETTERS[j]: choice for j, choice in enumerate(record["choices"])}
        gold_letter = OPTION_LETTERS[record["answer"]]
        items.append(MultipleChoiceItem(
            task_id=f"mmlu_{i:05d}",
            task_family=TaskFamily.MMLU,
            question=record["question"],
            options=options,
            gold_letter=gold_letter,
            category=record.get("subject", ""),
            key_terms=[],
        ))
    return items
