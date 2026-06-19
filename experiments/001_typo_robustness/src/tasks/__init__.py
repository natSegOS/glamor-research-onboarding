"""The tasks sub-package: reasoning items (GSM-Symbolic) and multiple-choice
items (MMLU-Pro).

Re-exports the public surface of both modules so callers can import from the
sub-package rather than drilling into the individual files:

    from tasks import generate_synthetic_reasoning_items
    from tasks import MultipleChoiceItem, make_demonstration_multiple_choice_items
"""

from tasks.reasoning import (
    REASONING_INSTRUCTION,
    SYNTHETIC_NAMES,
    OPERATION_WORDS,
    ReasoningTemplate,
    ReasoningItem,
    generate_synthetic_reasoning_items,
    load_official_gsm_symbolic,
    load_reasoning_jsonl,
)

from tasks.multiple_choice import (
    OPTION_LETTERS,
    MULTIPLE_CHOICE_INSTRUCTION,
    MultipleChoiceItem,
    load_official_mmlu_pro,
    load_multiple_choice_jsonl,
    make_demonstration_multiple_choice_items,
)

