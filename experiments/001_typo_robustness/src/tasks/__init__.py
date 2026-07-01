"""The tasks sub-package: reasoning items and multiple-choice items.

All items are loaded from pre-fetched JSONL files (tools/build_task_items.py).
The registry maps config keys to loaders; the orchestrator uses it exclusively.

Re-exports the public surface so callers can import from the sub-package:

    from tasks import ReasoningItem, load_reasoning_jsonl
    from tasks import MultipleChoiceItem, load_multiple_choice_jsonl
    from tasks import DATASET_REGISTRY, get_spec

Official HF fetchers (network, one-time pre-fetch):

    from tasks import load_official_gsm_symbolic, load_official_gsm8k
    from tasks import load_official_mmlu_pro, load_official_mmlu
"""

from tasks.reasoning import (
    REASONING_INSTRUCTION,
    ReasoningTemplate,
    ReasoningItem,
    generate_synthetic_reasoning_items,
    extract_instance_parameters,
    load_reasoning_jsonl,
    load_official_gsm_symbolic,
    load_official_gsm8k,
)

from tasks.multiple_choice import (
    OPTION_LETTERS,
    MULTIPLE_CHOICE_INSTRUCTION,
    MultipleChoiceItem,
    load_multiple_choice_jsonl,
    make_demonstration_multiple_choice_items,
    load_official_mmlu_pro,
    load_official_mmlu,
)

from tasks.registry import DatasetSpec, DATASET_REGISTRY, get_spec
