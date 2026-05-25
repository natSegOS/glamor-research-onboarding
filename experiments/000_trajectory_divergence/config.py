from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ExperimentConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    quant_bits: Optional[int] = None  # None = fp16, 4 = 4-bit, 8 = 8-bit
    max_new_tokens: int = 80
    top_p: float = 0.95
    runs_per_condition: int = 5
    output_dir: str = "results"


PROMPT_VARIANTS: List[Dict] = [
    {
        "prompt_id": "clean",
        "perturbation_type": "none",
        "prompt": "Explain unlearning in one sentence."
    },
    {
        "prompt_id": "typo",
        "perturbation_type": "typo",
        "prompt": "Explain unlearning in one sentnce."
    },
    {
        "prompt_id": "numeric",
        "perturbation_type": "wording",
        "prompt": "Explain unlearning in 1 sentence."
    },
    {
        "prompt_id": "uppercase",
        "perturbation_type": "formatting",
        "prompt": "EXPLAIN unlearning in one sentence."
    },
    {
        "prompt_id": "punctuation",
        "perturbation_type": "punctuation",
        "prompt": "Explain unlearning in one sentence!!"
    },
    {
        "prompt_id": "polite",
        "perturbation_type": "tone",
        "prompt": "Please explain unlearning in one sentence."
    },
    {
        "prompt_id": "compressed",
        "perturbation_type": "compressed",
        "prompt": "Unlearning, one sentence."
    },
]


TEMPERATURES = [0.2, 0.7, 1.2]
