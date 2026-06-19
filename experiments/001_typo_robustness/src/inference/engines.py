"""Inference engines: vLLM (primary) and HuggingFace transformers (fallback).

Provenance
----------
Greedy decoding everywhere in the confirmatory runs: temperature 0, top_p 1,
fixed max_new_tokens (design/05 §5.6). We study input perturbation, not sampling
randomness, so sampling noise is removed. vLLM gives continuous batching and
prefix caching (design/07 §7.2); the HuggingFace engine is the fallback for
machines without vLLM.

Chat templates
--------------
BOTH engines apply each model's OWN chat template before generation (design/05
§5.7). This is essential: instruction-tuned models are trained on chat-formatted
inputs and behave inconsistently on raw strings. Applying the template in BOTH
engines (not just vLLM) means results from the cluster and from the fallback are
comparable.

The heavy imports (vllm, torch, transformers) are guarded inside the engine
constructors so this module imports on any machine; the runner and tests use the
DeterministicDummyEngine in pipeline/runner.py instead of a real model.
"""

from __future__ import annotations

from typing import Optional, Sequence

from enums import Precision
from inference.roster import ModelSpecification


class VllmEngine:
    """Offline batched vLLM engine with continuous batching and prefix caching.
    Callers pre-sort prompts so shared-prefix families are adjacent, maximizing
    prefix-cache hits (design/07 §7.8)."""

    def __init__(self, specification: ModelSpecification, seed: int = 1729,
                 max_model_length: Optional[int] = None):
        from vllm import LLM, SamplingParams        # guarded: GPU-side only

        engine_arguments = dict(
            model=specification.huggingface_identifier,
            revision=specification.revision if specification.revision_is_pinned else None,
            gpu_memory_utilization=specification.gpu_memory_utilization,
            enable_prefix_caching=specification.enable_prefix_caching,
            seed=seed,
        )
        if specification.precision in (Precision.AWQ, Precision.GPTQ):
            engine_arguments["quantization"] = specification.precision
        if max_model_length:
            engine_arguments["max_model_len"] = max_model_length

        self._language_model = LLM(**engine_arguments)
        self._sampling_params_class = SamplingParams
        self.revision = specification.revision
        self.specification = specification
        self.tokenizer = self._language_model.get_tokenizer()

    def _greedy_sampling_params(self, max_new_tokens: int):
        return self._sampling_params_class(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)

    def apply_chat_template(self, user_message: str) -> str:
        """Wrap a user message in the model's own chat template (design/05 §5.7)."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        """Generate greedily for a batch of ALREADY CHAT-TEMPLATED prompts."""
        outputs = self._language_model.generate(
            list(prompts), self._greedy_sampling_params(max_new_tokens))
        return [output.outputs[0].text for output in outputs]


class HuggingFaceEngine:
    """transformers fallback: greedy, left-padded batched generation. Slower
    than vLLM; for environments without it. Applies the chat template too, so
    its outputs are comparable with the vLLM engine's (design/05 §5.7)."""

    def __init__(self, specification: ModelSpecification, device: str = "cuda",
                 batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        revision = specification.revision if specification.revision_is_pinned else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            specification.huggingface_identifier, revision=revision, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if specification.precision == Precision.AWQ:
            # bitsandbytes nf4 stands in for AWQ when AWQ wheels are unavailable
            # on a given machine (design/05 §5.4). The main sweep uses real AWQ
            # via vLLM; this is purely a fallback convenience.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            specification.huggingface_identifier, revision=revision,
            quantization_config=quantization_config,
            dtype=torch.float16 if quantization_config is None else None,
            device_map="auto")

        self.revision = specification.revision
        self.specification = specification
        self.batch_size = batch_size
        self._torch = torch

    def apply_chat_template(self, user_message: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        generations: list[str] = []
        for batch_start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[batch_start:batch_start + self.batch_size])
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)

            with self._torch.no_grad():
                generated_ids = self.model.generate(
                    **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id)

            prompt_length = encoded["input_ids"].shape[1]
            for generated_row in generated_ids:
                completion_ids = generated_row[prompt_length:]
                generations.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        return generations


def build_inference_engine(specification: ModelSpecification, backend: str = "vllm", **keyword_arguments):
    if backend == "vllm":
        return VllmEngine(specification, **keyword_arguments)
    if backend == "huggingface":
        return HuggingFaceEngine(specification, **keyword_arguments)
    raise ValueError(f"unknown backend {backend!r}")

