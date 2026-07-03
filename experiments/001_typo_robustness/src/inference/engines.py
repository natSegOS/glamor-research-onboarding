"""Inference engine: vLLM with continuous batching and prefix caching.

Greedy decoding everywhere (temperature 0, top_p 1, fixed max_new_tokens per
task family). vLLM is the only supported backend; runs on the USC GPU cluster.
The DeterministicDummyEngine in pipeline/runner.py is used for unit tests.

Chat templates
--------------
The engine applies each model's OWN chat template before generation (design/05
§5.7). Instruction-tuned models are trained on chat-formatted inputs and behave
inconsistently on raw strings.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence


from enums import Precision
from inference.roster import ModelSpecification


# All current precisions (FP16, AWQ, GPTQ) use float16 as the compute dtype.
# Passing this explicitly suppresses vLLM's deprecated torch_dtype auto-detection
# path. AWQ and GPTQ weight-packing is controlled by the separate ``quantization``
# constructor parameter; their unquantized layers (norms, embeddings) still run
# in float16 regardless.
_VLLM_COMPUTE_DTYPE = "float16"

_GREEDY_TEMPERATURE = 0.0
_GREEDY_TOP_P       = 1.0


class VllmEngine:
    """Offline batched vLLM engine with continuous batching and prefix caching.

    Callers pre-sort prompts so shared-prefix families are adjacent, maximising
    prefix-cache hits (design/07 §7.7).
    """

    def __init__(
            self,
            specification: ModelSpecification,
            seed: int = 1729,
            max_model_length: Optional[int] = None,
    ):
        from vllm import LLM, SamplingParams

        engine_arguments = dict(
            model=specification.huggingface_identifier,
            revision=specification.revision if specification.revision_is_pinned else None,
            dtype=_VLLM_COMPUTE_DTYPE,
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

        # Monotonic counter for vLLM engine request IDs (generate_streaming),
        # unique across every call for the lifetime of this engine instance —
        # several shards may stream through the same engine object in one run.
        self._next_request_id = 0

    def _greedy_sampling_params(self, max_new_tokens: int):
        return self._sampling_params_class(
            temperature=_GREEDY_TEMPERATURE,
            top_p=_GREEDY_TOP_P,
            max_tokens=max_new_tokens,
        )

    def apply_chat_template(
            self, user_message: str, system_message: Optional[str] = None) -> str:
        """Wrap a user message (and optionally a system message) in the model's
        own chat template (design/05 §5.7).

        ``system_message`` is included only when the model's tokenizer chat
        template declares a "system" role slot; otherwise it is prepended to the
        user turn with a blank line so the instruction is never silently dropped.
        """
        messages = []
        template_str = getattr(self.tokenizer, "chat_template", "") or ""
        supports_system = "system" in template_str
        if system_message:
            if supports_system:
                messages.append({"role": "system", "content": system_message})
            else:
                user_message = f"{system_message}\n\n{user_message}"
        messages.append({"role": "user", "content": user_message})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        """Generate greedily for a batch of ALREADY CHAT-TEMPLATED prompts.

        Blocks until every prompt in ``prompts`` has finished, then returns all
        outputs at once — appropriate for a single small one-off call (e.g. the
        LLM-judge in judge.py), but not for the main sweep: a caller that wants
        to persist each row as soon as it is ready (surviving a mid-run crash
        with minimal lost work) should use ``generate_streaming`` instead.
        """
        outputs = self._language_model.generate(
            list(prompts), self._greedy_sampling_params(max_new_tokens))
        return [output.outputs[0].text for output in outputs]

    def generate_streaming(
            self, prompts: Sequence[str], max_new_tokens: int,
    ) -> Iterator[tuple[int, str]]:
        """Generate greedily, yielding ``(index, generated_text)`` the instant
        each prompt's decoding finishes — not only once every prompt in
        ``prompts`` has, and not in submission order.

        Drives vLLM's engine directly (``add_request`` + ``step``) instead of
        the blocking bulk ``LLM.generate`` call that ``generate`` above uses.
        vLLM's own scheduler is unchanged — it still decides how many requests
        run concurrently via continuous batching — this only changes *when*
        the caller is handed a finished result: as soon as that one request is
        done, rather than after the slowest request in the batch. That lets a
        caller (``pipeline.runner.run_shard``) persist and flush each row
        immediately, so a crash loses only requests that were still in flight,
        not an entire submitted batch (design/07 §7.6).

        ``index`` is the prompt's position in the input ``prompts`` sequence,
        letting the caller map each finished generation back to the request it
        came from.
        """
        sampling_params = self._greedy_sampling_params(max_new_tokens)
        llm_engine = self._language_model.llm_engine

        index_by_request_id: dict[str, int] = {}
        for index, prompt in enumerate(prompts):
            request_id = str(self._next_request_id)
            self._next_request_id += 1
            index_by_request_id[request_id] = index
            llm_engine.add_request(request_id, prompt, sampling_params)

        remaining = len(index_by_request_id)
        while remaining > 0:
            for output in llm_engine.step():
                if not output.finished:
                    continue
                index = index_by_request_id.pop(output.request_id)
                yield index, output.outputs[0].text
                remaining -= 1


def build_inference_engine(
        specification: ModelSpecification,
        **keyword_arguments,
) -> VllmEngine:
    return VllmEngine(specification, **keyword_arguments)
