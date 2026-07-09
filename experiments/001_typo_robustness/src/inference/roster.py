"""The model roster and the revision-pinning machinery.

Provenance
----------
The roster and the rationale for each model are in design/05 §5.2; the model
identifiers are verified on HuggingFace (June 2026) and listed in
docs/PROVENANCE.md §4. Three families (Llama, Qwen, Mistral) across three scales
(1B, 3B, 7-8B) support claims about scale and family; AWQ counterparts of the
7-8B models support the quantization sub-study (design/05 §5.5).

Revision pinning
----------------
Every model's ``revision`` is the literal string ``PIN_ME`` until
pre-registration. Reproducibility requires the exact commit SHA, which whoever
has gated-model access fills in with:

    python -c "from huggingface_hub import HfApi; \\
               print(HfApi().model_info('<model_id>').sha)"

``assert_revisions_pinned`` refuses to let a confirmatory run start while any
revision is still ``PIN_ME`` (design/10 §10.5).
"""

from __future__ import annotations

from dataclasses import dataclass

from enums import Precision


REVISION_PLACEHOLDER = "PIN_ME"


@dataclass(frozen=True)
class ModelSpecification:
    """Everything the inference engines need to load a model reproducibly."""
    roster_key: str
    huggingface_identifier: str
    revision: str                          # pinned commit SHA, or REVISION_PLACEHOLDER
    precision: Precision = Precision.FP16
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True

    @property
    def revision_is_pinned(self) -> bool:
        return self.revision != REVISION_PLACEHOLDER


# The roster. Identifiers verified on HuggingFace, June 2026 (docs/PROVENANCE.md
# §4). Revisions are PIN_ME until pre-registration. Each spec's roster_key
# appears once here; MODEL_ROSTER below indexes this tuple by that key, so the
# key can never drift out of sync with its dict entry.
_MODEL_SPECIFICATIONS: tuple[ModelSpecification, ...] = (
    # Colab pilot (T4 GPU, ungated, fp16). Qwen2.5-1.5B fits a T4 comfortably
    # and needs no HF gating approval — use for the pilot; swap to a main-study
    # model for the confirmatory run.
    #
    # enable_prefix_caching=False: a real pilot run crashed vLLM 0.10.0's
    # prefix-prefill attention kernel (vllm/attention/ops/prefix_prefill.py,
    # context_attention_fwd) with "RuntimeError: PassManager::run failed" — a
    # Triton kernel compilation failure specific to the T4's Turing
    # architecture (Volta/Turing GPUs fall back to the XFormers attention
    # backend, since FlashAttention-2 isn't supported there; the crash
    # appeared once several thousand concurrent requests with shared
    # instruction-prefix prompts triggered a prefix-cache hit — a small
    # 2-request repro with the same shared prefix did not reproduce it).
    # Disabling prefix caching avoids that code path entirely, at the cost of
    # the throughput prefix caching would otherwise give. Scoped to this
    # entry only: main-study models targeting newer GPUs (design/07 §7.2)
    # are not known to hit this and keep prefix caching enabled.
    ModelSpecification(
        "qwen_1b5_pilot", "Qwen/Qwen2.5-1.5B-Instruct", REVISION_PLACEHOLDER, Precision.FP16,
        enable_prefix_caching=False),

    # Main study models (gated; pin revisions before a confirmatory run).
    ModelSpecification(
        "llama_1b", "meta-llama/Llama-3.2-1B-Instruct", REVISION_PLACEHOLDER, Precision.FP16),
    ModelSpecification(
        "llama_3b", "meta-llama/Llama-3.2-3B-Instruct", REVISION_PLACEHOLDER, Precision.FP16),
    ModelSpecification(
        "llama_8b", "meta-llama/Llama-3.1-8B-Instruct", REVISION_PLACEHOLDER, Precision.FP16),
    ModelSpecification(
        "llama_8b_awq", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        REVISION_PLACEHOLDER, Precision.AWQ),
    ModelSpecification(
        "qwen_7b", "Qwen/Qwen2.5-7B-Instruct", REVISION_PLACEHOLDER, Precision.FP16),
    ModelSpecification(
        "qwen_7b_awq", "Qwen/Qwen2.5-7B-Instruct-AWQ", REVISION_PLACEHOLDER, Precision.AWQ),
    ModelSpecification(
        "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.3", REVISION_PLACEHOLDER, Precision.FP16),

    # Cross-family regime-audit judge. Gemma 2 9B is from Google DeepMind —
    # a distinct pre-training corpus and architecture from every generation
    # model in this study (Llama = Meta, Qwen = Alibaba, Mistral = Mistral AI).
    # Cross-family selection is required so that the judge's own tendencies
    # are not correlated with the tendencies of the models being judged.
    # The judge always runs at temperature=0 (greedy) via run_judge() in
    # src/judge.py; all decisions are cached content-addressably so the judge
    # is called at most once per unique (judge_revision, prompt_version, input).
    ModelSpecification(
        "gemma2_9b_judge", "google/gemma-2-9b-it", REVISION_PLACEHOLDER, Precision.FP16,
        gpu_memory_utilization=0.90),
)

MODEL_ROSTER: dict[str, ModelSpecification] = {
    spec.roster_key: spec for spec in _MODEL_SPECIFICATIONS}


def get_model_specification(roster_key: str) -> ModelSpecification:
    if roster_key not in MODEL_ROSTER:
        raise KeyError(
            f"unknown model roster key {roster_key!r}; "
            f"known keys are {sorted(MODEL_ROSTER)}")
    return MODEL_ROSTER[roster_key]


def resolve_current_revision(huggingface_identifier: str) -> str:
    """Look up the current main-branch commit SHA of a model on the Hub. Use
    this to fill in the PIN_ME placeholders at pre-registration time. Requires
    network access and (for gated models) authentication."""
    from huggingface_hub import HfApi
    return HfApi().model_info(huggingface_identifier).sha


def assert_revisions_pinned(specifications) -> None:
    """Raise if any specification still has the PIN_ME placeholder. Called at the
    start of a confirmatory run so that no result is ever produced against an
    unpinned (and therefore non-reproducible) model revision (design/10 §10.5)."""
    unpinned = [spec.roster_key for spec in specifications if not spec.revision_is_pinned]
    if unpinned:
        raise RuntimeError(
            "these models still have unpinned revisions (PIN_ME): "
            f"{unpinned}. Pin them with resolve_current_revision() before a "
            "confirmatory run (docs/PROVENANCE.md §4).")
