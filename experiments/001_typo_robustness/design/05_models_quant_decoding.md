# 05 — Models, Quantization, and Decoding

This document fixes the model roster and the precision/decoding protocol so that nothing about *how the model was run* can be blamed for a result. The quantization choices do double duty: they make 7–8B models fit on small GPUs (Document 07) and they constitute the secondary contribution (RQ2).

---

## 5.1 Roster-selection principles

A roster must let us make claims about (a) **scale** — so we need at least three sizes; (b) **family** — so we need at least two, ideally three, tokenizer/training lineages, to avoid a single-family overclaim; and (c) **quantization** — so we need models we can run both fp16 and 4-bit. It must also be runnable on the compute we have (Document 07), which caps us at the 1B–8B range for an honest, reproducible, low-cost study. The repo already targets exactly this band (Llama-3.2-1B in Experiment 000; issue 02 explicitly asks for quantized 7–8B), so the roster is continuous with existing lab work.

## 5.2 The roster (locked)

| Model | Size | Family / tokenizer | Role | Precision(s) |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1B | Llama 3 (TikToken-BPE, 128k vocab) | scale floor, pilot model | fp16 |
| Llama-3.2-3B-Instruct | 3B | Llama 3 | mid-scale, same family | fp16 |
| Llama-3.1-8B-Instruct | 8B | Llama 3 | upper scale, same family | fp16 + AWQ-4bit |
| Qwen2.5-7B-Instruct | 7B | Qwen2 (BPE, 152k vocab) | second family | fp16 + AWQ-4bit |
| Mistral-7B-Instruct-v0.3 | 7B | Mistral (BPE, 32k vocab) | third family, *small vocab* | fp16 + AWQ-4bit |

**Why these five.** The three Llama sizes give a clean within-family scale axis (1B→3B→8B) with the tokenizer held fixed, which is essential for separating scale effects from tokenizer effects. Qwen2.5-7B and Mistral-7B-v0.3 add family diversity at a fixed ~7–8B scale. Crucially, **Mistral's 32k vocabulary versus Qwen's 152k and Llama's 128k** gives a natural contrast in subword granularity — a small-vocab tokenizer fragments words more aggressively, which is directly relevant to the tokenization-mediation contribution (RQ1): if fragmentation drives failure, the small-vocab model should show a steeper fragmentation effect. This turns the roster itself into evidence for the primary contribution.

**Pinned revisions.** Each model is pinned to a specific Hugging Face commit hash, recorded at pre-registration (Document 10) and logged on every output row (Document 08 §8.4). This defends against silent model updates.

## 5.3 The 1B pilot model's special role

Llama-3.2-1B-Instruct is the Stage-2 pilot model (Document 11 §11.2). It is the cheapest to run, it already works end-to-end in the repo, and it is the model on which we estimate the discordant-pair rate `p_d` that fixes the final per-cell `N` (Document 06 §6.3). It is in the confirmatory roster too, as the scale floor.

## 5.4 Quantization in the main sweep (locked: AWQ W4A16)

The main sweep runs 7–8B models in **AWQ 4-bit weight, 16-bit activation (W4A16)**. Rationale:
- AWQ (Activation-aware Weight Quantization) preserves accuracy better than naive round-to-nearest and is well-supported in vLLM, which we use for throughput (Document 07).
- It fits an 8B model in ~4.5–5 GB, comfortably within a free T4's 16 GB with room for the KV cache (Document 07 §7.3).
- It is deterministic given fixed weights, so it does not introduce a stochastic confound.

The quantization *method* is held constant within every cell of the main sweep so that any difference attributed to "quantization" is the difference between fp16 and *this one recipe*, not a mix of recipes. This is a confound-control decision (Document 03 §3.3).

## 5.5 The quantization sub-study (the RQ2 engine)

Holding one recipe constant answers "does AWQ-4bit change robustness," but a reviewer will ask whether the answer is AWQ-specific. The sub-study guards against that:

- **Design:** on a fixed subset (the three 7–8B models, Module 1 + Module 3-A conditions), run **fp16 vs AWQ-4bit vs GPTQ-4bit**.
- **Analysis:** the interaction of interest is precision × perturbation on `CCF`/`Δ`, *conditioned on clean accuracy* `A₀`. We never compare a quantized model to its fp16 self without conditioning on `A₀`, because a quantized model that is simply worse overall is a different phenomenon from one that is *less typo-robust*. Conditioning is done two ways for robustness: (a) the retention metric `R = A₁/A₀` (Document 02 §2.6, M5), which normalizes by baseline, and (b) clean-conditioned failure `CCF` (M4), which by construction only looks at items the model got right clean. Reporting both makes the conditioning transparent.
- **Pre-registered direction:** two-sided (H2, Document 01 §1.6). The only prior evidence (Fang et al., 2025, code generation) found quantized models *more* robust in 51.6% vs 42.9% of cases; we do not assume the same direction for nonword typos on reasoning, and saying so is itself a defensible, honest stance.
- **Recipe caveat in the paper:** we will state plainly that the interaction is characterized for AWQ and GPTQ at 4-bit and is not a universal claim about all quantization. Bounded claims are non-refutable claims (Document 10).

Supporting literature to cite: Fang et al. (2025, arXiv:2506.22776) and the mixed-precision AdvGLUE++ trustworthiness study (arXiv:2511.22483), which finds AWQ generally more robust than GPTQ at low precision — a useful prior for interpreting our sub-study.

## 5.6 Decoding protocol (locked: greedy)

```
do_sample      = False        # greedy
temperature    = 0            # (n/a under greedy; logged for clarity)
top_p          = 1
max_new_tokens = task-specific, frozen (Document 04)
seed           = fixed and logged
```

**Why greedy.** We are studying *input* perturbation, not *sampling* randomness. Greedy decoding makes `f(·)` a deterministic function of the input, so every paired comparison isolates the typo's effect with zero decoding variance. This is the single biggest confound-removal in the study: with sampling, a clean-vs-perturbed difference could be sampling noise; with greedy, it cannot. Note this is a deliberate change from Experiment 000, which used `do_sample=True` with a temperature sweep — appropriate there (the goal was trajectory divergence) but wrong here.

**The exploratory sampling check.** To preempt "your greedy result is a quirk of greedy," we run a small exploratory cell at temperature 0.7, top_p 0.95, with 3 fixed seeds, on one model and one task, and confirm the CCF ranking of conditions is preserved. This lives in the clearly-labeled exploratory section (Document 03 §3.2) and never feeds a confirmatory test.

## 5.7 Prompt templates and few-shot setup (held constant)

- Each model uses **its own chat template** (the `tokenizer.apply_chat_template` output), because cross-model prompt formatting differences are a known confound and the fair comparison is "each model in its intended format."
- Few-shot exemplars (if used) are a **fixed set per task**, identical across all conditions and all models, and are **never perturbed** (only the query is perturbed, except in the instruction-location module which perturbs the instruction text by design).
- A **second paraphrased template** per task is run as a robustness check (Document 03 §3.3) to show results are not an artifact of one phrasing; reported alongside the primary template.

## 5.8 Tokenizer handling for the mediation metrics

All token-inflation (`τ_tok`) and fragmentation (`Δsub`) quantities are computed with **the model's own tokenizer** (Document 02 §2.5). We never compare token counts across tokenizers as if they were the same unit. Because the roster spans 32k / 128k / 152k vocabularies, the mediation analysis is run per-model and the *pattern* (does more fragmentation predict more failure?) is what is compared across models, not the raw token counts. The small-vocab Mistral model is expected to show the strongest fragmentation effect, which is a built-in cross-check on the mechanism (Document 05 §5.2).

## 5.9 What could still go wrong, and the guard

| Risk | Guard |
|---|---|
| AWQ checkpoint unavailable for a model | fall back to GPTQ for that model in the main sweep, log the substitution, and note it; the sub-study already characterizes the AWQ/GPTQ gap |
| fp16 7–8B does not fit available GPU | run fp16 arm only on the paid tier (Document 07 §7.4); if no paid tier, restrict the quantization sub-study to one model (Document 03 §3.7 priority 3) |
| chat template changes between HF versions | pin `transformers` and tokenizer versions; log them |
| greedy decoding rejected by a model wrapper | force greedy via `generation_config`; if impossible, exclude that model and note it |
| quantized model clean accuracy collapses (bad recipe) | the `A₀` validity check (Document 04 §4.2) catches this before the main sweep; re-quantize or drop |
