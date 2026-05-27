# Experiment 001 — Typographical Perturbation Robustness: Design Suite

**Project:** GLAMOR Lab (USC) research onboarding — Experiment 001
**Working title:** *Selective Invariance Under Typographical Noise: A Tokenization-Mediated, Quantization-Aware, Matched-Pair Study of Robustness in Open Instruction LLMs*
**Repo path (proposed):** `experiments/001_typo_robustness/`
**Status:** Design / pre-registration phase
**Maintainer:** (you) · **Mentor:** Zizhao Hu

---

## 0.1 What this suite is

This is the complete experimental blueprint for Experiment 001. It is deliberately split into focused documents so that each can be read, reviewed, and revised independently with no missing logical links. Every numerical choice (sample sizes, edit budgets, model roster, compute tier) is traced back to a defensible justification, either statistical, literature-based, or hardware-based. Nothing in this suite is arbitrary; where a value is provisional and must be confirmed by a pilot, that is stated explicitly with the decision rule that resolves it.

The suite is written so that it can be lifted, lightly edited, and used directly as (a) the pre-registration document, (b) the methods section of the paper, and (c) the onboarding reference for anyone joining the project.

## 0.2 The documents, in reading order

| # | Document | What it answers |
|---|---|---|
| 00 | `00_README_index.md` (this file) | What is the whole thing, what was decided, where to look |
| 01 | `01_scientific_framing.md` | Why this study, what is genuinely novel, research questions, hypotheses, related-work positioning |
| 02 | `02_formal_framework.md` | The mathematics: edit space, perturbation state vector, semantic regimes, every metric defined formally |
| 03 | `03_experimental_design.md` | The factor structure, the modules, what varies, what is held constant, the full design matrix and cell counts |
| 04 | `04_tasks_and_scoring.md` | Datasets, contamination control, deterministic answer extraction, parser specifications |
| 05 | `05_models_quant_decoding.md` | The model roster, quantization recipes held constant, decoding protocol, the quantization sub-study |
| 06 | `06_statistics_and_power.md` | Sample-size derivation tied to claims, McNemar, BCa bootstrap, mixed-effects model, mediation, multiplicity |
| 07 | `07_compute_and_engineering.md` | The compute decisions: vLLM, batching, prefix caching, tiered hardware, throughput math, checkpointing |
| 08 | `08_pipeline_and_data_schema.md` | The perturbation engine specification, edit scripts, the canonical output schema, reproducibility, test plan |
| 09 | `09_human_audit_protocol.md` | The semantic-validity audit: annotators, agreement thresholds, audit sample size, exclusion rule |
| 10 | `10_defensibility_and_prereg.md` | Reviewer-attack table with pre-emptions, reproducibility checklist, bounded-claim language, pre-registration |
| 11 | `11_execution_roadmap.md` | Stage gates with quantitative decision thresholds, week-by-week plan |

## 0.3 One-page summary of the study

We measure whether current open-weight instruction-tuned LLMs exhibit **selective invariance**: staying correct when the input carries intent-preserving typographical noise, while still changing their answer when an edit genuinely changes the question. We do this with a **matched-pair** design — every perturbed prompt has a clean twin scored on the same item — so degradation is measured on the same problems rather than across populations.

The study is built around three contributions, in priority order:

1. **Tokenization-fragmentation mediation (primary).** We do not merely show that typos hurt (already well established). We decompose the typo-induced accuracy loss into the part that flows through *subword fragmentation* (measured by a per-item token-inflation ratio τ and subword-count change) and a residual part, using a matched, fragmentation-controlled counterfactual. No prior typo study delivers this causal decomposition; the closest works stop at correlation.
2. **Quantization × typo interaction (strong secondary).** We test whether 4-bit-quantized instruction models are more, equally, or less typo-robust than their fp16 counterparts *on matched items, controlling for clean accuracy*. The one adjacent result (code generation) found quantized models surprisingly *more* robust; whether that direction holds for nonword typos on reasoning is open.
3. **Three-regime selective-invariance audit (framing).** We separate intent-preserving nonword typos, context-recoverable real-word shifts, and meaning-changing controls, and report robustness only on the first as the primary endpoint, with the others as sensitivity and over-robustness diagnostics.

Tasks are deterministically scorable and contamination-controlled (GSM-Symbolic-style fresh reasoning items; MMLU-Pro multiple choice). Models span three families and three scales. All statistics are paired (McNemar with mid-p exact correction, BCa bootstrap confidence intervals, a crossed-random-effects mixed logistic model). The whole analysis plan is pre-registered before the held-out runs.

## 0.4 Locked decisions (the decision log)

These were chosen for you and are justified in the cited document. Change them only with a recorded reason (append to §0.5).

| Decision | Choice | Rationale doc |
|---|---|---|
| Primary contribution | Tokenization-fragmentation mediation | 01 §1.4, 06 §6.8 |
| Secondary contribution | Quantization × typo interaction | 01 §1.4, 05 §5.5 |
| Framing | Selective invariance / three regimes | 01 §1.3 |
| Primary endpoint | Paired item-level correctness degradation on intent-preserving nonword typos | 02 §2.6, 06 §6.2 |
| Primary statistical test | McNemar mid-p exact, per cell, Holm within model | 06 §6.4 |
| Effect-size CI | BCa item-paired bootstrap, B = 10,000 | 06 §6.5 |
| Full-design model | Mixed-effects logistic, maximal random-effects structure | 06 §6.6 |
| Multiplicity | Benjamini–Hochberg FDR (q = 0.05) across cells | 06 §6.7 |
| Reasoning task | GSM-Symbolic-style fresh templates (primary); GSM8K for contamination contrast | 04 §4.2 |
| MCQ task | MMLU-Pro (primary); MMLU for contamination contrast | 04 §4.3 |
| Model roster | Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3 (all -Instruct) | 05 §5.2 |
| Quantization (main sweep) | AWQ W4A16, held constant | 05 §5.4 |
| Quantization sub-study | fp16 vs AWQ-4bit vs GPTQ-4bit on a fixed model subset | 05 §5.5 |
| Decoding | Greedy (temperature 0, top_p 1), fixed max_new_tokens per task | 05 §5.6 |
| Inference engine | vLLM with continuous batching + prefix caching | 07 §7.2 |
| Hardware (default) | Colab Pro L4 24 GB; RunPod A40 48 GB for 7–8B bursts | 07 §7.4 |
| Hardware (fallback) | Free Colab T4: 1B/3B fp16 + 8B AWQ, drop conditions not items | 07 §7.6 |
| Edit budgets | k ∈ {1, 2, 4} primary; k = 8 exploratory | 03 §3.4 |
| Selection policy | Keyboard-neighbor (MulTypo) primary; uniform-random ablation; informative-word-targeted as a separate condition | 03 §3.5 |
| Per-cell sample size | Provisional 600 paired items (5 pp MDE); confirmed by Stage-2 pilot | 06 §6.3 |
| Human audit | 3 annotators, 385 items/regime, Fleiss κ ≥ 0.6 gate | 09 §9.3 |

## 0.5 Change log

Append any deviation from §0.4 here with date, reason, and the document section updated. (Empty at design freeze.)

- *(none yet)*

## 0.6 Relationship to the existing repo

Experiment 000 (`000_trajectory_divergence`) already proves the core loop: load an open instruction model on a Colab T4, generate, and write a tidy results CSV with `model_id` and `quant_bits` columns; `model.py` already accepts a `quant_bits` argument and builds a `BitsAndBytesConfig`. Experiment 001 reuses that scaffolding and adds: a perturbation engine, deterministic task scorers, a vLLM-based generation runner, a statistics module, and a human-audit harness. The Allegro layout (`paper/` submodule + self-contained `experiments/<name>/`) is preserved. Document 08 specifies the new modules; Document 07 specifies the engine migration from raw `transformers.generate` to vLLM.

## 0.7 Glossary of recurring symbols

| Symbol | Meaning |
|---|---|
| `x` | a clean prompt string |
| `x'` | a perturbed prompt string, `x' = τ(x)` |
| `τ` (operator) | a typo transformation |
| `τ` (ratio) | token-inflation ratio = tokens(x') / tokens(x) (context disambiguates) |
| `r` | perturbation state vector (operation, unit, scope, budget, policy, regime, seed) |
| `k` | edit budget (number of primitive edits) |
| `π(x)` | the intended task/meaning of `x` |
| `A₀`, `A₁` | clean accuracy, perturbed accuracy |
| `Δ` | paired absolute degradation, `A₀ − A₁` |
| `p₁₀`, `p₀₁` | discordant-pair probabilities (clean✓perturbed✗, clean✗perturbed✓) |
| `p_d` | discordant-pair rate, `p₁₀ + p₀₁` |
| Regime A / B / C | intent-preserving nonword typo / context-recoverable real-word shift / meaning-changing control |
