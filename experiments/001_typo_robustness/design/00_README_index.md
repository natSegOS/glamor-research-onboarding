# Experiment 001; Typographical Perturbation Robustness: Design Suite

**Project:** GLAMOR Lab (USC) research onboarding; Experiment 001
**Working title:** *When Voice Meets Text: Tokenization-Mediated LLM Robustness to ASR Transcription Errors and Typographical Noise*
**Repo path (proposed):** `experiments/001_typo_robustness/`
**Status:** Design / pre-registration phase
**Maintainer:** (you) · **Mentor:** Zizhao Hu

---

## 0.1 What this suite is

This is the complete experimental blueprint for Experiment 001. It is deliberately split into focused documents so that each can be read, reviewed, and revised independently with no missing logical links. Every numerical choice (sample sizes, edit budgets, model roster, compute tier) is traced back to a defensible justification, either statistical, literature-based, or hardware-based. Where a value is provisional and must be confirmed by a pilot, that is stated explicitly with the decision rule that resolves it.

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
| 12 | `12_acoustic_asr_arm_plan.md` | PROPOSAL (not registered): the acoustic ASR arm plan pending PI approval |

## 0.3 One-page summary of the study

Voice is an increasingly dominant way of interacting with AI systems, but ASR (automatic speech recognition) transcription introduces systematic errors before the LLM ever sees the prompt: acoustic confusions (real-word substitutions like "weather" for "whether"), disfluencies, filler words, and non-standard punctuation. Understanding exactly *how and why* these surface errors degrade LLM performance is a prerequisite for mitigating them. This study addresses that causal question.

We use a **matched-pair** design and combine **two complementary perturbation sources**: controlled keyboard-adjacency typos (the experimental baseline) and realistic ASR-transcription errors (the ecologically motivated application). The dual-source design lets us test whether the same causal mechanism operates across both noise types, making the findings directly actionable for voice-interface deployments.

The study is built around three contributions, in priority order:

1. **Tokenization-fragmentation mediation (primary).** We decompose noise-induced accuracy loss into the part flowing through *subword fragmentation* and a residual, using a fragmentation-matched counterfactual. No prior typo or ASR-noise study delivers this causal decomposition; the closest works stop at correlation.
2. **Quantization × noise interaction (strong secondary).** We test whether 4-bit-quantized models are more, equally, or less robust to transcription noise than fp16 counterparts, controlling for clean accuracy; directly relevant for deployed voice assistants running quantized models.
3. **Three-regime selective-invariance audit (framing).** We separate intent-preserving noise, context-recoverable real-word shifts (the dominant ASR error type), and meaning-changing controls, reporting primary robustness only on audited intent-preserving items.

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
| Hardware (default) | USC lab GPU cluster (confirmed); free Colab T4 fallback if access pending | 07 §7.4 |
| Hardware (fallback) | Free Colab T4: 1B/3B fp16 + 8B AWQ, drop conditions not items | 07 §7.6 |
| Edit budgets | k ∈ {1, 2, 4} primary; k = 8 exploratory | 03 §3.4 |
| Selection policy | Keyboard-neighbor (MulTypo) + ASR-transcription errors (primary); uniform-random ablation; informative-word-targeted | 03 §3.5 |
| Per-cell sample size | Provisional 600 paired items (5 pp MDE); confirmed by Stage-2 pilot | 06 §6.3 |
| Human audit | ≥3 annotators (sourcing TBD with Zizhao), 385 items/regime, Fleiss κ ≥ 0.6 gate | 09 §9.3 |

## 0.5 Change log

Append any deviation from §0.4 here with date, reason, and the document section updated. (Empty at design freeze.)

- **2025-05-29:** Meeting with Zizhao: (1) ASR transcription errors added as a second perturbation source and elevated to primary real-world motivation; (2) Regime B (context-recoverable real-word shift) elevated to co-primary perturbation type alongside Regime A nonword typos; (3) hardware confirmed as USC lab GPU cluster; (4) annotator sourcing TBD with Zizhao; (5) target venue confirmed as ACL Rolling Review → EMNLP 2026. Sections updated: 00 §0.3, §0.4, §0.5; 01 §1.1, §1.3, §1.5, §1.7, §1.8; 03 §3.2, §3.5, §3.8; 07 §7.4, §7.5, §7.6; 09 §9.2; 11 §11.7; 12.

- **2026-07-09:** First pilot (Qwen2.5-1.5B-Instruct, Colab T4, 6,202 rows) reviewed; four amendments before the pilot rerun and pre-registration:
  1. **ASR arm deferred.** The TTS+Whisper pipeline is judged too unrealistic; Zizhao is proposing an alternative from a prior project. The typo (keyboard) arm proceeds alone; ASR integrates later as its own amendment. No code changes needed (ASR was never implemented in `src/`); README updated.
  2. **Primary severity moved to k=2 (reasoning) / k=4 (MCQ).** Pilot p_d at the k=1 keyboard-substitution primary: reasoning ≈ 0.10–0.11 but observed δ ≈ 3 pp (below the 5 pp MDE; a well-powered null risk); MCQ p_d = 0.02–0.03, which fires the pre-registered p_d < 0.05 contingency (06 §6.3). At k=2 reasoning shows p_d ≈ 0.16, δ ≈ 6 pp (N = 600 powered). Parameterized as `primary_edit_budget_reasoning` / `primary_edit_budget_mcq` in `configs/*.yaml`; a one-line change if the rerun revises this. MCQ may remain a near-null even at k=4 (clean A₀ = 0.34 on MMLU-Pro leaves two-thirds of items unbreakable); if so it is reported as the selective-invariance arm, not a second primary endpoint.
  3. **Pilot model corrected to Llama-3.2-1B per 11 §11.2.** The first pilot ran Qwen2.5-1.5B (HF gating convenience); p_d gates are model-conditional and the roster's scale axis is Llama-anchored, so the rerun uses `llama_1b` with HF-token auth. `qwen_1b5_pilot` remains an ungated fallback.
  4. **Mediation reporting corrected (06 §6.8).** The pilot's pooled proportion-mediated (4.85, CI [−14.5, 25.4]) was an artifact: (a) clean rows were coded with token-inflation ratio 0.0 instead of the definitional 1.0, inflating α ≈ 30×; (b) pooling task families with opposite-signed deltas (GSM8K positive, GSM-Symbolic ≈ 0/negative) cancelled the total effect in the ratio's denominator. Mediation now runs per task family with the pooled fit as labeled supplementary; the primary quantity is the indirect effect with bootstrap CI; the proportion is withheld unless the total-effect CI excludes zero. The mediator enters as token-inflation *excess* (ratio − 1): because the mediator is definitionally 1.0 on clean rows, τ ≡ 1 + perturbed·(τ−1), so a separate `perturbed:τ` interaction column (06 §6.6) is perfectly collinear; the excess term *is* the H1 interaction.

- **2026-07-10:** Stage-1 pilot completed on the spec'd model (Llama-3.2-1B-Instruct, revision 9213176, Colab T4, 6,663 rows at code commit 71daea7; ~41k rows/hour measured, so the full main sweep is a few GPU-hours). Two latent implementation bugs surfaced and fixed during the run: letter-boundary mismatch between counterfactual word selection and application (crashed on "Python" in "Python3"), and scope spans applied in full-prompt coordinates to content-only text (excluded ~half of every dataset once the instruction grew an exemplar). Gate readout (`analysis/pilot/gates.json`):
  1. **Format compliance 0.61 vs the 0.95 gate; FAILED (the only failed gate).** One worked exemplar lifted compliance from 0.02 to 0.61; a 1B model needs 2–3 few-shot exemplars. Next iteration before any freeze; re-pilot is ~30 GPU-minutes.
  2. **A₀ = 0.35 / 0.21 / 0.43 / 0.26** (gsm8k / symbolic / mmlu / mmlu_pro); the true Llama-1B anchor; compresses all effects (60–80% of items unbreakable). Decision needed with Zizhao: accept, or lift A₀ via few-shot.
  3. **p_d at primary k**: 0.23 / 0.10 / 0.18 / 0.09 → implied N = 720 / 312 / 563 / 281. N=600 confirmed for three families; GSM8K needs N=720 or the pre-registered 6 pp MDE relaxation; decision to record here before prereg. GSM8K's k=2 discordance is symmetric (δ≈−0.01, churn); its clean signal appears at k=4 (Δ=15 pp, p=0.0015).
  4. **H1 evidence (first supportive, honestly computed):** GSM8K fragmentation indirect effect −0.043, bootstrap CI [−0.071, −0.014] (excludes zero), β=−1.34 in the predicted direction; proportion-mediated correctly withheld (suppression: direct +0.028).
  5. **Contamination × fragmentation interaction replicated across both pilot models** (Qwen and Llama): indirect effect negative on contaminated GSM8K, positive on fresh GSM-Symbolic (+0.019); consistent with fragmentation disrupting memorized surface matching rather than reasoning. Candidate for pre-registered-hypothesis status.
  6. **Method A yield bound:** ~25% of items admit a Low/High pair under Llama's 128k-vocab tokenizer; clean-correct restriction leaves 2–10 pairs/cell at pilot scale. Prereg must either adopt fragmentation-aware target-word selection or frame Method A cross-model (Mistral's 32k vocab expected generous).
  Remaining before the confirmatory run: the prompting iteration + re-pilot (item 1), the Zizhao decisions (items 2, 3, 6), Stage-2 human audit (κ ≥ 0.60 gate), Stage-3 OSF lock, and formal revision pinning in the roster. Engineering is complete: run outputs are versioned (`results/pilot/`), regeneration is explicit (`run_generation.py --fresh`), and the offline suite stands at 441 tests.

- **2026-07-20: Pre-registration hardening.** Full audit of code, statistics, and references, implemented and verified by the rebuilt test suite and a Colab rerun of the pilot analysis. Summary (details in the referenced sections):
  1. **Confirmatory model is now a true logistic GLMM.** The previous code fit a linear `statsmodels.MixedLM` on the binary outcome and reported `exp(coef)` as an odds ratio (the pilot's "OR 0.965, p=.019" was a linear risk difference of about -3.5 pp). New estimator: `lme4::glmer` (binomial, bobyqa) via a direct rpy2 bridge, with the full §6.6 ladder (maximal, then `||`, then drop by-model slope, then intercepts-only, then fixed-effects logistic GLM) and every rejected rung recorded. The linear model survives only as a labeled risk-difference appendix. pymer4 rejected (pandas version lag).
  2. **Mediation follows Imai's general algorithm** (§6.8): mixed linear mediator model plus mixed logistic outcome model, effects by quasi-Bayesian Monte Carlo (1,000 draws, conditional on the median item). The linear product shortcut survives only as the labeled offline fallback (item-demeaned within estimator, B=1,000 cluster bootstrap). A pooled outcome model flips the mediator coefficient's sign on pilot data (between-item confounding); pinned as a regression test.
  3. **Method A demoted to supporting evidence.** Pilot yield: 607/740 exclusions were "no Low/High pair", 3 to 8 pairs per cell. Selection is now pair-aware (up to 8 candidates, longest first); Mistral-7B (32k vocab) is the cross-model anchor. Method B is the primary mediation quantity.
  4. **H1b registered** (§6.8): mediation fitted separately on keyboard vs filler conditions. Fillers inflate tokens without fragmenting words, so a fragmentation-specific mechanism predicts near-zero mediated share for fillers. First readout supports it (keyboard indirect -0.020, CI excludes 0; filler -0.014, CI includes 0).
  5. **Multiplicity implemented** as locked in §0.4: Holm within each model's primary family, BH-FDR q=0.05 across the exploratory grid, as adjusted-p columns in `cell_table.csv`.
  6. **Endpoint discipline:** GSM-Symbolic is the primary reasoning endpoint. GSM8K makes no primary claims (k=2 discordance is symmetric churn, delta about +1 pp; reported as the contamination contrast). MMLU-Pro stays co-primary, two-sided, with the p_d contingency. Low pilot A0 accepted; prompts frozen; CCF leads reporting.
  7. **New conditions** in main.yaml: `regime_b_homophone` (CMU-exact homophones, the acoustic-confusion proxy, HIVE #13 crosswalk) and `regime_a_missed_space` (whitespace merge through nonword rejection sampling, HIVE #21), joining `regime_a_filler` (HIVE #9). The acoustic ASR arm stays deferred; the proposed plan is design/12.
  8. **Bugs fixed** (each with a regression test): Regime-B phonetic pool silently empty in the pilot (pronouncing lazy-init); filler edits did not replay through `apply_edit_script`; falsy-zero gold misrouted by or-chains (a recomputed Regime-C gold of 0 scored against the old gold); judge decisions misaligned onto wrong audit rows in `sample_for_audit.py` (now row-aligned); judge flags now route to analysis-time exclusion via the audit gate, never generation-time; `build_task_items.py` defaults to `--gsm-config p1`; `--fresh` deletes only known output suffixes; rpy2 result conversion (call under the default converter).
  9. **References rebuilt against the PDFs** (docs/REFERENCES.md): wrong MMLU PDF replaced (was the MATH paper); about 12 rows had wrong titles, authors, editions, or IDs; five where/why attributions corrected (informative-words to Sun 2020, ASR-realism to Wang 2024, Regime-C numeric-edit motivation to Xie 2024, judge verbosity/self-enhancement biases to Zheng 2023, sample-size table values to the simple approximation with Connor exact values 626/870/312 alongside). References added for every locked method (Fleiss, Cohen, Landis & Koch, Benjamini-Hochberg, Holm, Fagerland, Efron, vLLM, AWQ, GPTQ, Zou, Aliannejadi, Pezeshkpour & Hruschka). `tools/verify_references.py` checks every row against its PDF (52/52). Four paywalled classics await institutional fetch.
  10. **Test suite refactored:** 12 modules, 442 tests, into 9 failure-class modules, 275 tests, each named for the failure class it guards.

- **2026-07-21:** Colab verification complete (suite green with the R bridge, GPU smoke 82/82 rows through vLLM across every new condition, glmer pilot artifacts committed: GLMM rung `glmer_intercepts_only`, perturbation OR 0.722, severity OR 0.845 with p=0.0019; GSM8K quasi-Bayes indirect CI [-0.077, -0.004] excludes zero). Acoustic ASR arm plan recorded as design/12 (proposal, pending PI approval). Full-run dress rehearsal config added as `configs/rehearsal.yaml`.

- **2026-07-22: Dress rehearsal (T4 half) executed and audited; three amendments.** All five T4-eligible models ran the full main-study grid (3,086 rows each: llama_1b, llama_3b, llama_8b_awq, qwen_1b5_pilot, qwen_7b_awq; fp16 7B-class trio deferred to the cluster). Resume machinery verified across VM reclaims; no duplicate rows, one revision per model. Analysis on the real estimators (glmer `glmer_intercepts_only` rung, non-singular; Imai quasi-Bayes mediation): compliance 0.988 ≥ 0.95 gate; edit-budget OR 0.905/edit (p ≈ 7e-6); mediation shows the H1b dissociation (keyboard-neighbor indirect −0.039, CI excludes 0; filler indirect +0.086, CI excludes 0 on the positive side) and the family split consistent with the contamination hypothesis (indirect negative with CI excluding 0 on gsm8k and gsm_symbolic, null on both MCQ families). Amendments from the audit:
  1. **MCQ max_new_tokens 256 → 512** (04 §4.3, §4.9; `configs/rehearsal.yaml`, `configs/main.yaml`). At 256 the Llama models truncated 21–38% of MMLU-Pro chains of thought (678/682 truncated rows wrong, accuracy pinned at the 10-option chance floor), contaminating the mmlu_pro gate readout. One budget across families also removes the budget as a family confound; the cost is bounded (greedy stops at EOS).
  2. **Truncated generations are never classified clarification/refusal** (04 §4.5; `pipeline/runner.py`, regression-tested). The rehearsal's truncated rows produced 73 spurious "refusals" from dangling first-person clauses; `finish_reason = length` with no parsed answer now stays `unparseable`.
  3. **Resume idempotence hardened** (07 §7.7; regression-tested): the exclusion sidecar no longer re-appends recomputed exclusion records on resume (resumed rehearsal models logged each exclusion twice), and the shard manifest records the token budgets and refuses to resume a directory whose budgets differ (row_id does not encode budgets, so a silent resume would mix them).
  Also: compute budget re-baselined to ~550k generations, 2.2× the planning figure (07 §7.5, measured throughputs recorded); mmlu sits exactly at the N=600 boundary for the 5 pp MDE (implied N = 600, `raise_n_or_relax_mde` bucket): decision to record with Zizhao at prereg alongside the GSM8K N=720 item. **The v1 rehearsal outputs (256-token MCQ budget) are superseded for gate purposes: rerun the five T4 models at the 512 budget into a fresh `rehearsal_v2` directory before freezing gates**, then add the cluster trio and rerun the combined analysis.

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
