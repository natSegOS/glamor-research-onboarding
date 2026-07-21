# Experiment 001 — Typographical Perturbation Robustness: Design Suite

**Project:** GLAMOR Lab (USC) research onboarding — Experiment 001
**Working title:** *When Voice Meets Text: Tokenization-Mediated LLM Robustness to ASR Transcription Errors and Typographical Noise*
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

Voice is an increasingly dominant way of interacting with AI systems, but ASR (automatic speech recognition) transcription introduces systematic errors before the LLM ever sees the prompt: acoustic confusions (real-word substitutions like "weather" for "whether"), disfluencies, filler words, and non-standard punctuation. Understanding exactly *how and why* these surface errors degrade LLM performance is a prerequisite for mitigating them. This study addresses that causal question.

We use a **matched-pair** design and combine **two complementary perturbation sources**: controlled keyboard-adjacency typos (the experimental baseline) and realistic ASR-transcription errors (the ecologically motivated application). The dual-source design lets us test whether the same causal mechanism operates across both noise types, making the findings directly actionable for voice-interface deployments.

The study is built around three contributions, in priority order:

1. **Tokenization-fragmentation mediation (primary).** We decompose noise-induced accuracy loss into the part flowing through *subword fragmentation* and a residual, using a fragmentation-matched counterfactual. No prior typo or ASR-noise study delivers this causal decomposition; the closest works stop at correlation.
2. **Quantization × noise interaction (strong secondary).** We test whether 4-bit-quantized models are more, equally, or less robust to transcription noise than fp16 counterparts, controlling for clean accuracy — directly relevant for deployed voice assistants running quantized models.
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

- **2025-05-29** — Meeting with Zizhao: (1) ASR transcription errors added as a second perturbation source and elevated to primary real-world motivation; (2) Regime B (context-recoverable real-word shift) elevated to co-primary perturbation type alongside Regime A nonword typos; (3) hardware confirmed as USC lab GPU cluster; (4) annotator sourcing TBD with Zizhao; (5) target venue confirmed as ACL Rolling Review → EMNLP 2026. Sections updated: 00 §0.3, §0.4, §0.5; 01 §1.1, §1.3, §1.5, §1.7, §1.8; 03 §3.2, §3.5, §3.8; 07 §7.4, §7.5, §7.6; 09 §9.2; 11 §11.7; 12.

- **2026-07-09** — First pilot (Qwen2.5-1.5B-Instruct, Colab T4, 6,202 rows) reviewed; four amendments before the pilot rerun and pre-registration:
  1. **ASR arm deferred.** The TTS+Whisper pipeline is judged too unrealistic; Zizhao is proposing an alternative from a prior project. The typo (keyboard) arm proceeds alone; ASR integrates later as its own amendment. No code changes needed (ASR was never implemented in `src/`); README updated.
  2. **Primary severity moved to k=2 (reasoning) / k=4 (MCQ).** Pilot p_d at the k=1 keyboard-substitution primary: reasoning ≈ 0.10–0.11 but observed δ ≈ 3 pp (below the 5 pp MDE — a well-powered null risk); MCQ p_d = 0.02–0.03, which fires the pre-registered p_d < 0.05 contingency (06 §6.3). At k=2 reasoning shows p_d ≈ 0.16, δ ≈ 6 pp (N = 600 powered). Parameterized as `primary_edit_budget_reasoning` / `primary_edit_budget_mcq` in `configs/*.yaml` — a one-line change if the rerun revises this. MCQ may remain a near-null even at k=4 (clean A₀ = 0.34 on MMLU-Pro leaves two-thirds of items unbreakable); if so it is reported as the selective-invariance arm, not a second primary endpoint.
  3. **Pilot model corrected to Llama-3.2-1B per 11 §11.2.** The first pilot ran Qwen2.5-1.5B (HF gating convenience); p_d gates are model-conditional and the roster's scale axis is Llama-anchored, so the rerun uses `llama_1b` with HF-token auth. `qwen_1b5_pilot` remains an ungated fallback.
  4. **Mediation reporting corrected (06 §6.8).** The pilot's pooled proportion-mediated (4.85, CI [−14.5, 25.4]) was an artifact: (a) clean rows were coded with token-inflation ratio 0.0 instead of the definitional 1.0, inflating α ≈ 30×; (b) pooling task families with opposite-signed deltas (GSM8K positive, GSM-Symbolic ≈ 0/negative) cancelled the total effect in the ratio's denominator. Mediation now runs per task family with the pooled fit as labeled supplementary; the primary quantity is the indirect effect with bootstrap CI; the proportion is withheld unless the total-effect CI excludes zero. The mediator enters as token-inflation *excess* (ratio − 1): because the mediator is definitionally 1.0 on clean rows, τ ≡ 1 + perturbed·(τ−1), so a separate `perturbed:τ` interaction column (06 §6.6) is perfectly collinear — the excess term *is* the H1 interaction.

- **2026-07-10** — Stage-1 pilot completed on the spec'd model (Llama-3.2-1B-Instruct, revision 9213176, Colab T4, 6,663 rows at code commit 71daea7; ~41k rows/hour measured, so the full main sweep is a few GPU-hours). Two latent implementation bugs surfaced and fixed during the run: letter-boundary mismatch between counterfactual word selection and application (crashed on "Python" in "Python3"), and scope spans applied in full-prompt coordinates to content-only text (excluded ~half of every dataset once the instruction grew an exemplar). Gate readout (`analysis/pilot/gates.json`):
  1. **Format compliance 0.61 vs the 0.95 gate — FAILED (the only failed gate).** One worked exemplar lifted compliance from 0.02 to 0.61; a 1B model needs 2–3 few-shot exemplars. Next iteration before any freeze; re-pilot is ~30 GPU-minutes.
  2. **A₀ = 0.35 / 0.21 / 0.43 / 0.26** (gsm8k / symbolic / mmlu / mmlu_pro) — the true Llama-1B anchor; compresses all effects (60–80% of items unbreakable). Decision needed with Zizhao: accept, or lift A₀ via few-shot.
  3. **p_d at primary k**: 0.23 / 0.10 / 0.18 / 0.09 → implied N = 720 / 312 / 563 / 281. N=600 confirmed for three families; GSM8K needs N=720 or the pre-registered 6 pp MDE relaxation — decision to record here before prereg. GSM8K's k=2 discordance is symmetric (δ≈−0.01, churn); its clean signal appears at k=4 (Δ=15 pp, p=0.0015).
  4. **H1 evidence (first supportive, honestly computed):** GSM8K fragmentation indirect effect −0.043, bootstrap CI [−0.071, −0.014] (excludes zero), β=−1.34 in the predicted direction; proportion-mediated correctly withheld (suppression: direct +0.028).
  5. **Contamination × fragmentation interaction replicated across both pilot models** (Qwen and Llama): indirect effect negative on contaminated GSM8K, positive on fresh GSM-Symbolic (+0.019) — consistent with fragmentation disrupting memorized surface matching rather than reasoning. Candidate for pre-registered-hypothesis status.
  6. **Method A yield bound:** ~25% of items admit a Low/High pair under Llama's 128k-vocab tokenizer; clean-correct restriction leaves 2–10 pairs/cell at pilot scale. Prereg must either adopt fragmentation-aware target-word selection or frame Method A cross-model (Mistral's 32k vocab expected generous).
  Remaining before the confirmatory run: the prompting iteration + re-pilot (item 1), the Zizhao decisions (items 2, 3, 6), Stage-2 human audit (κ ≥ 0.60 gate), Stage-3 OSF lock, and formal revision pinning in the roster. Engineering is complete: run outputs are versioned (`results/pilot/`), regeneration is explicit (`run_generation.py --fresh`), and the offline suite stands at 441 tests.

- **2026-07-20 — Pre-registration hardening** (full audit of code, statistics, and references; all changes below verified by the rebuilt test suite and the Colab rerun of the pilot analysis):
  1. **Confirmatory model corrected to a true logistic GLMM.** The previous `fit_crossed_mixed_effects_logistic` fit a LINEAR `statsmodels.MixedLM` on the binary outcome and reported `exp(coef)` as an "odds ratio" (the pilot's "OR 0.965, p=.019" was really a −3.5 pp linear risk difference). The confirmatory estimator is now `lme4::glmer` (binomial, logit, bobyqa) via rpy2 (`requirements-stats.txt`), with the §6.6 Barr ladder implemented in full (maximal → `||` → drop by-model slope → intercepts-only → fixed-effects logistic GLM) and every rejected rung's reason recorded. The linear model survives only as a labeled risk-difference appendix with no OR fields. pymer4 was rejected for its pandas-major-version lag; the bridge is direct rpy2.
  2. **Mediation estimator corrected to Imai's general algorithm.** Method B now fits a mixed linear mediator model and a mixed LOGISTIC outcome model (by-item intercepts) and computes probability-scale effects by the quasi-Bayesian Monte Carlo algorithm (1,000 parameter draws, conditional on the median item); the linear α·β shortcut (the case Imai et al. 2010 p. 316 warns about) survives only as the labeled offline fallback (item-demeaned "within" estimator + B=1,000 cluster bootstrap). Empirical justification: a pooled outcome model FLIPS the mediator coefficient's sign on the pilot data (between-item confounding) — pinned as a regression test with a Simpson's-paradox construction.
  3. **Method A demoted to supporting evidence; Method B is the primary mediation quantity.** Pilot yield: 607/740 exclusions were "no Low/High pair"; surviving cells held 3–8 pairs. Target-word selection is now pair-aware (tries up to 8 candidates, longest first — was: single longest word); Mistral-7B (32k vocab) is the designated cross-model anchor.
  4. **H1b registered** (§6.8): mediation fitted separately on keyboard-neighbor vs filler-word conditions — fillers inflate tokens without fragmenting words, so a fragmentation-specific mechanism predicts ≈0 mediated share for fillers. First pilot readout is directionally supportive (keyboard indirect −0.020, CI excl. 0; filler −0.014, CI incl. 0).
  5. **Multiplicity implemented as locked** (§0.4): Holm within each model's primary family + BH-FDR q=0.05 across the exploratory grid, as adjusted-p columns in `cell_table.csv`.
  6. **Endpoint discipline:** GSM-Symbolic is the primary reasoning endpoint; **GSM8K makes no primary claims** (its k=2 discordance is symmetric churn, δ≈+1 pp — reported as the contamination-contrast finding). MMLU-Pro stays co-primary two-sided with the pre-registered p_d contingency. The low pilot A₀ anchor is accepted; prompts are frozen (no post-hoc accuracy tuning); CCF leads the reporting.
  7. **New pre-registered conditions** (main.yaml): `regime_b_homophone` (CMU-exact homophones only — the pure acoustic-confusion proxy; HIVE #13 crosswalk) and `regime_a_missed_space` (whitespace merge through the standard nonword rejection sampling; HIVE #21 crosswalk), joining `regime_a_filler` (HIVE #9) as the text-side voice proxies. **The acoustic ASR arm remains deferred** (2026-07-09 entry); docs/PROVENANCE.md §3 now labels it explicitly.
  8. **Bugs fixed:** the Regime-B phonetic-homophone pool was silently EMPTY in the pilot (`pronouncing` lazy-init never triggered — pilot Regime B was orthographic-only, now fixed and tested); filler-word edits did not replay through `apply_edit_script` (contract clause 5 — the Edit recorded the particle without its trailing space); falsy-zero gold answers were misrouted by `or`-chains in the request builder (a recomputed Regime-C gold of 0 would have been scored against the OLD gold); `run_judge_on_sample` now returns row-aligned decisions (the previous silent skip misaligned judge labels onto wrong audit items in `sample_for_audit.py`); judge flag hand-off redefined as analysis-time exclusion through the audit-outcomes gate (never generation-time; an LLM judge cannot veto items — design/09 §9.7); `build_task_items.py` defaults to `--gsm-config p1` (with "main" there are no Apple templates and Regime C reasoning is silently empty); `--fresh` deletes only known output suffixes.
  9. **Reference manifest rebuilt against the PDFs** (docs/REFERENCES.md): the MMLU PDF was the MATH paper (replaced with arXiv:2009.03300); ~12 rows had wrong titles/authors/editions/IDs; five "where/why used" attributions were wrong (informative-words finding re-attributed from Pruthi to Sun et al. 2020; ASR-realism motivation from WikiTypo to Wang et al. 2024; Regime-C numeric-edit motivation added as Xie et al. 2024; judge verbosity/self-enhancement biases re-attributed from Shi to Zheng et al. 2023; sample-size table values 628/873/314 correctly attributed to the simple planning approximation, with Connor eq. (3) exact values 626/870/312 alongside). Added references for every locked-but-uncited method: Fleiss 1971, Cohen 1960, Landis & Koch 1977, Benjamini–Hochberg 1995, Holm 1979, Fagerland et al. 2013 (mid-p), Efron 1987 (BCa), Kwon et al. (vLLM), Lin et al. (AWQ), Frantar et al. (GPTQ), Zou et al. 2023, Aliannejadi et al. 2019, Pezeshkpour & Hruschka 2023. `tools/verify_references.py` now mechanically checks every manifest row against its PDF's first pages (52/52 verified). Four paywalled classics await institutional fetch (rows marked; Fleiss/Cohen/Landis & Koch/Holm).
  10. **Test suite refactored** from 12 modules to 8 failure-class-oriented modules (perturbation contracts incl. adversarial/unicode fuzzing, regimes+gold integrity, tasks+scoring, pipeline runner, end-to-end, statistics goldens, GLMM+mediation simulated recovery, judge+audit) — every test names the failure class it guards.

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
