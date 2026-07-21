# 03 — Experimental Design

This document specifies what varies, what is held constant, how the conditions are organized into modules, and the exact cell counts that drive the compute budget. It is the bridge between the formalism (Document 02) and the statistics (Document 06) and compute (Document 07) plans.

---

## 3.1 Design philosophy: modules, not one giant factorial

A full crossing of every factor (operation × unit × location × budget × policy × regime × task × model × quantization) would be tens of thousands of cells, most of them uninterpretable and statistically underpowered. Instead we run **independent modules**, each isolating one scientific question with the other factors fixed at sensible defaults. This is the standard way to estimate main effects and the few interactions we actually care about without a factorial explosion, and it keeps every reported cell adequately powered (Document 06).

The defaults that hold across modules unless a module explicitly varies them:

| Factor | Default |
|---|---|
| unit `u` | character |
| location `ℓ` | content |
| selection `s` | keyboard-neighbor (MulTypo) for controlled cells; asr-transcription for ecological cells |
| regime `c` | A (intent-preserving) |
| decoding | greedy, temperature 0 (Document 05 §5.6) |
| quantization | AWQ W4A16 (Document 05 §5.4) |
| task | both primary tasks (Document 04) |

## 3.2 The modules

**Module 1 — Tokenization mediation (PRIMARY, serves RQ1/H1).**
Varies: fragmentation stratum (Low vs High `Δsub`) within fragmentation-matched counterfactual sets; edit budget `k ∈ {1, 2}`.
Fixed: Regime A, content location, keyboard-neighbor on the target word, all models.
Output: CCF and Δ by fragmentation stratum; the mediation estimate (Document 06 §6.8); CCF and Δ by token-inflation quartile.
This is the module the paper is built around.

**Module 2 — Quantization interaction (SECONDARY, serves RQ2/H2).**
Varies: precision ∈ {fp16, AWQ-4bit} in the main sweep; the sub-study adds GPTQ-4bit (Document 05 §5.5).
Fixed: Regime A, content, keyboard-neighbor, `k ∈ {1, 2, 4}`, fixed model subset (the three 7–8B models).
Output: Δ and CCF for quantized vs fp16, conditioned on clean accuracy; interaction term in the mixed model.

**Module 3 — Selective-invariance audit (FRAMING, serves RQ3/H3).**
Varies: regime `c ∈ {A, B, C}`; perturbation source ∈ {keyboard-neighbor, asr-transcription}; edit budget `k ∈ {1, 2, 4}`; model scale.
Fixed: content location.
Output: CCF/Δ/AFR/ICR on A and B, separately for keyboard and ASR sources; ACR/ORR on C; the CCF(A)-vs-ORR(C) selectivity plot; a keyboard-vs-ASR profile comparison.
Note: ASR-transcription errors predominantly land in Regime B (real-word acoustic confusions) rather than Regime A (nonword typos), because ASR rarely produces nonwords. This is the key structural difference between the two noise sources and is itself an expected finding.

**Module 4 — Edit structure (DESCRIPTIVE, serves RQ4/H4).**
Varies: operation `o ∈ {I, D, S, T}`; location `ℓ ∈ {instruction, content, answer-critical}`; selection `s ∈ {keyboard-neighbor, asr-transcription, uniform, informative-word-targeted}`; edit budget `k ∈ {1, 2, 4}`.
Fixed: Regime A.
Output: CCF/Δ by operation, by location, by policy, by budget; token-inflation by operation; a direct keyboard-vs-ASR degradation comparison at matched severity.

**Exploratory addenda (not pre-registered as confirmatory):** `k = 8` heavy-stress budget; a single byte-level reference point (e.g., a ByT5- or BLT-class model if one is runnable) to contextualize the BPE results; a decoding-noise check (temperature 0.7, fixed seeds) to confirm greedy results are not a sampling artifact. These are reported in a clearly labeled exploratory section so they never contaminate the confirmatory claims.

## 3.3 What is held constant (the confound register)

Every one of these is fixed and logged so that a reviewer cannot attribute a result to it. This register is the backbone of Document 10's defensibility argument.

| Held constant | Value / rule | Why it would otherwise confound |
|---|---|---|
| Decoding | greedy, temp 0, top_p 1, fixed max_new_tokens per task | sampling variance masquerading as robustness difference |
| Quantization method (within a cell) | AWQ W4A16 | recipe-specific accuracy artifacts |
| Prompt template | model's own chat template + a fixed paraphrase as robustness check | template idiosyncrasy |
| Few-shot exemplars | identical, fixed set per task, never perturbed | exemplar leakage / variation |
| Gold answers & scorer | frozen regex/exact-match per task (Document 04) | scoring drift |
| Item set | identical clean items across all conditions (matched pairs) | population differences |
| Seeds | logged per perturbation `ρ` and per run | non-reproducibility |
| Model revision | pinned HF commit hash per model | silent model updates |
| Tokenizer | the model's own; token metrics computed with it | cross-tokenizer comparison errors |
| Max context / truncation | set above 99th percentile of clean-correct length | truncation-induced failure |

## 3.4 Edit budgets and their justification

```
k ∈ {0, 1, 2, 4}   primary
k = 8              exploratory stress
```

- `k = 0`: the clean control (defines `A₀` and the matched pair).
- `k = 1`: the minimal nonzero perturbation — the single-typo regime that R²ATA shows already moves accuracy measurably.
- `k = 2`: tests immediate compounding beyond one typo.
- `k = 4`: a moderate multi-typo regime.
- `k = 8`: heavy stress, exploratory only.

Powers of two are chosen to estimate the *shape* of the response curve efficiently: if degradation is linear in `k`, doubling reveals the slope; if it accelerates, doubling reveals the threshold; with four points {1,2,4,8} we resolve linear-vs-superlinear with the fewest cells. We additionally log edit density `δ = k / |E(x, ℓ)|` (Document 02 §2.6) so that a fixed `k` is interpretable across prompts of different length. Using an exact edit count rather than "5%, 15%, 30% of tokens" is what makes the severity axis non-arbitrary and exactly reproducible.

## 3.5 Selection policies and their justification

```
s ∈ {keyboard-neighbor, uniform, informative-word-targeted, real-word, whitespace}
```

- **keyboard-neighbor (primary):** substitutions/insertions drawn from a QWERTY adjacency graph; this is the ecologically realistic policy and follows what MulTypo implements and validated for human naturalness (>=15 raters per language; Zhao et al., 2025). It is the default for Modules 1–3.
- **uniform (ablation):** edits drawn uniformly over the alphabet; the "arbitrary noise" baseline, included so we can show human-realistic noise differs from arbitrary noise (a question MulTypo motivates).
- **informative-word-targeted:** edits placed on the most task-relevant content words (operationalized in Document 04 §4.6 via task-specific key-term lists, not gradient saliency, to keep it model-agnostic and reproducible). This is the upper-bound-of-damage policy and tests H4, extending Pruthi et al. (2019).
- **real-word:** edit sequences whose result is a valid word — used to *construct* Regime B items, drawing on the WikiTypos and GitHub Typo Corpus distributions.
- **whitespace:** split/merge of spaces — a common human/OCR error, used in Module 4's location analysis.
- **asr-transcription:** errors drawn from a real ASR error distribution (§3.5a below) — acoustic confusions, disfluencies, filler insertions, and missing punctuation. This is the ecologically motivated noise source and produces primarily Regime B items.

### 3.5a ASR error source (decided)

We use **real ASR output errors** rather than synthetic acoustic-confusion models, for two reasons: realism (actual ASR errors reflect the full noise distribution including disfluencies and language-model-in-ASR artifacts that synthetic models miss) and defensibility (a reviewer cannot say "your ASR simulation is unrealistic"). The source is Whisper (Radford et al., OpenAI 2022) transcriptions of spoken English, compared against reference transcripts. Concretely:

- We take items from the primary tasks (GSM-Symbolic reasoning, MMLU-Pro MCQ), read them aloud via a TTS voice (e.g., edge-tts or Google TTS, a single fixed voice per task to avoid speaker-variation confounds), and run Whisper (large-v3, greedy decoding) to transcribe. The difference between the original text and the Whisper transcription is the ASR error.
- Because ASR errors are item-specific and not controllable the way keyboard edits are, **the edit budget `k` for ASR items is the measured Damerau-Levenshtein distance** between original and transcription, logged per item rather than set in advance. We then select ASR items whose edit distance falls within our k-bands {1–2, 3–5, 6+} to maintain comparability with the keyboard-typo severity axis.
- The TTS voice, Whisper version, and decoding settings are pinned and logged.
- A separate noise condition: **environmental-noise stress** — we also run Whisper on TTS audio degraded with babble noise at a fixed SNR (e.g., 10 dB), to represent realistic deployment conditions. This gives a "quiet ASR" vs "noisy ASR" comparison within the ASR arm.

## 3.6 The design matrix and cell counts

A "cell" is one (module condition × model × task) combination for which we report a metric. Per-cell sample size `N` (paired items) is fixed by the power analysis in Document 06 §6.3; the provisional value is **600** (5 pp minimum detectable effect, expected discordant rate ≈ 0.2), to be confirmed by the Stage-2 pilot. We tabulate the generation count, since that is what the compute budget consumes. Each perturbed cell also consumes its matched clean baseline, but the clean baseline is shared across all conditions for the same (model, task), so it is counted once per (model, task), not once per cell.

Let `N = 600` paired items per cell, two tasks, and the model roster of Document 05 (five models; the 7–8B trio doubles under the quantization module). Generations = perturbed runs + shared clean runs.

**Module 1 (mediation):** strata {Low, High} × `k ∈ {1,2}` × 5 models × 2 tasks = 40 perturbed cells.
Perturbed generations: 40 × 600 = 24,000.

**Module 2 (quantization):** precision {fp16, AWQ} × `k ∈ {1,2,4}` × 3 models (7–8B) × 2 tasks = 36 perturbed cells.
Perturbed generations: 36 × 600 = 21,600. (fp16 7–8B requires the A40/L4 tier — Document 07.)

**Module 3 (selectivity):** regime {A,B,C} × `k ∈ {1,2,4}` × 5 models × 2 tasks = 90 perturbed cells.
Perturbed generations: 90 × 600 = 54,000.

**Module 4 (structure):** the largest; to keep it tractable we reduce its per-cell N to 400 (it serves descriptive RQ4, where a 6–7 pp MDE is acceptable) and we do not fully cross all four sub-factors. We run: operations {I,D,S,T} × `k∈{1,2,4}` (12) + locations {instr, content, ac} × `k∈{1,2,4}` (9, content reused) + policies {kbd, uniform, infoword} × `k∈{1,2,4}` (9, kbd reused). Net distinct cells ≈ 24 × 5 models × 2 tasks = 240 cells.
Perturbed generations: 240 × 400 = 96,000.

**Shared clean baselines:** one clean run per item per (model, task). The union of items across modules per (model, task) is at most the largest single-module item pool plus the audit pool; budget ≈ 1,500 unique clean items × 5 models × 2 tasks = 15,000 clean generations. (Re-used across modules; not multiplied per cell.)

**Total (full design, USC GPU cluster):**
```
Keyboard arm:  24,000 + 21,600 + 54,000 + 96,000 + 15,000 ≈ 210,600
ASR arm (Modules 3+4 only, shared clean baselines): ~40,000
Total: ~250,000 generations
```

This is the full-design footprint. The USC lab GPU cluster (confirmed by Zizhao) makes this scale straightforwardly feasible; Document 07 §7.4 covers the logistics of remote submission if cluster access for an external collaborator is pending. It is *not* what we run if compute is constrained; §3.7 and Document 07 §7.6 specify how it collapses.

## 3.7 Graceful degradation (drop conditions, never items)

If compute forces a smaller study, we shrink by removing *conditions* (cells), never by shrinking `N` per cell, because shrinking `N` is what destroys statistical power and invites the "underpowered" critique. The priority order for what to keep:

1. **Always keep:** Module 1 (mediation, the primary contribution) at full `N`, both tasks, all five models at AWQ-4bit, **both keyboard and ASR noise sources**. This alone is ~30,000 perturbed + ~12,000 clean ≈ 42,000 generations — feasible on the USC cluster or free T4 for the 1B/3B models plus AWQ-8B.
2. **Keep next:** Module 3 regimes A and C only (drop B), `k ∈ {1, 4}` only. Establishes selectivity with the over-robustness control intact.
3. **Keep next:** Module 2 quantization on a single 7–8B model (Qwen2.5-7B) fp16-vs-AWQ.
4. **Drop first under pressure:** Module 4's policy and location sweeps; `k=8`; the second task; the fp16 arm of the other 7–8B models; the GPTQ sub-study.

The "worst-case free-T4" study is therefore Module 1 full + Module 3 (A/C, k∈{1,4}) + Module 2 (one model) ≈ 30–40k generations, which Document 07 §7.6 shows fits in a few free Colab sessions and **still supports the primary mediation claim and a bounded selectivity claim** at full per-cell power. That is the floor below which we do not go, because below it the primary contribution loses power.

## 3.8 The planned figures (so the design serves the paper)

Designing the figures now ensures every cell we run earns its place. The confirmatory figures are:

- **Fig 1 — Pipeline schematic:** clean item → state vector `r` → perturbed item → audit → model → scorer → paired metrics. (Document 08 has the data flow.)
- **Fig 2 — Mediation (the money figure):** CCF vs token-inflation quartile, and the Low-vs-High fragmentation-matched contrast, per model. Serves RQ1/H1.
- **Fig 3 — Quantization interaction:** Δ (and CCF) for fp16 vs AWQ per 7–8B model, with BCa CIs, conditioned on matched clean accuracy. Serves RQ2/H2.
- **Fig 4 — Selectivity scatter:** CCF(Regime A) on the x-axis vs ORR(Regime C) on the y-axis, one point per (model, scale), ideal at bottom-left. Serves RQ3/H3. *This is the conceptual centerpiece.*
- **Fig 5 — Severity curves:** CCF vs `k` by operation, with the {1,2,4,8} points showing linear-vs-superlinear shape. Serves RQ4/H4.
- **Fig 6 — Location/policy heatmap:** CCF by (location × task) and by (policy × task). Serves RQ4.
- **Fig 7 — Keyboard vs ASR profile:** side-by-side CCF and Δ for keyboard-neighbor (Regime A) vs ASR-transcription (Regime B) at matched severity bands, across models. The headline "does the same mechanism explain both noise types?" figure.

Every module in §3.2 maps to at least one figure, and every figure maps to a research question. No cell is run that does not feed a planned figure or a pre-registered test.
