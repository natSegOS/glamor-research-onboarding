# 11: Execution Roadmap

A stage-gated plan from where the repo is now to a submittable paper. Each stage has a concrete deliverable and a **quantitative decision threshold** that says whether to proceed, revise, or branch. The gates keep the project from burning compute on a design that a pilot would have shown to be underpowered or a novelty search would have shown to be closed.

---

## 11.1 Stage 0: Design freeze and novelty re-check (this week)

- **Do:** finalize this suite; re-run a focused literature check on the two open pillars (tokenization mediation, quantization × typo) to confirm nothing published in the last few months has closed them.
- **Deliverable:** the locked design suite + a one-paragraph novelty memo for Zizhao (Document 12 has the draft).
- **Decision threshold:**
  - If a 2025–2026 paper has already done the *formal mediation* on tokenization → pivot the headline to the quantization interaction (still open) and demote mediation to a replication-plus-extension. (Doc 01 §1.4 ranks both so the pivot is cheap.)
  - If both pillars are closed (unlikely) → fall back to the three-regime selective-invariance map on open small models with full paired stats as the contribution, which is still a clean methods paper.
  - Otherwise → proceed with mediation as primary.

## 11.2 Stage 1: Engineering + Stage-2 pilot (weeks 1–2)

- **Do:** build the `src/` modules including `asr_generate.py` (Doc 08), wire vLLM (Doc 07), write the unit tests. Run the pilot in two parts: (1) keyboard arm: Llama-3.2-1B on 200 paired GSM-Symbolic items at `k=1`, Regime A, keyboard-neighbor; (2) ASR arm: run the TTS+Whisper pipeline on the same 200 items (quiet + 10 dB noisy), produce Regime B candidates, run the 1B model, confirm the pipeline produces sensible transcriptions and parseable outputs.
- **Deliverables:** passing test suite; pilot results; a 1-hour throughput benchmark on the target GPU.
- **Decision thresholds (these fix the locked-but-provisional numbers):**
  - **Discordant rate `p_d`** → fixes `N` (Doc 06 §6.3): `p_d ≤ 0.19` confirms `N=600`; `0.19<p_d≤0.30` raises `N` or relaxes MDE to ~6 pp; `p_d<0.05` moves the primary condition to `k=3/4`.
  - **Throughput** → confirms or revises the compute tier (Doc 07 §7.5). If measured tok/s is far below the conservative estimate, drop to the fallback design earlier.
  - **`max_new_tokens`** → set to the 99th percentile of clean-correct lengths and freeze (Doc 04 §4.9).
  - **Clean `A₀`** → validate fresh-instance comparability against GSM-Symbolic bands (Doc 04 §4.2); if `A₀` is implausibly low, fix prompting/scoring before proceeding.
- **Gate:** do not proceed to the main sweep until the tests pass, `N` is fixed, and `A₀` is validated.

## 11.3 Stage 2: Human audit + generator validation (weeks 3–4, parallel with Stage 3)

- **Do:** generate the full perturbation set (keyboard arm + ASR arm); audit 385 stratified items per regime with ≥3 annotators (Doc 09). **Annotator sourcing confirmed with Zizhao before this stage.** The ASR arm contributes primarily to Regime B, so weight the Regime B audit sample accordingly.
- **Deliverable:** Fleiss κ per regime; the Regime-A intent-preserved validity rate with ±5 pp CI; the exclusion list.
- **Decision threshold:** **Fleiss κ ≥ 0.60 per regime** to proceed. If below, revise the generator/guideline for that regime and re-audit a fresh sample. If the Regime-A validity rate is low, tighten the nonword generator or restrict the primary endpoint to the audited subset (pre-registered contingency, Doc 09 §9.6).

## 11.4 Stage 3: Pre-registration (week 3)

- **Do:** lock the OSF pre-registration (Doc 10 §10.5) *after* the pilot fixes `N`/`max_new_tokens`/contingencies and *before* the held-out confirmatory runs.
- **Deliverable:** timestamped pre-registration.
- **Gate:** no confirmatory run touches the held-out items before this is locked. (The pilot and audit-sample items are kept separate from the held-out confirmatory set so the pilot does not contaminate the confirmatory claim.)

## 11.5 Stage 4: Main sweep (weeks 4–8)

- **Do:** run the modules (Doc 03 §3.2) at the chosen compute tier, idempotent shards with manifest (Doc 07 §7.7), checkpointing every ≤500 generations, shard-parallel across GPUs where available.
- **Deliverable:** complete generation rows (full schema) for all confirmatory cells.
- **Order of execution (so the headline is secured first):**
  1. **ASR data generation:** run TTS+Whisper pipeline on all task items (CPU/small GPU, before the cluster sweep); done once, artifacts cached.
  2. Module 1 (mediation): keyboard arm, then ASR arm; the primary contribution; run first and in full.
  3. Module 3 regimes A/C (selectivity + over-robustness), both keyboard and ASR sources.
  4. Module 2 (quantization) including the fp16 arm.
  5. Module 3 regime B; Module 4 (descriptive); exploratory `k=8`, noisy-ASR stress, sampling check.
- **Branch:** if compute runs short, stop after the highest-priority modules per Doc 03 §3.7; the primary claim is already secured because Module 1 ran first.

## 11.6 Stage 5: Analysis (weeks 8–10)

- **Do:** run the pre-registered analyses first (McNemar + BCa per cell; the mixed model; the mediation Methods A and B; the quantization interaction), then the exploratory analyses in a separately labeled section. Generate all figures with CIs (Doc 03 §3.8).
- **Deliverable:** results tables, figures, fitted models; a results memo for Zizhao.
- **Decision threshold (interpretation, not gating):** report whatever the data show, including nulls. A null quantization interaction or a small mediated fraction is a publishable, honest result; the paper's value rests on the rigor of the measurement, not on the effect being large. Do not re-run or re-slice to find significance: that would break the pre-registration.

## 11.7 Stage 6: Writing and venue (weeks 10+)

- **Do:** write the paper using these documents as the methods backbone; the framing (Doc 01), formal framework (Doc 02), and defensibility table (Doc 10) translate almost directly into Intro/Methods/Limitations sections.
- **Venue selection:**
  - **Confirmed primary target: ACL Rolling Review → EMNLP 2026 main** (evaluation and analysis track). Zizhao confirmed this target in the post-design-suite meeting.
  - **Fallback:** NAACL 2027 main if the ARR round does not land in time; ACL Findings as a strong secondary outcome.
  - **Publication path:** submit to ARR, receive reviews, attend the associated conference (EMNLP 2026 or NAACL 2027) if accepted. If reviews are negative, revise based on feedback and resubmit to the next ARR cycle with a response document. This iterative path is standard and expected.
  - **A single-pillar short version** (keyboard-typo mediation only) fits a workshop or short-paper track as a safety net.

## 11.8 The week-by-week view

| Week | Stage | Milestone |
|---|---|---|
| 0 | 0 | Design freeze; novelty memo to Zizhao |
| 1–2 | 1 | `src/` + tests + vLLM; ASR pipeline built and piloted; **N fixed, throughput measured, A₀ validated, ASR transcriptions cached** |
| 3 | 4 | Pre-registration locked |
| 3–4 | 2 | Full perturbation set + audit; **κ ≥ 0.60 confirmed** |
| 4–8 | 4 | Main sweep, Module 1 first; checkpointed, shard-parallel |
| 8–10 | 5 | Pre-registered analysis; figures with CIs; results memo |
| 10+ | 6 | Draft; ARR/EMNLP submission |

## 11.9 Risk register (project-level)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Novelty closed by a new paper | medium | two-pillar design; pivot rule (Stage 0); re-check at submission |
| Pilot shows tiny `p_d` (underpowered single typo) | medium | move primary condition to `k=3/4` (pre-registered, Doc 06 §6.3) |
| Compute shortfall | medium | fallback-T4 design preserves primary claim (Doc 07 §7.6) |
| Low annotator agreement | low–med | revise generator/guideline, re-audit (Stage 2 gate) |
| Mixed model won't converge | medium | pre-registered fallback ladder to fixed-effect model (Doc 06 §6.6) |
| Quantized clean accuracy collapses | low | A₀ validity check catches it pre-sweep (Doc 05 §5.9) |
| Time overrun | medium | single-pillar short-paper safety net (§11.7) |

## 11.10 Definition of done

The project is done when: the pre-registered analyses are complete with CIs; every claim in the abstract maps to a number with its interval; the reviewer-attack table (Doc 10 §10.2) has a built-in answer for each row; the reproduction artifacts (Doc 10 §10.4) are released; and Zizhao has signed off on the framing and the bounded claims. At that point the paper is submittable to ARR/EMNLP, and, independently of acceptance, the result is a credible, reproducible contribution to the understanding of how typographical perturbations affect LLMs.
