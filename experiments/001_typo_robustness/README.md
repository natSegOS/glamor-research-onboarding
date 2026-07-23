# Experiment 001: Typo Robustness of Instruction LLMs

A matched-pair robustness study of controlled keyboard-adjacency typos on open instruction models, built around three contributions:

1. **Tokenization-fragmentation mediation** (primary): how much of the accuracy lost to intent-preserving noise flows through subword fragmentation. Method B (Imai quasi-Bayesian mediation over mixed models) is the primary quantity; Method A (fragmentation-matched counterfactual) corroborates it. See design/06 §6.8.
2. **Quantization x noise interaction** (secondary): fp16 vs AWQ-4bit vs GPTQ-4bit, pre-registered two-sided.
3. **Three-regime selective-invariance audit** (framing): intent-preserving nonwords (A), context-recoverable real-word shifts (B), and meaning-changing controls (C), with humans as the final regime authority.

The full blueprint lives in `design/` (13 documents, start at `design/00_README_index.md`). Every numeric and procedural choice is justified there; the code is the executable form of that spec.

## Status

Stage-1 pilot complete on Llama-3.2-1B, all gates pass (readout: `analysis/pilot/gates.json`, report: `results/pilot/report.html`). The 2026-07-20 hardening pass corrected the confirmatory statistics (true logistic GLMM via lme4, Imai mediation), rebuilt and mechanically verified the reference manifest, fixed the audit tooling, and refactored the test suite. Full decision log: design/00 §0.5.

Remaining before the confirmatory run: PI sign-off on four flagged decisions, Stage-2 human audit, Stage-3 OSF lock (see "Pre-registration gates" below).

The acoustic ASR arm is deferred. Its text-side proxies run in this arm (homophone-only Regime B, filler insertion, whitespace merge), and the proposed acoustic plan awaiting PI approval is `design/12_acoustic_asr_arm_plan.md`.

## Quickstart (no GPU)

```bash
python3 -m pytest tests/ -q        # 306 tests, offline, no network or GPU
python3 tools/verify_references.py # bibliography manifest vs actual PDFs
```

The suite exercises everything except a real model: perturbation contracts, regime construction, request building, sharding and resume, inline scoring, statistics goldens, the GLMM and mediation estimators on simulated data, and the audit tooling. A deterministic dummy engine stands in for the GPU.

## GPU runs (Colab / local / USC cluster)

One entrypoint drives the whole pipeline, the same way on Colab and locally:

```bash
python3 tools/run_pipeline.py setup   # install deps
python3 tools/run_pipeline.py all     # data -> generate -> analyze -> report
```

Every user-facing setting (which models, which config, where output goes,
whether to reuse committed data) lives in `configs/run_profile.yaml`. On
Colab, open `colab_driver.ipynb` instead: it's the same commands in four
cells (bootstrap, configure, run, download). Every stage resumes instead of
recomputing (run manifest plus per-row deterministic IDs), and a model
that's already fully generated is skipped without even being loaded.

Full walkthrough, including per-stage commands, GPU sharding, and
parallelizing across accounts: **`RUNBOOK.md`**.

Configs:

| Config | Purpose |
|---|---|
| `configs/pilot.yaml` | Stage-1 pilot (frozen; 100 items, one model) |
| `configs/rehearsal.yaml` | Full-run dress rehearsal: every condition and dataset of the main run at small N, intended for the whole model roster |
| `configs/main.yaml` | Confirmatory run (720 items per dataset; refuses to start with unpinned revisions) |

## Pre-registration gates

1. Pin every `PIN_ME` revision in `src/inference/roster.py` (`tools/pin_revisions.py`) and pin package versions. Confirmatory runs refuse to start unpinned.
2. Read `analysis/pilot/gates.json` against the Stage-1 gates (design/06 §6.3, design/11 §11.2).
3. Stage-2 human audit: 385 items per regime, Fleiss kappa >= 0.60 gate (`src/analysis/audit.py`, `tools/sample_for_audit.py`).
4. Lock the OSF pre-registration, then touch the held-out set.

## Dependencies

| File | Contents |
|---|---|
| `requirements.txt` | CPU core: statistics, perturbation, annotation, analysis. The full offline test suite runs with these alone |
| `requirements-gpu.txt` | vLLM generation stack |
| `requirements-stats.txt` | rpy2 bridge to R lme4 for the confirmatory GLMM and mediation. Analysis degrades to labeled pure-Python fallbacks without it (`src/analysis/models.py`) |
