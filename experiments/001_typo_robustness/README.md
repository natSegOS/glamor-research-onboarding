# Experiment 001 - Typo Robustness of Instruction LLMs

Implementation of the design suite in `design/` (12 documents; start at `design/00_README_index.md`).

**Acoustic ASR arm: deferred** (design/00 §0.5, 2026-07-09). The TTS+Whisper pipeline was judged too unrealistic; a replacement approach is pending from the PI. Its **text-side proxies run in this arm**: homophone-only Regime B (CMU acoustic confusions), filler-word insertion (disfluency), and whitespace merge ("missed-space") — each crosswalks to a HIVE voice-arm operator (design/00 §0.5, 2026-07-20 item 7).

**Status (2026-07-20):** Stage-1 pilot v2 complete on Llama-3.2-1B — all gates pass (compliance 0.9825, results in `results/pilot/`, readout in `analysis/pilot/gates.json`, report in `results/pilot/report.html`) — followed by the pre-registration hardening pass: confirmatory statistics corrected to a true logistic GLMM (lme4 via rpy2) + Imai quasi-Bayesian mediation, multiplicity implemented, reference manifest rebuilt and mechanically verified (`tools/verify_references.py`), audit tooling bugs fixed, and the test suite refactored into 8 failure-class modules. Full decision log: design/00 §0.5, 2026-07-20 entry. Before the confirmatory run: PI sign-off on the four flagged decisions, Stage-2 human audit, Stage-3 OSF lock.

## What this is

A matched-pair robustness study of controlled keyboard-adjacency typos, built around a tokenization-fragmentation **mediation** analysis (primary contribution: Method B — Imai quasi-Bayesian mediation over mixed models — corroborated by the Method A fragmentation-matched counterfactual, design/06 §6.8), a quantization x noise interaction (secondary), and a three-regime selective-invariance audit (framing). Every numeric and procedural choice is justified in the design docs; the code is the executable form of that spec.

## Quickstart (no GPU needed)

```bash
python3 -m pytest tests/ -q       # full offline suite, no network or GPU
```

The suite exercises everything except a real model: item generation, regime construction (A/B/C + the fragmentation-matched counterfactual), request building, idempotent sharding, inline scoring, the cell table, gates, and figures — via the deterministic dummy engine.

## GPU runs (USC cluster / Colab)

Open `colab_driver.ipynb` and run the cells top to bottom: clone/install -> HF auth (gated Llama) -> build items -> build dictionary -> generate -> analyze + download. Re-running any cell resumes rather than recomputes (manifest + per-row deterministic IDs). The equivalent CLI:

```bash
python3 tools/run_generation.py --config configs/pilot.yaml --model llama_1b --output-directory results/pilot
python3 tools/run_analysis.py --generations results/pilot/pilot_generations.jsonl \
    --output-directory analysis/pilot --config configs/pilot.yaml
```

`analysis/pilot/gates.json` is the Stage-1 pass/fail readout: per-family `p_d` at the primary k with the implied per-cell N, clean accuracy A0, reasoning format compliance (target >= 0.95), truncation rate, and the p99 clean-correct output length that freezes `max_new_tokens`.

Before committing cluster time, size the run with a real throughput number:

```bash
python3 tools/benchmark_throughput.py --config configs/pilot.yaml --model llama_1b --limit 200
```

**Proxy workflow** Send Zizhao the experiment folder + this command `python3 tools/run_generation.py --config configs/main.yaml --model <roster_key> --output-directory results/main` and get back `results/`. Shards are idempotent, so partial runs transfer safely.

## Before any confirmatory run (pre-registration gates)

1. Pin every `PIN_ME` revision in `src/inference/roster.py` to an HF commit hash (`inference.roster.resolve_current_revision`), and pin package versions. Non-confirmatory runs resolve and stamp the current SHA automatically; confirmatory runs refuse to start unpinned.
2. Run the pilot; read `analysis/pilot/gates.json` against the Stage-1 gates (design/06 §6.3, design/11 §11.2). The primary severity is `primary_edit_budget_reasoning` / `primary_edit_budget_mcq` in the config (currently k=2 / k=4 per the 2026-07-09 amendment).
3. Human audit (385/regime, kappa >= 0.60 gate) via `src/analysis/audit.py`.
4. Lock the OSF pre-registration, then touch the held-out set.

## Dependencies

Core: `requirements.txt` (statistics, perturbation, annotation, analysis — the full offline test suite runs with these alone).
GPU side: `requirements-gpu.txt` (vLLM stack).
Confirmatory GLMM/mediation: `requirements-stats.txt` (rpy2 bridge to R's lme4; the analysis degrades to loudly-labeled pure-Python fallbacks without it — see `src/analysis/models.py`).
