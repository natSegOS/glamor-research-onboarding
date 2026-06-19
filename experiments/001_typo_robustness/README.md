# Experiment 001 - Typo/ASR Robustness of Instruction LLMs

Implementation of the design suite in `design/` (12 documents; start at `design/00_README_index.md`). Working title: *When Voice Meets Text: Tokenization-Mediated LLM Robustness to ASR Transcription Errors and Typographical Noise.*

## What this is

A matched-pair robustness study with two perturbation sources: controlled keyboard-adjacency typos and realistic ASR transcription errors, built around a tokenization-fragmetation **mediation** analysis (primary contribution), a quantization x noise interaction (secondary), and a three-regime selective-invariance audit (framing). Every numeric and procedural choice is justified in the design docs; the code is the executable form of that spec.

## Layout

```
001_typo_robustness/

```

## Quickstart (no GPU needed)

```bash
python3 tools/run_tests.py  # 78 tests, offline-safe
python3 src/run_experiment.py configs/pilot.yaml --engine dummy  # full loop, fake engine
```

The dummy run exercises everything except the model: item generation, regime construction, request building, idempotent sharding, scoring, the cell table, and the CCF-vs-k figure. On GPU, swap `--engine vllm`

## GPU runs (USC cluster / Colab)

Open `colab_driver.ipynb`. Order matters: tests -> ASR pre-processing (once, casched, CPU-fine) -> main sweep -> pull results. Re-running any cell resumes rather than recomputes (manifest + per-row deterministic IDs).

**Proxy workflow** Send Zizhao the experiment folder + this command `python3 src/run_experiment.py configs/main.yaml --engine vllm` and get back `results/`. Shards are idempotent, so partial runs transfer safely.

## Before any confirmatory run (pre-registration gates)

1. Pin every `revision: PINE_ME` in `configs/` to an HF commit hash, and pin package versions
2. Run the pilot; record discorant rate `p_d`, throughput, and clean accuracy against the Stage-1 gates. `stats.mcnemar_sample_size` turns the measured `p_d` into the confirmed per-cell N.
3. Swap the demo wordlist (`data/wordlist_demo.txt`, smoke tests only) for the pinned full dictionary, and the demo MCQ items for the MMLU-Pro subsample.
4. Human audit (385/regime, kappa >= 0.60 gate) via `src/audit.py`.
5. Lock the OSF pre-registration then touch the held-out set.

## Dependencies

Core: `numpy scipy pyyaml` (analysis), `matplotlib` (figures, optional).
GPU side: `vllm transformers accelerate`. ASR arm: `openai-whisper edge-tts soundfile`. Tests run with zero non-core dependencies via `tools/run_tests.py`
