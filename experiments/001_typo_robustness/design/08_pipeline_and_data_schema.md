# 08: Pipeline, Perturbation Engine, and Data Schema

This document specifies the software the study runs on, in enough detail to implement without further design decisions: the module layout, the perturbation-engine contract, the canonical output schema (one row per generation), and the unit-test plan that makes the perturbations auditable and reproducible. It does not contain final code; it is the spec the code is written against.

---

## 8.1 Repo layout (extends the Allegro convention)

```
experiments/001_typo_robustness/
├── design/                      ← this document suite (the .md files)
├── src/
│   ├── perturb.py               ← the perturbation engine (§8.3)
│   ├── regimes.py               ← regime A/B/C construction + nonword/real-word checks (§8.6)
│   ├── asr_generate.py          ← TTS synthesis + Whisper transcription pipeline (Doc 04 §4.7a)
│   ├── tokenize_metrics.py      ← τ_tok, Δsub, fragmentation strata (Doc 02 §2.5)
│   ├── tasks.py                 ← task loaders, GSM-Symbolic instancing, MMLU-Pro sampling (Doc 04)
│   ├── scoring.py               ← deterministic answer extractors + g(·) (Doc 04)
│   ├── models.py                ← vLLM loader, precision/decoding config (Doc 05, 07)
│   ├── run_generation.py        ← idempotent shard runner + manifest (Doc 07 §7.7)
│   ├── stats.py                 ← McNemar, BCa bootstrap, mixed model, mediation (Doc 06)
│   ├── audit.py                 ← human-audit harness + agreement stats (Doc 09)
│   └── analyze.py               ← figures and tables (Doc 03 §3.8)
├── tests/
│   ├── test_perturb.py
│   ├── test_regimes.py
│   ├── test_tokenize_metrics.py
│   ├── test_scoring.py
│   └── test_reproducibility.py
├── configs/
│   ├── pilot.yaml               ← Stage-2 pilot config
│   ├── main.yaml                ← full design
│   └── fallback_t4.yaml         ← free-T4 study (Doc 07 §7.6)
├── data/
│   ├── items/                   ← frozen task items + gold answers
│   ├── perturbations/           ← generated state vectors + edit scripts
│   └── audit/                   ← audit assignments + labels
├── results/                     ← generation rows (Parquet/CSV) + manifest
├── analysis/                    ← figures, tables, fitted models
└── colab_driver.ipynb           ← thin GPU driver (Doc 07 §7.4)
```

This mirrors the layout the literature-review change-doc already sketched, with vLLM and the regime/token modules added.

## 8.2 End-to-end data flow

```
task loader → clean items (x_i, y*_i)
     │
     ├─ regimes.py → for each item, build A/B/C perturbed candidates
     │       using perturb.py (state vector r) + nonword/real-word checks
     │
     ├─ tokenize_metrics.py → τ_tok, Δsub per (x, x'); assign Low/High strata
     │
     ├─ audit.py → human audit assigns/confirms regime; exclusions applied
     │
     ├─ run_generation.py (vLLM) → model_output per (model, x') and clean x
     │
     ├─ scoring.py → parsed_answer, correct, parse_status
     │
     └─ stats.py / analyze.py → paired tests, CIs, mixed model, mediation, figures
```

Every arrow writes a typed artifact to `data/` or `results/` so each stage is independently re-runnable and inspectable.

## 8.3 The perturbation engine contract (`perturb.py`)

The engine is the scientific instrument; its contract is fixed so results are reproducible and auditable.

```
perturb(
    text: str,
    operation: {"insert","delete","substitute","transpose"},
    unit: {"char","word","span"},
    scope: {"instruction","content","answer_critical","anywhere"},
    edit_budget: int,                 # k
    selection_policy: {"uniform","keyboard_neighbor","informative_word","real_word","whitespace"},
    semantic_class: {"A","B","C"},
    seed: int,
    protected_spans: list[span] = None,   # never edit these (e.g., few-shot exemplars, the #### answer)
    key_terms: list[str] = None,          # for informative_word / answer_critical
) -> (perturbed_text: str, edit_script: list[Edit])
```

Guarantees the engine must satisfy (each is a unit test, §8.7):
1. **Determinism:** same arguments + same seed → byte-identical `perturbed_text` and `edit_script`.
2. **Budget exactness:** exactly `k` primitive edits are applied (or the call fails loudly if `k` exceeds eligible positions), so the severity axis is exact, not approximate.
3. **k = 0 is identity:** `perturb(..., edit_budget=0)` returns `text` unchanged with an empty edit script.
4. **Protected spans are inviolable:** no edit ever touches a protected span (this is how few-shot exemplars and the gold-answer marker stay clean; Document 05 §5.7, Document 04).
5. **Reconstructibility:** applying `edit_script` to `text` reproduces `perturbed_text` exactly, and the script records, per edit, `(op, word_index, char_index, before, after, old_char, new_char)`.
6. **Policy fidelity:** keyboard_neighbor draws only from the QWERTY adjacency graph; uniform draws from the full alphabet; real_word guarantees a valid-word result; informative_word/answer_critical only edit positions inside `key_terms`.

**Upstream library:** the keyboard-neighbor policy wraps **MulTypo** (Zhao et al., 2025; github.com/cisnlp/multypo), which provides validated keyboard-layout-based typo generation; we cite it and use its adjacency model rather than reinventing one. We add the state-vector logging, edit-script reconstruction, protected-span handling, and regime hooks around it. The uniform policy uses the simpler Pruthi et al. (2019) swap/drop/key/add primitives for the ablation baseline.

## 8.4 The canonical output schema (one row per generation)

Every generation, clean or perturbed, produces exactly one row with these fields. This is the single source of truth that `stats.py` and `analyze.py` read. (Extends the Experiment-000 CSV schema, which already carries `model_id` and `quant_bits`.)

```
# provenance
run_id, git_commit, timestamp, manifest_shard_id, row_id (deterministic hash)
# model
model_id, model_revision (HF commit hash), quant_method ("fp16"|"awq"|"gptq"), quant_bits,
vllm_version, transformers_version, torch_version, tokenizer_revision, device
# task / item
task_family ("gsm_symbolic"|"gsm8k"|"mmlu_pro"|"mmlu"), task_id, item_template_id,
clean_prompt, expected_answer (y*), expected_answer_changed (y'* for Regime C, else null)
# perturbation state vector r
is_clean (bool), operation, unit, scope, edit_budget (k), edit_density (δ),
selection_policy, semantic_class (A|B|C), seed (ρ), perturbed_prompt,
edit_script (JSON), damerau_levenshtein (d_DL),
asr_source ("keyboard"|"asr_clean"|"asr_noisy"|null), tts_voice_id, whisper_model_revision, snr_db
# tokenization metrics
clean_token_count, perturbed_token_count, token_inflation (τ_tok),
edited_word, clean_subwords, perturbed_subwords, delta_subwords (Δsub), frag_stratum ("Low"|"High")
# audit
audit_regime (human-confirmed A|B|C), audit_intent_preserved (bool), audit_status ("included"|"excluded"|"reassigned")
# generation + scoring
decoding ("greedy"), max_new_tokens, model_output, parsed_answer,
correct (0|1), parse_status ("valid"|"unparseable"|"clarification"|"refusal")
# performance (optional, for the cost story)
latency_sec, output_token_count, tokens_per_second
```

Anything a reviewer might ask to slice by must be a column. The matched-pair join is `row_clean ↔ row_perturbed` on `(model_revision, task_id, item_template_id)` with `is_clean` distinguishing them.

## 8.5 Tokenization-metric module (`tokenize_metrics.py`)

Computes, per (clean, perturbed) pair and per model tokenizer (Document 05 §5.8): `τ_tok`, `Δsub` on the edited word, and the Low/High fragmentation stratum used by the mediation counterfactual (Document 06 §6.8). It also builds the **fragmentation-matched counterfactual sets** `E_k(w)` (Document 02 §2.5): for each target word and budget, enumerate keyboard-plausible Regime-A realizations and partition by `Δsub`. This module is what makes Contribution 1 possible, so it is tested as carefully as the perturbation engine.

## 8.6 Regime construction (`regimes.py`)

Implements Document 04 §4.7: Regime A via nonword keyboard typos on non-key words (nonword check against a pinned `hunspell`/`wordfreq` en_US list); Regime B via real-word single-edit neighbors or WikiTypos/GitHub-Typo-Corpus draws; Regime C via key-operand swap (reasoning, with `y'*` recomputed from the template) or negation/entity swap (MCQ). Outputs candidate perturbations tagged with their *intended* regime; the human audit (Document 09) confirms or reassigns, and `audit_status` records the outcome.

## 8.7 The test plan (`tests/`)

Minimum coverage:

`test_perturb.py`
- determinism (same seed → identical output), idempotent k=0, exact budget, protected-span inviolability, edit-script reconstruction, each operation type, each policy's fidelity (keyboard_neighbor only QWERTY neighbors, real_word always valid, etc.).

`test_regimes.py`
- Regime A results are always nonwords; Regime B results are always valid words; Regime C reasoning items have a correctly recomputed `y'*`; Regime C MCQ items have a changed gold option.

`test_tokenize_metrics.py`
- `τ_tok` and `Δsub` computed with the correct (model-specific) tokenizer; Low/High strata partition correctly; counterfactual sets hold word/k/position fixed.

`test_scoring.py`
- GSM extractor: `####` priority, last-number fallback, comma/`$`/decimal normalization, unparseable labeling; MMLU extractor: marker regex, last-letter fallback, exact compare, unparseable labeling. Includes adversarial outputs (multiple numbers, refusals, clarifications).

`test_reproducibility.py`
- a fixed (model, item, r, seed) reproduces an identical generation under pinned versions (greedy); the rare vLLM batch-composition flip is detected and bounded (Document 07 §7.9).

`test_asr_generate.py`
- a fixed (item, tts_voice, whisper_model, snr_db) produces identical transcriptions under pinned versions; TTS audio is cached to avoid re-synthesis; Whisper decoding is greedy (deterministic); the edit script from original to transcription reconstructs the transcription exactly.

CI runs the non-GPU tests on every commit; the reproducibility test runs on the GPU tier before the main sweep.

## 8.8 Reproducibility artifacts released with the paper

- All state vectors `r`, edit scripts, and token-metric logs (so every perturbation is reconstructible from the clean item).
- The generation rows (full schema §8.4), the run manifests, and the fitted statistical models.
- Code (perturbation engine, scorers, runner, stats), `configs/`, pinned versions, and model commit hashes.
- Where a dataset license blocks redistributing source text, the **generation scripts + seeds** that reconstruct identical items (Document 04 §4.8).
- **TTS audio files** for the ASR arm (machine-generated, not copyrightable), so the Whisper transcription step is reproducible without re-running TTS. Stored as 16 kHz mono WAV, one file per item per noise condition.
This is the concrete content behind the reproducibility claim in Document 10.
