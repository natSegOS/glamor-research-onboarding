# 10: Defensibility, Pre-Registration, and Bounded Claims

This document consolidates everything that makes the study hard to refute: the reviewer-attack table (each likely objection paired with the design feature that pre-empts it), the bounded-claim language, the reproducibility checklist, and the pre-registration plan. It is the place to look when asking "could a reviewer kill this, and is the answer already built in?"

---

## 10.1 The non-refutability strategy in one idea

A claim cannot be refuted *on the basis of the findings* if:

- (a) the claim is bounded to exactly what the data support;
- (b) every confound a reviewer could name is held constant and logged;
- (c) the central judgments (intent preservation, correctness) are data or deterministic rather than authorial;
- (d) the whole analysis plan was fixed before the confirmatory runs so nothing was fished.

The study is engineered around these four properties; this document maps each design feature to the objection it neutralizes.

## 10.2 The reviewer-attack table

| # | Likely objection | Pre-emption (and where it lives) |
|---|---|---|
| 1 | "Your typos changed the meaning: you measured task redefinition." | Human audit, Fleiss κ≥0.60 gate, primary endpoint restricted to audited intent-preserving items, audit-failed items excluded by a pre-registered rule. (Doc 09) |
| 2 | "Your benchmark is contaminated: the drop is a memorization artifact." | Fresh GSM-Symbolic-style instances + MMLU-Pro as primary; standard GSM8K/MMLU only as contamination contrast; clean A₀ validated against published bands. (Doc 04 §4.2–4.3) |
| 3 | "The differences are within noise." | Paired McNemar (mid-p exact), BCa 95% CIs on every effect, pre-registered MDE and power; raw discordant counts reported. (Doc 06) |
| 4 | "Underpowered, like most NLP studies." | N derived from a 5 pp MDE at 80% power, pilot-gated on measured discordant rate; cite Card et al. 2020. (Doc 06 §6.3) |
| 5 | "Cherry-picked model or quantization recipe." | 3 families × 3 scales; AWQ held constant in main sweep; fp16/AWQ/GPTQ sub-study guards against recipe artifacts. (Doc 05) |
| 6 | "Greedy-vs-sampling confound." | All confirmatory runs greedy (temp 0); exploratory sampling check confirms the condition ranking survives. (Doc 05 §5.6) |
| 7 | "Prompt-template idiosyncrasy." | Each model in its own chat template + a second paraphrased template reported alongside. (Doc 05 §5.7) |
| 8 | "Your mediation is just correlation." | Fragmentation-matched counterfactual (holds word/k/position/regime fixed, varies only fragmentation) plus population mediation; both reported, agreement shown. (Doc 06 §6.8) |
| 9 | "Quantized model is just worse overall, not less typo-robust." | Always condition on clean accuracy via CCF (clean-correct items only) and R=A₁/A₀; matched-item analysis. (Doc 05 §5.5, Doc 06 §6.9) |
| 10 | "You compare token counts across different tokenizers." | All token metrics use the model's own tokenizer; the *pattern* is compared across models, not raw counts; small-vocab Mistral is a built-in cross-check. (Doc 05 §5.8) |
| 11 | "Refusals/clarifications inflate your failure rate ambiguously." | Dual accounting: clarifications count against accuracy (conservative) but are logged separately as ICR. (Doc 04 §4.5) |
| 12 | "LLM-judge is unreliable." | LLM-judge excluded from the primary endpoint and regime assignment; humans final; judge only pre-screens / confirms deterministic Regime-C comparisons. (Doc 09 §9.7) |
| 13 | "Not reproducible." | Released code, configs, seeds, model commit hashes, pinned versions, run manifests, all state vectors + edit scripts; generation scripts where licenses block source release. (§10.4, Doc 08 §8.8) |
| 14 | "You're just restating CheckList / Niu & Bansal." | Explicit attribution of the selective-invariance concept; contribution framed as mechanism + quantization + paired stats, not the concept. (Doc 01 §1.2–1.3) |
| 15 | "Multiple comparisons: you found significance by running many cells." | Primary endpoint + 2 interactions designated and protected; everything else BH-FDR at q=0.05; Holm within a model's family. (Doc 06 §6.7) |
| 16 | "Edit budgets / percentages are arbitrary." | Exact primitive-edit counts (not %), powers of two justified by response-curve estimation, edit density logged. (Doc 03 §3.4) |
| 17 | "Single task, can't generalize." | Two task archetypes (reasoning + MCQ) spanning the literature's disagreement about which is more fragile. (Doc 04 §4.1) |
| 18 | "No baseline defense, so what?" | Report at least one denoising baseline (spell-check pre-pass / ScRNN-style) and the re-pass strategy from Wang et al. 2024. (§10.6) |
| 19 | "vLLM batching makes it non-deterministic." | Greedy removes sampling variance; versions pinned and logged; reproducibility test bounds rare batch-composition flips; manifests released. (Doc 07 §7.9) |
| 20 | "You haven't shown it matters in the real world." | ASR transcription is the primary real-world motivation; noise in voice interfaces is structural and unavoidable. The mediation result is actionable: it tells you whether the fix should be at the tokenizer level, the acoustic model level, or via input normalization. (Doc 01 §1.9) |
| 21 | "Your ASR errors are from a controlled TTS+Whisper setup, not real spontaneous speech." | Acknowledged as a scope limitation stated explicitly in §1.7. TTS+Whisper is a reproducible, pinned, fully auditable noise source; real spontaneous speech is not. The noisy-ASR condition (babble noise at 10 dB SNR) approximates realistic ambient conditions. The keyboard-typo arm serves as the controlled baseline that makes the mechanism findings hold regardless of ASR realism. |
| 22 | "Your ASR and keyboard errors are not the same kind of noise: you can't compare them." | Correct, and the paper never claims they are the same. The comparison is: does the same causal mechanism (tokenization fragmentation) explain accuracy loss under *both* noise types? If yes, it is a general mechanism. If no, that is itself a finding. The dual-source design is designed to ask this question, not to equate the sources. |

If a new objection appears in review, it is added here with its pre-emption; the table is the living defense.

## 10.3 Bounded-claim language (use verbatim in the paper)

The difference between a refutable and a non-refutable paper is often one adjective. Approved phrasings:

- ✅ "We provide the first **matched-pair decomposition** of typo-induced accuracy loss into a tokenization-fragmentation channel and a residual channel, **within a fragmentation-matched counterfactual scope**."
- ✅ "Holding clean accuracy fixed, 4-bit AWQ quantization **changes** typo robustness by [X] ([direction]) **for the models and recipes studied**."
- ✅ "Current open instruction LLMs in the 1B–8B range exhibit [low/moderate] clean-conditioned failure on intent-preserving noise and [degree] over-robustness on meaning-changing controls."
- ❌ Avoid: "LLMs cannot reason," "definitively," "first ever," "all quantization methods," "proves that tokenization causes," any unbounded universal.

The rule: every sentence in the abstract and conclusion must be checkable against a number in the results with its CI. If it is not, it is cut or qualified.

## 10.4 Reproducibility checklist (Pineau et al. 2021 / NeurIPS / ACL aligned)

Released with the paper:
- [ ] Code: perturbation engine, regime construction, scorers, vLLM runner, stats, audit harness.
- [ ] Configs: `pilot.yaml`, `main.yaml`, `fallback_t4.yaml`.
- [ ] Pinned versions: `transformers`, `vllm`, `torch`, tokenizer, spaCy, wordlist, all in a lockfile.
- [ ] Model commit hashes (HF revision) per model.
- [ ] Seeds: per perturbation `ρ`, per run, per bootstrap.
- [ ] All perturbation state vectors + edit scripts (every `x'` reconstructible from `x`).
- [ ] Generation rows (full schema, Doc 08 §8.4) + run manifests.
- [ ] Fitted statistical models + analysis notebooks producing every figure/table.
- [ ] Audit guideline, worked examples, anonymized labels, agreement computation.
- [ ] Where licenses block source-text release: generation scripts + seeds that reconstruct identical items.
- [ ] A one-command reproduction script for the fallback-T4 study (so a reviewer can re-run the primary claim cheaply).

## 10.5 Pre-registration plan

Before any confirmatory run on the held-out evaluation items, we lock an OSF (or AsPredicted-style) pre-registration containing:
- The research questions and **directional hypotheses** (H1, H3, H4 directional; H2 two-sided) (Doc 01 §1.6).
- The **primary endpoint** and the two **primary interactions** (mediation, quantization) (Doc 06).
- The **statistical tests**, the multiplicity scheme, and the convergence-fallback ladder (Doc 06 §6.4–6.7).
- The **sample sizes** and the pilot decision rule for `N` (Doc 06 §6.3).
- The **exclusion rule** (audit-failed and ambiguous items) (Doc 09 §9.5).
- The **held-constant confound register** (Doc 03 §3.3).
- The **fallback design** if compute is constrained (Doc 03 §3.7, Doc 07 §7.6).

The pilot (Stage 2, Doc 11) is explicitly *exploratory* and may inform parameter choices; the pre-registration is locked *after* the pilot fixes `N`, `max_new_tokens`, and the discordance contingency, and *before* the held-out confirmatory runs. This sequencing (pilot to set parameters, then lock, then confirm) is standard and is what lets us call the main results confirmatory.

## 10.6 Baseline defenses to include

So that "so what / is it fixable?" is answered, we run at least one mitigation and report whether it closes the gap:
- **Spell-check / robust-word-recognition pre-pass** (ScRNN-style, Pruthi et al. 2019): correct the input before the model sees it; report residual CCF.
- **"Re-pass" self-denoising** (Wang et al. 2024): ask the model to clean the input first; report whether it helps for open small models (Wang et al. found it weak for open models; replicating that is itself informative).
- Optionally, a single **byte-level reference point** (ByT5/BLT-class) to show the architectural ceiling.
These turn the paper from "here is a problem" into "here is a mechanism *and* what does/does not fix it," which is stronger and harder to dismiss.

## 10.7 Ethics, data, and compute disclosure

- **Data:** all tasks are public benchmarks; no personal data; licenses honored (Doc 04 §4.8).
- **Annotators:** if labmates/peers, acknowledge; if paid, report compensation at or above local norms.
- **Compute:** report GPU type, GPU-hours, and approximate cost (the study is < $50 of burst compute, Doc 07 §7.5), a transparency point reviewers increasingly expect.
- **Risk:** the study reveals a brittleness that could be misused to craft adversarial typos, but the effect is already public (R²ATA et al.); the mitigation analysis (§10.6) is the responsible counterweight. State this briefly.
