# 04: Tasks, Datasets, and Scoring

The statistical edifice rests on `g(·)` being a deterministic, defensible, exact-match correctness judgment. This document pins down which tasks, which data, how contamination is neutralized, and exactly how answers are extracted and scored, to the level that a second engineer could implement the scorer without asking a question.

---

## 4.1 Task-selection principles

Tasks must be:

- **Objectively and deterministically scorable**: no LLM-as-judge in the primary endpoint.
- **Contamination-controllable**: we can run on fresh instances whose answers the model could not have memorized.
- **Diverse in failure mode**: at least one open-ended reasoning task and one closed-form discriminative task, because the literature disagrees on which is more typo-fragile (MulTypo vs WikiTypos); spanning both avoids a task-specific overclaim.
- **English**, per the scope statement (Document 01 §1.7).

Two primary tasks, each with a contamination-contrast partner:

| Role | Primary (fresh) | Contamination contrast (standard) |
|---|---|---|
| Reasoning / numeric | GSM-Symbolic-style fresh templates | GSM8K test |
| Closed-form MCQ | MMLU-Pro | MMLU |

A third optional task (BBH subset, for cross-replication with R²ATA) is included only if compute allows (Document 03 exploratory).

## 4.2 Reasoning task: GSM-Symbolic-style fresh instances

**Why not raw GSM8K alone.** GSM8K (Cobbe et al., 2021) is in many pretraining corpora; a "robustness" drop could be confounded with memorization of the clean form. Mirzadeh et al. (Apple, 2024, GSM-Symbolic, arXiv:2410.05229) show that re-instantiating GSM8K problems from symbolic templates with fresh names and numbers drops accuracy by 0.3% (GPT-4o) to ~9.2% (Mistral-7B), and that adding an irrelevant clause (GSM-NoOp) drops some models by up to 65%. This is direct evidence that the *clean* form carries contamination and that fresh instances are necessary for a clean baseline.

**What we use.** We generate fresh instances from GSM-Symbolic-style templates (the public Apple `ml-gsm-symbolic` templates, or equivalently constructed templates if license/format requires), instantiating new numeric values and names per item so the clean problem is novel. The gold answer is the template's computed numeric result. This neutralizes the "your model memorized GSM8K" attack: our clean baseline `A₀` is measured on instances the model has not seen.

**Defensibility check.** We verify that clean `A₀` on our fresh instances falls within the published GSM-Symbolic confidence band for each model class; if it does, the instances are valid and comparable. We then apply typo perturbations *on top of* these fresh instances. We additionally run the same perturbations on a sample of standard GSM8K so the paper can report the contamination contrast (Δ on fresh vs Δ on standard) as a robustness check.

**Answer extraction (frozen scorer).** GSM-Symbolic and the original GSM8K format put the final answer after `####`. The scorer, in priority order:
1. If the generation contains `####`, take the first numeric token after the last `####`.
2. Else take the last number in the generation (regex `-?\$?\d[\d,]*\.?\d*`, strip `$` and commas).
3. Normalize: strip commas, trailing `.0`, leading `$`; compare as a rational/float with tolerance 0 (exact) for integer answers, and `abs(a−b) < 1e-6` for decimal answers.
4. If no number is found → label `unparseable` (counts toward ICR, not toward correctness).

The reference implementation in Apple's `ml-gsm-symbolic` repo uses exactly this `####`/last-number regex; we match it so our scoring is comparable to published numbers.

**max_new_tokens.** Set to cover >99% of clean-correct chain-of-thought lengths empirically measured in the pilot (provisionally 512). Logged and frozen.

## 4.3 MCQ task: MMLU-Pro

**Why MMLU-Pro.** MMLU (Hendrycks et al., 2021) is heavily contaminated and has a 4-option format with known label artifacts. MMLU-Pro (Wang et al., 2024) extends to 10 options, is harder, and is more discriminative. We use it as the primary MCQ task and standard MMLU as the contamination contrast, subsampled subject-stratified to control clean-accuracy heterogeneity.

**Answer extraction (frozen scorer).**
1. Look for an explicit answer marker: regex for `answer is \(?([A-J])\)?` or `^\(?([A-J])\)?[).:]` at the start of a line, case-insensitive.
2. Else take the last standalone capital letter in {A–J} in the generation.
3. Compare to the gold option letter exactly.
4. If no valid letter found → `unparseable` (ICR).

The prompt template instructs the model to answer with a single letter, which maximizes parseability. The instruction itself is part of the held-constant template (Document 03 §3.3) and, except in the instruction-location module, is never perturbed.

**max_new_tokens.** Provisionally 256 (MMLU-Pro benefits from brief reasoning before the letter). Logged and frozen.

## 4.4 Why deterministic scoring matters for the statistics

McNemar, the binomial, and the mixed logistic model all require a clean binary outcome per item. Any softness in `g(·)` (a fuzzy judge, a length-sensitive metric) would introduce a noise source the matched-pair design cannot remove and would expose the study to "your metric is arbitrary." Restricting the primary endpoint to tasks where correctness is a regex-exact comparison against a gold answer makes `Y_{i,m,c}` a true Bernoulli outcome, which is what Document 06's tests assume. This is a design choice in service of non-refutability, not just convenience.

## 4.5 Handling the `unparseable` / clarification outcomes

A typo can cause a model to ask "did you mean France?" rather than answer. This is *not* the same as a wrong answer and must not be silently scored as either correct or incorrect. We therefore:

- Score `unparseable`/clarification/refusal as **incorrect for accuracy purposes** (it did not produce the right answer), AND
- Log it separately so ICR (Document 02 §2.6, M9) captures the interactional-failure rate distinctly.

This dual accounting keeps the headline accuracy numbers conservative (clarifications count against the model) while the ICR diagnostic preserves the qualitative distinction. A reviewer asking "did the drop come from wrong answers or from refusals?" is answered by data.

## 4.6 Key-term lists for informative-word targeting and answer-critical location

Module 4's informative-word-targeted policy and answer-critical location require knowing which words are task-critical. We operationalize this *without* model-internal saliency, so it is model-agnostic and reproducible:

- **Reasoning:** the key terms are the numeric quantities, the operation words ("more", "each", "twice", "total", "remaining"), and the named entities that bind quantities to referents. These are extractable deterministically from the GSM-Symbolic template structure (the template knows which tokens are operands and operators). This is a major side-benefit of templated data: the answer-critical tokens are *known by construction*, not guessed.
- **MCQ:** the key terms are the question's focus noun phrase and any negation/quantifier, extracted by a fixed POS+dependency rule (spaCy, pinned version) and stored per item. Negation insertion/deletion on MCQ doubles as a natural Regime-C (meaning-changing) construction.

This determinism is what lets us claim the "answer-critical" condition is exactly that, rather than an author's subjective pick.

## 4.7 Constructing the three regimes per task

- **Regime A (intent-preserving nonword):** apply MulTypo keyboard-neighbor edits to non-key content words (reasoning) or to the question's content words (MCQ), constrained so the result is a nonword (dictionary check against a frozen wordlist, e.g., `wordfreq`/`hunspell` en_US, pinned). Audited to confirm intent preservation (Document 09).
- **Regime B (context-recoverable real-word shift):** items are sourced primarily from **ASR-transcription errors** (Document 03 §3.5a): Whisper transcriptions of TTS-spoken task items, where the recognized form is a valid word differing from the original (e.g., acoustic confusions). Real-word substitutions from WikiTypos / GitHub Typo Corpus serve as a supplementary source and for cross-validation. Audited; items where audit says intent is *not* recoverable are reassigned to C or dropped.
- **Regime C (meaning-changing control):** for reasoning, swap a key numeric operand to a different value and recompute the gold answer `y'*` from the template (trivial because the template computes the answer); for MCQ, insert/delete a negation or swap the focus entity so the correct option changes. The new gold `y'*` is known by construction.

Templated reasoning data letting us recompute `y'*` exactly for Regime C is a second major reason to use GSM-Symbolic-style templates rather than raw GSM8K: the meaning-changing control is gold-labeled without a human guessing the new answer.

### 4.7a ASR data generation procedure (Regime B primary source)

1. For each clean item `(x_i, y*_i)`, synthesize audio using a single fixed TTS voice (pinned model/voice ID, e.g., `edge-tts en-US-AriaNeural`) at a fixed speaking rate.
2. Run **Whisper large-v3** (greedy decoding, English language forced, pinned commit hash) on the audio to produce a transcription `x'_i`.
3. Log the Damerau-Levenshtein distance `d_DL(x_i, x'_i)` and all word-level diffs.
4. Classify each changed token: real-word substitution (→ Regime B candidate), nonword (→ Regime A candidate, rare for ASR), deletion/insertion (logged by operation type).
5. Also generate a **noisy-ASR variant**: add babble noise at 10 dB SNR before transcription, to simulate realistic ambient-noise deployment conditions.
6. Both the quiet and noisy ASR transcriptions enter the pipeline as separate conditions, tagged `asr_clean` and `asr_noisy` in the `selection_policy` field.
7. The TTS audio files are retained and released with the paper (they are not copyrightable as machine-generated audio of public benchmark text) to make the ASR arm fully reproducible without re-running TTS.

## 4.8 Data provenance, licensing, and release

- GSM8K: MIT-licensed. GSM-Symbolic templates: check the Apple repo license at use time and cite. If the license precludes redistribution, we release our *generation scripts and seeds* so others can regenerate identical fresh instances rather than redistributing instances.
- MMLU / MMLU-Pro: cite and follow their licenses; release subject-stratified item IDs and our perturbation scripts, not necessarily the source text, depending on license.
- We release: all perturbation state vectors `r`, edit scripts, token-metric logs, model generations, and scorer code. Where a license blocks redistributing source items, we release everything needed to reconstruct them deterministically (templates + seeds). This satisfies reproducibility (Document 10) without violating licenses.

## 4.9 Pilot-confirmed parameters (placeholders until Stage 2)

These are set in the Stage-2 pilot (Document 11) and then frozen:

- `max_new_tokens` per task (provisional 512 reasoning / 256 MCQ).
- The exact subject-stratified MMLU-Pro subsample size and composition.
- The wordlist and POS-tagger versions for nonword and key-term checks.
- The empirical clean accuracy `A₀` per model, used to (a) validate fresh-instance comparability and (b) condition the quantization analysis.

Once frozen at pre-registration (Document 10), they do not change; any later change is logged in Document 00 §0.5 with a reason.
