# 01 — Scientific Framing, Novelty, and Hypotheses

**Read this first.** It explains what we are doing, why it is worth publishing, exactly what is novel versus what is already known, and the precise questions and hypotheses everything else serves. If a reviewer is going to reject the paper, it will most likely be on the grounds covered here, so this document is deliberately conservative about what we claim.

---

## 1.1 The problem in one paragraph

Voice is becoming the dominant interface for AI assistants. But the transcription step — converting speech to the text the model actually receives — is systematically noisy. ASR systems produce acoustic confusions ("weather" for "whether"), disfluencies, run-on phrasing, absent punctuation, and filler words. These are not occasional edge cases; they are a structural property of the voice pipeline, and they arrive at the LLM as corrupted text. The question this study asks is not "does this hurt performance" (it does; noisy-instruction work has established that) but **why it hurts and through what mechanism** — because only a mechanistic answer can guide mitigation. Keyboard typos provide the controlled experimental baseline for the same causal question, and ASR errors provide the ecologically motivated application. The study combines both.

## 1.2 What is already known (so we do not claim it)

The literature from 2018 to 2026 has firmly established several things. We must cite these and must not present them as our findings.

- **Typos degrade LLM task performance, sometimes sharply.** R²ATA (Gan et al., 2024, arXiv:2411.05345) reports Mistral-7B-Instruct dropping from 43.7% to 38.6% on GSM8K with a *single* adversarial character edit, and to 19.2% with eight. MulTypo (Liu et al., 2025, arXiv:2510.09536) shows consistent degradation across 18 multilingual LLMs and five tasks, worst on generative and reasoning tasks. WikiTypos (Aliakbarzadeh et al., 2025, arXiv:2501.08322) shows a 2.3–4.3 point average drop from real-world Wikipedia-edit noise across nine models.
- **Even information-preserving surface noise hurts.** ArithmAttack (Abedin et al., 2025, arXiv:2501.08203) shows punctuation-only noise — which removes no words — still degrades math reasoning.
- **Tokenization is implicated.** "Tokenization Falling Short" / the curse of tokenization (Chai et al., Findings of EMNLP 2024, arXiv:2406.11687) shows subword tokenization makes models brittle to typographical variation and that scaling helps but does not cure it; BPE-dropout (Provilkov et al., ACL 2020) mitigates it.
- **The invariance-vs-sensitivity dichotomy is old.** Niu & Bansal (CoNLL 2018) named over-sensitivity ("should-not-change") and over-stability ("should-change"). CheckList (Ribeiro et al., ACL 2020 best paper) formalized invariance (INV) and directional (DIR) behavioral tests. Most recently, Ismailov & Asanova (2025, arXiv:2507.15868, "Small Edits, Big Consequences") argued explicitly for *selective* robustness and found, on code prompts, ~85% over-robustness under 90% prompt deletion and only 54% sensitivity to a quantifier flip.
- **Tokenizer-free architectures are more robust.** The Byte Latent Transformer (Pagnoni et al., Meta FAIR, ACL 2025, arXiv:2412.09871) matches Llama 3 at 8B scale while being markedly more robust to noisy input.
- **There are "typo neurons."** Tsuji et al. (2025, arXiv:2502.19669) localize internal typo-repair behavior to specific middle-layer neurons and attention heads.

The honest consequence: **"we show LLMs are brittle to typos" is not publishable in 2026.** Neither is "LLMs should be invariant to benign noise but sensitive to meaning changes" as a *concept* — that is Niu & Bansal and CheckList. The original plan's framing, taken at face value, would be rejected as a restatement.

## 1.3 The framing we will actually use

We keep the selective-invariance vocabulary because it organizes the experiment cleanly and is reader-friendly, but we **explicitly attribute the concept** to Niu & Bansal (2018), CheckList (2020), and Ismailov & Asanova (2025), and we position our contribution as the *measurement* and *mechanism*, not the idea. Concretely, our framing sentence is:

> ASR transcription introduces systematic noise into voice-LLM pipelines. We provide the first matched-pair, statistically disciplined decomposition of noise-induced accuracy loss into a tokenization-fragmentation channel and a residual channel — using both controlled keyboard-adjacency typos and realistic ASR-transcription errors — and the first controlled test of whether 4-bit quantization changes robustness to transcription noise on reasoning tasks. A three-regime design separates intent-preserving noise from meaning-changing edits throughout, so robustness is never confused with altered task definition.

Everything defensible in this study lives in three words: **matched-pair, mechanism, controlled.**

## 1.4 The three contributions, ranked

We lead with the contribution that is both genuinely open and hard to refute, and we keep the weaker ones as support.

### Contribution 1 (primary) — Tokenization-fragmentation mediation

**Claim we will be able to defend:** "The accuracy lost under intent-preserving typos is partly mediated by how badly the typo fragments the affected word into more subword tokens. We estimate the mediated fraction with a matched, fragmentation-controlled counterfactual."

Why it is open: Chai et al. (2024) show a *correlation* between tokenization and brittleness; Tsuji et al. (2025) show internal repair *mechanisms*; neither delivers an item-level causal decomposition of *how much of the accuracy drop runs through fragmentation.* That decomposition is what we add. It is mechanistic, quantitative, and the kind of result reviewers respect because it explains rather than merely measures.

How we make it non-refutable: we use a **fragmentation-matched counterfactual** (Document 02 §2.5, Document 06 §6.8). For a given target word and edit budget, multiple keyboard-plausible typos exist; some fragment the word into many more subword pieces than others. We compare model accuracy on high-fragmentation vs low-fragmentation typos *of the same word at the same edit distance*, so the comparison holds meaning, position, and edit count fixed and varies only the tokenization consequence. This converts "tokenization correlates with failure" into "holding everything else fixed, more fragmentation causes more failure," which is a causal statement within a clearly bounded scope.

### Contribution 2 (strong secondary) — Quantization × typo interaction

**Claim:** "Holding clean accuracy and the perturbation fixed, 4-bit AWQ quantization changes typo robustness by [measured amount], in [direction]."

Why it is open: the only adjacent result is Fang et al. (2025, arXiv:2506.22776, "Smaller = Weaker?"), which found quantized code-generation models *more* adversarially robust in 51.6% of cases versus 42.9% — a surprising direction. Whether that generalizes to nonword typos on math/MCQ reasoning is unknown. The lab already cares about quantized 7–8B models (repo issue 02), so this contribution is aligned with existing lab interest and reuses the `quant_bits` plumbing already in `experiments/000_trajectory_divergence/model.py`.

How we make it non-refutable: we hold the quantization *method* constant (AWQ) within the main sweep, and run a dedicated sub-study (Document 05 §5.5) that compares fp16 vs AWQ vs GPTQ on a fixed model subset so that any interaction is not an artifact of one quantization recipe. We always condition on clean accuracy, because a quantized model that is simply worse overall is not the same as one that is *less typo-robust*.

### Contribution 3 (framing / hygiene) — Three-regime selective-invariance audit with paired statistics

**Claim:** "We report robustness only on audited intent-preserving items, measure over-robustness on meaning-changing controls, and do so with matched-pair tests and confidence intervals — a level of statistical discipline absent from most typo-robustness papers."

Why it still counts: no prior typo paper combines (a) a human-audited three-regime separation with (b) McNemar + BCa + crossed-random-effects mixed models. It is not a flashy contribution, but it is what makes contributions 1 and 2 trustworthy, and reviewers in the ACL/EMNLP "evaluation and analysis" track reward it.

## 1.5 Research questions

- **RQ1 (primary, mechanism).** Of the accuracy lost under intent-preserving noise — whether keyboard-typo or ASR-transcription — what fraction is attributable to subword fragmentation, holding the target word, edit count, and position fixed?
- **RQ2 (secondary, quantization).** Does 4-bit quantization change an instruction model's typo robustness relative to its fp16 counterpart, after conditioning on clean accuracy, and in which direction?
- **RQ3 (framing, selectivity).** Do current open instruction LLMs exhibit selective invariance — low clean-conditioned failure on intent-preserving noise yet high appropriate-change on meaning-changing controls — and how does this depend on model scale, family, edit budget, and edit location?
- **RQ4 (descriptive, structure).** Which primitive operations, edit locations, and selection policies (keyboard-neighbor, ASR-transcription, uniform, informative-word-targeted) produce the most clean-conditioned failure, and do keyboard-typo and ASR-error conditions produce similar degradation profiles?

RQ1 and RQ2 are the publishable core. RQ3 and RQ4 are the map that contextualizes them and that any reviewer expects to see.

## 1.6 Hypotheses (directional, pre-registered)

These are committed *before* the held-out runs (Document 10, Document 11 Stage 3). Stating direction in advance is what lets us claim a confirmatory rather than exploratory result.

- **H1 (mediation).** Clean-conditioned failure increases monotonically with token-inflation quartile, and the fragmentation-matched counterfactual shows a positive, significant fragmentation effect (more fragmentation → more failure) within matched word/edit-count strata. *Mechanism prediction.*
- **H2 (quantization).** Direction is left open (two-sided) because the only prior evidence (code generation) points the counterintuitive way; we pre-register a two-sided test and will report whichever direction emerges with its CI. We do **not** pre-commit to "quantization hurts." *Honest uncertainty is itself defensible.*
- **H3 (selectivity).** Clean-conditioned failure on Regime A (intent-preserving) is low at k=1 and rises with k; appropriate-change rate on Regime C (meaning-changing) is high; over-robustness rate (clinging to the old answer when the meaning changed) is non-negligible and rises with model scale — i.e., bigger models are *more* over-robust, echoing Ismailov & Asanova (2025) and MulTypo's observation that instruction tuning can increase brittleness/rigidity.
- **H4 (structure).** Answer-critical and informative-word-targeted edits produce substantially higher clean-conditioned failure than uniform-random edits at the same k; deletion and substitution fragment tokenization more than transposition and therefore (via H1) hurt more. *Replicates and extends Pruthi et al. (2019).*

## 1.7 Scope statement (what we deliberately do not claim)

A paper is hard to refute partly because of what it refuses to say. We bound the scope explicitly:

- We study **English** only, including English ASR transcription errors. We do not claim multilingual generality (MulTypo owns that).
- We study **open-weight instruction-tuned LLMs in the 1B–8B range**. We do not claim anything about frontier or closed models.
- We study **BPE/subword-tokenized** models. Byte-level models (BLT, ByT5) are discussed as the contrasting architecture but are out of experimental scope except as an optional single-point reference.
- We measure **task accuracy** on deterministically scorable tasks. We do not study end-to-end ASR pipeline optimization, open-ended generation quality, or safety under noise.
- The mediation claim is **causal only within the fragmentation-matched counterfactual's bounded scope** — we do not claim a complete causal account of typo brittleness, only that fragmentation is a measurable, manipulable channel.
- The quantization claim is about **one quantization family per cell**, with a sub-study to guard against recipe-specific artifacts; we do not claim all quantization methods behave identically.

## 1.8 Positioning against the closest competitors (the differentiation table)

| Competitor | What they did | What we do differently |
|---|---|---|
| R²ATA (Gan 2024) | Adversarial gradient-guided typos on reasoning; severity curve | Naturalistic keyboard typos, matched-pair stats, mechanism (fragmentation), quantization |
| MulTypo (Liu 2025) | Multilingual keyboard typo generator + broad eval | English-only depth: mediation + quantization + paired stats; we *use* MulTypo as the generator |
| WikiTypos (2025) | Real-world noise, multilingual, 3 tasks | We separate real-word shifts as their own audited regime; we add mechanism + quantization |
| Tokenization Falling Short (Chai 2024) | Tokenization correlates with brittleness | We move from correlation to a fragmentation-matched counterfactual decomposition |
| Tsuji et al. (2025) | Internal "typo neurons/heads" | Behavioral, item-level, quantitative mediation rather than circuit localization |
| Ismailov & Asanova (2025) | Selective robustness on code, frontier models, no stats | Reasoning+MCQ, open small models, nonword typos, full paired-stats stack, mechanism |
| GeoRepEval (2026) | Paired McNemar+bootstrap on geometry | Same statistical spirit, applied to typo robustness with the mediation and quantization contributions |

The cell that no competitor fills: **matched-pair fragmentation-mediation + quantization interaction, dual keyboard+ASR noise sources, three audited regimes, open small models, English reasoning + MCQ.** That is our claim to a place in the literature, and it is narrow enough to defend.

## 1.9 Why this is worth a journal/conference slot

Three reasons, in the order a reviewer weighs them. First, it answers a *mechanistic* question (RQ1) rather than adding another benchmark number — the field is moving from "models fail" to "models fail *because*," and we are on the right side of that. Second, it is *statistically disciplined* in a subfield that is often not, which directly raises its credibility. Third, it is *cheap to reproduce* (open small models, deterministic scoring, released code/configs/seeds), which reviewers increasingly reward and which makes the claims verifiable rather than take-our-word-for-it. The confirmed target venue is ACL Rolling Review → EMNLP 2026 main (evaluation and analysis track); NAACL 2027 is the fallback. Document 11 §11.7 covers submission timing.
