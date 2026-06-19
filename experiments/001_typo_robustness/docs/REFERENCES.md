# Bibliography manifest — GLAMOR Lab Exp 001

One row per PDF in `references/`. Columns: **citation key** used in this project's design docs and code comments | **full reference** with arXiv/DOI identifier | **where / why used** (design sections and/or code modules). Verified 2026-06-19.

> **Citation style throughout this project:** Surname et al. (Year, arXiv:ID) or Surname (Year, Venue).

---

## Adversarial / noise robustness of LLMs

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `AdversarialOverSensitivity_Niu` | Niu et al. (2024). *LLMs Are Not Robust Few-Shot Learners.* arXiv:2404.04935 | `src/scoring.py` module docstring — justifies the clarification/refusal phrase proxy as a conservative under-count (over-sensitivity framing); design/04 §4.5 |
| `ArithmAttack_Abedin` | Abedin et al. (2025). *Arithmetic Attack: Exploring Adversarial Robustness of LLMs on Math Benchmarks.* arXiv:2501.08203 | design/01 §1.3 — evidence that numerical changes break LLMs; motivation for Regime C operand swap |
| `CombatingAdversarialMisspellings_Pruthi` | Pruthi et al. (ACL 2019). *Combating Adversarial Misspellings with Robust Word Recognition.* ACL 2019 anthology P19-1561 | design/03 §3.2 — source of the uniform substitution policy (`SelectionPolicy.UNIFORM`); `src/perturbation.py` |
| `CheckList_Ribeiro` | Ribeiro et al. (ACL 2020, best paper). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL 2020 anthology 2020.acl-main.442 | design/01 §1.5 — INV/DIR behavioral test framing; motivates the paired clean/perturbed design |
| `MixedPrecisionAdvGlue_Lu` | Lu et al. (2025). *Enhancing Trustworthiness with Mixed Precision: Benchmarks, Opportunities, and Challenges.* arXiv:2511.22483 | design/05 §5.5 — prior showing AWQ more robust than GPTQ at 4-bit and 3-bit on AdvGLUE++ trustworthiness metrics |
| `MulTypo_Liu` | Liu et al. (2025). *MulTypo: Multilingual Typo Robustness Benchmark.* arXiv:2510.09536 | design/01 §1.3, design/03 §3.2 — validates keyboard-layout adjacency model; we cite and reuse adjacency structure rather than reinventing |
| `R2ATA_Gan` | Gan et al. (2024). *R2ATA: Towards Realistic and Reliable Adversarial Text Attacks.* arXiv:2411.05345 | design/01 §1.3 — reports Mistral-7B dropping from 43.7% to 38.6% on GSM8K with single adversarial edit |
| `ResilienceOfLLMsForNoisyInstructions_Wang` | Wang et al. (2024). *Resilience of LLMs for Noisy Instructions.* (venue TBD) | `src/scoring.py` module docstring — supports conservative proxy rationale; design/04 §4.5 |
| `SmallEditsBigConsequences_Ismailov` | Ismailov & Asanova (2025). *Small Edits, Big Consequences: Selective Robustness in Frontier LLMs.* arXiv:2507.15868 | design/01 §1.3 — frontier model selective robustness on code; contrasted with our open small-model focus |
| `WikiTypos_Aliakbarzadeh` | Aliakbarzadeh et al. (2025). *WikiTypos: A Real-World Multilingual Typo Dataset.* arXiv:2501.08322 | design/01 §1.3, design/03 §3.3 — motivates ASR pipeline as a realistic second noise source; real-world typo distribution |

---

## Tokenization and subword representations

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `BPEDropout_Provilkov` | Provilkov et al. (ACL 2020). *BPE-Dropout: Simple and Effective Subword Regularization.* ACL 2020 anthology 2020.acl-main.170 | design/03 §3.4 — cited as the regularization technique we do NOT use (we measure tokenization effect rather than regularizing against it) |
| `ByteLatentTransformer_Pagnoni` | Pagnoni et al. (ACL 2025). *Byte Latent Transformer: Patches Scale Better Than Tokens.* arXiv:2412.09871 | design/03 §3.4 — future-facing note on byte-level architectures; motivates fragmentation stratum as a temporary construct |
| `TokenizationFallingShort_Chai` | Chai et al. (Findings EMNLP 2024). *Tokenization Falling Short: The Curse of Tokenization.* arXiv:2406.11687 | design/03 §3.4 — shows tokenization correlates with brittleness; we move from correlation to counterfactual decomposition; `src/tokenization.py` |
| `TypoNeurons_Tsuji` | Tsuji et al. (2025). *Typo Neurons: How LLMs Internally Represent Typographic Noise.* arXiv:2502.19669 | design/01 §1.4 — mechanistic (circuit) explanation of typo failures; contrasted with our behavioral / statistical approach |

---

## Edit distance and spelling correction

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `Damerau` | Damerau, F. J. (1964). *A Technique for Computer Detection and Correction of Spelling Errors.* Communications of the ACM 7(3):171–176 | design/03 §3.1 — defines the four error types (substitution, insertion, deletion, transposition); transposition counts as 1 edit. `src/perturbation.py` DL metric |
| `GitHubTypoCorpus_Hagiwara` | Hagiwara & Mita (2020). *GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors.* arXiv:2011.09040 | design/03 §3.2 — empirical distribution of real-world edit types |

---

## Benchmarks and tasks

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `GeoRepEval_Jawandhia` | Jawandhia et al. (2026, arXiv:2604.16421). *Measuring Representation Robustness in Large Language Models for Geometry.* | design/01 §1.5 — uses the same paired McNemar+bootstrap statistical spirit applied to geometry representation; comparable methodology |
| `GSM8K_Cobbe` | Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K).* arXiv:2110.14168 | design/04 §4.2 — the original GSM8K benchmark; `src/tasks/reasoning.py`; contamination risk noted in design/01 |
| `GSMSymbolic_Mirzadeh` | Mirzadeh et al. (Apple, ICLR 2025). *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs.* arXiv:2410.05229 | design/04 §4.2 — fragility framing (up to 65% drop with NoOp clause); synthetic template design; `src/tasks/reasoning.py` generator |
| `HumansOrLLMs_Chen` | Chen et al. (EMNLP 2024). *Humans or LLMs as the Judge? A Study on Judgement Biases.* ACL anthology 2024.emnlp-main.474 | design/08 §8.2 — broad judgment biases of LLM-as-judge; motivates human audit |
| `JudgingTheJudges_Shi` | Shi et al. (2024). *Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges.* arXiv:2406.07791 | design/08 §8.2 — 15 judges, ~150k evaluations; cited alongside Chen et al. for human audit rationale |
| `LimitationsOfLLMAsJudge_Szymanski` | Szymanski et al. (ACM IUI 2025). *The Limitations of LLM-as-a-Judge.* arXiv:2410.20266 | design/08 §8.2 — domain-expert disagreement with LLM judges; reinforces human audit |
| `MMLU_Hendrycks` | Hendrycks et al. (ICLR 2021). *Measuring Massive Multitask Language Understanding.* arXiv:2009.03300 | design/04 §4.3 — original MMLU; noted as contaminated and 4-option; we use MMLU-Pro |
| `MMLUPro_Wang` | Wang et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* arXiv:2406.01574 | design/04 §4.3 — primary MCQ task; `src/tasks/multiple_choice.py` |
| `SmallerWeaker_Fang` | Fang, Ding, Mastropaolo & Xu (2025). *Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation.* arXiv:2506.22776 | design/01 §1.6, design/05 §5.4 — 51.59% vs 42.86% cases showing quantized models more robust (rounded to 51.6% vs 42.9% in prose); motivates two-sided H2 |

---

## ASR and speech

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `Whisper_Radford` | Radford et al. (OpenAI, 2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* arXiv:2212.04356 | `src/asr.py` — temperature fallback mechanism (§4.5): tuple `(0.0, 0.2, …, 1.0)` escalates when log-prob < −1 or gzip ratio > 2.4; scalar `temperature=0.0` disables it for reproducibility |

---

## Statistical methods

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `DifferenceConfidenceIntervalSOA_Bestgen` | Bestgen (2022). *Confidence Interval for the Difference Between Two Proportions: State of the Art.* arXiv:2205.11134 | `src/analysis/statistics.py` — BCa bootstrap CI, 10,000-resample convention; design/06 §6.5 |
| `HitchhikersGuideToTestingStatisticalSignificantInNLP_Dror` | Dror et al. (ACL 2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP.* ACL anthology P18-1128 | design/06 §6.4 — McNemar test selection, Type-I error control; `src/analysis/statistics.py` |
| `McNemarSampleSize_Connor` | Connor, R. J. (1987). *Sample Size for Testing Differences in Proportions for the Paired-Sample Design.* Biometrics 43(1):207–211. https://www.jstor.org/stable/2531961 | `src/analysis/statistics.py` `mcnemar_sample_size()` — equation (3) is implemented exactly; design table values 628/873/314 verified |
| `ReplicabilityAnalysisForNLP_Dror` | Dror et al. (2017). *Replicability Analysis for Natural Language Processing.* TACL 5:1–13 | design/06 §6.8 — replicability framing; motivates full state-vector logging for exact reproduction |
| `SampleSizeTablesForClinicalStudies_Machin` | Machin et al. *Sample Size Tables for Clinical Studies.* 3rd ed. Wiley-Blackwell, 2009 | design/06 §6.3 — design table cross-check for audit_sample_size values (385/1068 at 5%/3% margin) |
| `WithLittlePowerComesGreatResponsibility_Card` | Card et al. (EMNLP 2020). *With Little Power Comes Great Responsibility.* arXiv:2010.06595 | design/06 §6.3 — documents NLP comparisons are systematically underpowered; Type-M and Type-S error inflation; justifies our power analysis |

---

## Mixed-effects modeling and mediation

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `AGeneralApproachToCausalMediationAnalysis_Imai` | Imai et al. (2010). *A General Approach to Causal Mediation Analysis.* Psychological Methods 15(4):309–334 | design/07 §7.2 — causal mediation framework for the fragmentation-mediation decomposition |
| `CategoricalDataAnalysis_Jaeger` | Jaeger, T. F. (2008). *Categorical Data Analysis: Away from ANOVAs (Transformation or Not) and Towards Logit Mixed Models.* Journal of Memory and Language 59(4):434–446 | design/07 §7.3 — logit mixed model rationale |
| `MixedEffectsModeling_Baayen` | Baayen, Davidson & Bates (2008). *Mixed-Effects Modeling with Crossed Random Effects for Subjects and Items.* Journal of Memory and Language 59(4):390–412 | design/07 §7.3 — crossed random effects for items and models |
| `RandomEffectsStructure_Barr` | Barr et al. (2013). *Random Effects Structure for Confirmatory Hypothesis Testing: Keep It Maximal.* Journal of Memory and Language 68(3):255–278 | design/07 §7.3 — maximal random effects structure justification |

---

## Other supporting references

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `HumansOrLLMs_Chen` | *(see Benchmarks and tasks section above)* | — |
| `R2ATA_Gan` | *(see Adversarial section above)* | — |

---

## Verification status of load-bearing quantitative claims

All claims below were verified against the cited PDF on 2026-06-19:

| Claim | Source | Status |
|-------|--------|--------|
| `mcnemar_sample_size(0.05, 0.20, "simple") == 628` | Connor (1987) eq. (3) with independent-sample bound | ✓ derived independently; matches code |
| `mcnemar_sample_size(0.03, 0.10, "simple") == 873` | Connor (1987) | ✓ |
| `mcnemar_sample_size(0.05, 0.10, "simple") == 314` | Connor (1987) | ✓ |
| `audit_sample_size(0.05) == 385` | Wald worst-case formula z²·0.25/m² | ✓ |
| `audit_sample_size(0.03) == 1068` | Wald worst-case formula | ✓ |
| Connor method ≤ Simple method | Connor (1987) §2 — n is a more conservative upper bound | ✓ |
| Transposition = 1 edit (not 2) | Damerau (1964) — defines four single-error types including adjacent transposition | ✓ |
| Whisper temperature tuple `(0.0, 0.2, …, 1.0)` escalates when log-prob < −1 or gzip ratio > 2.4 | Radford et al. (2022) §4.5 | ✓ scalar disables escalation |
| AWQ more robust than GPTQ at 4-bit and 3-bit | Lu et al. (2025) Fig. 1 caption and §III.C.2 | ✓ |
| Quantized models more robust in 51.59% vs 42.86% of adversarial cases | Fang et al. (2025) abstract | ✓ (rounded to 51.6% vs 42.9% in design docs) |
| GSM-Symbolic performance drops up to 65% with NoOp clauses | Mirzadeh et al. (ICLR 2025) §4.4 | ✓ |
