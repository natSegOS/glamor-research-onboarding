# Bibliography manifest: GLAMOR Lab Exp 001

One row per reference. Columns: **citation key** (= PDF filename in `references/`, minus `.pdf`) | **full reference** with arXiv/DOI identifier | **where / why used** (design sections and/or code modules).

> **Citation style throughout this project:** Surname et al. (Year, arXiv:ID) or Surname (Year, Venue).
>
> **Verification protocol:** every row below was checked against the actual PDF's first page (title, authors, identifier) and against the design/code loci it claims, on **2026-07-20**. `tools/verify_references.py` re-checks the title/identifier match mechanically; run it whenever a PDF is added or replaced. Five classic statistics papers are paywalled; their rows are marked *(PDF pending institutional fetch)* until the PDF lands; the row metadata is verified against the publisher's record.

---

## Adversarial / noise robustness of LLMs

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `AdvBERT_Sun` | Sun, Hashimoto, Yin, Asai, Li, Yu & Xiong (2020). *Adv-BERT: BERT is not robust on misspellings! Generating nature adversarial samples on BERT.* arXiv:2003.04985 | docs/PROVENANCE.md §2.3, design/01 (H4): typos on **informative words** damage more than typos on random words; the precedent for the `informative_word` targeting policy |
| `AdversarialOverSensitivity_Niu` | Niu & Bansal (CoNLL 2018). *Adversarial Over-Sensitivity and Over-Stability Strategies for Dialogue Models.* arXiv:1809.02079 | design/01 §1.2, design/10 (attack 14): names over-sensitivity (Should-Not-Change) vs. over-stability (Should-Change); the origin of the selective-invariance framing behind the three-regime design |
| `ArithmAttack_Abedin` | Abedin, Qamar, Flek & Karimi (2025). *ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving.* arXiv:2501.08203 | design/01 §1.2: punctuation-only noise (no words added or removed) still degrades math reasoning on GSM8K/MultiArith across 8 LLMs |
| `CombatingAdversarialMisspellings_Pruthi` | Pruthi, Dhingra & Lipton (ACL 2019). *Combating Adversarial Misspellings with Robust Word Recognition.* ACL anthology P19-1561 | design/03 §3.5: the swap/drop/keyboard/add attack primitive family (internal characters, words ≥ 4 chars, stopwords excluded); design/10: the ScRNN word-recognition defense as a mitigation baseline. Note: its substitution primitive is **keyboard-adjacent**, and the informative-word finding belongs to `AdvBERT_Sun`, not this paper |
| `CheckList_Ribeiro` | Ribeiro, Wu, Guestrin & Singh (ACL 2020, best paper). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.* ACL anthology 2020.acl-main.442, arXiv:2005.04118 | design/01 §1.2–1.3: INV (invariance) / DIR (directional) behavioral test types; typos appear as an INV robustness perturbation, which grounds the paired clean/perturbed design |
| `MixedPrecisionAdvGlue_Lu` | Lu, Chen, Que, Luk & Fan (2025). *Enhancing Trustworthiness with Mixed Precision: Benchmarks, Opportunities, and Challenges.* arXiv:2511.22483 | design/05 §5.5: prior showing AWQ more robust than GPTQ at 4-bit and 3-bit on AdvGLUE++ (LLaMA-2-Chat 7B/13B, weight-only PTQ, classification tasks) |
| `MulTypo_Zhao` | Zhao, Liu, Altinger, Schütze & Hedderich (2025). *Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors.* arXiv:2510.09536 (MulTypo is the method/package name) | design/03 §3.5, `src/perturbation/keyboard.py`: the keyboard-layout-adjacency replacement operation we mirror for English QWERTY; its number-exclusion rule is convergent support for our protected numeric spans; validated for human naturalness in 6/7 languages |
| `R2ATA_Gan` | Gan, Zhao, Cheng, Mao, Goyal, Kawaguchi, Kan & Shieh (2024). *Reasoning Robustness of LLMs to Adversarial Typographical Errors.* arXiv:2411.05345 | design/01 §1.2, design/03 §3.4: the severity curve motivating the edit-budget ladder (Mistral-7B GSM8K 43.7% → 38.6% at 1 edit → 19.2% at 8); gradient-saliency (white-box) targeting of query-important words |
| `ResilienceOfLLMsForNoisyInstructions_Wang` | Wang, Wei, Liu, Lin & Chen (Findings of EMNLP 2024, pp. 11939–11950). *Resilience of Large Language Models for Noisy Instructions.* | design/10 §10.6: the "re-pass to self-denoise" baseline is weak for open models (helps ChatGPT, hurts Llama-2); also the ASR-noise realism motivation (builds a Whisper-based error-injection pipeline over 5 noise types) |
| `SmallEditsBigConsequences_Ismailov` | Ismailov & Asanova (2025). *Small Edits, Big Consequences: Telling Good from Bad Robustness in Large Language Models.* arXiv:2507.15868 | design/01: frontier-model selective robustness on code (≥85% pass under 90% prompt deletion vs. 54% sensitivity to a quantifier flip); contrasted with our open small-model + paired-statistics scope |
| `WikiTypos_Aliakbarzadeh` | Aliakbarzadeh, Flek & Karimi (2025). *Exploring Robustness of Multilingual LLMs on Real-World Noisy Data.* arXiv:2501.08322 (introduces the **WikiTypo** dataset) | design/03 §3.5: real-world (Wikipedia-edit-history) misspelling pairs as a real-word-shift source and empirical typo distribution; 2.3–4.3 pt average drops across nine fine-tuned NLU models |

---

## Tokenization and subword representations

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `BPEDropout_Provilkov` | Provilkov, Emelianenko & Voita (ACL 2020). *BPE-Dropout: Simple and Effective Subword Regularization.* ACL anthology 2020.acl-main.170, arXiv:1910.13267 | design/01 §1.2: the regularization technique we deliberately do NOT use (we measure the tokenization effect rather than regularizing against it) |
| `ByteLatentTransformer_Pagnoni` | Pagnoni, Pasunuru, Rodriguez, et al. (2024). *Byte Latent Transformer: Patches Scale Better Than Tokens.* arXiv:2412.09871 | design/01 §1.2, design/03 (exploratory): byte-level architectures match Llama 3 at 8B FLOP-controlled while more robust to noise; motivates the fragmentation stratum as a temporary construct |
| `TokenizationFallingShort_Chai` | Chai, Fang, Peng & Li (Findings of EMNLP 2024). *Tokenization Falling Short: On Subword Robustness in Large Language Models.* arXiv:2406.11687 (v3; the v1 preprint was subtitled "The Curse of Tokenization") | design/01 §1.4: shows subword tokenization correlates with typo brittleness (correlation only; no causal decomposition), which is exactly the gap the fragmentation-mediation contribution addresses; `src/tokenization.py` |
| `TypoNeurons_Tsuji` | Tsuji, Hiraoka, Cheng, Aramaki & Iwakura (2025). *Investigating Neurons and Heads in Transformer-based LLMs for Typographical Errors.* arXiv:2502.19669 | design/01 §1.4: mechanistic (neuron/attention-head) account of typo processing: early/late-layer typo neurons handle local context, middle layers global; contrasted with our behavioral/statistical approach |

---

## Edit distance and spelling correction

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `Damerau` | Damerau, F. J. (1964). *A Technique for Computer Detection and Correction of Spelling Errors.* Communications of the ACM 7(3):171–176 | design/02 §2.2: over 80% of observed errors fall into four single-error classes (wrong / missing / extra letter, adjacent transposition), with transposition counted as ONE error; `src/perturbation/engine.py` |
| `GitHubTypoCorpus_Hagiwara` | Hagiwara & Mita (LREC 2020). *GitHub Typo Corpus: A Large-Scale Multilingual Dataset of Misspellings and Grammatical Errors.* arXiv:1911.12893 | design/03 §3.5, design/04 §4.7: supplementary real-world misspelling pairs used to cross-validate the synthetic Regime B construction (see docs/PROVENANCE.md §1.3) |

---

## Benchmarks and tasks

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `GeoRepEval_Jawandhia` | Jawandhia, Sinha, Mandal, Pal & Kumar (2026). *Measuring Representation Robustness in Large Language Models for Geometry.* arXiv:2604.16421 | design/01 §1.8 (differentiation table): same paired McNemar + B=10,000 bootstrap spirit applied to geometry representation; methodology comparator only |
| `GSM8K_Cobbe` | Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168 (introduces GSM8K) | design/04 §4.2: the GSM8K benchmark (contamination-contrast partner); `src/tasks/reasoning.py`; `src/tasks/_shared.py` answer format |
| `GSMSymbolic_Mirzadeh` | Mirzadeh, Alizadeh, Shahrokhi, Tuzel, Bengio & Farajtabar (ICLR 2025). *GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs.* arXiv:2410.05229 | design/04 §4.2: fragility framing (up to 65% drop with NoOp clauses, §4.4); symbolic-template generation; contamination rationale (quoted verbatim in docs/PROVENANCE.md §1.1); `src/tasks/reasoning.py` template parser |
| `HumansOrLLMs_Chen` | Chen, Chen, Liu, Jiang & Wang (EMNLP 2024). *Humans or LLMs as the Judge? A Study on Judgement Bias.* ACL anthology 2024.emnlp-main.474 | design/09 §9.7: judgement biases in both LLM and human judges (misinformation-oversight, gender, authority, beauty); part of the LLM-judge-caution rationale for the human audit |
| `JudgingLLMAsAJudge_Zheng` | Zheng, Chiang, Sheng, et al. (NeurIPS 2023 Datasets & Benchmarks). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* arXiv:2306.05685 | design/09 §9.7: the canonical source for **position, verbosity, and self-enhancement** biases of LLM judges |
| `JudgingTheJudges_Shi` | Shi, Ma, Liang, Diao, Ma & Vosoughi (2024). *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge.* arXiv:2406.07791 | design/09 §9.7: systematic **position-bias** study: 15 LLM judges, MT-Bench + DevBench, >150,000 evaluation instances (this paper covers position bias only; verbosity/self-enhancement are `JudgingLLMAsAJudge_Zheng`) |
| `LimitationsOfLLMAsJudge_Szymanski` | Szymanski, Ziems, Eicher-Miller, Li, Jiang & Metoyer (ACM IUI 2025). *Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks.* arXiv:2410.20266 (PDF on disk is the arXiv v1 preprint) | design/09 §9.7: subject-matter experts agreed with LLM judges only ~64% (mental health) / ~68% (dietetics); reinforces human authority in the audit |
| `MMLU_Hendrycks` | Hendrycks, Burns, Basart, Zou, Mazeika, Song & Steinhardt (ICLR 2021). *Measuring Massive Multitask Language Understanding.* arXiv:2009.03300 | design/04 §4.3: original MMLU; used as the contamination-contrast MCQ partner; noted as contaminated and 4-option (we use MMLU-Pro as primary) |
| `MMLUPro_Wang` | Wang, Ma, Zhang, et al. (NeurIPS 2024 Datasets & Benchmarks). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* arXiv:2406.01574 | design/04 §4.3: primary MCQ task; 10 options (avg 9.47 after false-negative removal; do not assume a uniform 1/10 guess rate), prompt sensitivity 4–5% → 2%, 12,032 test items; `src/tasks/multiple_choice.py` |
| `SmallerWeaker_Fang` | Fang, Ding, Mastropaolo & Xu (2025). *Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation.* arXiv:2506.22776 | design/01 §1.6, design/05 §5.5: quantized models MORE adversarially robust in 51.59% vs 42.86% of cases (code generation; **bitsandbytes NF4/FP4**, not AWQ/GPTQ, an adjacent prior on a different recipe and task); motivates the two-sided H2 |
| `AdversarialMathWordProblems_Xie` | Xie, Huang, Wang & Dhingra (2024). *Adversarial Math Word Problem Generation.* arXiv:2402.17916 (Findings of EMNLP 2024) | design/04 §4.7: AST-based **numeric-value** edits that preserve structure/difficulty but break LLM solving; the motivation for the Regime C operand-swap control |
| `MCQOptionOrder_Pezeshkpour` | Pezeshkpour & Hruschka (2023). *Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions.* arXiv:2308.11483 (Findings of NAACL 2024) | `src/regimes.py`: 13–75% performance gaps under option reordering; grounds the Regime C option-permutation control (a model relying on position rather than content will fail it) |

---

## ASR and speech (deferred acoustic arm)

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `Whisper_Radford` | Radford, Kim, Xu, Brockman, McLeavey & Sutskever (OpenAI, 2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* arXiv:2212.04356 | design/03 §3.5a and docs/PROVENANCE.md §3: the **deferred** acoustic-arm design (2026-07-09 amendment, design/00 §0.5). §4.5 documents the temperature-escalation strategy: escalate 0.0→1.0 by 0.2 when avg log-prob < −1.0 or gzip compression ratio > 2.4. The fact that a *scalar* `temperature=0.0` disables the escalation is an **openai-whisper implementation behavior** (verified against the official repo, June 2026), not a claim of the paper |

---

## Statistical methods

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `DifferenceConfidenceIntervalSOA_Bestgen` | Bestgen, Y. (2022). *Please, Don't Forget the Difference and the Confidence Interval when Seeking for the State-of-the-Art Status.* arXiv:2205.11134 | design/06 §6.5, `src/analysis/statistics.py`: argues for bootstrap CIs on system differences in NLP; uses BCa with B=10,000 in its case studies (and notes ≥1,000–5,000 suffices, its fn. 2 citing Hesterberg). Our per-cell B=10,000 is a registered value within that practice, not an external "convention" |
| `HitchhikersGuideToTestingStatisticalSignificantInNLP_Dror` | Dror, Baumer, Shlomov & Reichart (ACL 2018). *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing.* ACL anthology P18-1128 | design/06 §6.4: McNemar is the appropriate sampling-free non-parametric test for paired binary labels (§3.2.2); Type-I error framing; `src/analysis/statistics.py` |
| `McNemarSampleSize_Connor` | Connor, R. J. (1987). *Sample Size for Testing Differences in Proportions for the Paired-Sample Design.* Biometrics 43(1):207–211. JSTOR 2531961 | `src/analysis/statistics.py::mcnemar_sample_size`: eq. (3) implemented exactly (with the standard two-sided z_{1−α/2} substitution; the paper's setup is one-sided). The design-table values 628/873/314 come from the **simple planning approximation**, not from eq. (3); see the verification table below |
| `ReplicabilityAnalysisForNLP_Dror` | Dror, Baumer, Bogomolov & Reichart (2017). *Replicability Analysis for Natural Language Processing: Testing Significance with Multiple Datasets.* TACL 5:471–486 | design/06 §6.7: BH-style FDR / partial-conjunction control for multiple comparisons in NLP (NOT about exact-reproduction logging; the paper explicitly distinguishes replicability from reproducibility) |
| `SampleSizeTablesForClinicalStudies_Machin` | Machin, Campbell, Fayers & Pinol (1997). *Sample Size Tables for Clinical Studies.* 2nd ed., Blackwell Science | design/06 §6.3: McNemar paired-design sample-size lineage (Ch. 4; note the bound-in errata slip correcting eq. 4.1/4.2 in this scan); Ch. 6 Table 6.1 is consistent with the Wald audit sizes (390 at table precision vs. our computed 385) |
| `WithLittlePowerComesGreatResponsibility_Card` | Card, Henderson, Khandelwal, Jia, Mahowald & Jurafsky (EMNLP 2020). *With Little Power Comes Great Responsibility.* arXiv:2010.06595 | design/06 §6.3: NLP comparisons are systematically underpowered; Type-M / Type-S error inflation; endorses McNemar for paired classifier comparison (§3.1); its power-analysis tooling seeds design/06 §6.9 |
| `MeasuringNominalScaleAgreement_Fleiss` | Fleiss, J. L. (1971). *Measuring nominal scale agreement among many raters.* Psychological Bulletin 76(5):378–382. doi:10.1037/h0031619 *(PDF pending institutional fetch)* | `src/analysis/audit.py::fleiss_kappa`: the agreement statistic behind the locked κ ≥ 0.60 Stage-2 gate (design/09 §9.4) |
| `CoefficientOfAgreementNominalScales_Cohen` | Cohen, J. (1960). *A coefficient of agreement for nominal scales.* Educational and Psychological Measurement 20(1):37–46. doi:10.1177/001316446002000104 *(PDF pending institutional fetch)* | design/09 §9.4: the pairwise κ cross-check; human–judge agreement metric promised in `src/judge.py` |
| `ObserverAgreementCategoricalData_LandisKoch` | Landis, J. R. & Koch, G. G. (1977). *The measurement of observer agreement for categorical data.* Biometrics 33(1):159–174. doi:10.2307/2529310 *(PDF pending institutional fetch)* | design/09 §9.4: the interpretation scale for κ (0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect) |
| `SequentiallyRejectiveMultipleTest_Holm` | Holm, S. (1979). *A simple sequentially rejective multiple test procedure.* Scandinavian Journal of Statistics 6(2):65–70. JSTOR 4615733 *(PDF pending institutional fetch)* | design/06 §6.7, `tools/run_analysis.py`: Holm step-down correction within a model's pre-registered primary family |
| `ControllingTheFalseDiscoveryRate_BenjaminiHochberg` | Benjamini, Y. & Hochberg, Y. (1995). *Controlling the false discovery rate: a practical and powerful approach to multiple testing.* Journal of the Royal Statistical Society B 57(1):289–300 *(PDF: scan hosted on Y. Benjamini's TAU homepage; no text layer; verify_references skips text matching for it)* | design/06 §6.7, `tools/run_analysis.py`: BH-FDR at q=0.05 across the exploratory cell grid (a locked decision, design/00 §0.4) |
| `BetterBootstrapConfidenceIntervals_Efron` | Efron, B. (1987). *Better bootstrap confidence intervals.* Journal of the American Statistical Association 82(397):171–185. doi:10.1080/01621459.1987.10478410 *(PDF on disk: Stanford Statistics TR-226 preprint (1984) from the Stanford Digital Repository; the typeset JASA version is paywalled; scan has no text layer)* | `src/analysis/statistics.py`: the BCa (bias-corrected and accelerated) interval itself, computed via `scipy.stats.bootstrap(method="BCa")` |
| `MidPMcNemar_Fagerland` | Fagerland, Lydersen & Laake (2013). *The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional.* BMC Medical Research Methodology 13:91. doi:10.1186/1471-2288-13-91 (open access) | design/06 §6.4, `src/analysis/statistics.py::mcnemar_test`: the mid-p exact variant used when discordant pairs < 25 |

---

## Mixed-effects modeling and mediation

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `AGeneralApproachToCausalMediationAnalysis_Imai` | Imai, Keele & Tingley (2010). *A General Approach to Causal Mediation Analysis.* Psychological Methods 15(4):309–334 | design/06 §6.8, `src/analysis/models.py`: the causal mediation framework; our binary-outcome estimator follows the paper's general algorithm (counterfactual prediction under a logistic outcome model), NOT the linear product-of-coefficients shortcut the paper cautions against for binary outcomes (p. 316) |
| `CategoricalDataAnalysis_Jaeger` | Jaeger, T. F. (2008). *Categorical Data Analysis: Away from ANOVAs (Transformation or Not) and Towards Logit Mixed Models.* Journal of Memory and Language 59(4):434–446 | design/06 §6.6, `src/analysis/models.py`: why the confirmatory model is a **logit** mixed model rather than any linear model on accuracy |
| `MixedEffectsModeling_Baayen` | Baayen, Davidson & Bates (2008). *Mixed-Effects Modeling with Crossed Random Effects for Subjects and Items.* Journal of Memory and Language 59(4):390–412 | design/06 §6.6: crossed (not nested) random effects; our item × model specification is the direct analogue of subjects × items |
| `RandomEffectsStructure_Barr` | Barr, Levy, Scheepers & Tily (2013). *Random Effects Structure for Confirmatory Hypothesis Testing: Keep It Maximal.* Journal of Memory and Language 68(3):255–278 | design/06 §6.6, `src/analysis/models.py`: maximal random-effects structure and the pre-registered convergence ladder (its "coping with failures to converge" guidance, pp. 275–276, is what the ladder implements) |

---

## Systems and inference

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `VllmPagedAttention_Kwon` | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang & Stoica (SOSP 2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* arXiv:2309.06180 | design/07 §7.2: the locked inference engine (vLLM): continuous batching + paged KV-cache + prefix caching; `src/inference/engines.py` |
| `AWQ_Lin` | Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan & Han (MLSys 2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* arXiv:2306.00978 | design/05 §5.4: the locked main-sweep quantization recipe (AWQ W4A16) |
| `GPTQ_Frantar` | Frantar, Ashkboos, Hoefler & Alistarh (ICLR 2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323 | design/05 §5.5: the second recipe in the fp16 / AWQ-4bit / GPTQ-4bit sub-study |

---

## Scoring-rule sources

| Key | Full reference | Where / why used |
|-----|---------------|------------------|
| `AskingClarifyingQuestions_Aliannejadi` | Aliannejadi, Zamani, Crestani & Croft (SIGIR 2019). *Asking Clarifying Questions in Open-Domain Information-Seeking Conversations.* arXiv:1907.06554 | `src/scoring.py`: the clarification-question taxonomy that the CLARIFICATION parse status formalizes (design/04 §4.5) |
| `UniversalAdversarialAttacks_Zou` | Zou, Wang, Carlini, Nasr, Kolter & Fredrikson (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043 | `src/scoring.py`, `tests/fixtures/refusal_phrases.txt`: the refusal-phrase methodology whose positive examples are the frozen validation oracle for the REFUSAL detector |

---

## Verification status of load-bearing quantitative claims

All claims below were re-verified against the cited PDF on 2026-07-20:

| Claim | Source | Status |
|-------|--------|--------|
| `mcnemar_sample_size(0.05, 0.20, SIMPLE) == 628`; `(0.03, 0.10) == 873`; `(0.05, 0.10) == 314` | The **simple planning approximation** (z_{α/2}+z_β)²·p_d/δ², used for the design tables; this formula is *not* in Connor's paper | ✓ matches code |
| Connor eq. (3) exact values for the same inputs: **626 / 870 / 312** | Connor (1987) eq. (3), p. 208 (two-sided z substitution) | ✓ derived independently; matches the `CONNOR` branch |
| Connor eq. (3) ≤ simple approximation | √(ψ−δ²) ≤ √ψ, always | ✓ mathematically; note Connor's own §2 comparison is against Miettinen's approximation, not ours |
| `audit_sample_size(0.05) == 385`; `(0.03) == 1068` | Wald worst-case z²·0.25/m²; cross-checked against Machin (2nd ed. 1997) Table 6.1 (390 at table precision for the 5 pp margin) | ✓ |
| Transposition = 1 edit (not 2) | Damerau (1964), p. 171 four single-error classes; Table 4 tallies "Transposed pair" under Single Error | ✓ |
| Whisper escalation thresholds: temperature ladder (0.0→1.0 by 0.2) when avg log-prob < −1.0 or gzip ratio > 2.4 | Radford et al. (2022) §4.5 | ✓ |
| Scalar `temperature=0.0` disables the escalation | **openai-whisper implementation** (`whisper.transcribe`), verified June 2026, not a claim of the paper | ✓ implementation fact |
| AWQ more robust than GPTQ at 4-bit and 3-bit | Lu et al. (2025) Fig. 1 caption and §III.C.2 | ✓ |
| Quantized models more robust in 51.59% vs 42.86% of adversarial cases | Fang et al. (2025) abstract (bitsandbytes NF4/FP4 quantization, code generation) | ✓ (rounded to 51.6% vs 42.9% in design prose) |
| GSM-Symbolic performance drops up to 65% with NoOp clauses | Mirzadeh et al. (ICLR 2025) §4.4 | ✓ |
| "the popularity and prevalence of GSM8K can increase the risk of inadvertent data contamination" | Mirzadeh et al., p. 2, verbatim | ✓ |
| MMLU-Pro: prompt sensitivity 4–5% → 2%; 12,032 test items; 10 options (avg 9.47) | Wang et al., abstract / §3.1 / Table 1 discussion | ✓ |
| 15 LLM judges, >150,000 evaluation instances (position bias) | Shi et al., abstract | ✓ |
| SME–LLM-judge agreement ~64% (mental health) / ~68% (dietetics) | Szymanski et al., abstract | ✓ |
| Mistral-7B GSM8K 43.7% → 38.6% (1 edit) → 19.2% (8 edits) | Gan et al., abstract + Table 2 | ✓ |
| Typos on informative words damage more | Sun et al. (2020), abstract finding (i) | ✓ |
