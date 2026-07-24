# 06: Statistics and Power

This document ties every sample size to a target detectable effect, matches every test to the data structure, and bounds every claim with a confidence interval. It is the study's statistical defense. Read alongside Document 02 (metric definitions).

---

## 6.1 The shape of the data

For each item `i`, model `m`, and condition `c`, the data are a matched pair of binary outcomes `(Y_{i,m,0}, Y_{i,m,c})` (clean, perturbed) under deterministic greedy decoding (Document 05 §5.6). Because both outcomes come from the *same item*, the correct analysis is **paired/matched**, not a comparison of two independent accuracy numbers. Using unpaired tests here would be a methodological error. This single fact dictates every test below:

- McNemar (paired binary test)
- BCa bootstrap clustered by item (paired CI)
- mixed-effects logistic model with item as a random effect (paired regression)

The 2×2 paired table per cell:

```
                  perturbed ✓   perturbed ✗
   clean ✓           p11           p10        (a, b in McNemar notation)
   clean ✗           p01           p00        (c, d)
```

- `p10` (clean✓, perturbed✗) = typo-induced failures. **The cell we care most about.**
- `p01` (clean✗, perturbed✓) = accidental improvements (typo happened to help).
- Discordant pairs are `p10` and `p01`; concordant pairs (`p11`, `p00`) carry no information about *change*.

## 6.2 The primary endpoint, restated statistically

**Primary endpoint:** paired item-level correctness degradation on audited Regime-A (intent-preserving) items.

- **Effect size:** `Δ = A₀ − A₁` on the same items = `p10 − p01` (Document 02, M3).
- **Diagnostic:** clean-conditioned failure `CCF = p10 / (p10 + p11)` (M4).
- **Test:** McNemar on `(p10, p01)`.

The endpoint is computed *only* on items the human audit (Document 09) confirmed are intent-preserving, so "your typo changed the meaning" is answered before the statistics run.

## 6.3 Sample size, derived from the claim

We tie `N` to a *minimum detectable effect* (MDE) so that the number is a consequence of what we want to claim, not a guess.

**McNemar sample-size formula** (Connor 1987, *Biometrics* 43:207–211; Machin, Campbell, Fayers & Pinol 1997). With discordant-pair probabilities `p10`, `p01`, discordant rate `p_d = p10 + p01`, and effect `δ = p10 − p01`:

```
N = [ z_{1−α/2} · √(p_d) + z_{1−β} · √(p_d − δ²) ]² / δ²
```

A convenient planning approximation (used for the table below):

```
N ≈ (z_{1−α/2} + z_{1−β})² · p_d / δ²
```

At α = 0.05 two-sided, power 1−β = 0.80: `z_{1−α/2} = 1.96`, `z_{1−β} = 0.84`, so `(1.96 + 0.84)² = 7.85`.

Only discordant pairs carry information, so the *effective* sample is `N · p_d`. A low discordant rate inflates `N`. This is why we pilot `p_d` before committing (Document 11 Stage 2) rather than assuming it.

**Planning table (paired items per cell):**

| MDE on δ (paired pp) | p_d ≈ 0.10 | p_d ≈ 0.15 | p_d ≈ 0.20 | p_d ≈ 0.30 |
|---|---|---|---|---|
| 5 pp | ~314 | ~471 | ~628 | ~942 |
| 4 pp | ~491 | ~736 | ~981 | ~1472 |
| 3 pp | ~872 | ~1308 | ~1744 | ~2616 |
| 2 pp | ~1962 | ~2943 | ~3924 | ~5886 |

**The locked decision and its justification.** We set the **primary-endpoint MDE at 5 pp** and provisionally **`N = 600` paired items per cell**, which covers `p_d` up to ≈ 0.19 at 5 pp / 80% power. Rationale: a 5 pp paired degradation is the smallest effect that is *practically* meaningful for a robustness claim (a model that loses 5 of every 100 previously-correct answers to a single typo is meaningfully brittle); resolving finer than 5 pp would multiply compute 2–4× (see the 3 pp row) for a difference of limited practical import, especially on free/cheap hardware. For the *descriptive* Module 4, where a 6–7 pp MDE is acceptable, we use `N = 400` (Document 03 §3.6).

**The pilot gate.** In Stage 2 (Document 11 §11.2) we measure the empirical `p_d` on Llama-3.2-1B at `k=1`. Then:

- If `p_d ≤ 0.19`: `N = 600` is confirmed; proceed.
- If `0.19 < p_d ≤ 0.30`: raise `N` to ≈ 942 for the primary cells, or accept a slightly larger MDE (~6 pp) at `N = 600`, decision recorded in Document 00 §0.5.
- If `p_d < 0.05`: the effect is so small that McNemar is underpowered at any feasible `N`; we then make the *primary* condition a higher budget (`k=3` or `k=4`) where `p_d` is larger, rather than chase an undetectable single-typo effect. This is a pre-registered contingency, not a post-hoc move.

**Gate outcome (2026-07-23; full record in Document 00 §0.5).** The pilot and rehearsal v2 measured implied N of 338–600 across the four families (mmlu at exactly 600; the pilot's GSM8K readout implied 720). `N` is set to **720 items per dataset, uniform**: this covers the worst measured family with ≥20% margin against the sampling error of the implied-N estimates and satisfies the standing GSM8K contingency. The uniform value applies to every condition — the provisional Module-4 `N = 400` economy was not carried into the implemented grid (`configs/main.yaml` sets a single `item_count: 720` per dataset).

**Power literature.** Card et al. (EMNLP 2020, "With Little Power Comes Great Responsibility," arXiv:2010.06595) document that most NLP comparisons are underpowered and suffer Type-M (magnitude exaggeration) and Type-S (sign-error) inflation; their power-analysis tooling (github.com/dallascard/NLP-power-analysis) is the reference. Dror et al. (ACL 2018, "The Hitchhiker's Guide to Testing Statistical Significance in NLP," P18-1128) give the test-selection decision tree that lands us on McNemar for paired binary data. Citing both pre-empts the "underpowered NLP study" critique.

## 6.4 Primary test: McNemar, mid-p exact

Per cell, the McNemar statistic on discordant counts `b = #(clean✓ perturbed✗)`, `c = #(clean✗ perturbed✓)`:

```
χ² = (b − c)² / (b + c)     (asymptotic, 1 df)
```

We use the **mid-p exact** version (Fagerland, Lydersen & Laake 2013) rather than the asymptotic χ² whenever `b + c < 25`: the asymptotic test is unreliable with few discordants, and the mid-p variant is less conservative than the pure exact binomial while controlling Type I error. Implementation: `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)` for the exact binomial, with the mid-p correction applied, or the equivalent in R (`exact2x2::mcnemar.exact` / mid-p). The exact-vs-asymptotic choice is determined by the `b+c` count rule, fixed in advance, not chosen after seeing the p-value.

We report, per cell: `b`, `c`, the McNemar p-value, and the effect `Δ = (b − c)/N` with its CI (§6.5). Reporting the raw discordant counts lets the reader recompute the test.

## 6.5 Effect-size CIs: BCa bootstrap, item-paired

For every reported effect size (`Δ`, `R`, `CCF`, `ACR`, `ORR`), we report a **bias-corrected and accelerated (BCa) bootstrap 95% CI** with **B = 10,000 resamples**, resampling **items with replacement while keeping each item's clean and perturbed outcomes together** (cluster/paired bootstrap by item).

- **Why paired bootstrap:** resampling rows independently would break the matched-pair structure and underestimate the CI; resampling *items* preserves the within-item correlation that the matched design exploits.
- **Why BCa over percentile:** BCa corrects for bias (`ẑ₀`) and skew (acceleration `â`) in the bootstrap distribution, which matters for bounded proportions near 0 or 1 (CCF can sit near 0; ORR can sit near 1).
- **Why B = 10,000:** the percentile method is stable at 1,000–5,000 (Hesterberg et al. 2003), but BCa's bias/acceleration estimates have higher variance and benefit from more resamples. Bestgen (2022, arXiv:2205.11134) uses BCa with B = 10,000 in its case studies while noting the smaller counts suffice for percentile intervals; we register **B = 10,000** as our fixed value rather than claiming an external field standard. The BCa method itself is Efron (1987, JASA 82:171–185).
- **Implementation:** `scipy.stats.bootstrap(data, statistic, method='BCa', n_resamples=9999, paired=True, random_state=<logged seed>)` or R `boot::boot.ci(type='bca')`.
- **Degenerate-case guard:** SciPy warns BCa returns NaN when the bootstrap distribution is degenerate (e.g., all-agreement cells). For cells with `b + c` very small, we fall back to the exact McNemar p-value plus a percentile (or Clopper–Pearson) interval and flag the cell as low-information. This contingency is fixed in advance.

## 6.6 The full-design model: mixed-effects logistic regression

Per-cell McNemar answers "did this condition hurt this model on this task." To estimate *which factors explain degradation across the whole design* (RQ4) and to test the interactions (RQ1 mediation, RQ2 quantization), we fit one mixed-effects logistic model with crossed random effects. The maximal-random-effects specification (Barr, Levy, Scheepers & Tily 2013) for the main model:

```
correct_{i,m,c} ~ 1
    + perturbed                      # clean vs perturbed (the within-item contrast)
    + edit_budget_k                  # severity
    + operation                      # I/D/S/T
    + location                       # instruction/content/answer-critical
    + selection_policy               # kbd/uniform/infoword
    + token_inflation_tau            # the mediation covariate
    + precision                      # fp16 vs AWQ (quantization arm)
    + perturbed:token_inflation_tau  # RQ1 interaction (mediation)
    + perturbed:precision            # RQ2 interaction (quantization)
    + (1 + perturbed | item)
    + (1 + perturbed | model)
```

- `correct` is the binary outcome. The random intercepts and **random slopes for `perturbed`** by both `item` and `model` prevent anticonservative p-values and pseudoreplication (Barr et al. 2013; Baayen, Davidson & Bates 2008; Jaeger 2008).
- **Estimator:** `lme4::glmer` (binomial, logit link, Laplace approximation, bobyqa optimizer with a raised evaluation budget) reached from Python through the rpy2 bridge (`requirements-stats.txt`). The pymer4 wrapper was considered and rejected: its conversion layer has historically lagged pandas major versions, and this repo pins pandas 3.x.
- **Convergence contingencies (fixed in advance, and implemented):** if the maximal model is singular or fails to converge (common with only five models), the ladder is: (1) drop random-slope correlations (`||` syntax), then (2) drop the by-model random slope, then (3) intercepts-only random effects, then (4) treat `item` and `model` as *fixed* categorical effects in a logistic GLM (five levels is borderline for a random effect anyway; Gelman & Hill's rule of thumb is ≥5–8 levels). A rung is accepted only when it converges with a non-singular random-effects estimate; every rejected rung's reason is recorded in the output (`ladder_notes`). Rung (4) doubles as the offline fallback when no R installation is present, loudly labeled via `ConvergenceMethod` (`src/analysis/models.py`). With a single model (the pilot case) the by-model terms are structurally absent, since one grouping level cannot support a variance estimate, and the ladder deduplicates accordingly.
- **Reporting:** fixed-effect odds ratios with Wald z standard errors and p-values from the accepted rung; profile-likelihood CIs are computed once for the camera-ready confirmatory fit, where their cost is paid a single time. The two interaction terms (the mediator-slope coding of `perturbed:token_inflation_tau`, see §6.8, and `perturbed:precision`) are the formal tests of H1 and H2 at the population level, complementing the per-cell and counterfactual analyses. The linear-probability mixed model formerly fitted here survives only as a clearly-labeled risk-difference robustness appendix (`fit_linear_probability_mixed_model`); its coefficients are never exponentiated.

## 6.7 Multiplicity: FDR across cells, FWER within a model

We run many cells, so we control false discoveries explicitly:

- **Within a single model's pre-specified family** of ≤10 comparisons (e.g., the regimes × budgets for that model): **Holm–Bonferroni** for strong FWER control.
- **Across the full grid** of (model × regime × budget × operation) cells: **Benjamini–Hochberg FDR at q = 0.05** (Dror, Baumer, Shlomov & Reichart 2017, "Replicability Analysis for NLP," TACL). FDR is the right tool when the family is large and we accept a controlled proportion of false positives among discoveries.
- The **primary endpoint** (Regime-A degradation) and the **two pre-registered interactions** (H1, H2) are designated *primary* comparisons and are *not* diluted by the exploratory family. They are tested at α = 0.05 with their own pre-registered correction; everything else is exploratory/FDR-controlled. Designating primaries in advance keeps the headline claims from being penalized for the breadth of the descriptive sweep.

## 6.8 The mediation analysis (RQ1, the primary contribution)

This is the most scrutinized analysis, so it is specified in full and uses two complementary, mutually-reinforcing methods.

**Method A: fragmentation-matched counterfactual (the causal-within-scope estimate).**
For each target word `w` and budget `k`, we have Low and High fragmentation realizations (Document 02 §2.5) that hold `w`, `k`, position, and Regime-A status fixed. We compute, per (model, task):

```
ΔCCF_frag = CCF(High fragmentation) − CCF(Low fragmentation)
```

paired by word, with a paired bootstrap CI and a McNemar-style test on the matched word set. A positive, significant `ΔCCF_frag` is direct evidence that, *holding meaning/edit-count/position fixed, more subword fragmentation causes more failure*. This is the cleanest, least-attackable form of the mediation claim because it is a controlled contrast, not a regression on observational covariates.

**Method B: population mediation (the magnitude estimate; the primary mediation quantity).**
On Regime-A rows plus their matched clean rows, fit a mixed **linear** model for the mediator (`token_inflation_excess = τ − 1`, definitionally 0 on clean rows, so its slope IS the §6.6 H1 interaction) and a mixed **logistic** model for the outcome, each with a by-item random intercept, and compute the decomposition with the **quasi-Bayesian Monte Carlo algorithm of Imai, Keele & Tingley (2010)**: 1,000 draws from the fitted coefficients' sampling distribution, effects evaluated on the probability scale conditional on the median item, so `total = direct + indirect` holds exactly in percentage points. Imai et al. warn (p. 316) that the linear product-of-coefficients shortcut does not generalize to binary outcomes, so it is used only as the labeled offline fallback (an item-demeaned "within" linear estimator with a by-item cluster bootstrap at B = 1,000, Hesterberg's sufficiency range; the B = 10,000 registered for the cheap per-cell BCa intervals is not the relevant convention for refit-based intervals). **The by-item structure is mandatory, not stylistic: on the pilot data a pooled outcome model flips the mediator coefficient's sign through between-item confounding.** The `estimator` field of every serialized result names the path that produced it. The **indirect effect with its interval** is the primary reported quantity; the **proportion mediated** is reported only when the total-effect interval excludes zero (the ratio is unstable near a zero denominator), with a machine-readable reason otherwise.

**H1b: inflation vs. fragmentation (pre-registered contrast).** Filler-word insertion (design/03) inflates the token count *without corrupting any word*, while keyboard typos inflate by *fragmenting* words. Method B is therefore fitted separately on the keyboard-neighbor and filler-word Regime-A conditions: if the mechanism is fragmentation-specific rather than length-specific, the mediated share should be ≈ 0 for fillers and substantial for keyboard typos at matched token inflation. This contrast doubles as the keyboard arm's contribution to the cross-modal comparison with the voice arm's disfluency operators.

**Why both methods:** Method A is causally clean but local (word-by-word), and its yield is structurally bounded (~25% of items admit a Low/High pair under a 128k vocabulary), so it serves as the **design-based corroboration** of Method B, reported for sign and rough magnitude agreement, never as an independently powered primary test; its highest-yield setting is small-vocab Mistral (32k), the designated cross-model anchor. Method B is population-level but leans on modeling assumptions. Reporting both pre-empts "your mediation is just a regression artifact" (Method A is not a regression) and "your counterfactual is too narrow to generalize" (Method B is population-level). The cross-model prediction, that small-vocab Mistral shows the largest fragmentation effect (Document 05 §5.8), is a third, independent line of support.

## 6.9 The quantization analysis (RQ2, secondary contribution)

The formal test is the `perturbed:precision` interaction in the §6.6 model, conditioned on clean accuracy via both `CCF` (which only looks at clean-correct items) and `R = A₁/A₀` (which normalizes by baseline). We additionally report a matched-item analysis: among items both the fp16 and AWQ versions solved clean, compare their `p10` rates with McNemar. Two-sided per H2. The sub-study (Document 05 §5.5) adds GPTQ so the interaction is not AWQ-specific. We report the direction and magnitude with CIs and bound the claim to 4-bit AWQ/GPTQ.

## 6.10 Reporting standards (locked)

Every reported metric carries: point estimate, BCa 95% CI, the raw discordant counts `(b, c)` where applicable, the test p-value, and the multiplicity adjustment applied. Every figure shows CIs. We report effect sizes prominently and p-values secondarily (per Dror et al. and Card et al., effect size + uncertainty matters more than a bare significance star). Negative and null results are reported with the same prominence as positive ones: a null quantization interaction is a perfectly publishable, honest result and reporting it that way is part of why the study cannot be accused of fishing.

## 6.11 Summary: every number's justification

| Number | Value | Justified by |
|---|---|---|
| Per-cell N (primary) | 720 (gate outcome; provisional 600) | 5 pp MDE, 80% power, α=0.05; ≥20% margin over worst measured implied N (§6.3, 00 §0.5) |
| Per-cell N (descriptive) | 720 (uniform; provisional 400 superseded) | single `item_count` in the implemented grid (§6.3 gate outcome) |
| Bootstrap resamples | 10,000 | BCa stability + NLP standard (§6.5) |
| MDE | 5 pp | smallest practically-meaningful brittleness (§6.3) |
| Power | 0.80 | field convention; explicitly stated |
| α | 0.05 two-sided | field convention; primaries protected, rest FDR (§6.7) |
| Audit N | 385/regime | ±5 pp Wald margin on intent-preserved rate (Document 09) |
| Edit budgets | {1,2,4,8} | response-curve shape estimation (Document 03 §3.4) |
