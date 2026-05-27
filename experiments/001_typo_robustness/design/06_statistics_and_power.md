# 06 — Statistics and Power

This is the document that makes the numbers non-arbitrary. Every sample size traces to a target detectable effect; every test matches the data structure; every claim is bounded by a confidence interval. If a reviewer attacks the study, this is the document that answers them. Read with Document 02 (metric definitions) at hand.

---

## 6.1 The shape of the data

For each item `i`, model `m`, and condition `c`, we have a matched pair of binary outcomes `(Y_{i,m,0}, Y_{i,m,c})` (clean, perturbed) under deterministic greedy decoding (Document 05 §5.6). Because the pair is on the *same item*, the correct analysis is **paired/matched**, not a comparison of two independent accuracy numbers. This single fact dictates every test below: McNemar (paired binary test), BCa bootstrap clustered by item (paired CI), and a mixed-effects logistic model with item as a random effect (paired regression). Using unpaired tests here would be a methodological error and a reviewer would catch it.

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

## 6.3 Sample size, derived from the claim (the core argument)

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

**The key insight that makes this honest:** only discordant pairs carry information, so the *effective* sample is `N · p_d`. A low discordant rate inflates `N`. This is why we pilot `p_d` before committing (Document 11 Stage 2) rather than assuming it.

**Planning table (paired items per cell):**

| MDE on δ (paired pp) | p_d ≈ 0.10 | p_d ≈ 0.15 | p_d ≈ 0.20 | p_d ≈ 0.30 |
|---|---|---|---|---|
| 5 pp | ~314 | ~471 | ~628 | ~942 |
| 4 pp | ~491 | ~736 | ~981 | ~1472 |
| 3 pp | ~872 | ~1308 | ~1744 | ~2616 |
| 2 pp | ~1962 | ~2943 | ~3924 | ~5886 |

**The locked decision and its justification.** We set the **primary-endpoint MDE at 5 pp** and provisionally **`N = 600` paired items per cell**, which covers `p_d` up to ≈ 0.19 at 5 pp / 80% power. Rationale: a 5 pp paired degradation is the smallest effect that is *practically* meaningful for a robustness claim (a model that loses 5 of every 100 previously-correct answers to a single typo is meaningfully brittle); resolving finer than 5 pp would multiply compute 2–4× (see the 3 pp row) for a difference of limited practical import, especially on free/cheap hardware. For the *descriptive* Module 4, where a 6–7 pp MDE is acceptable, we use `N = 400` (Document 03 §3.6).

**The pilot gate (this is what makes 600 non-arbitrary).** In Stage 2 (Document 11 §11.2) we measure the empirical `p_d` on Llama-3.2-1B at `k=1`. Then:
- If `p_d ≤ 0.19`: `N = 600` is confirmed; proceed.
- If `0.19 < p_d ≤ 0.30`: raise `N` to ≈ 942 for the primary cells, or accept a slightly larger MDE (~6 pp) at `N = 600`, decision recorded in Document 00 §0.5.
- If `p_d < 0.05`: the effect is so small that McNemar is underpowered at any feasible `N`; we then make the *primary* condition a higher budget (`k=3` or `k=4`) where `p_d` is larger, rather than chase an undetectable single-typo effect. This is a pre-registered contingency, not a post-hoc move.

This pilot-gated derivation is the answer to "why 600 and not 1000 or 100": 600 is the smallest N that detects the smallest practically-meaningful effect at the discordance rate we will measure, and the rule for revising it is fixed in advance.

**Power literature.** Card et al. (EMNLP 2020, "With Little Power Comes Great Responsibility," arXiv:2010.06595) document that most NLP comparisons are underpowered and suffer Type-M (magnitude exaggeration) and Type-S (sign-error) inflation; their power-analysis tooling (github.com/dallascard/NLP-power-analysis) is the reference. Dror et al. (ACL 2018, "The Hitchhiker's Guide to Testing Statistical Significance in NLP," P18-1128) give the test-selection decision tree that lands us on McNemar for paired binary data. Citing both pre-empts the "underpowered NLP study" critique.

## 6.4 Primary test — McNemar, mid-p exact

Per cell, the McNemar statistic on discordant counts `b = #(clean✓ perturbed✗)`, `c = #(clean✗ perturbed✓)`:

```
χ² = (b − c)² / (b + c)     (asymptotic, 1 df)
```

We use the **mid-p exact** version (Fagerland, Lydersen & Laake 2013) rather than the asymptotic χ² whenever `b + c < 25`, because the asymptotic test is unreliable with few discordants and the mid-p variant is less conservative than the pure exact binomial while controlling Type I error. Implementation: `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)` for the exact binomial, with the mid-p correction applied, or the equivalent in R (`exact2x2::mcnemar.exact` / mid-p). The choice (exact-vs-asymptotic) is determined by the `b+c` count rule, fixed in advance, not chosen after seeing the p-value.

We report, per cell: `b`, `c`, the McNemar p-value, and the effect `Δ = (b − c)/N` with its CI (§6.5). Reporting the raw discordant counts is itself a defensibility move — the reader can recompute the test.

## 6.5 Effect-size CIs — BCa bootstrap, item-paired

For every reported effect size (`Δ`, `R`, `CCF`, `ACR`, `ORR`), we report a **bias-corrected and accelerated (BCa) bootstrap 95% CI** with **B = 10,000 resamples**, resampling **items with replacement while keeping each item's clean and perturbed outcomes together** (cluster/paired bootstrap by item). 

- **Why paired bootstrap:** resampling rows independently would break the matched-pair structure and underestimate the CI; resampling *items* preserves the within-item correlation that the matched design exploits.
- **Why BCa over percentile:** BCa corrects for bias (`ẑ₀`) and skew (acceleration `â`) in the bootstrap distribution, which matters for bounded proportions near 0 or 1 (CCF can sit near 0; ORR can sit near 1).
- **Why B = 10,000:** the percentile method is stable at 1,000–5,000 (Hesterberg et al. 2003), but BCa's bias/acceleration estimates have higher variance and need more resamples; 9,999/10,000 is the de-facto NLP standard (Bestgen 2022, arXiv:2205.11134, "Please, Don't Forget the Difference and the Confidence Interval when Seeking for the State-of-the-Art Status"). Using the field-standard count is itself a small defensibility point.
- **Implementation:** `scipy.stats.bootstrap(data, statistic, method='BCa', n_resamples=9999, paired=True, random_state=<logged seed>)` or R `boot::boot.ci(type='bca')`.
- **Degenerate-case guard:** SciPy warns BCa returns NaN when the bootstrap distribution is degenerate (e.g., all-agreement cells). For cells with `b + c` very small, we fall back to the exact McNemar p-value plus a percentile (or Clopper–Pearson) interval and flag the cell as low-information. This contingency is fixed in advance.

## 6.6 The full-design model — mixed-effects logistic regression

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

- `correct` is the binary outcome; the random intercepts and **random slopes for `perturbed`** by both `item` and `model` are what prevent anticonservative p-values and pseudoreplication (Barr et al. 2013; Baayen, Davidson & Bates 2008; Jaeger 2008).
- Fit with `lme4::glmer` (R) or `pymer4`/`statsmodels.MixedLM`-logit (Python), Laplace approximation, or `nAGQ=10` adaptive Gauss-Hermite quadrature for higher accuracy on the binary outcome.
- **Convergence contingencies (fixed in advance):** if the maximal model is singular or fails to converge — common with only five models — we (1) drop random-slope correlations (`||` syntax), then (2) drop the by-model random slope, then (3) if `model` random effects remain unstable, treat `model` as a *fixed* categorical effect (five levels is borderline for a random effect anyway; Gelman & Hill's rule of thumb is ≥5–8 levels). The fallback ladder is pre-registered so the choice is not seen as cherry-picking.
- We report fixed-effect odds ratios with profile-likelihood CIs, and we report the two interaction terms (`perturbed:token_inflation_tau`, `perturbed:precision`) as the formal tests of H1 and H2 at the population level, complementing the per-cell and counterfactual analyses.

## 6.7 Multiplicity — FDR across cells, FWER within a model

We run many cells, so we control false discoveries explicitly:
- **Within a single model's pre-specified family** of ≤10 comparisons (e.g., the regimes × budgets for that model): **Holm–Bonferroni** for strong FWER control.
- **Across the full grid** of (model × regime × budget × operation) cells: **Benjamini–Hochberg FDR at q = 0.05** (Dror, Baumer, Shlomov & Reichart 2017, "Replicability Analysis for NLP," TACL). FDR is the right tool when the family is large and we accept a controlled proportion of false positives among discoveries.
- The **primary endpoint** (Regime-A degradation) and the **two pre-registered interactions** (H1, H2) are designated *primary* comparisons and are *not* diluted by the exploratory family — they are tested at α = 0.05 with their own pre-registered correction, and everything else is exploratory/FDR-controlled. Designating primaries in advance is what keeps the headline claims from being penalized for the breadth of the descriptive sweep.

## 6.8 The mediation analysis (RQ1, the primary contribution)

This is the most important and most scrutinized analysis, so it is specified in full and uses two complementary, mutually-reinforcing methods.

**Method A — fragmentation-matched counterfactual (the causal-within-scope estimate).**
For each target word `w` and budget `k`, we have Low and High fragmentation realizations (Document 02 §2.5) that hold `w`, `k`, position, and Regime-A status fixed. We compute, per (model, task):
```
ΔCCF_frag = CCF(High fragmentation) − CCF(Low fragmentation)
```
paired by word, with a paired bootstrap CI and a McNemar-style test on the matched word set. A positive, significant `ΔCCF_frag` is direct evidence that, *holding meaning/edit-count/position fixed, more subword fragmentation causes more failure*. This is the cleanest, least-attackable form of the mediation claim because it is a controlled contrast, not a regression on observational covariates.

**Method B — population mediation (the magnitude estimate).**
Fit, on perturbed Regime-A items:
```
correct_perturbed ~ token_inflation_tau + correct_clean + (1|item) + (1|model)
```
and decompose the total `perturbed` effect from §6.6 into the part flowing through `token_inflation_tau` (the mediated/indirect effect) and the residual (direct) effect, using a mediation estimator appropriate for binary outcomes (e.g., the counterfactual/`mediation` framework of Imai, Keele & Tingley 2010, or the `medflex`/`CMAverse` implementations). We report the **proportion mediated** with a bootstrap CI. This gives the headline "X% of the typo-induced loss flows through fragmentation" number.

**Why both:** Method A is causally clean but local (word-by-word); Method B is population-level but leans on modeling assumptions. Reporting both, and showing they agree in sign and roughly in magnitude, is far more convincing than either alone, and it pre-empts "your mediation is just a regression artifact" (Method A is not a regression) and "your counterfactual is too narrow to generalize" (Method B is population-level). The cross-model prediction — that small-vocab Mistral shows the largest fragmentation effect (Document 05 §5.8) — is a third, independent line of support.

## 6.9 The quantization analysis (RQ2, secondary contribution)

The formal test is the `perturbed:precision` interaction in the §6.6 model, conditioned on clean accuracy via both `CCF` (which only looks at clean-correct items) and `R = A₁/A₀` (which normalizes by baseline). We additionally report a matched-item analysis: among items both the fp16 and AWQ versions solved clean, compare their `p10` rates with McNemar. Two-sided per H2. The sub-study (Document 05 §5.5) adds GPTQ so the interaction is not AWQ-specific. We report the direction and magnitude with CIs and bound the claim to 4-bit AWQ/GPTQ.

## 6.10 Reporting standards (locked)

Every reported metric carries: point estimate, BCa 95% CI, the raw discordant counts `(b, c)` where applicable, the test p-value, and the multiplicity adjustment applied. Every figure shows CIs. We report effect sizes prominently and p-values secondarily (per Dror et al. and Card et al., effect size + uncertainty matters more than a bare significance star). Negative and null results are reported with the same prominence as positive ones — a null quantization interaction is a perfectly publishable, honest result and reporting it that way is part of why the study cannot be accused of fishing.

## 6.11 Summary: every number's justification

| Number | Value | Justified by |
|---|---|---|
| Per-cell N (primary) | 600 (pilot-gated) | 5 pp MDE, 80% power, α=0.05, p_d≤0.19 (§6.3) |
| Per-cell N (descriptive) | 400 | 6–7 pp MDE acceptable for RQ4 (§6.3) |
| Bootstrap resamples | 10,000 | BCa stability + NLP standard (§6.5) |
| MDE | 5 pp | smallest practically-meaningful brittleness (§6.3) |
| Power | 0.80 | field convention; explicitly stated |
| α | 0.05 two-sided | field convention; primaries protected, rest FDR (§6.7) |
| Audit N | 385/regime | ±5 pp Wald margin on intent-preserved rate (Document 09) |
| Edit budgets | {1,2,4,8} | response-curve shape estimation (Document 03 §3.4) |
