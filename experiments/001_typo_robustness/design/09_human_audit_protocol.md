# 09 — Human Audit Protocol

The single most likely reason a typo-robustness paper gets rejected is the objection "your perturbations changed the meaning, so you measured task-redefinition, not robustness." This document is the defense: a human-audited, agreement-quantified, pre-registered procedure that assigns every perturbed item to a semantic regime by human judgment, with a fixed exclusion rule. The primary endpoint is computed only on audited intent-preserving items.

---

## 9.1 What the audit decides

For each perturbed item the audit answers two binary questions:
1. **Intent preserved?** Would a competent English reader recover the original intended question from the perturbed text?
2. **Gold answer unchanged?** Should the correct answer be the same as for the clean item?

The pair of answers maps to a regime:
- Yes / Yes → **Regime A or B** (intent-preserving; A if the corrupted token is a nonword, B if a real word — the nonword/real-word split is deterministic from `regimes.py`, the audit only confirms intent).
- No / No → **Regime C** (meaning genuinely changed).
- Disagreement or "ambiguous" → flagged for adjudication, then included or excluded per §9.5.

The audit's job is to validate the *intended* regime from `regimes.py` (Document 08 §8.6), correcting mislabels. This is what operationalizes the intent projection `π` (Document 02 §2.4) as data rather than authorial assertion.

## 9.2 Who audits

- **Three independent annotators**, fluent English readers, blind to model outputs and to each other's labels. Blinding to model outputs is essential: an annotator who has seen the model fail might rationalize the item as "meaning-changed."
- Annotators receive a **written guideline** with worked examples of each regime (including the hard cases: a real-word typo that *does* change meaning → C; a nonword typo on a non-key word → A; a number swap → C).
- A **fourth person adjudicates** disagreements (§9.5).
- For a course/onboarding setting, annotators can be the student plus two labmates; the protocol is identical regardless of who fills the roles, and the agreement statistics (not the identities) are what the paper reports.

## 9.3 How many items to audit (sample size tied to a margin)

The audit estimates, per regime, the rate at which the *intended* regime label is correct (the "intent-preserved rate" for A/B; the "meaning-changed rate" for C). For a binary proportion with a Wald normal-approximation 95% CI and worst-case `p = 0.5`:

```
n = z²·p(1−p) / margin²  =  1.96² · 0.25 / margin²
```

- **±5 pp margin → n ≈ 385 items per regime.** (Locked.)
- ±3 pp margin → n ≈ 1,068 per regime (used only if a tighter bound is needed and annotator time allows).

**Locked decision: audit 385 stratified-random items per regime** (A, B, C) = ~1,155 items, each by 3 annotators = ~3,465 judgments. At a few seconds per binary judgment this is a few hours per annotator — a realistic one-week task. Stratify the sample across tasks, edit budgets, and operations so the audited rate is representative of the full perturbation set, not just the easy cells.

This `n = 385` is derived from the ±5 pp margin exactly as `N = 600` is derived from the 5 pp MDE (Document 06 §6.3) — both numbers are consequences of a stated precision target, not round-number guesses.

## 9.4 Agreement thresholds (the quality gate)

We report **Fleiss' κ** (three raters, nominal regime labels) per regime and overall, and interpret it on the Landis & Koch (1977) scale: 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect.

**Locked gate: Fleiss' κ ≥ 0.60 (substantial) per regime is required to proceed.** If κ < 0.60 for a regime, the perturbation generator for that regime is too ambiguous; we revise the guideline and/or the generator (e.g., tighten the real-word neighbor selection for B) and re-audit a fresh sample before the main sweep. We also report Cohen's κ pairwise as a cross-check. Reporting κ — and gating on it — is what converts "we think these are intent-preserving" into "annotators substantially agree these are intent-preserving, κ = X."

## 9.5 Adjudication and the exclusion rule (locked)

- An item with **3-of-3 or 2-of-3 agreement** on its regime takes the majority label.
- An item with **no majority** (all three disagree, or a 2-1 split that the adjudicator overturns) goes to the **fourth-person adjudicator**, whose label is final.
- An item the adjudicator marks **ambiguous** (intent genuinely unclear) is **excluded from the primary endpoint** and analyzed separately as an "ambiguity" set.
- **Crucially:** any item intended as Regime A but audited as meaning-changed is **removed from the primary endpoint** (and may be reassigned to C for the over-robustness analysis). This is the mechanism that guarantees the primary endpoint contains only genuinely intent-preserving items.

The exclusion rule is pre-registered (Document 10) so it cannot be seen as post-hoc data selection: we commit, before running models, to dropping audit-failed items from the primary endpoint.

## 9.6 Extrapolating the audit to the full set

We audit a 385-item-per-regime sample, not every perturbation. To justify applying the audited intent-preserved rate to the full (unaudited) Regime-A set, we (a) stratify the audit sample to match the full set's distribution over tasks/budgets/operations, and (b) report the audited intent-preserved rate with its ±5 pp CI as the *validity rate* of the generator. If the validity rate is high (say ≥95%) with a tight CI, the full Regime-A set is trustworthy; if it is lower, we either tighten the generator or restrict the primary endpoint to the audited subset (a smaller but fully-clean endpoint), with the choice pre-registered as a contingency. Either way the claim is bounded by audited evidence.

## 9.7 The LLM-as-judge boundary (locked)

LLM-as-judge is **not used for the primary endpoint or for regime assignment.** Documented reliability problems make it unsuitable as a final arbiter: position/verbosity/self-enhancement bias (Shi et al. 2024, "Judging the Judges," arXiv:2406.07791, 15 judges over ~150k evaluations), broad judgment biases (Chen et al., EMNLP 2024, aclanthology 2024.emnlp-main.474), and domain-expert disagreement (Szymanski et al., ACM IUI 2025 / arXiv:2410.20266: SMEs agreed with LLM judges only ~64% in mental health and ~68% in dietetics).

Where an LLM judge *may* assist (clearly labeled, never final):
- **Pre-screening** the audit pool to prioritize likely-ambiguous items for human attention.
- The **appropriate-change-rate diagnostic** on Regime-C items (M7), where the new gold `y'*` is already known by construction (Document 04 §4.7), so the judge is only confirming a deterministic comparison — and even there, humans validate a 200-item subset.

Humans are the final arbiter on every regime label and on the primary endpoint. This boundary is exactly what the reliability literature recommends and is stated plainly in the paper.

## 9.8 What the paper reports from the audit

- Fleiss' κ per regime and overall, with the Landis-Koch interpretation.
- The intent-preserved validity rate of the Regime-A generator, with its ±5 pp CI.
- The number of items excluded from the primary endpoint by the audit, and a breakdown of why.
- The guideline and worked examples (in an appendix), so the audit is reproducible by others.
- Confirmation that the primary endpoint contains only audited intent-preserving items.

Reporting all of this turns the audit from an invisible assumption into a visible, quantified, reproducible part of the method — which is precisely what makes the "your typos changed the meaning" objection answerable with a number instead of a rebuttal.
