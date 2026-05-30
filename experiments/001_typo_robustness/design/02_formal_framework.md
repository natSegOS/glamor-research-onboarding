# 02 — Formal Framework

This document gives the mathematics the rest of the suite refers to. Every metric used anywhere in the paper is defined here once, unambiguously, so that scoring code and statistics code have a single source of truth. Notation follows Document 00 §0.7.

---

## 2.1 The objects

A **prompt** is a finite string over an alphabet `Σ`:

```
x ∈ Σ*
```

A **typo transformation** is a (possibly stochastic, but seeded-deterministic) map

```
τ : Σ* → Σ*,    x' = τ(x)
```

The **model-plus-decoding** procedure is a deterministic function under greedy decoding (Document 05 §5.6):

```
f : Σ* → Y
```

where `Y` is the space of generated strings. A **scorer** maps a generation to a binary correctness label against a gold answer `y*`:

```
g : Y → {0, 1},    g(f(x)) = 1  iff the answer is correct
```

The atomic unit of analysis is the **item**: a (task instance, gold answer) pair, indexed by `i`. For each item, each model `m`, and each perturbation condition `c`, we obtain a single binary outcome under greedy decoding:

```
Y_{i,m,c} = g(f_m(τ_c(x_i))) ∈ {0, 1}
```

and the clean baseline `Y_{i,m,0} = g(f_m(x_i))`. The matched pair is `(Y_{i,m,0}, Y_{i,m,c})` — the same item scored clean and perturbed by the same model.

## 2.2 The primitive edit basis

The edit basis is the Damerau–Levenshtein operation set:

```
O = { I, D, S, T }
```

- `I` insertion: `cat → caat`
- `D` deletion: `cat → ct`
- `S` substitution: `cat → cot`
- `T` adjacent transposition: `cat → cta`

This basis is principled, not chosen for convenience. Damerau (1964, *CACM* 7(3):171–176) established that more than 80% of human single-word spelling errors are exactly one of these four operations, and the Damerau–Levenshtein distance is the standard minimum-cost edit metric incorporating adjacent transposition. Higher-level "kinds" of typo are **not** separate primitives; they are constrained subspaces:

| Human-readable kind | Formal interpretation |
|---|---|
| casing error | substitution within the upper/lower case map |
| punctuation insertion | insertion where the inserted char ∈ punctuation set |
| whitespace split/merge | insertion/deletion of a space char |
| keyboard-neighbor typo | substitution constrained by a QWERTY adjacency graph |
| real-word typo | a sequence of edits whose result is a valid dictionary word |

Defining the basis this way means the edit budget `k` is an exact, auditable count of primitive operations, not an approximate percentage. This is the foundation of every "the numbers are not arbitrary" argument in Document 06.

## 2.3 The perturbation state vector

Every generated perturbation is fully described by a seven-component state vector:

```
r = (o, u, ℓ, k, s, c, ρ)
```

| Component | Meaning | Domain |
|---|---|---|
| `o` | primitive operation | {I, D, S, T} |
| `u` | unit | {char, word, span} |
| `ℓ` | location/scope | {instruction, content, answer-critical, anywhere} |
| `k` | edit budget | {0, 1, 2, 4, 8} |
| `s` | selection policy | {uniform, keyboard-neighbor, informative-word-targeted, real-word, whitespace, asr-transcription} |
| `c` | semantic regime | {A intent-preserving, B context-recoverable, C meaning-changing} |
| `ρ` | seed | ℤ |

This is the minimal complete representation: removing any component merges conditions that the experiment must keep distinct (one deletion in "please" is not one deletion in "France"; a keyboard-neighbor substitution is not a uniform-random one; `k=1` is not `k=8`). The state vector is logged on every row of the output (Document 08 §8.4) together with an **edit script** (the exact list of operations with character indices and before/after characters) so that `x'` is fully reconstructible from `x` and `r`. Reconstructibility is what makes the perturbation set auditable rather than a black box.

## 2.4 The three semantic regimes

Let `π : Σ* → I` project a prompt onto its intended task/meaning.

**Regime A — intent-preserving nonword typo.** The edit yields an invalid word, but the intended word is recoverable.
```
π(x') = π(x),   the corrupted token is not a valid dictionary word
Desired behavior:  g(f(x')) = g(f(x)) = 1
```
Example: "capital of *Frnace*" → still Paris. *This is the primary-endpoint regime.*

**Regime B — context-recoverable real-word shift.** The edit yields a different *valid* word, but context still makes the original intent recoverable to a competent human.
```
the corrupted token IS a valid dictionary word
π(x') ≈ π(x)  (recoverable but weakened)
```
Example: "capital of *Finance*" → a human infers France. **This is the dominant error type produced by ASR transcription** — the recognizer outputs a valid word that sounds like the intended word. Because ASR is the study's primary real-world motivation, Regime B is elevated to co-primary status alongside Regime A, and ASR-sourced items are the main population of Regime B. WikiTypos also contains real-word substitutions of this type (the reason it must be separated from Regime A). Desired behavior is less absolute than Regime A, so it is reported as a secondary endpoint.

**Regime C — meaning-changing control.** The edit changes the intended task.
```
π(x') ≠ π(x)
Desired behavior:  g_{new}(f(x')) = 1  against the NEW gold answer y'*,  not y*
```
Example: "capital of *France*" → "capital of *Germany*" → Berlin. *This is the over-invariance control, not a robustness test.* It is what lets us measure whether a model is *appropriately* sensitive rather than blindly invariant.

The audit (Document 09) assigns every perturbed item to exactly one regime by human judgment, and the primary endpoint is computed only on audited Regime-A items.

## 2.5 Tokenization quantities (the mediation machinery)

For a tokenizer `Tok` (model-specific), define for each clean/perturbed item:

**Token-inflation ratio:**
```
τ_tok(x, x') = |Tok(x')| / |Tok(x)|
```
(We write `τ_tok` to avoid clashing with the transformation operator `τ`.)

**Local subword fragmentation change** for the single edited word `w → w'`:
```
Δsub(w, w') = |Tok(w')| − |Tok(w)|
```

**Fragmentation-matched counterfactual set.** For a target word `w` at edit budget `k`, let `E_k(w)` be the set of keyboard-plausible perturbed forms reachable in `k` edits that remain in Regime A (nonword, intent-preserving by audit). Partition `E_k(w)` by `Δsub`:
```
Low(w, k)  = { w' ∈ E_k(w) : Δsub(w, w') ≤ 0 }
High(w, k) = { w' ∈ E_k(w) : Δsub(w, w') ≥ 1 }
```
The mediation contrast for word `w` at budget `k` compares model accuracy on `Low` vs `High` realizations of the *same word at the same edit count* (Document 06 §6.8). Because `w`, `k`, position, and semantic regime are held fixed and only the fragmentation consequence varies, a difference in accuracy is attributable to fragmentation within this scope.

This is the single most important construction in the study; it is what turns a correlational tokenization story into a controlled one.

## 2.6 The metric definitions

All probabilities are estimated as empirical means over items within a cell, paired by item where indicated.

**M1 — Clean accuracy.**
```
A₀ = P(g(f(x)) = 1) = mean_i Y_{i,0}
```
Baseline competence. Robustness relative to a weak clean model is meaningless, so this always accompanies any degradation number.

**M2 — Perturbed accuracy.**
```
A₁(r) = P(g(f(x')) = 1 | r)
```
(For Regimes A and B, scored against `y*`. For Regime C, see M7.)

**M3 — Paired absolute degradation (PRIMARY EFFECT SIZE).**
```
Δ(r) = A₀ − A₁(r),   computed on the SAME items
```
The headline quantity. Reported with a paired BCa 95% CI (Document 06 §6.5).

**M4 — Clean-conditioned failure rate (PRIMARY DIAGNOSTIC).**
```
CCF(r) = P(g(f(x')) = 0 | g(f(x)) = 1, r)
```
Of the items the model solved clean, the fraction the typo broke. This isolates typo-induced failure from baseline ignorance — a model that was already wrong cannot have been "broken by the typo." In matched-pair terms, `CCF = p₁₀ / (p₁₀ + p₁₁)` where `p₁₁` is the clean✓perturbed✓ cell. This is the quantity the McNemar test operates on.

**M5 — Retention.**
```
R(r) = A₁(r) / A₀
```
Fraction of clean competence surviving. Normalizes degradation by baseline so that different models (and quantized vs fp16) are comparable even when their clean accuracies differ.

**M6 — Answer-flip rate.**
```
AFR(r) = P(answer(x') ≠ answer(x) | r)
```
Behavioral instability regardless of correctness. A model can stay correct yet unstable; that still matters for trust.

**M7 — Appropriate-change rate (Regime C only).**
```
ACR(r) = P(answer(x') = y'* | y'* ≠ y*, r)
```
When the meaning genuinely changed, did the model update to the new correct answer?

**M8 — Over-robustness rate (Regime C only).**
```
ORR(r) = P(answer(x') = y* | y'* ≠ y*, r)
```
When the meaning changed, did the model wrongly cling to the *old* answer? High ORR is the signature of blind invariance and is the quantitative core of RQ3/H3.

**M9 — Invalid / clarification / refusal rate.**
```
ICR(r) = P(answer(x') ∈ {unparseable, clarification, refusal} | r)
```
A different failure mode from confidently-wrong: "could you clarify?" is operationally distinct from a wrong number, and conflating them would hide a real behavior.

**Diagnostic quantities (not endpoints):** token-inflation ratio `τ_tok` (M2.5, §2.5), local fragmentation `Δsub`, edit density `δ = k / |E(x, ℓ)|` (edits per eligible editable unit, so one edit in a 5-word prompt ≠ one edit in a 60-word prompt), and Damerau–Levenshtein distance `d_DL(x, x')`.

## 2.7 The selectivity targets

A model with **good selective invariance** satisfies, simultaneously:
```
CCF(Regime A) ≈ 0      (ignores benign noise)
ACR(Regime C) ≈ 1      (responds to real meaning change)
ORR(Regime C) ≈ 0      (does not cling to the old answer)
```
The study's RQ3 answer is essentially the joint distribution of these three numbers across models, scales, and budgets. Plotting CCF(A) against ORR(C) per model is the conceptual centerpiece figure (Document 03 §3.8): the bottom-left corner is the ideal, and we expect larger models to drift rightward (more over-robust) even as they drift downward (less benign-noise failure).

## 2.8 Why this formalization is enough — and bounded

Three deliberate limits keep the formal claims defensible. First, correctness `g` is binary and deterministic (greedy decoding + exact-match scoring), so every metric is a clean Bernoulli mean amenable to McNemar and the binomial; we do not invent soft metrics that invite "your metric is arbitrary" critiques. Second, the mediation claim is scoped to the fragmentation-matched counterfactual of §2.5 and is never stated more broadly than that construction supports. Third, the intent projection `π` is operationalized by *human audit* (Document 09), not asserted by the authors, so "your typo changed the meaning" is answered by data rather than by claim. The formalization is intentionally minimal: every symbol here is used downstream, and nothing here is decorative.
