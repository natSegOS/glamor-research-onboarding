# HIVE perturbation suite — evaluation data

One file, one schema: **`hive_all_instances.jsonl.gz`** — every scored instance from the
HIVE input-perturbation study. 550,370 rows, 70 MB gzipped (~1.2 GB raw).

Every published number is reproducible from this file alone. No joins required.

---

## What's in the run

| axis | values |
|---|---|
| benchmarks (6) | `gsm8k`, `gsm_symbolic`, `gsm1k`, `mmlu_pro`, `truthfulqa`, `humaneval` |
| conditions (20) | `clean` + 19 perturbation operators (below) |
| models (5) | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-8B, phi-4 (14B) |
| seeds | 5 |
| items | 200 per benchmark per cell (164 for HumanEval) |

Row counts per benchmark: gsm1k 94,920 · gsm8k 94,700 · gsm_symbolic 94,740 ·
mmlu_pro 96,755 · truthfulqa 95,430 · humaneval 73,825.

**GSM-Symbolic and GSM1k are never-trained contamination controls** — an effect that
reproduces there is not memorization.

Qwen3-32B was attempted and is **excluded**: its 4-bit generation path deadlocks
reproducibly (~3.2k rows in, then sits in `RUNNING` while hung), so its cells are
truncated and not comparable. Recovering it needs a non-4-bit 2-GPU rerun.

---

## Schema

One JSON object per line:

| field | meaning |
|---|---|
| `model`, `seed` | which cell this instance came from |
| `benchmark`, `qid` | item identity; `qid` is stable across conditions |
| `condition` | which perturbation was applied (`clean` = unperturbed reference) |
| `question` | **the exact stem shown to the model** (already perturbed) |
| `gold` | gold answer |
| `choices` | option list for MCQ benchmarks, else `null` |
| `score` | 0/1 as graded |
| `clean_score` | the same `(model, seed, benchmark, qid)` under `clean` |
| `flip` | `break` (clean right → perturbed wrong), `fix`, or `same` |
| `meaning_kept` | did this variant pass the meaning guard (see below) |
| `completion` | the model's full generation |

`clean_score` and `flip` are precomputed so paired analysis needs no self-join.
For `clean` rows, `condition == "clean"` and `flip == "same"` by construction.

### Conditions

*Reference / controls*: `clean`, `clean_qfirst` (question fronted before context),
`ctrl_option_perm` (MCQ options permuted, gold remapped by content — `gold`/`choices`
already reflect the remap).

*Voice, LLM-rewritten*: `spoken_casual`, `spoken_formal` (register transfer),
`spoken_recast` (compress, free to reorder), `spoken_reflow` (compress, clause order
kept), `spoken_reflow_llama` (same with a Llama rewriter).

*Voice, deterministic*: `spoken_filler_stripped`, `clean_fillers`, `clean_numwords`,
`clean_nofunc`, `clean_nocase`, `clean_homophone`.

*Keyboard, deterministic*: `kbd_neighbor` (QWERTY-adjacent substitution),
`kbd_random` (arbitrary letter — the adjacency control), `kbd_swap`, `kbd_repeat`,
`kbd_fatfinger`, `kbd_nospace`. Default: one edit on every 5th eligible word; numeric
tokens are never targeted; seeded per `qid`.

---

## The meaning guard — read before citing any number

`meaning_kept` is a **routed** verdict, because the two available guards disagree
sharply and are not interchangeable:

- **LLM judge** — the semantic check. Rejects ~1.5% of `spoken_casual`. Governs the
  LLM-rewritten conditions (`spoken_casual/formal/recast/reflow*`, `spoken_filler_stripped`).
- **Deterministic validator** — a numeric/length check. Correct for the deterministic
  operators, but **~96% false-positive on LLM-rewritten text** (74 of 77 failures were
  judge-equivalent): it reads number-words used as articles or pronouns — "each **one**
  eats five pounds" — as the numeral 1 and reports a spurious "number added".

So: judge where a judge verdict exists, deterministic validator otherwise. AND-ing both
drops ~38% of spoken items and biases every spoken delta (it made GSM8K −5.5 look like
−1.8 and flipped HumanEval's sign). Judge coverage is complete, 180/180
(benchmark × condition × seed).

Headline numbers are computed on `meaning_kept == true`. Note the original pipeline
computed both guards and **applied neither**; that join is done here.

Caveat: for keyboard conditions `meaning_kept` comes from the deterministic validator,
which checks numbers and length but **not entities**. A legitimate adjacent-key typo can
still corrupt meaning — e.g. `month of June` → `month of Mune` passes the guard. Worth
partitioning on if you care about that distinction.

---

## Reproducing the headline results

```python
import gzip, json, collections
rows = [json.loads(l) for l in gzip.open("hive_all_instances.jsonl.gz", "rt")]

# meaning-guarded accuracy delta vs clean, per (benchmark, condition)
acc = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if not r["meaning_kept"]:
        continue
    a = acc[(r["benchmark"], r["condition"])]
    a[0] += r["score"]; a[1] += 1

# paired flip rates for a keyboard operator
kb = [r for r in rows if r["condition"] == "kbd_neighbor"]
brk = sum(r["flip"] == "break" for r in kb) / len(kb)
fix = sum(r["flip"] == "fix"   for r in kb) / len(kb)
print(f"break {brk:.1%}  fix {fix:.1%}  net {fix-brk:+.1%}  churn {brk+fix:.1%}")
```

---

## Findings this data supports

### Voice
- **spoken-casual costs 4.6–5.8 pp on every benchmark**, including both contamination
  controls, and survives the guard essentially untouched (98.9% kept, −5.5 → −5.4). Real
  structural difficulty, not meaning drift.
- **spoken-recast is the worst operator**: −11 to −14 on the GSM family and HumanEval,
  ~0 on MCQ. Also the meaning-riskiest (~71% kept) — about half its raw −24.7 was drift;
  −11.1 survives filtering. Free reordering specifically breaks multi-step reasoning.
- **clean+numwords is the only gain** (+0.5 to +8.1). Writing numbers as words helps.
- Function-word dropping and homophone substitution are ≈0. Benign.

### Keyboard — instability, not degradation
Mean deltas are small (−0.4 to −3.6 pp), which reads as "typos barely matter". Paired
analysis (29,100 items per operator) says otherwise:

| operator | break% | fix% | net | **churn** |
|---|---|---|---|---|
| kbd_random | 10.50 | 6.92 | −3.58 | **17.41** |
| kbd_neighbor | 10.02 | 7.00 | −3.02 | **17.01** |
| kbd_swap | 8.42 | 6.80 | −1.62 | 15.22 |
| kbd_fatfinger | 8.04 | 6.93 | −1.12 | 14.97 |
| kbd_repeat | 7.17 | 6.40 | −0.76 | 13.57 |
| kbd_nospace | 6.32 | 5.88 | −0.44 | 12.20 |

**12–17% of items change correctness while the mean moves under 4 pp.** Typos
destabilize far more than they degrade: ~6–10% break, ~6–7% fix, and they largely
cancel. A mean accuracy delta is nearly blind to this, and the instability is arguably
the more important robustness fact — a user retyping the same question with a different
typo gets a different answer much more often than the aggregate implies.

**Adjacency does not hold.** `kbd_random` (arbitrary letter) is equal to or *worse* than
`kbd_neighbor` (physically adjacent key) on 4 of 6 benchmarks. An earlier single-seed run
showed the opposite; 5 seeds removed it. Whatever damages these models is not keyboard
geometry.

**Churn tracks model, separately from damage:**

| model | churn% | net |
|---|---|---|
| mistral-7b | **18.20** | −1.14 |
| llama31-8b | 15.90 | −2.49 |
| qwen25-7b | 15.17 | −2.27 |
| qwen3-8b | 14.34 | −2.23 |
| phi4-14b | **11.71** | **−0.65** |

phi-4 (14B) is both most stable and least damaged. Mistral-7B is the *least* stable yet
shows only −1.14 net, because its breaks and fixes are near-symmetric. Net damage and
instability are separate axes: a model can look robust on means while being highly
non-deterministic under input noise.

TruthfulQA is the most typo-sensitive benchmark throughout, and its spoken conditions
have the lowest guard-kept rates (60–75%) — adversarial phrasing is fragile under *any*
rewrite, so read its deltas with care.

---

## Open questions

1. **Is churn just sampling noise?** These are single-sample-per-item runs. A
   clean-vs-clean re-run at identical decoding settings would establish the noise floor.
   If clean↔clean churn is also ~12%, most of the typo effect is decode variance and the
   interpretation changes substantially. **Check this first — it gates everything else.**
2. **What is churn made of?** Every broken item's full reasoning trace is in
   `completion`. Do breaks cluster on typo position (first character? a number-adjacent
   word?) or on particular reasoning steps?
3. **Why is random ≥ neighbor?** Plausibly neighbor-substitutions more often yield real
   words (`form`→`dorm`) that read as intentional, while random ones yield obvious
   non-words the model silently repairs. Testable by dictionary-checking the edited token.
4. **Why is Mistral unstable yet undamaged?** High symmetric churn suggests answers
   sitting near a decision boundary rather than genuine comprehension failure.
5. **Does churn predict the voice penalty?** spoken-casual's −5 may be the same mechanism
   at higher intensity, or something categorically different.

---

## Provenance

Code: `zizhao-hu/human-input-variations`, `experiments/002_voice_variations/`.
This file was produced by `export_all_instances.py`; aggregates by `analyze_suite.py`.

Run `verify_matrix.py` before trusting any aggregate rebuilt from raw results: on the
source cluster, preemption cancels rather than requeues, and jobs can sit in `RUNNING`
while hung — both silently produced partial data during this run and both were caught
only by checking row counts, never by job status.
