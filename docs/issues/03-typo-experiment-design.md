# Design the typo experiment

## Why

So far the trajectory-divergence pipeline has been driven by a tiny ad-hoc prompt set with two perturbation types (`none`, `synonym_swap`). To turn this into a real study, we need a principled experimental design around *typos* — what kinds, how many, where, and on what prompts. This issue is the design + scaffolding; it doesn't yet need to run at full scale.

## Experimental dimensions

Each generation is parameterized by the product of:

### 1. Typo type

- **Character-level**
  - `swap` — adjacent character transposition (`teh` ← `the`)
  - `delete` — drop a character (`th` ← `the`)
  - `insert` — duplicate or random insertion (`thee` ← `the`)
  - `substitute` — keyboard-neighbor swap (`tge` ← `the`)
- **Word-level**
  - `homophone` — `there` ↔ `their` ↔ `they're`
  - `dropped_word` — remove a function word
  - `word_repetition` — duplicate a word
- **Casing / punctuation**
  - `mixed_case` — rAndOm CaSe
  - `missing_punct` — strip `.`, `,`, `?`
  - `extra_punct` — `!!!`, `??`

### 2. Typo degree

How many typos per prompt:

- `0` (control)
- `1` (single edit)
- `low` (≈ 5% of tokens)
- `medium` (≈ 15% of tokens)
- `high` (≈ 30% of tokens)

### 3. Typo scope

*Where* in the prompt the typos land:

- `instruction_only` — only the imperative ("Difene overfitting…")
- `content_only` — only the task content (the noun phrase, the example, etc.)
- `anywhere` — uniform over the prompt
- `last_word` — the final salient word only (tests recency sensitivity)

### 4. Prompt type

Drawn from a few canonical categories so we can see whether typo robustness varies by task:

- **Definitional** — "Define overfitting in one paragraph."
- **Q&A** — "What causes overfitting?"
- **Instruction-following** — "List three regularization techniques as bullets."
- **Reasoning** — "If a model has training loss 0.01 and validation loss 0.8, what is happening?"
- **Code completion** — short Python prompt with a function header.
- **Roleplay / persona** — "You are a TA. Explain overfitting to a freshman."

5–10 prompts per type, ~30–50 total.

### 5. Sampling

Temperatures `{0.2, 0.7, 1.2}` × `n_runs_per_cell ≥ 5` (need enough samples to see distributional effects, not just one-shot luck).

## Tasks

- [ ] Add `prompts/` (or `experiments/typo_robustness/prompts/`) with the canonical prompt set as YAML or JSON, one record per prompt with `id`, `category`, `text`, and `target_tokens` (the spans typos may operate on for `last_word` / `content_only` scopes).
- [ ] Implement `perturb(text, typo_type, degree, scope, seed) -> perturbed_text` as a pure function with unit tests. Each `(text, typo_type, degree, scope, seed)` quadruple must produce one deterministic output.
- [ ] Extend the experiment config to express the full grid:

  ```yaml
  prompts: prompts/canonical_v1.yaml
  models: [meta-llama/Llama-3.1-8B-Instruct]   # depends on #2
  quantization: { enabled: true, bits: 4 }
  typo:
    types: [swap, delete, substitute, homophone, missing_punct]
    degrees: [0, 1, low, medium]
    scopes: [instruction_only, content_only, anywhere]
  temperatures: [0.2, 0.7, 1.2]
  n_runs_per_cell: 5
  seed: 0
  ```

- [ ] Estimate the total cell count before running. Sketch the math in the experiment README: `len(prompts) × len(types) × len(degrees) × len(scopes) × len(temps) × n_runs`. If it exceeds ~5,000 generations, propose a fractional design (Latin square, or sweep one axis at a time holding the others at defaults) instead of full factorial.
- [ ] Write the *design doc* (`experiments/typo_robustness/README.md`) before writing the runner. It should state: the hypothesis, the dimensions above, the cell count, what would count as evidence for/against the hypothesis, what metrics decide.
- [ ] Only then add the runner that walks the grid and emits one row per (prompt, perturbation, model, temperature, run) into the CSV.

## Acceptance criteria

- [ ] `experiments/typo_robustness/README.md` documents hypothesis, dimensions, cell count, and stopping criteria.
- [ ] `perturb()` has unit tests covering each typo type and scope.
- [ ] The runner can execute a *small* slice of the grid (e.g. 2 prompts × 2 typo types × 2 degrees) end-to-end on a T4.
- [ ] CSV columns include `typo_type`, `typo_degree`, `typo_scope`, `prompt_category` in addition to the existing trajectory-divergence columns.
- [ ] A first pass of results across one full axis (e.g. degree, holding type/scope fixed) is committed and discussed in the README.

Depends on #1 and #2.
