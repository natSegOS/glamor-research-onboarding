# Design the typo experiment

Turn the ad-hoc prompt perturbations into a real experimental design around typos. There are four axes: **type** (character-level swap / delete / insert / substitute, word-level homophone / dropped word, casing / punctuation), **degree** (0, 1, low ≈5%, medium ≈15%, high ≈30% of tokens), **scope** (instruction-only, content-only, anywhere, last-word — testing recency sensitivity), and **prompt category** (definitional, Q&A, instruction-following, reasoning, code, roleplay). Each generation lives at the cross-product of these axes; before running, estimate the cell count and pick a design (full factorial vs. one-axis-at-a-time) that doesn't blow the GPU budget.

Write the design doc (hypothesis, axes, cell count, what counts as evidence) before the runner. Ask Claude Code to draft the doc and the `perturb(text, type, degree, scope, seed)` function with unit tests, then run a small slice end-to-end before scaling up.
