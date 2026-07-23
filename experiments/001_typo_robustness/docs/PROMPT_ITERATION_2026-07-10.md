# Prompt iteration for reasoning format compliance (2026-07-10)

Goal: lift Llama-3.2-1B-Instruct reasoning format compliance from the failed
Stage-1 gate value (0.6086 vs the 0.95 target in `analysis/pilot/gates.json`)
to ≥0.95. Method: three smoke-test rounds on a Colab T4. Each round ran all
200 clean reasoning items (100 GSM8K + 100 GSM-Symbolic) per prompt variant
through the repo's own `VllmEngine` (greedy, `max_new_tokens=512`) and the
frozen scorer (`scoring.extract_reasoning_answer`). Compliance is the share
of generations whose answer extracts at the `hash_delimited` tier, the same
definition `src/analysis/gates.py` uses, restricted to clean rows.

The winning scaffold (round 3, `r3`) is implemented in commits `337561e` +
`a884e29`: eight hand-written chat-form exemplar turns
(`REASONING_CHAT_EXEMPLAR_TURNS`) inserted at chat-template time, plus a
post-question reminder line (`REASONING_INSTRUCTION_REMINDER`).

## Round 1: inline exemplars in the instruction text

| variant | pooled compliance | pooled accuracy | gsm8k / gsm_symbolic compliance |
|---|---|---|---|
| v0 current 1-shot (pilot prompt) | 0.650 | 0.285 | 0.64 / 0.66 |
| v1 1-shot + reminder | 0.760 | 0.280 | 0.77 / 0.75 |
| v2 3-shot inline | 0.555 | 0.240 | 0.58 / 0.53 |
| v3 3-shot inline + reminder | 0.770 | 0.230 | 0.78 / 0.76 |

Finding: the post-question reminder alone is worth ~+11pp, while **more
inline exemplars hurt** (v2 < v0). Inline few-shot plateaus near 0.77.

## Round 2: chat-form exemplars (user/assistant turns)

| variant | pooled compliance | pooled accuracy | gsm8k / gsm_symbolic compliance |
|---|---|---|---|
| m1 chat 1-shot | 0.520 | 0.270 | 0.51 / 0.53 |
| m2 chat 3-shot | 0.905 | 0.220 | 0.93 / 0.88 |
| m3 chat 3-shot + reminder | 0.945 | 0.205 | 0.97 / 0.92 |
| m4 single-turn 3-shot + 'Solution:' prefill | 0.795 | 0.185 | 0.88 / 0.71 |

Finding: chat-form exemplars (assistant turns literally ending in
`#### <number>`) are the mechanism that works, but the terse one-line
solutions taught the model to skip its own reasoning (accuracy 0.285→0.205).

## Round 3: chat-form with rich multi-step (CoT) solutions

| variant | pooled compliance | pooled accuracy | gsm8k / gsm_symbolic compliance |
|---|---|---|---|
| r1 chat 3-shot rich + reminder | 0.955 | 0.240 | 0.96 / 0.95 |
| r2 chat 5-shot rich + reminder | 0.960 | 0.220 | 0.98 / 0.94 |
| r3 chat 8-shot rich + reminder (ADOPTED) | **0.990** | 0.250 | 1.00 / 0.98 |
| r4 chat 3-shot rich, no reminder | 0.930 | 0.270 | 0.97 / 0.89 |

Findings: rich solutions recover most of the accuracy loss; compliance scales
with exemplar count; the reminder is still worth ~+2.5pp at 3-shot. `r3`
gives ~4pp headroom over the 0.95 gate before pooling perturbed rows.
Accuracy 0.250 vs the 0.285 1-shot baseline is within sampling noise at n=200
(SE ≈ 0.03).

## Implementation notes (for the confirmatory run and prereg)

- Exemplar turns are a fixed scaffold applied per reasoning request at
  chat-template time (`pipeline.runner.chat_exemplar_turns_for_family`);
  MCQ prompts are untouched. The reminder is appended after the question by
  `ReasoningItem.full_prompt` and sits outside both perturbable scope spans.
- The embedded `instruction` field in `data/items/{gsm8k,gsm_symbolic}.jsonl`
  was refreshed in `aedd153`. The loader prefers the stored field, so stale
  items silently ignore prompt-constant changes.
- Row IDs do not hash prompt text: any prompt change requires
  `run_generation.py --fresh`, otherwise committed rows are kept as-is.
- The exemplar set must be frozen at OSF pre-registration
  (design/10 §pilot-exploratory; design/05 §5.7).
