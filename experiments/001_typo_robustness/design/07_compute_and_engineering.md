# 07 — Compute and Engineering

This document makes the compute decisions for you and shows the arithmetic behind them, so the study runs efficiently on cheap hardware and the generation budget is provably feasible. The headline decisions: switch from raw `transformers.generate` to **vLLM** with continuous batching and prefix caching, default to a **Colab Pro L4** with **RunPod A40** bursts for fp16 7–8B, and keep a fully-specified **free-T4 fallback** that still supports the primary claim.

---

## 7.1 The efficiency problem stated as arithmetic

The full design is ~210k generations (Document 03 §3.6). The naive Experiment-000 loop generates one prompt at a time with `transformers.generate` — fine for 315 generations, hopeless for 210k. The governing equation for wall-clock time is:

```
time ≈ (number_of_generations × mean_output_tokens) / effective_throughput_tok_per_s / 3600   [hours]
```

The three levers are: reduce generations (Document 03 §3.7 graceful degradation), reduce mean output tokens (cap `max_new_tokens` tightly, Document 04 §4.9), and **raise effective throughput** — which is what this document is mostly about. Throughput is where a 10–50× win is available essentially for free, so it is the first lever to pull.

## 7.2 Decision: vLLM with continuous batching + prefix caching

We replace per-prompt `transformers.generate` with **vLLM** as the inference engine. Three reasons, each tied to this study's structure:

1. **Continuous batching.** vLLM schedules many sequences through the GPU concurrently with paged-attention KV-cache management, instead of one-at-a-time. vLLM's own v0.6 benchmark reports ~2.7× throughput and ~5× lower latency over its prior version on Llama-3-8B; the practical effect for an offline batch of thousands of prompts is an order-of-magnitude speedup over a serial HF loop. For an *offline batch* workload like ours (we have all prompts up front), this is the single biggest win.
2. **Prefix caching (`--enable-prefix-caching`).** This is unusually valuable *here* because of the matched-pair design: a clean prompt and its perturbed twin share a long common prefix (chat template + fixed few-shot exemplars + the unchanged portion of the query), and many perturbations differ from the clean prompt by only a few characters. vLLM caches the KV for shared prefixes, so the second and subsequent prompts in a shared-prefix group skip recomputing the prefix. Expected 1.5–3× on top of batching for our workload. HuggingFace `transformers` does not do this natively — a concrete reason the migration pays off.
3. **Quantization support.** vLLM runs AWQ and GPTQ 4-bit directly, which is exactly the main-sweep recipe (Document 05 §5.4), so the quantization plumbing and the throughput engine are the same tool.

**Engineering note (offline, not server):** we use vLLM in **offline batched mode** (`LLM.generate(list_of_prompts, sampling_params)`), not the OpenAI-compatible server, because we have a fixed prompt set and want maximum throughput, not low-latency serving. We sort prompts by length and group shared-prefix families together before submitting, so the scheduler and prefix cache work optimally.

## 7.3 What fits where (VRAM reality)

| Model | fp16 weights | 4-bit (AWQ) weights | Fits free T4 (16 GB)? |
|---|---|---|---|
| Llama-3.2-1B | ~2.5 GB | — (run fp16) | yes, easily; large batches |
| Llama-3.2-3B | ~6.5 GB | — (run fp16) | yes; moderate batches |
| Llama-3.1-8B | ~16 GB | ~5 GB | fp16 **no**; AWQ-4bit **yes** |
| Qwen2.5-7B | ~15 GB | ~5 GB | fp16 **no**; AWQ-4bit **yes** |
| Mistral-7B-v0.3 | ~14.5 GB | ~4.5 GB | fp16 **no**; AWQ-4bit **yes** |

The KV cache also consumes VRAM and grows with batch size × context length; vLLM's paged attention uses it efficiently, but on a 16 GB T4 with an AWQ-8B model we set `gpu_memory_utilization ≈ 0.85` and a modest `max_num_seqs` to avoid OOM. The fp16 7–8B arm (needed for the quantization sub-study, Document 05 §5.5) **does not fit a T4** and is the reason the default tier includes a larger GPU.

## 7.4 Decision: default hardware tier

**Default: Colab Pro (L4 24 GB) for routine work + RunPod A40 48 GB bursts for fp16 7–8B and the big Module-4 sweep.**

Justification and current (May 2026) pricing:
- **Colab Pro** is ~$10/month with compute units (~$0.10/unit; an L4 burns roughly 2–3 units/hour), giving an L4 24 GB that runs AWQ 7–8B comfortably and fp16 3B easily, with longer session limits than free. This is the everyday driver and matches the repo's existing Colab-as-GPU workflow (the tutorial in `docs/tutorial.html`).
- **RunPod Community Cloud** A40 48 GB at ~$0.20/GPU-hr or A6000 48 GB at ~$0.44/hr handles the fp16 7–8B arm and the largest sweeps; **Vast.ai** RTX 3090 from ~$0.17/hr or RTX 4090 from ~$0.40/hr are cheaper spot alternatives if 24 GB suffices.
- The whole 210k-generation study, at the throughput estimates in §7.5, is on the order of **tens of GPU-hours**, i.e. **well under $50** of burst compute even at the higher per-hour rates. This is the "relatively cheap, temporary upgrade" you said you could do, and it is the recommended path because it unlocks the fp16 arm (and thus the full RQ2 contribution).

You set up the GPU side exactly as the existing tutorial describes (Claude Code writes the driver notebook locally; you upload to Colab/RunPod, pull the repo, `pip install -e .`, run). The only change is that the runner now calls vLLM.

## 7.5 Throughput estimates and the time budget

Throughput on small GPUs is workload-dependent; we use deliberately conservative estimates and confirm them in a 1-hour pilot before committing (Document 11 Stage 2). Conservative offline-batched output throughput:

| GPU | Model (precision) | Est. output tok/s (batched) |
|---|---|---|
| T4 16 GB | 1B fp16 | 300–600 |
| T4 16 GB | 3B fp16 | 120–250 |
| T4 16 GB | 8B AWQ-4bit | 40–90 |
| L4 24 GB | 8B AWQ-4bit | 120–250 |
| A40 48 GB | 8B fp16 | 300–600 |

Time for a given generation count, at mean output 200 tokens (a deliberately high estimate; MCQ is far shorter), using `time = gens × 200 / tok_s / 3600`:

- **Free-T4 fallback study** (~36k generations, Document 03 §3.7), mostly 1B/3B fp16 + 8B AWQ: at a blended ~150 tok/s, ≈ 36,000 × 200 / 150 / 3600 ≈ **13 GPU-hours**, i.e. 2–3 free Colab sessions with checkpointing. Feasible.
- **Default-tier full study** (~210k generations) on L4+A40 at a blended ~250 tok/s: ≈ 210,000 × 200 / 250 / 3600 ≈ **47 GPU-hours**, i.e. a few days of part-time bursts, well under $50.
- Because MCQ answers are short (≤256 tokens, often <50 with the single-letter instruction) and prefix caching cuts the effective prefill cost, the *real* numbers will be better than these estimates. We treat the estimates as upper bounds and re-baseline after the pilot.

## 7.6 The free-T4-only fallback (fully specified)

If no paid tier is available, the study still runs and still supports the primary contribution. The plan (Document 03 §3.7, made concrete here):
- **Models:** Llama-3.2-1B (fp16), Llama-3.2-3B (fp16), Llama-3.1-8B (AWQ-4bit). The fp16 7–8B arm is dropped, so the quantization sub-study shrinks to "AWQ-8B vs the fp16 small models" framing — weaker, but the primary mediation contribution does not need it.
- **Modules:** Module 1 (mediation) at full `N=600`, both tasks, all three runnable models. Module 3 regimes A and C, `k∈{1,4}`. Module 2 collapses to a single qualitative quantization observation.
- **Budget:** ~36k generations ≈ 13 GPU-hours ≈ 2–3 free sessions.
- **Session-limit handling:** free Colab caps sessions near 12 h and can disconnect; §7.7 checkpointing makes this safe.
- **What survives:** the full RQ1 mediation claim (the paper's headline) at full statistical power, plus a bounded RQ3 selectivity claim. What is lost: the strong RQ2 quantization contribution and the descriptive RQ4 breadth. The paper is still publishable as a mechanism paper; it is just narrower. This is the floor (Document 03 §3.7).

## 7.7 Checkpointing and idempotent runs (mandatory from day one)

Long batches on preemptible/free hardware will be interrupted, so the runner is built to resume:
- **Idempotent unit of work:** one (model, task, condition-cell) shard. Each generation row is keyed by a deterministic ID `hash(model_revision, task_id, r_state_vector, seed)`.
- **Append-and-skip:** before generating, the runner checks the output store for that ID and skips if present. A killed session loses at most the in-flight batch.
- **Frequent flush:** write completed rows to disk (and push to the repo / cloud storage) every ≤500 generations.
- **Manifest:** a per-run manifest records which (model, task, cell) shards are complete, so resuming is "run everything not in the manifest."
- This also makes the run **embarrassingly parallel** across hardware: two GPUs can take different shards of the manifest with no coordination beyond the shared output store (§7.8).

## 7.8 Parallelization strategy (decided)

We exploit three independent axes of parallelism, in priority order:

1. **Within-GPU continuous batching (vLLM).** The primary speedup; the engine keeps the GPU saturated with concurrent sequences. No extra engineering beyond using vLLM's batched `generate`.
2. **Shard-level parallelism across GPUs/sessions.** Because shards are idempotent (§7.7), we can run different (model, task) shards on different rented GPUs simultaneously and merge via the shared output store. For the default tier this is how the 7–8B fp16 arm runs on an A40 while the 1B/3B fp16 runs on the L4, in parallel.
- **Prefix-grouped submission order.** Within a shard, we submit clean+perturbed prompt families together and sort by length so the prefix cache and the scheduler are maximally effective. This is a free 1.5–3× on top of batching for our shared-prefix workload.

We deliberately do **not** use tensor/pipeline parallelism (multi-GPU single model) — our models fit on one GPU, so model-sharding would add complexity and inter-GPU overhead for no benefit. Data/shard parallelism is the right tool for a "many independent generations" workload.

## 7.9 Determinism and the throughput/reproducibility tension

vLLM with continuous batching can introduce tiny numerical non-determinism (batch composition affects floating-point reduction order), which under greedy decoding *usually* does not change the argmax token but occasionally can at a near-tie. We manage this so reproducibility is not compromised:
- Greedy decoding (Document 05 §5.6) removes sampling variance, the dominant source.
- We pin vLLM, CUDA, and model versions, and log them per row.
- We set vLLM's `seed` and, for the subset of analyses most sensitive to it (the fragmentation-matched counterfactual, Document 06 §6.8), we verify on a sample that re-running reproduces identical outputs; any rare flips are logged and shown to be negligible relative to the effect sizes.
- For the strictest reproducibility statement in the paper, we note that exact bitwise reproduction may require a fixed batch composition, and we release the run manifests so others can reproduce the *same* batching. This is an honest, bounded reproducibility claim rather than an overstated one.

## 7.10 The engineering migration checklist (repo)

From the Experiment-000 scaffolding to the Experiment-001 runner:
- Keep: the Allegro layout, the `model_id`/`quant_bits` result columns, the Colab-as-GPU workflow, the tidy-CSV philosophy.
- Replace: `model.generate_once` (serial HF) → a vLLM offline-batched generation function.
- Add: the perturbation engine, the deterministic scorers, the token-metric logger, the idempotent shard runner with manifest, and the statistics/audit modules. Full module spec is Document 08.
- Pin: `transformers`, `vllm`, `torch`, tokenizer versions, and model commit hashes in `pyproject.toml` / a lockfile; the current `pyproject.toml` already lists torch/transformers/accelerate/bitsandbytes and adding `vllm` is a one-line change.

## 7.11 Compute decisions summary

| Question | Decision | Why |
|---|---|---|
| Inference engine | vLLM offline batched | 10×+ over serial HF; prefix caching fits matched-pair design |
| Default hardware | Colab Pro L4 + RunPod A40 bursts | runs AWQ everywhere + fp16 7–8B; whole study < $50 |
| Fallback hardware | Free Colab T4 | 1B/3B fp16 + 8B AWQ; ~13 GPU-h; supports primary claim |
| Parallelism | continuous batching + idempotent shards across GPUs | matches "many independent generations" workload |
| Model parallelism | none | models fit on one GPU |
| Checkpointing | idempotent per-row IDs + manifest, flush ≤500 | survives free-tier disconnects; enables shard parallelism |
| Output-token cap | tight per-task, pilot-confirmed | linear lever on wall-clock time |
