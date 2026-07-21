# 07: Compute and Engineering

This document records the compute decisions and the arithmetic behind them, so the study runs efficiently on cheap hardware and the generation budget is provably feasible. Headline decisions: replace raw `transformers.generate` with **vLLM** (continuous batching + prefix caching, §7.2), default to the hardware tier decided in §7.4, and keep a fully-specified **free-T4 fallback** (§7.6) that still supports the primary claim. All decisions are summarized in §7.11.

---

## 7.1 The efficiency problem stated as arithmetic

The full design is ~210k generations (Document 03 §3.6). The naive Experiment-000 loop generates one prompt at a time with `transformers.generate`: fine for 315 generations, hopeless for 210k. The governing equation for wall-clock time is:

```
time ≈ (number_of_generations × mean_output_tokens) / effective_throughput_tok_per_s / 3600   [hours]
```

The three levers:

- reduce generations (Document 03 §3.7 graceful degradation)
- reduce mean output tokens (cap `max_new_tokens` tightly, Document 04 §4.9)
- **raise effective throughput**, the main subject of this document

Throughput offers a 10–50× win essentially for free, so it is the first lever to pull.

## 7.2 Decision: vLLM with continuous batching + prefix caching

We replace per-prompt `transformers.generate` with **vLLM** as the inference engine. Three reasons, each tied to this study's structure:

1. **Continuous batching.** vLLM schedules many sequences through the GPU concurrently with paged-attention KV-cache management, instead of one-at-a-time. vLLM's own v0.6 benchmark reports ~2.7× throughput and ~5× lower latency over its prior version on Llama-3-8B; the practical effect for an offline batch of thousands of prompts is an order-of-magnitude speedup over a serial HF loop. For an *offline batch* workload like ours (we have all prompts up front), this is the single biggest win.
2. **Prefix caching (`--enable-prefix-caching`).** Unusually valuable *here* because of the matched-pair design: a clean prompt and its perturbed twin share a long common prefix (chat template + fixed few-shot exemplars + the unchanged portion of the query), and many perturbations differ from the clean prompt by only a few characters. vLLM caches the KV for shared prefixes, so the second and subsequent prompts in a shared-prefix group skip recomputing the prefix. Expected 1.5–3× on top of batching for our workload. HuggingFace `transformers` does not do this natively, which is a concrete reason the migration pays off.
3. **Quantization support.** vLLM runs AWQ and GPTQ 4-bit directly, which is exactly the main-sweep recipe (Document 05 §5.4), so the quantization plumbing and the throughput engine are the same tool.

**Engineering note (offline, not server):** we use vLLM in **offline batched mode** (`LLM.generate(list_of_prompts, sampling_params)`), not the OpenAI-compatible server, because we have a fixed prompt set and want maximum throughput, not low-latency serving. We sort prompts by length and group shared-prefix families together before submitting, so the scheduler and prefix cache work optimally.

## 7.3 What fits where (VRAM reality)

| Model | fp16 weights | 4-bit (AWQ) weights | Fits free T4 (16 GB)? |
|---|---|---|---|
| Llama-3.2-1B | ~2.5 GB | n/a (run fp16) | yes, easily; large batches |
| Llama-3.2-3B | ~6.5 GB | n/a (run fp16) | yes; moderate batches |
| Llama-3.1-8B | ~16 GB | ~5 GB | fp16 **no**; AWQ-4bit **yes** |
| Qwen2.5-7B | ~15 GB | ~5 GB | fp16 **no**; AWQ-4bit **yes** |
| Mistral-7B-v0.3 | ~14.5 GB | ~4.5 GB | fp16 **no**; AWQ-4bit **yes** |

The KV cache also consumes VRAM and grows with batch size × context length. vLLM's paged attention uses it efficiently, but on a 16 GB T4 with an AWQ-8B model we set `gpu_memory_utilization ≈ 0.85` and a modest `max_num_seqs` to avoid OOM. The fp16 7–8B arm (needed for the quantization sub-study, Document 05 §5.5) **does not fit a T4** and is the reason the default tier includes a larger GPU.

## 7.4 Decision: default hardware tier

**Default: USC lab GPU cluster (confirmed by Zizhao).**

Zizhao confirmed in the post-design-suite meeting that the lab has access to a more powerful GPU through USC. This makes the full ~250k-generation design (Document 03 §3.6) straightforwardly feasible with no personal compute cost.

**Access logistics.** Zizhao is checking whether external collaborators (you, as a non-USC student) can be granted direct cluster access. Two workflows, depending on the outcome:

- **Direct access (preferred):** you push code to the repo; you submit the job yourself via the cluster's scheduler (SLURM or equivalent). The existing Allegro layout (code locally, GPU remotely) maps cleanly onto this.
- **Proxy workflow:** if direct access is unavailable, you prepare a fully self-contained job script and send it to Zizhao with a one-command run instruction. The idempotent shard design (§7.7) means intermediate results transfer back safely. He runs it; results come back as a results CSV + manifest. You analyze locally. This is slower but requires no design changes.

**Fallback if cluster access is delayed:** Colab Pro (L4 24 GB) at ~$10/month or RunPod A40 at ~$0.20/GPU-hr for short bursts, well under $50 for the primary Module 1 arm. Use the free-T4 fallback (§7.6) if neither is available.

**ASR data generation** (Document 03 §3.5a, 04 §4.7a): TTS synthesis and Whisper transcription are a separate pre-processing step that runs once on CPU/small GPU before the main sweep. Whisper large-v3 runs on a single T4 in minutes per batch; this step does not need the cluster and can be done locally or on free Colab in advance.

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
- **Full study** (~250k generations, keyboard + ASR arms) on the USC cluster at a blended ~400+ tok/s on a server-class GPU: ≈ 250,000 × 200 / 400 / 3600 ≈ **35 GPU-hours**, i.e. a single overnight batch run.
- Because MCQ answers are short (≤256 tokens, often <50 with the single-letter instruction) and prefix caching cuts the effective prefill cost, the *real* numbers will be better than these estimates. We treat the estimates as upper bounds and re-baseline after the pilot.

## 7.6 The free-T4-only fallback (if USC cluster access is delayed)

If the USC cluster is unavailable and no paid alternative is accessible, the study still runs and still supports the primary contribution. The plan (Document 03 §3.7, made concrete here):

- **Models:** Llama-3.2-1B (fp16), Llama-3.2-3B (fp16), Llama-3.1-8B (AWQ-4bit). The fp16 7–8B arm is dropped, so the quantization sub-study shrinks to an "AWQ-8B vs the fp16 small models" framing: weaker, but the primary mediation contribution does not need it.
- **Modules:** Module 1 (mediation) at full `N=600`, both tasks, all three runnable models. Module 3 regimes A and C, `k∈{1,4}`. Module 2 collapses to a single qualitative quantization observation.
- **Budget:** see §7.5 (≈ **13 GPU-hours**, 2–3 free sessions).
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
3. **Prefix-grouped submission order.** Within a shard, submit clean+perturbed prompt families together and sort by length (see §7.2) so the prefix cache and the scheduler are maximally effective.

We deliberately do **not** use tensor/pipeline parallelism (multi-GPU single model): our models fit on one GPU, so model-sharding would add complexity and inter-GPU overhead for no benefit. Data/shard parallelism is the right tool for a "many independent generations" workload.

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
- Pin: `transformers`, `vllm`, `torch`, `whisper`, `edge-tts`, tokenizer versions, and model commit hashes in `pyproject.toml` / a lockfile; the current `pyproject.toml` already lists torch/transformers/accelerate/bitsandbytes and adding `vllm` and `openai-whisper` is a two-line change.

## 7.11 Compute decisions summary

| Question | Decision | Why |
|---|---|---|
| Inference engine | vLLM offline batched | 10×+ over serial HF; prefix caching fits matched-pair design |
| Default hardware | USC lab GPU cluster (confirmed) | full ~250k-gen study in a single overnight run; no personal cost |
| Fallback hardware | Free Colab T4 / Colab Pro | 1B/3B fp16 + 8B AWQ; ~13 GPU-h; supports primary claim if cluster delayed |
| Parallelism | continuous batching + idempotent shards across GPUs | matches "many independent generations" workload |
| Model parallelism | none | models fit on one GPU |
| Checkpointing | idempotent per-row IDs + manifest, flush ≤500 | survives free-tier disconnects; enables shard parallelism |
| Output-token cap | tight per-task, pilot-confirmed | linear lever on wall-clock time |
