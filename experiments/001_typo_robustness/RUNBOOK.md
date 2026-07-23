# Runbook

Full detail for running the pipeline: the complete `run_profile.yaml` field
table, per-stage commands, Colab specifics, local GPU and sharding setup,
and the throughput check to run before committing cluster time.

If you're looking for the *scientific* design (perturbation grid, models,
statistics, pre-registration), start at `design/00_README_index.md` instead.
This doc is only about running the code.

## Quickstart (no GPU)

See the README Quickstart for the offline test commands (`pytest`,
`verify_references.py`). To see every runnable model key:

```bash
python3 tools/run_pipeline.py list-models
```

## Everything configurable, in one place

Open `configs/run_profile.yaml`. Every user-facing knob is there, with
inline comments:

| Field | What it controls |
|---|---|
| `experiment_config` | Which of `configs/{pilot,rehearsal,main}.yaml` to run (datasets, perturbation grid, token budgets) |
| `models` | Which roster keys to run this session (`list-models` for the full list) |
| `output_root` | Where generations go (one subdirectory per model). Point this at Google Drive on Colab to survive a runtime restart |
| `analysis_dir` | Where the combined analysis + report go |
| `rebuild_data` | `false` (default) reuses the committed `data/items/` + dictionary; `true` rebuilds from HuggingFace/SCOWL. Confirmatory configs always rebuild regardless: see `run_profile.yaml` for the exact rule |
| `skip_if_complete` | `true` (default): a model whose generations are already complete is never even loaded |

Override the model list without editing the file:

```bash
python3 tools/run_pipeline.py generate --models qwen_7b,mistral_7b
# or
TYPO_MODELS=qwen_7b,mistral_7b python3 tools/run_pipeline.py generate
```

This is also how to parallelize across Google accounts: one model per copy
(see the `models` comment in `run_profile.yaml`).

## Running everything

```bash
python3 tools/run_pipeline.py setup   # install deps (uv; Drive-cached on Colab)
python3 tools/run_pipeline.py all     # data -> generate -> analyze -> report
```

`all` prints a plan first, listing which models will run, which are already
complete, and whether data is being reused, before any GPU time is spent.

Or run one stage at a time:

```bash
python3 tools/run_pipeline.py data       # build/reuse task items + dictionary
python3 tools/run_pipeline.py generate   # run every model in the profile
python3 tools/run_pipeline.py analyze    # cell table, GLMM, mediation, figures
python3 tools/run_pipeline.py report     # the HTML report
```

Every stage resumes instead of recomputing (run manifest plus per-row
deterministic IDs). Killing and re-running any command is safe.

## Colab

Open `colab_driver.ipynb` and run its four cells top to bottom: bootstrap
(clone/auth/install), configure (edit the profile dict), run (`... all`),
download. It calls the exact same `tools/run_pipeline.py` commands as above.

## Local GPU box / cluster (no notebook)

Same commands as "Running everything" above, run from a terminal in a repo
clone with `requirements-gpu.txt` and `requirements-stats.txt` installed.
Set `output_root` in the profile to a local or NFS path instead of a Drive
path.

With multiple GPUs, the simplest speedup is one model per GPU: every
process runs the same pipeline with a different model override, into the
same `output_root` (per-model subdirectories never collide):

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/run_pipeline.py generate --models qwen_7b   &
CUDA_VISIBLE_DEVICES=1 python3 tools/run_pipeline.py generate --models mistral_7b &
CUDA_VISIBLE_DEVICES=2 python3 tools/run_pipeline.py generate --models llama_8b  &
wait
python3 tools/run_pipeline.py analyze
python3 tools/run_pipeline.py report
```

To split one model across multiple GPUs/processes, use
`tools/run_generation.py` directly (`run_pipeline.py generate` does not
shard); pin each worker to its own GPU:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python3 tools/run_generation.py \
      --config configs/main.yaml --model llama_8b_awq \
      --output-directory results/main/llama_8b_awq \
      --shard-index $i --shard-count 4 &
done
wait
```

Workers write disjoint `..._wIofN_*` files under the same output directory;
`analyze`/`report` pick up every shard automatically via their glob.

## Throughput check before committing cluster time

```bash
python3 tools/benchmark_throughput.py --config configs/pilot.yaml --model llama_1b --limit 200
```

## Pre-registration gates

See `README.md` § Pre-registration gates for the full checklist (revision
pinning, Stage-1 gates, human audit, OSF lock). The confirmatory-rebuild
rule for `run_pipeline.py data` is documented in `run_profile.yaml`, next
to `rebuild_data`.
