# Migrate code to `.py` files; use Colab as a GPU resource only

## Why

Right now the work lives inside Colab notebooks (`notebooks/llama_pipeline.ipynb`, `notebooks/trajectory_divergence.ipynb`). Notebooks are great for proving a pipeline runs on a GPU and bad for everything else: diffs are noisy, cells run out of order, code can't be reused, and Claude Code works best on plain `.py` files.

Going forward: **all editing happens locally in Claude Code on `.py` files**. Colab is only a place we rent a GPU.

## Goal

Adopt the [Allegro project layout](../tutorial.html):

```
glamor-research-onboarding/
├── paper/          ← git submodule → paper repo
└── experiments/    ← one folder per experiment
```

Each `experiments/<name>/` is self-contained — code, config, results, and a thin Colab driver notebook all live in that folder.

## Tasks

- [ ] Add `paper/` as a git submodule (placeholder repo is fine for now).
- [ ] Create `experiments/trajectory_divergence/` and port the temperature sweep from `notebooks/llama_pipeline.ipynb` into a `.py` entry point in that folder.
- [ ] Move the visualization logic out of `scripts/build_visualization.py` into the same experiment folder; update its hardcoded paths to be script-relative.
- [ ] Move `results/` and `visualizations/trajectory_divergence.html` under the experiment folder.
- [ ] Add a 5-cell `colab.ipynb` inside the experiment folder that clones the repo, installs deps, runs the `.py` entry point on a T4, and downloads the CSV. No experiment logic in the notebook.
- [ ] Replace the broken `requirements.txt` (it's an accidental `pip freeze` — `torch==2.11.0` isn't a real release) with a real dependency list (`torch`, `transformers`, `pandas`, `huggingface_hub`, `pyyaml`).
- [ ] Delete the old `notebooks/`, `scripts/`, top-level `results/`, and `visualizations/` once everything is moved over.

## Acceptance criteria

- [ ] `git submodule status` shows `paper/`.
- [ ] Running the experiment's `.py` entry point on a Colab T4 regenerates a CSV with the same columns as the current `results/generations.csv`.
- [ ] Running the visualization `.py` locally regenerates the experiment's `visualization.html`.
- [ ] `colab.ipynb` has ≤5 cells and contains no experiment logic.

## Tutorial

See `docs/tutorial.html` for the prompts to walk through this with Claude Code.
