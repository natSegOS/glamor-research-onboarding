# GLAMOR Research Onboarding

Public repository tracking my onboarding progress for the GLAMOR Lab at USC under Zizhao Hu.

## Layout

```
.
├── paper/          ← git submodule → the paper's GitHub repo (placeholder for now)
└── experiments/    ← one folder per experiment
    └── 000_trajectory_divergence/
```

See [`docs/index.html`](docs/index.html) for the weekly log — sidebar tabs link to each week's deliverable. Week 1 is the trajectory-divergence visualization; Week 2 is the onboarding tutorial for migrating into this layout.

## Experiments

### `000_trajectory_divergence`

End-to-end inference with `meta-llama/Llama-3.2-1B-Instruct` on Google Colab (Tesla T4, 14.56 GB VRAM): authenticate with Hugging Face, load the model, generate at temperatures 0.2 / 0.7 / 1.2, and measure tokens/sec.

Across temperatures, throughput stayed around ~40.5 tokens/sec — temperature changes the sampling distribution, not the forward-pass cost. Output became more varied at higher temperature; lower temperature stayed close to a direct definitional answer.

Colab notebook used during the initial run:
https://colab.research.google.com/drive/1hlmsd2qhBLDaMtXJLa3WE6F-zK85KWmH?usp=sharing
