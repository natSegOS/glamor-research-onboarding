# Migrate code to `.py` files; use Colab as a GPU resource only

The work currently lives inside Colab notebooks, which are great for proving a pipeline runs on a GPU and bad for everything else (noisy diffs, hidden cell-order state, no reuse, no good tooling). The next step is the **Allegro project layout**: two top-level folders — `paper/` (a git submodule pointing at the paper's GitHub repo) and `experiments/` (one self-contained subfolder per experiment, holding its code, config, results, visualization, and a thin Colab driver notebook). All editing happens locally in Claude Code on plain `.py` files; Colab is rented only when a GPU is actually needed.

Open Claude Code in the repo and ask it to walk you through this — `docs/tutorial.html` has the prompts.
