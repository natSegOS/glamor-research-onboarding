# GLAMOR Research Onboarding

This is a public repository to show my progress in the onboarding process for the GLAMOR Lab at USC under Zizhao Hu.

Below is a Google Colab link where I determined I have a little under 15 GB available, which is more than enough for Llama-3.2-1B inference.

https://colab.research.google.com/drive/1hlmsd2qhBLDaMtXJLa3WE6F-zK85KWmH?usp=sharing

## Onboarding Milestone

Completed end-to-end inference with `meta-llama/Llama-3.2-1B-Instruct` on Google Colab using a Tesla T4 GPU.

Pipeline:
1. Checked CUDA availability and GPU memory
2. Authenticated with Hugging Face
3. Loaded tokenizer and causal language model
4. Tokenized prompt
5. Generated outputs at temperatures 0.2, 0.7, and 1.2
6. Measured generation speed in tokens/second

Hardware:
- GPU: Tesla T4
- VRAM: 14.56 GB

Observation:
Across temperature values, generation speed stayed around ~40.5 tokens/sec because temperature changes the sampling distribution, not the model forward-pass cost. Output content became more varied at higher temperature, while lower temperature stayed closer to a direct definitional answer.
