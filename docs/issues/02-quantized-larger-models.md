# Test the pipeline on quantized 7B / 8B models

## Why

The current pipeline runs `meta-llama/Llama-3.2-1B-Instruct` because it fits a Colab T4 in fp16 with room to spare (~14.5 GB free, model is ~2.5 GB). The interesting research questions need bigger models — at least 7B / 8B class. A T4 can't hold those in fp16, but it *can* hold them in 4-bit / 8-bit quantization (via `bitsandbytes`). This issue is to prove the pipeline still works there.

## Models to try

At minimum:

- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Optional, if time: a 7B/8B base (non-Instruct) model for comparison.

## Tasks

- [ ] Add a `quantization` block to the experiment config:

  ```yaml
  quantization:
    enabled: true
    bits: 4              # or 8
    compute_dtype: float16
  ```

- [ ] In the experiment's loader, when `quantization.enabled` is true, build a `BitsAndBytesConfig` and pass it to `from_pretrained`. Verify the model fits a T4 (target: <13 GB VRAM, leaves headroom for activations).
- [ ] Run the existing trajectory-divergence experiment with each model at 4-bit and 8-bit. Record VRAM peak and tokens/sec in the results CSV (same columns as the 1B run, plus `model`, `quant_bits`).
- [ ] Compare 1B fp16 vs 8B 4-bit vs 8B 8-bit on the *same* prompts and temperatures. Is the qualitative behavior (lexical diversity, branching) similar? Different?
- [ ] Document gotchas in the experiment's README: which model needs which `bnb_4bit_quant_type`, any tokenizer special-token surprises, any prompts that overflow the context.

## Acceptance criteria

- [ ] At least one 7B-class and one 8B-class model run end-to-end on a T4 via the same `.py` entry point as 1B, controlled only by `config.yaml`.
- [ ] Results CSV contains rows for `{Llama-3.2-1B, Llama-3.1-8B, Mistral-7B}` × `{fp16 (1B only), 4-bit, 8-bit}` over the existing prompt/temperature grid.
- [ ] Visualization can filter by model.
- [ ] README notes the VRAM peak and tokens/sec for each (model × quant) combination.

Depends on #1.
