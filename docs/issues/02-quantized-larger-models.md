# Test the pipeline on quantized 7B / 8B models

The current pipeline runs Llama-3.2-1B because it fits a Colab T4 in fp16 with room to spare. Interesting research questions need bigger models — at least 7B / 8B class (Llama-3.1-8B-Instruct, Mistral-7B-Instruct). A T4 can't hold those in fp16, but it can in 4-bit or 8-bit via `bitsandbytes`. The work here is to confirm the existing pipeline still runs end-to-end with a `BitsAndBytesConfig` passed to `from_pretrained`, measure VRAM and tokens/sec, and add `model` + `quant_bits` columns to the results so we can compare 1B fp16 against 7B/8B quantized on the same prompts.

Ask Claude Code to set up the quantization config and run a small slice first to confirm it fits the T4.
