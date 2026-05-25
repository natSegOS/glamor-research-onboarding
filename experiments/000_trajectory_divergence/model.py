import math
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any, Optional


def load_model(model_id: str, quant_bits: Optional[int] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = None
    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.float16 if bnb_config is None else None,
        device_map="auto",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def generate_once(tokenizer, model, prompt, temperature, top_p, max_new_tokens) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_tokens = output_ids.shape[-1]
    generated_tokens = total_tokens - input_tokens
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

    return {
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed if elapsed > 0 else math.nan,
        "full_text": full_text,
        "generated_text": generated_text,
    }
