import os
import torch
import pandas as pd
from typing import List, Dict, Any

from config import ExperimentConfig, PROMPT_VARIANTS, TEMPERATURES
from model import load_model, generate_once
from metrics import lexical_diversity, sentence_count, classify_behavior


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer, model = load_model(config.model_id)

    # warmup
    warmup_inputs = tokenizer(PROMPT_VARIANTS[0]["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**warmup_inputs, max_new_tokens=5,
                       pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id)

    records: List[Dict[str, Any]] = []
    total = len(PROMPT_VARIANTS) * len(TEMPERATURES) * config.runs_per_condition
    completed = 0

    for prompt_info in PROMPT_VARIANTS:
        for temperature in TEMPERATURES:
            for run_id in range(config.runs_per_condition):
                completed += 1
                print(f"[{completed}/{total}] prompt={prompt_info['prompt_id']} temp={temperature} run={run_id}")

                result = generate_once(tokenizer, model, prompt_info["prompt"],
                                       temperature, config.top_p, config.max_new_tokens)
                gt = result["generated_text"]
                records.append({
                    "model_id": config.model_id,
                    "prompt_id": prompt_info["prompt_id"],
                    "perturbation_type": prompt_info["perturbation_type"],
                    "prompt": prompt_info["prompt"],
                    "temperature": temperature,
                    "top_p": config.top_p,
                    "run_id": run_id,
                    **result,
                    "word_count": len(gt.split()),
                    "char_count": len(gt),
                    "sentence_count": sentence_count(gt),
                    "lexical_diversity": lexical_diversity(gt),
                    "behavior_label": classify_behavior(gt),
                })

    return pd.DataFrame(records)


def create_summaries(df: pd.DataFrame, output_dir: str) -> None:
    df.to_csv(os.path.join(output_dir, "generations.csv"), index=False)

    (df.groupby("temperature")
       .agg(n_generations=("generated_text","count"),
            avg_tokens_per_second=("tokens_per_second","mean"),
            std_tokens_per_second=("tokens_per_second","std"),
            avg_generated_tokens=("generated_tokens","mean"),
            avg_word_count=("word_count","mean"),
            avg_sentence_count=("sentence_count","mean"),
            avg_lexical_diversity=("lexical_diversity","mean"))
       .reset_index()
       .to_csv(os.path.join(output_dir, "summary_by_temperature.csv"), index=False))

    (df.groupby(["prompt_id","perturbation_type","temperature"])
       .agg(n_generations=("generated_text","count"),
            avg_tokens_per_second=("tokens_per_second","mean"),
            avg_word_count=("word_count","mean"),
            avg_lexical_diversity=("lexical_diversity","mean"))
       .reset_index()
       .to_csv(os.path.join(output_dir, "summary_by_prompt.csv"), index=False))

    (df.groupby(["temperature","behavior_label"])
       .size()
       .reset_index(name="count")
       .sort_values(["temperature","count"], ascending=[True,False])
       .to_csv(os.path.join(output_dir, "behavior_label_summary.csv"), index=False))


if __name__ == "__main__":
    config = ExperimentConfig()
    df = run_experiment(config)
    create_summaries(df, config.output_dir)
    print("Done.")
