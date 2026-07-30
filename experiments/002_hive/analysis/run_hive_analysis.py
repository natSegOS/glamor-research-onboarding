"""Compute every table, interval, and claim-check for the HIVE report.

Reads the slim parquet tables produced by build_slim_dataset.py and writes CSV/JSON
aggregates into the same outputs directory. All confidence intervals are 95% cluster
bootstraps resampling items (qid within benchmark), because the five models and five
seeds repeat the same items and per-row intervals would be spuriously tight.

Usage:
    python run_hive_analysis.py --output-directory outputs
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_CONDITION = "clean"
FLIP_BREAK, FLIP_FIX, FLIP_SAME = "break", "fix", "same"
KEYBOARD_CONDITIONS = [
    "kbd_neighbor", "kbd_random", "kbd_swap", "kbd_repeat", "kbd_fatfinger", "kbd_nospace",
]
SPOKEN_LLM_CONDITIONS = [
    "spoken_casual", "spoken_formal", "spoken_recast", "spoken_reflow", "spoken_reflow_llama",
    "spoken_filler_stripped",
]
CONTROL_CONDITIONS = ["clean_qfirst", "ctrl_option_perm"]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260729
CONFIDENCE_PERCENTILES = (2.5, 97.5)
PERCENTAGE_POINTS = 100.0

README_CLAIMS = {
    "spoken_casual_delta_range_pp": (-5.8, -4.6),
    "spoken_recast_gsm_humaneval_range_pp": (-14.0, -11.0),
    "clean_numwords_delta_range_pp": (0.5, 8.1),
    "keyboard_operator_table": {
        "kbd_random": {"break": 10.50, "fix": 6.92, "net": -3.58, "churn": 17.41},
        "kbd_neighbor": {"break": 10.02, "fix": 7.00, "net": -3.02, "churn": 17.01},
        "kbd_swap": {"break": 8.42, "fix": 6.80, "net": -1.62, "churn": 15.22},
        "kbd_fatfinger": {"break": 8.04, "fix": 6.93, "net": -1.12, "churn": 14.97},
        "kbd_repeat": {"break": 7.17, "fix": 6.40, "net": -0.76, "churn": 13.57},
        "kbd_nospace": {"break": 6.32, "fix": 5.88, "net": -0.44, "churn": 12.20},
    },
    "model_churn_table": {
        "mistralai_Mistral-7B-Instruct-v0.3": {"churn": 18.20, "net": -1.14},
        "meta-llama_Llama-3.1-8B-Instruct": {"churn": 15.90, "net": -2.49},
        "Qwen_Qwen2.5-7B-Instruct": {"churn": 15.17, "net": -2.27},
        "Qwen_Qwen3-8B": {"churn": 14.34, "net": -2.23},
        "microsoft_phi-4": {"churn": 11.71, "net": -0.65},
    },
    "random_worse_or_equal_benchmarks_of_six": 4,
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args()


def cluster_bootstrap_interval_of_ratio(numerators, denominators, replicate_count=BOOTSTRAP_REPLICATES):
    """95% percentile interval for sum(numerators)/sum(denominators) resampling
    clusters with replacement. Inputs are per-cluster arrays."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    cluster_count = len(numerators)
    sampled_indices = generator.integers(0, cluster_count, size=(replicate_count, cluster_count))
    replicate_ratios = (
        numerators[sampled_indices].sum(axis=1) / denominators[sampled_indices].sum(axis=1)
    )
    low, high = np.percentile(replicate_ratios, CONFIDENCE_PERCENTILES)
    return float(low), float(high)


def mean_with_item_cluster_interval(frame, value_column):
    """Mean of value_column with a 95% CI clustered on (benchmark, qid)."""
    per_cluster = frame.groupby(["benchmark", "qid"], observed=True)[value_column].agg(["sum", "count"])
    low, high = cluster_bootstrap_interval_of_ratio(
        per_cluster["sum"].to_numpy(float), per_cluster["count"].to_numpy(float))
    return float(frame[value_column].mean()), low, high


def with_flip_indicator_columns(frame):
    return frame.assign(
        is_break=(frame["flip"] == FLIP_BREAK).astype(float),
        is_fix=(frame["flip"] == FLIP_FIX).astype(float),
        is_churn=(frame["flip"] != FLIP_SAME).astype(float),
        net_flip=(frame["flip"] == FLIP_FIX).astype(float) - (frame["flip"] == FLIP_BREAK).astype(float),
        paired_delta=(frame["score"] - frame["clean_score"]).astype(float),
    )


def guard_kept_rates(perturbed_frame):
    return (
        perturbed_frame.groupby(["condition", "benchmark"], observed=True)["meaning_kept"]
        .agg(kept_rate="mean", rows="size").reset_index()
    )


def accuracy_delta_table(perturbed_frame):
    """Per (condition, benchmark): raw and meaning-guarded paired accuracy deltas,
    guarded delta with an item-clustered 95% CI."""
    records = []
    for (condition, benchmark), group in perturbed_frame.groupby(
            ["condition", "benchmark"], observed=True):
        kept = group[group["meaning_kept"]]
        delta_mean, delta_low, delta_high = mean_with_item_cluster_interval(kept, "paired_delta")
        records.append({
            "condition": condition, "benchmark": benchmark,
            "rows": len(group), "kept_rows": len(kept),
            "kept_rate": float(group["meaning_kept"].mean()),
            "clean_accuracy_on_kept": float(kept["clean_score"].mean()),
            "perturbed_accuracy_on_kept": float(kept["score"].mean()),
            "raw_delta_pp": float(group["paired_delta"].mean()) * PERCENTAGE_POINTS,
            "guarded_delta_pp": delta_mean * PERCENTAGE_POINTS,
            "guarded_delta_ci_low_pp": delta_low * PERCENTAGE_POINTS,
            "guarded_delta_ci_high_pp": delta_high * PERCENTAGE_POINTS,
        })
    return pd.DataFrame(records)


def flip_rate_table(frame, grouping_columns):
    """break/fix/net/churn percentages per group, churn and net with item-clustered CIs,
    plus a naive McNemar sign-test p-value (rows treated as independent — optimistic)."""
    from scipy.stats import binomtest

    records = []
    for group_key, group in frame.groupby(grouping_columns, observed=True):
        churn_mean, churn_low, churn_high = mean_with_item_cluster_interval(group, "is_churn")
        net_mean, net_low, net_high = mean_with_item_cluster_interval(group, "net_flip")
        break_count = int(group["is_break"].sum())
        fix_count = int(group["is_fix"].sum())
        flipping_rows = break_count + fix_count
        record = dict(zip(grouping_columns, group_key if isinstance(group_key, tuple) else (group_key,)))
        records.append({
            **record,
            "rows": len(group),
            "break_pct": float(group["is_break"].mean()) * PERCENTAGE_POINTS,
            "fix_pct": float(group["is_fix"].mean()) * PERCENTAGE_POINTS,
            "net_pp": net_mean * PERCENTAGE_POINTS,
            "net_ci_low_pp": net_low * PERCENTAGE_POINTS,
            "net_ci_high_pp": net_high * PERCENTAGE_POINTS,
            "churn_pct": churn_mean * PERCENTAGE_POINTS,
            "churn_ci_low_pct": churn_low * PERCENTAGE_POINTS,
            "churn_ci_high_pct": churn_high * PERCENTAGE_POINTS,
            "mcnemar_naive_p": float(
                binomtest(break_count, flipping_rows).pvalue) if flipping_rows else 1.0,
        })
    return pd.DataFrame(records)


def random_versus_neighbor_contrast(keyboard_frame):
    """Paired per-item contrast between kbd_random and kbd_neighbor, overall and per
    benchmark, on churn and net, with item-clustered CIs on the difference."""
    both = keyboard_frame[keyboard_frame["condition"].isin(["kbd_random", "kbd_neighbor"])]
    per_item = both.pivot_table(
        index=["benchmark", "qid", "model", "seed"], columns="condition",
        values=["is_churn", "net_flip"], observed=True).dropna()
    per_item.columns = [f"{measure}__{condition}" for measure, condition in per_item.columns]
    per_item = per_item.reset_index()
    per_item["churn_difference"] = (
        per_item["is_churn__kbd_random"] - per_item["is_churn__kbd_neighbor"])
    per_item["net_difference"] = (
        per_item["net_flip__kbd_random"] - per_item["net_flip__kbd_neighbor"])

    def contrast_record(scope_label, scope_frame):
        churn_mean, churn_low, churn_high = mean_with_item_cluster_interval(
            scope_frame, "churn_difference")
        net_mean, net_low, net_high = mean_with_item_cluster_interval(
            scope_frame, "net_difference")
        return {
            "scope": scope_label, "paired_rows": len(scope_frame),
            "churn_random_minus_neighbor_pp": churn_mean * PERCENTAGE_POINTS,
            "churn_difference_ci_low_pp": churn_low * PERCENTAGE_POINTS,
            "churn_difference_ci_high_pp": churn_high * PERCENTAGE_POINTS,
            "net_random_minus_neighbor_pp": net_mean * PERCENTAGE_POINTS,
            "net_difference_ci_low_pp": net_low * PERCENTAGE_POINTS,
            "net_difference_ci_high_pp": net_high * PERCENTAGE_POINTS,
        }

    records = [contrast_record("all_benchmarks", per_item)] + [
        contrast_record(benchmark, benchmark_frame)
        for benchmark, benchmark_frame in per_item.groupby("benchmark", observed=True)
    ]
    return pd.DataFrame(records)


def clean_decode_noise_floor(clean_frame, keyboard_frame):
    """Across-seed disagreement on clean items: for each (model, benchmark, qid) the
    fraction of seed pairs whose 0/1 scores differ. This is the decode-variance floor
    the keyboard churn numbers must clear (README open question 1). Seeds sample
    different item subsets, so the floor is only measurable on items clean-scored under
    at least two seeds; keyboard churn restricted to that same item set is reported
    alongside for a like-for-like comparison."""
    per_item = clean_frame.groupby(["model", "benchmark", "qid"], observed=True)["score"].agg(
        ["sum", "count"])
    multi_seed = per_item[per_item["count"] >= 2].reset_index()
    multi_seed["seed_pair_disagreement"] = (
        multi_seed["sum"] * (multi_seed["count"] - multi_seed["sum"])
        / (multi_seed["count"] * (multi_seed["count"] - 1) / 2))

    floor_item_keys = multi_seed[["model", "benchmark", "qid"]]
    keyboard_on_floor_items = keyboard_frame.merge(floor_item_keys)

    def scope_summary(frame):
        return float(frame["seed_pair_disagreement"].mean()) * PERCENTAGE_POINTS

    def churn_summary(frame):
        return float(frame["is_churn"].mean()) * PERCENTAGE_POINTS

    def per_group(grouping_column):
        return {
            str(group_key): {
                "clean_floor_pct": scope_summary(group),
                "items_measured": int(len(group)),
                "keyboard_churn_on_same_items_pct": churn_summary(
                    keyboard_on_floor_items[
                        keyboard_on_floor_items[grouping_column] == group_key]),
            }
            for group_key, group in multi_seed.groupby(grouping_column, observed=True)
        }

    return {
        "overall": {
            "clean_floor_pct": scope_summary(multi_seed),
            "keyboard_churn_on_same_items_pct": churn_summary(keyboard_on_floor_items),
            "keyboard_churn_all_items_pct": churn_summary(keyboard_frame),
        },
        "per_model": per_group("model"),
        "per_benchmark": per_group("benchmark"),
        "model_benchmark_items_with_2plus_seeds": int(len(multi_seed)),
        "model_benchmark_items_total": int(len(per_item)),
        "items_scored_on_all_five_seeds": int((per_item["count"] == 5).sum()),
    }


def real_word_edit_analysis(keyboard_edit_frame):
    """README open question 3: do neighbor typos yield real words more often than random
    ones, and do rows containing a real-word edit break more? Token-level real-word rates
    per substitution operator, then break rate by has-any-real-word-edit among
    clean-correct rows."""
    substitutions = keyboard_edit_frame[
        keyboard_edit_frame["condition"].isin(["kbd_neighbor", "kbd_random"])]
    records = []
    for condition, condition_frame in substitutions.groupby("condition", observed=True):
        token_level_real_rate = (
            condition_frame["real_word_edit_count"].sum() / condition_frame["edit_count"].sum())
        clean_correct = with_flip_indicator_columns(
            condition_frame[condition_frame["clean_score"] == 1]).assign(
            has_real_word_edit=lambda frame: frame["real_word_edit_count"] > 0)
        for has_real_word, word_class_frame in clean_correct.groupby(
                "has_real_word_edit", observed=True):
            break_mean, break_low, break_high = mean_with_item_cluster_interval(
                word_class_frame, "is_break")
            records.append({
                "condition": condition,
                "token_level_real_word_rate": float(token_level_real_rate),
                "row_has_real_word_edit": bool(has_real_word),
                "share_of_clean_correct_rows": len(word_class_frame) / len(clean_correct),
                "clean_correct_rows": len(word_class_frame),
                "break_pct": break_mean * PERCENTAGE_POINTS,
                "break_ci_low_pct": break_low * PERCENTAGE_POINTS,
                "break_ci_high_pct": break_high * PERCENTAGE_POINTS,
            })
    return pd.DataFrame(records)


def edit_feature_break_analysis(keyboard_edit_frame):
    """README open question 2, structural slice: break rate by edit count and by
    number adjacency, clean-correct rows only."""
    clean_correct = with_flip_indicator_columns(
        keyboard_edit_frame[keyboard_edit_frame["clean_score"] == 1]).copy()
    clean_correct["edit_count_band"] = pd.cut(
        clean_correct["edit_count"], bins=[0, 2, 4, 6, np.inf],
        labels=["1-2", "3-4", "5-6", "7+"])

    def break_rate_records(grouping_column):
        records = []
        for group_key, group in clean_correct.groupby(grouping_column, observed=True):
            break_mean, break_low, break_high = mean_with_item_cluster_interval(group, "is_break")
            records.append({
                "grouping": grouping_column, "group": str(group_key),
                "clean_correct_rows": len(group),
                "break_pct": break_mean * PERCENTAGE_POINTS,
                "break_ci_low_pct": break_low * PERCENTAGE_POINTS,
                "break_ci_high_pct": break_high * PERCENTAGE_POINTS,
            })
        return records

    return pd.DataFrame(
        break_rate_records("edit_count_band") + break_rate_records("any_edit_neighbors_a_number"))


def item_break_concentration(keyboard_frame):
    """README open question 2, item slice: do breaks concentrate on a fragile minority
    of items? Per-(benchmark, qid) break rates among clean-correct keyboard rows, the
    share of breaks carried by the top decile of items, and a binomial overdispersion
    index (variance of item break counts over binomial expectation)."""
    clean_correct = keyboard_frame[keyboard_frame["clean_score"] == 1]
    per_item = clean_correct.groupby(["benchmark", "qid"], observed=True)["is_break"].agg(
        breaks="sum", exposures="count")
    per_item = per_item[per_item["exposures"] >= 10]
    per_item["break_rate"] = per_item["breaks"] / per_item["exposures"]

    pooled_rate = per_item["breaks"].sum() / per_item["exposures"].sum()
    binomial_variance = (per_item["exposures"] * pooled_rate * (1 - pooled_rate)).sum()
    observed_variance = (
        (per_item["breaks"] - per_item["exposures"] * pooled_rate) ** 2).sum()

    sorted_items = per_item.sort_values("break_rate", ascending=False)
    top_decile_count = max(1, len(sorted_items) // 10)
    return {
        "items_with_at_least_10_clean_correct_exposures": int(len(per_item)),
        "pooled_break_rate_pct": float(pooled_rate) * PERCENTAGE_POINTS,
        "share_of_breaks_in_top_decile_of_items_pct": float(
            sorted_items.head(top_decile_count)["breaks"].sum() / per_item["breaks"].sum()
        ) * PERCENTAGE_POINTS,
        "share_of_items_with_zero_breaks_pct": float(
            (per_item["breaks"] == 0).mean()) * PERCENTAGE_POINTS,
        "overdispersion_index": float(observed_variance / binomial_variance),
        "median_item_break_rate_pct": float(per_item["break_rate"].median()) * PERCENTAGE_POINTS,
        "p90_item_break_rate_pct": float(
            per_item["break_rate"].quantile(0.9)) * PERCENTAGE_POINTS,
    }


def within_range(value, value_range):
    low, high = min(value_range), max(value_range)
    return bool(low - 0.25 <= value <= high + 0.25)


def verify_readme_claims(delta_table, operator_flips, model_flips, benchmark_contrasts):
    """Recompute each quantitative README claim and record computed-versus-claimed."""
    guarded = delta_table.set_index(["condition", "benchmark"])["guarded_delta_pp"]

    spoken_casual_deltas = guarded.loc["spoken_casual"].to_dict()
    recast_hard = guarded.loc["spoken_recast"][
        ["gsm8k", "gsm_symbolic", "gsm1k", "humaneval"]].to_dict()
    numwords_deltas = guarded.loc["clean_numwords"].to_dict()

    operator_rows = operator_flips.set_index("condition")
    operator_check = {
        operator: {
            "claimed": claimed,
            "computed": {
                "break": round(float(operator_rows.loc[operator, "break_pct"]), 2),
                "fix": round(float(operator_rows.loc[operator, "fix_pct"]), 2),
                "net": round(float(operator_rows.loc[operator, "net_pp"]), 2),
                "churn": round(float(operator_rows.loc[operator, "churn_pct"]), 2),
            },
        }
        for operator, claimed in README_CLAIMS["keyboard_operator_table"].items()
    }

    model_rows = model_flips.set_index("model")
    model_check = {
        model_name: {
            "claimed": claimed,
            "computed": {
                "churn": round(float(model_rows.loc[model_name, "churn_pct"]), 2),
                "net": round(float(model_rows.loc[model_name, "net_pp"]), 2),
            },
        }
        for model_name, claimed in README_CLAIMS["model_churn_table"].items()
    }

    per_benchmark = benchmark_contrasts[benchmark_contrasts["scope"] != "all_benchmarks"]
    random_worse_or_equal = int((per_benchmark["churn_random_minus_neighbor_pp"] >= 0).sum())

    return {
        "spoken_casual": {
            "claimed_range_pp": README_CLAIMS["spoken_casual_delta_range_pp"],
            "computed_per_benchmark_pp": {
                benchmark: round(delta, 2) for benchmark, delta in spoken_casual_deltas.items()},
            "all_within_claimed_range": all(
                within_range(delta, README_CLAIMS["spoken_casual_delta_range_pp"])
                for delta in spoken_casual_deltas.values()),
        },
        "spoken_recast_gsm_and_humaneval": {
            "claimed_range_pp": README_CLAIMS["spoken_recast_gsm_humaneval_range_pp"],
            "computed_pp": {benchmark: round(delta, 2) for benchmark, delta in recast_hard.items()},
            "all_within_claimed_range": all(
                within_range(delta, README_CLAIMS["spoken_recast_gsm_humaneval_range_pp"])
                for delta in recast_hard.values()),
        },
        "clean_numwords": {
            "claimed_range_pp": README_CLAIMS["clean_numwords_delta_range_pp"],
            "computed_per_benchmark_pp": {
                benchmark: round(delta, 2) for benchmark, delta in numwords_deltas.items()},
            "all_within_claimed_range": all(
                within_range(delta, README_CLAIMS["clean_numwords_delta_range_pp"])
                for delta in numwords_deltas.values()),
        },
        "keyboard_operator_table": operator_check,
        "model_churn_table": model_check,
        "random_worse_or_equal_benchmarks": {
            "claimed_of_six": README_CLAIMS["random_worse_or_equal_benchmarks_of_six"],
            "computed_of_six": random_worse_or_equal,
        },
    }


def main():
    arguments = parse_arguments()
    outputs = arguments.output_directory

    slim = pd.read_parquet(outputs / "slim_instances.parquet")
    for column in ["model", "seed", "benchmark", "condition", "flip"]:
        slim[column] = slim[column].astype("category")

    perturbed = with_flip_indicator_columns(slim[slim["condition"] != CLEAN_CONDITION])
    keyboard = perturbed[perturbed["condition"].isin(KEYBOARD_CONDITIONS)]
    keyboard_kept = keyboard[keyboard["meaning_kept"]]
    clean = slim[slim["condition"] == CLEAN_CONDITION]

    guard_kept_rates(perturbed).to_csv(outputs / "guard_kept_rates.csv", index=False)
    delta_table = accuracy_delta_table(perturbed)
    delta_table.to_csv(outputs / "accuracy_deltas.csv", index=False)

    operator_flips = flip_rate_table(keyboard, ["condition"])
    operator_flips.to_csv(outputs / "keyboard_operator_flips.csv", index=False)
    operator_flips_kept = flip_rate_table(keyboard_kept, ["condition"])
    operator_flips_kept.to_csv(outputs / "keyboard_operator_flips_meaning_kept.csv", index=False)
    flip_rate_table(keyboard, ["condition", "benchmark"]).to_csv(
        outputs / "keyboard_operator_by_benchmark_flips.csv", index=False)
    model_flips = flip_rate_table(keyboard, ["model"])
    model_flips.to_csv(outputs / "keyboard_model_flips.csv", index=False)
    flip_rate_table(keyboard, ["model", "condition"]).to_csv(
        outputs / "keyboard_model_by_operator_flips.csv", index=False)

    benchmark_contrasts = random_versus_neighbor_contrast(keyboard)
    benchmark_contrasts.to_csv(outputs / "random_versus_neighbor.csv", index=False)

    (outputs / "clean_decode_noise_floor.json").write_text(
        json.dumps(clean_decode_noise_floor(clean, keyboard), indent=2))
    (outputs / "item_break_concentration.json").write_text(
        json.dumps(item_break_concentration(keyboard), indent=2))

    keyboard_edits = pd.read_parquet(outputs / "keyboard_edit_features.parquet")
    real_word_edit_analysis(keyboard_edits).to_csv(
        outputs / "real_word_edit_breaks.csv", index=False)
    edit_feature_break_analysis(keyboard_edits).to_csv(
        outputs / "edit_feature_breaks.csv", index=False)

    model_condition_deltas = perturbed[perturbed["meaning_kept"]].groupby(
        ["model", "condition"], observed=True)["paired_delta"].mean().mul(
        PERCENTAGE_POINTS).round(2).reset_index()
    model_condition_deltas.to_csv(outputs / "model_condition_guarded_deltas.csv", index=False)

    clean_accuracy = clean.groupby(["model", "benchmark"], observed=True)["score"].mean()
    clean_accuracy.mul(PERCENTAGE_POINTS).round(2).reset_index().to_csv(
        outputs / "clean_accuracy_by_model_benchmark.csv", index=False)

    (outputs / "headline_verification.json").write_text(json.dumps(
        verify_readme_claims(delta_table, operator_flips, model_flips, benchmark_contrasts),
        indent=2))
    print("analysis complete")


if __name__ == "__main__":
    main()
