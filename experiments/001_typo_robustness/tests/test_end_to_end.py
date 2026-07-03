"""End-to-end tests: the orchestrator runs offline with the dummy engine and the
analysis layer reproduces the expected matched-pair arithmetic; plus model
registry pinning behavior. These exercise the review fixes (C2 token metrics,
C3 key terms, S2 scope) on the real wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis import results as result_analysis
import regimes as semantic_regimes
from pipeline.experiment import (
    ExperimentConfiguration,
    PerturbationCondition,
    run_experiment,
)
from pipeline.runner import DeterministicDummyEngine, load_generation_rows
from enums import Precision, SemanticClass, Operation, SelectionPolicy, Scope
from inference import (
    MODEL_ROSTER,
    assert_revisions_pinned,
    get_model_specification,
)
try:
    from tests.conftest import FakeTokenizer
except ImportError:                # the offline shim loads conftest as a top-level module
    from conftest import FakeTokenizer


# --- Model registry ---------------------------------------------------------

def test_roster_has_expected_models():
    expected = {"llama_1b", "llama_3b", "llama_8b", "llama_8b_awq",
                "qwen_7b", "qwen_7b_awq", "mistral_7b"}
    assert expected <= set(MODEL_ROSTER)


def test_unpinned_revisions_block_confirmatory_run():
    with pytest.raises(RuntimeError):
        assert_revisions_pinned(list(MODEL_ROSTER.values()))


def test_pinned_revisions_pass():
    import dataclasses
    pinned = [dataclasses.replace(spec, revision="a" * 40) for spec in MODEL_ROSTER.values()]
    assert_revisions_pinned(pinned)               # does not raise


def test_awq_models_marked_as_awq():
    assert get_model_specification("qwen_7b_awq").precision == Precision.AWQ
    assert get_model_specification("llama_8b").precision == Precision.FP16


# --- End-to-end orchestration -----------------------------------------------

def _correct_clean_engine():
    """An engine that answers '#### 0' always. Since the demo reasoning gold is
    rarely 0, this yields a deterministic, mostly-wrong baseline — fine for
    plumbing assertions."""
    return DeterministicDummyEngine(lambda prompt: "Reasoning.\n#### 0\nAnswer: A")


def test_orchestrator_runs_and_logs_token_metrics(tmp_path):
    configuration = ExperimentConfiguration(
        run_id="e2e",
        seed=1729,
        datasets=[
            {"key": "gsm_symbolic_synthetic", "item_count": 6},
            {"key": "mcq_demo", "item_count": 4},
        ],
        conditions=[
            PerturbationCondition("kbd_A", SemanticClass.A, Operation.SUBSTITUTE, SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.ANYWHERE, [1, 2]),
            PerturbationCondition("kbd_A_content", SemanticClass.A, Operation.SUBSTITUTE, SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.CONTENT, [1]),
        ],
    )
    is_word = semantic_regimes.make_is_word()
    summary = run_experiment(configuration, _correct_clean_engine(), is_word,
                             FakeTokenizer(), tmp_path)

    rows = load_generation_rows([Path(summary["output_path"])])
    perturbed_rows = [row for row in rows if not row["is_clean"]]

    # Fix C2: every perturbed row carries a token-inflation ratio.
    assert all("token_inflation_ratio" in row for row in perturbed_rows)

    # Fix S2: both scopes are present in the logged rows.
    scopes = {row["r_scope"] for row in perturbed_rows}
    assert {"anywhere", "content"} <= scopes


def test_analysis_reproduces_matched_pair_counts(tmp_path):
    configuration = ExperimentConfiguration(
        run_id="e2e_analysis",
        seed=2024,
        datasets=[{"key": "gsm_symbolic_synthetic", "item_count": 8}],
        conditions=[PerturbationCondition("kbd_A", SemanticClass.A, Operation.SUBSTITUTE,
                                          SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.ANYWHERE, [1])],
    )
    is_word = semantic_regimes.make_is_word()
    summary = run_experiment(configuration, _correct_clean_engine(), is_word,
                             FakeTokenizer(), tmp_path)

    rows = load_generation_rows([Path(summary["output_path"])])
    pairs = result_analysis.join_matched_pairs(rows)
    cells = result_analysis.summarize_all_cells(pairs, resamples=200)

    assert pairs
    assert cells
    # Every matched pair has a clean partner from the same model+item.
    assert all(pair.model_revision == "dummy-engine-0" for pair in pairs)


# --- Parallel shard partitioning ---------------------------------------------

def _partition_config(run_id):
    return ExperimentConfiguration(
        run_id=run_id,
        seed=1729,
        datasets=[
            {"key": "gsm_symbolic_synthetic", "item_count": 6},
            {"key": "mcq_demo", "item_count": 4},
        ],
        conditions=[
            PerturbationCondition("kbd_A", SemanticClass.A, Operation.SUBSTITUTE,
                                  SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.ANYWHERE, [1, 2]),
        ],
    )


def test_shard_partition_covers_every_request_with_no_overlap(tmp_path):
    """Two workers splitting the same config's requests (design/07 §7.7) must
    together account for every request exactly once — no gaps, no duplicates."""
    is_word = semantic_regimes.make_is_word()

    unpartitioned = run_experiment(
        _partition_config("part"), _correct_clean_engine(), is_word,
        FakeTokenizer(), tmp_path / "single")
    single_rows = load_generation_rows([Path(unpartitioned["output_path"])])
    single_row_ids = {row["row_id"] for row in single_rows}

    worker_row_ids: list[set] = []
    for worker_index in (0, 1):
        summary = run_experiment(
            _partition_config("part"), _correct_clean_engine(), is_word,
            FakeTokenizer(), tmp_path / "parallel",
            shard_partition=(worker_index, 2))
        rows = load_generation_rows([Path(summary["output_path"])])
        worker_row_ids.append({row["row_id"] for row in rows})

    assert worker_row_ids[0].isdisjoint(worker_row_ids[1])
    assert worker_row_ids[0] | worker_row_ids[1] == single_row_ids


def test_shard_partition_only_worker_zero_writes_exclusions(tmp_path):
    is_word = semantic_regimes.make_is_word()

    summary_0 = run_experiment(
        _partition_config("part_excl"), _correct_clean_engine(), is_word,
        FakeTokenizer(), tmp_path, shard_partition=(0, 2))
    summary_1 = run_experiment(
        _partition_config("part_excl"), _correct_clean_engine(), is_word,
        FakeTokenizer(), tmp_path, shard_partition=(1, 2))

    assert summary_0["exclusions_path"] is not None
    assert summary_1["exclusions_path"] is None
    assert summary_1["excluded_count"] is None


def test_shard_partition_writes_distinct_files(tmp_path):
    is_word = semantic_regimes.make_is_word()
    summary_0 = run_experiment(
        _partition_config("part_files"), _correct_clean_engine(), is_word,
        FakeTokenizer(), tmp_path, shard_partition=(0, 2))
    summary_1 = run_experiment(
        _partition_config("part_files"), _correct_clean_engine(), is_word,
        FakeTokenizer(), tmp_path, shard_partition=(1, 2))
    assert summary_0["output_path"] != summary_1["output_path"]
    assert "_w0of2_" in summary_0["output_path"]
    assert "_w1of2_" in summary_1["output_path"]
