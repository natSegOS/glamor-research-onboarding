"""End-to-end tests: the orchestrator runs offline with the dummy engine and the
analysis layer reproduces the expected matched-pair arithmetic; model registry
pinning behavior; build_requests parallelization determinism; and the
tokenization metrics (src/tokenization.py) those end-to-end rows carry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tokenization as tm
from analysis import results as result_analysis
import regimes as semantic_regimes
from pipeline.experiment import (
    ExperimentConfiguration,
    PerturbationCondition,
    run_experiment,
)
from pipeline.runner import DUMMY_ENGINE_REVISION, DeterministicDummyEngine, load_generation_rows
from enums import FragmentationStratum, Precision, SemanticClass, Operation, SelectionPolicy, Scope
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
    assert {Scope.ANYWHERE, Scope.CONTENT} <= scopes


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
    assert all(pair.model_revision == DUMMY_ENGINE_REVISION for pair in pairs)


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


# --- build_requests: ProcessPoolExecutor parallelization ---------------------

def test_build_requests_parallel_matches_sequential(monkeypatch):
    """build_requests must produce byte-identical output whether it runs
    sequentially or across a ProcessPoolExecutor — parallelization is a
    performance detail (see its docstring for the disjoint-slice design),
    never allowed to change what gets generated."""
    import pipeline.experiment as experiment_module

    is_word = semantic_regimes.make_is_word()
    configuration = _partition_config("parallel_check")
    task_items = experiment_module.load_task_items(configuration)
    tokenizer = FakeTokenizer()
    assert len(task_items) >= 2  # otherwise worker_count==1 and this proves nothing

    monkeypatch.setattr(experiment_module, "_MINIMUM_ITEMS_FOR_PARALLEL_BUILD", 10 ** 9)
    sequential_requests = experiment_module.build_requests(
        task_items, configuration.conditions, is_word, tokenizer, seed=1729)

    monkeypatch.setattr(experiment_module, "_MINIMUM_ITEMS_FOR_PARALLEL_BUILD", 1)
    parallel_requests = experiment_module.build_requests(
        task_items, configuration.conditions, is_word, tokenizer, seed=1729)

    assert sequential_requests == parallel_requests


def test_build_requests_parallel_writes_same_exclusions_as_sequential(monkeypatch, tmp_path):
    """Exclusion records collected from parallel workers must match what a
    sequential run would have logged — same records, same order."""
    import pipeline.experiment as experiment_module

    is_word = semantic_regimes.make_is_word()
    configuration = _partition_config("parallel_excl_check")
    task_items = experiment_module.load_task_items(configuration)
    tokenizer = FakeTokenizer()

    monkeypatch.setattr(experiment_module, "_MINIMUM_ITEMS_FOR_PARALLEL_BUILD", 10 ** 9)
    sequential_sidecar = experiment_module.ExclusionSidecar(tmp_path / "sequential.jsonl")
    experiment_module.build_requests(
        task_items, configuration.conditions, is_word, tokenizer, seed=1729,
        exclusion_sidecar=sequential_sidecar)

    monkeypatch.setattr(experiment_module, "_MINIMUM_ITEMS_FOR_PARALLEL_BUILD", 1)
    parallel_sidecar = experiment_module.ExclusionSidecar(tmp_path / "parallel.jsonl")
    experiment_module.build_requests(
        task_items, configuration.conditions, is_word, tokenizer, seed=1729,
        exclusion_sidecar=parallel_sidecar)

    assert sequential_sidecar.count == parallel_sidecar.count


# ---------------------------------------------------------------------------
# Tokenization metrics (src/tokenization.py) — the per-row fields these
# end-to-end runs attach to every perturbed request.
# ---------------------------------------------------------------------------

class TestFragmentationStratum:

    @pytest.mark.parametrize("subword_count_change,expected", [
        (-100, FragmentationStratum.LOW), (-1, FragmentationStratum.LOW),
        (0, FragmentationStratum.LOW),                                     # last LOW value
        (1, FragmentationStratum.HIGH),                                    # first HIGH value
        (3, FragmentationStratum.HIGH), (100, FragmentationStratum.HIGH),
    ])
    def test_boundary_is_exactly_at_zero(self, subword_count_change, expected):
        assert tm.fragmentation_stratum(subword_count_change) == expected

    def test_returns_the_enum_type_not_a_plain_string(self):
        assert isinstance(tm.fragmentation_stratum(1), FragmentationStratum)


class TestTokenInflationRatio:

    def test_identical_text_has_ratio_one(self, fake_tokenizer):
        assert tm.token_inflation_ratio(fake_tokenizer, "the cat sat", "the cat sat") == 1.0

    def test_more_fragmentation_gives_a_higher_ratio(self, fake_tokenizer):
        ratio = tm.token_inflation_ratio(fake_tokenizer, "cat", "c@t!!!")  # non-alpha -> extra pieces
        assert ratio >= 1.0

    @pytest.mark.parametrize("text", ["hello", "the quick brown fox", "x"])
    def test_is_always_positive(self, fake_tokenizer, text):
        assert tm.token_inflation_ratio(fake_tokenizer, text, text) > 0

    def test_a_shorter_perturbed_sequence_does_not_crash_and_stays_positive(self, fake_tokenizer):
        assert tm.token_inflation_ratio(fake_tokenizer, "hello world", "hi world") > 0


class TestSubwordCountChange:

    def test_more_fragmented_variant_is_positive(self, fake_tokenizer):
        assert tm.subword_count_change(fake_tokenizer, "cat", "c@t") >= 1

    def test_identical_text_is_zero(self, fake_tokenizer):
        assert tm.subword_count_change(fake_tokenizer, "cat", "cat") == 0

    def test_a_shorter_variant_returns_an_int_without_crashing(self, fake_tokenizer):
        assert isinstance(tm.subword_count_change(fake_tokenizer, "longer", "lo"), int)


class TestFragmentationMatchedPair:

    def test_is_deterministic_across_repeated_calls_and_across_budgets_and_seeds(
            self, is_word, fake_tokenizer):
        for word, budget, seed in [("capital", 1, 5), ("example", 1, 5),
                                   ("example", 2, 10), ("example", 3, 15)]:
            first = tm.build_fragmentation_matched_pair(fake_tokenizer, word, budget, seed, is_word)
            second = tm.build_fragmentation_matched_pair(fake_tokenizer, word, budget, seed, is_word)
            assert first == second

    def test_low_never_fragments_more_than_high_when_both_variants_exist(self, is_word, fake_tokenizer):
        pair = tm.build_fragmentation_matched_pair(fake_tokenizer, "remaining", 1, 9, is_word)
        if pair is not None:
            assert pair.low_fragmentation_subword_change <= 0
            assert pair.high_fragmentation_subword_change >= 1
            assert pair.low_fragmentation_subword_change < pair.high_fragmentation_subword_change
            assert pair.low_fragmentation_variant != pair.high_fragmentation_variant
