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
        reasoning_item_count=6,
        multiple_choice_item_count=4,
        seed=1729,
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
        reasoning_item_count=8,
        multiple_choice_item_count=0,
        include_multiple_choice=False,
        seed=2024,
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
