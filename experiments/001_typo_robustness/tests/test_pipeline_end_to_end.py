"""Full offline pipeline runs with the dummy engine over a configuration that
exercises every condition class (Regime A keyboard substitute/delete, filler
insertion, whitespace merge, fragmentation-matched, Regime B real-word and
homophone, Regime C), asserting the rows-or-exclusions accounting identity,
clean-row uniqueness, parallel-build determinism, shard-partition coverage,
and that the analysis layer summarises the produced rows.

Replaces test_end_to_end.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import regimes as semantic_regimes
from analysis import results as result_analysis
from enums import Operation, Scope, SelectionPolicy, SemanticClass
from pipeline.experiment import (
    ExperimentConfiguration,
    PerturbationCondition,
    load_task_items,
    run_experiment,
)
from pipeline.runner import DUMMY_ENGINE_REVISION, DeterministicDummyEngine, load_generation_rows

try:
    from tests.conftest import FakeTokenizer
except ImportError:                # the offline shim loads conftest as a top-level module
    from conftest import FakeTokenizer


_REASONING_ITEM_COUNT = 6
_MCQ_ITEM_COUNT = 4
_EDIT_BUDGET = 1
_SEED = 1729
_BOOTSTRAP_RESAMPLES = 100

# One condition per class the design can express, keyed by condition name.
# The (semantic_class, operation, selection_policy) triple is what generation
# rows record (r_-prefixed), so it maps rows back to their condition here.
_EVERY_CONDITION_CLASS = (
    PerturbationCondition("keyboard_substitute_A", SemanticClass.A, Operation.SUBSTITUTE,
                          SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("keyboard_delete_A", SemanticClass.A, Operation.DELETE,
                          SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("filler_insertion_A", SemanticClass.A, Operation.INSERT,
                          SelectionPolicy.FILLER_WORD, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("whitespace_merge_A", SemanticClass.A, Operation.DELETE,
                          SelectionPolicy.WHITESPACE, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("fragmentation_matched_A", SemanticClass.A, Operation.SUBSTITUTE,
                          SelectionPolicy.FRAGMENTATION_MATCHED, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("real_word_B", SemanticClass.B, Operation.WORD_SUBSTITUTE,
                          SelectionPolicy.REAL_WORD, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("homophone_B", SemanticClass.B, Operation.WORD_SUBSTITUTE,
                          SelectionPolicy.HOMOPHONE, Scope.ANYWHERE, [_EDIT_BUDGET]),
    PerturbationCondition("meaning_change_C", SemanticClass.C, Operation.WORD_SUBSTITUTE,
                          SelectionPolicy.INFORMATIVE_WORD, Scope.ANSWER_CRITICAL, [_EDIT_BUDGET]),
)

# Conditions whose constructions are guaranteed to succeed on the demo
# vocabulary and items: character-level Regime A edits and both Regime C
# builders (synthetic reasoning items carry templates; MCQ permutes options).
_CONDITIONS_GUARANTEED_TO_PRODUCE_ROWS = frozenset({
    "keyboard_substitute_A", "keyboard_delete_A", "filler_insertion_A",
    "whitespace_merge_A", "meaning_change_C",
})


def _full_condition_configuration(run_id: str) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        run_id=run_id,
        seed=_SEED,
        datasets=[
            {"key": "gsm_symbolic_synthetic", "item_count": _REASONING_ITEM_COUNT},
            {"key": "mcq_demo", "item_count": _MCQ_ITEM_COUNT},
        ],
        conditions=list(_EVERY_CONDITION_CLASS),
    )


def _condition_state_triple(condition: PerturbationCondition) -> tuple:
    return (condition.semantic_class, condition.operation, condition.selection_policy)


def _row_state_triple(row: dict) -> tuple:
    return (row["r_semantic_class"], row["r_operation"], row["r_selection_policy"])


class _RareLetterFragmentingTokenizer:
    """Fragments a word by its rare-letter count, so keyboard substitutions can
    move a variant between fragmentation strata. FakeTokenizer's length-only
    rule cannot (substitutions never change length, so no High variant would
    ever exist and the fragmentation-matched condition could never emit rows)."""

    _RARE_LETTERS = "qxzjkw"

    def encode(self, text):
        return [word
                for word in text.split()
                for _piece in range(1 + sum(character in self._RARE_LETTERS
                                            for character in word.lower()))]


def _answer_every_prompt_engine():
    """A dummy engine emitting both answer formats, so reasoning and MCQ rows
    all parse as VALID and the analysis layer gets scoreable pairs."""
    return DeterministicDummyEngine(lambda prompt: "Reasoning.\n#### 0\nAnswer: A")


@pytest.fixture(scope="module")
def full_pipeline_run(tmp_path_factory):
    """One full offline run over every condition class, shared by the
    assertion tests below (the run itself is the expensive part)."""
    output_directory = tmp_path_factory.mktemp("full_pipeline")
    summary = run_experiment(
        _full_condition_configuration("e2e_full"), _answer_every_prompt_engine(),
        semantic_regimes.make_is_word(), _RareLetterFragmentingTokenizer(),
        output_directory)
    rows = load_generation_rows([Path(summary["output_path"])])
    exclusion_records = load_generation_rows([Path(summary["exclusions_path"])])
    return summary, rows, exclusion_records


# ---------------------------------------------------------------------------
# Rows-or-exclusions accounting: every (item, condition) attempt must end as
# generation rows or as an exclusion record, never vanish silently.
# ---------------------------------------------------------------------------

class TestConditionCoverage:

    def test_every_condition_class_yields_rows_or_exclusion_records(self, full_pipeline_run):
        """Breaking this means a whole condition class silently produced
        nothing: neither data nor an auditable record of why."""
        _summary, rows, exclusion_records = full_pipeline_run
        excluded_condition_names = {record["condition_name"] for record in exclusion_records}
        produced_state_triples = {_row_state_triple(row) for row in rows if not row["is_clean"]}

        for condition in _EVERY_CONDITION_CLASS:
            state_triple = tuple(str(part) for part in _condition_state_triple(condition))
            has_rows = state_triple in produced_state_triples
            has_exclusions = condition.name in excluded_condition_names
            assert has_rows or has_exclusions, (
                f"condition {condition.name!r} produced neither rows nor exclusions")

    def test_guaranteed_constructible_conditions_produce_rows_for_every_item(
            self, full_pipeline_run):
        """Breaking this means a construction that should always succeed on
        this vocabulary started failing (an engine or regime regression)."""
        _summary, rows, _exclusions = full_pipeline_run
        all_task_ids = {row["task_id"] for row in rows}
        assert len(all_task_ids) == _REASONING_ITEM_COUNT + _MCQ_ITEM_COUNT

        rows_by_state_triple: dict = {}
        for row in (row for row in rows if not row["is_clean"]):
            rows_by_state_triple.setdefault(_row_state_triple(row), set()).add(row["task_id"])

        for condition in _EVERY_CONDITION_CLASS:
            if condition.name not in _CONDITIONS_GUARANTEED_TO_PRODUCE_ROWS:
                continue
            state_triple = tuple(str(part) for part in _condition_state_triple(condition))
            assert rows_by_state_triple.get(state_triple) == all_task_ids, (
                f"condition {condition.name!r} missing rows for some items")

    def test_every_item_condition_attempt_is_accounted_for(self, full_pipeline_run):
        """The accounting identity: for each (item, condition) either rows
        exist or an exclusion record names that item and condition. Breaking
        it reintroduces the silent-drop bug the sidecar was built to close."""
        _summary, rows, exclusion_records = full_pipeline_run
        all_task_ids = {row["task_id"] for row in rows}
        item_ids_with_rows_by_triple: dict = {}
        for row in (row for row in rows if not row["is_clean"]):
            item_ids_with_rows_by_triple.setdefault(
                _row_state_triple(row), set()).add(row["task_id"])
        excluded_item_ids_by_condition_name: dict = {}
        for record in exclusion_records:
            excluded_item_ids_by_condition_name.setdefault(
                record["condition_name"], set()).add(record["task_id"])

        for condition in _EVERY_CONDITION_CLASS:
            state_triple = tuple(str(part) for part in _condition_state_triple(condition))
            accounted_item_ids = (
                item_ids_with_rows_by_triple.get(state_triple, set())
                | excluded_item_ids_by_condition_name.get(condition.name, set()))
            assert accounted_item_ids >= all_task_ids, (
                f"condition {condition.name!r} lost items "
                f"{sorted(all_task_ids - accounted_item_ids)} without a trace")

    def test_fragmentation_matched_rows_come_in_low_high_pairs_per_item(
            self, full_pipeline_run):
        """Breaking this destroys the Method A counterfactual contrast: an
        item with only one stratum has no within-item comparison."""
        _summary, rows, _exclusions = full_pipeline_run
        fragmentation_rows = [
            row for row in rows if not row["is_clean"]
            and row["r_selection_policy"] == SelectionPolicy.FRAGMENTATION_MATCHED]
        assert fragmentation_rows, "no fragmentation-matched pair was constructible"

        strata_by_item: dict = {}
        for row in fragmentation_rows:
            strata_by_item.setdefault(row["task_id"], set()).add(
                row["r_fragmentation_stratum"])
            assert row["counterfactual_target_word"]
        assert all(strata == {"Low", "High"} for strata in strata_by_item.values())

    def test_every_perturbed_row_carries_the_tokenization_metric(self, full_pipeline_run):
        """Breaking this starves the primary mediation analysis of its input
        on whichever condition stopped logging the metric."""
        _summary, rows, _exclusions = full_pipeline_run
        assert all("token_inflation_ratio" in row
                   for row in rows if not row["is_clean"])

    def test_clean_rows_are_unique_per_item(self, full_pipeline_run):
        """Breaking this gives some item two clean baselines (ambiguous
        matched-pair join) or none (every perturbed row unpaired)."""
        _summary, rows, _exclusions = full_pipeline_run
        clean_rows = [row for row in rows if row["is_clean"]]
        clean_task_ids = [row["task_id"] for row in clean_rows]
        assert len(clean_task_ids) == len(set(clean_task_ids))
        assert set(clean_task_ids) == {row["task_id"] for row in rows}
        assert len({row["row_id"] for row in rows}) == len(rows)


# ---------------------------------------------------------------------------
# Analysis over the produced rows.
# ---------------------------------------------------------------------------

class TestAnalysisOverProducedRows:

    def test_matched_pairs_join_and_cells_carry_the_expected_columns(self, full_pipeline_run):
        """Breaking this means generation output and the analysis layer have
        drifted apart: the study would run but not be summarisable."""
        _summary, rows, _exclusions = full_pipeline_run
        matched_pairs = result_analysis.join_matched_pairs(rows)
        assert matched_pairs
        assert all(pair.model_revision == DUMMY_ENGINE_REVISION for pair in matched_pairs)

        cell_summaries = result_analysis.summarize_all_cells(
            matched_pairs, resamples=_BOOTSTRAP_RESAMPLES)
        assert cell_summaries
        expected_columns = set(result_analysis.CELL_DIMENSION_KEYS) | {
            "answer_flip_rate", "invalid_or_clarification_rate",
            "delta_valid_only", "mcnemar_p_valid_only", "n_audit_excluded"}
        for cell_summary in cell_summaries:
            assert expected_columns <= set(cell_summary)
            assert cell_summary["n_audit_excluded"] == 0


# ---------------------------------------------------------------------------
# Parallel-build determinism and shard partitioning.
# ---------------------------------------------------------------------------

class TestParallelismDeterminism:

    def test_parallel_and_sequential_request_builds_are_identical(self, monkeypatch):
        """Breaking this makes what gets generated depend on worker count.
        Parallelization is a performance detail, never a semantic one."""
        import pipeline.experiment as experiment_module

        configuration = _full_condition_configuration("parallel_check")
        task_items = load_task_items(configuration)
        is_word = semantic_regimes.make_is_word()
        tokenizer = FakeTokenizer()
        assert len(task_items) >= 2   # otherwise a single worker proves nothing

        monkeypatch.setattr(experiment_module, "_MINIMUM_ITEMS_FOR_PARALLEL_BUILD", 10 ** 9)
        sequential_requests = experiment_module.build_requests(
            task_items, configuration.conditions, is_word, tokenizer, seed=_SEED)

        monkeypatch.setattr(experiment_module, "_MINIMUM_ITEMS_FOR_PARALLEL_BUILD", 1)
        parallel_requests = experiment_module.build_requests(
            task_items, configuration.conditions, is_word, tokenizer, seed=_SEED)

        assert sequential_requests == parallel_requests

    def test_shard_partition_covers_every_request_exactly_once(self, tmp_path):
        """Breaking this loses rows (gaps) or double-generates them (overlap)
        when two workers split the same configuration."""
        is_word = semantic_regimes.make_is_word()
        tokenizer = _RareLetterFragmentingTokenizer()

        unpartitioned_summary = run_experiment(
            _full_condition_configuration("partition"), _answer_every_prompt_engine(),
            is_word, tokenizer, tmp_path / "single")
        unpartitioned_row_ids = {
            row["row_id"]
            for row in load_generation_rows([Path(unpartitioned_summary["output_path"])])}

        per_worker_row_ids = []
        for worker_index in (0, 1):
            worker_summary = run_experiment(
                _full_condition_configuration("partition"), _answer_every_prompt_engine(),
                is_word, tokenizer, tmp_path / "parallel",
                shard_partition=(worker_index, 2))
            per_worker_row_ids.append({
                row["row_id"]
                for row in load_generation_rows([Path(worker_summary["output_path"])])})

        assert per_worker_row_ids[0].isdisjoint(per_worker_row_ids[1])
        assert per_worker_row_ids[0] | per_worker_row_ids[1] == unpartitioned_row_ids

    def test_content_scope_perturbs_questions_shorter_than_the_instruction(self, tmp_path):
        """Breaking this reintroduces the second Llama pilot's regression:
        full-prompt scope spans applied to content_text coordinates left no
        eligible positions in any question shorter than the exemplar-bearing
        instruction, excluding half of every dataset."""
        content_scope_configuration = ExperimentConfiguration(
            run_id="e2e_content_scope",
            seed=_SEED,
            datasets=[{"key": "gsm_symbolic_synthetic", "item_count": _REASONING_ITEM_COUNT}],
            conditions=[PerturbationCondition(
                "keyboard_substitute_A_content", SemanticClass.A, Operation.SUBSTITUTE,
                SelectionPolicy.KEYBOARD_NEIGHBOR, Scope.CONTENT, [_EDIT_BUDGET])],
        )
        task_items = load_task_items(content_scope_configuration)
        assert all(len(item.question_text) < len(item.instruction) for item in task_items), (
            "fixture assumption: questions shorter than the exemplar-bearing instruction")

        summary = run_experiment(
            content_scope_configuration, _answer_every_prompt_engine(),
            semantic_regimes.make_is_word(), FakeTokenizer(), tmp_path)
        perturbed_rows = [
            row for row in load_generation_rows([Path(summary["output_path"])])
            if not row["is_clean"]]
        assert len(perturbed_rows) == _REASONING_ITEM_COUNT
        assert all(row["perturbed_prompt"] != row["clean_prompt"] for row in perturbed_rows)

    def test_only_worker_zero_writes_the_exclusion_sidecar(self, tmp_path):
        """Breaking this duplicates every exclusion record once per worker,
        inflating the exclusion accounting the audit relies on."""
        is_word = semantic_regimes.make_is_word()
        tokenizer = _RareLetterFragmentingTokenizer()

        worker_zero_summary = run_experiment(
            _full_condition_configuration("partition_exclusions"),
            _answer_every_prompt_engine(), is_word, tokenizer, tmp_path,
            shard_partition=(0, 2))
        worker_one_summary = run_experiment(
            _full_condition_configuration("partition_exclusions"),
            _answer_every_prompt_engine(), is_word, tokenizer, tmp_path,
            shard_partition=(1, 2))

        assert worker_zero_summary["exclusions_path"] is not None
        assert worker_one_summary["exclusions_path"] is None
        assert worker_one_summary["excluded_count"] is None


# ---------------------------------------------------------------------------
# Counterfactual word selection/application agreement: the two functions the
# fragmentation-matched pipeline path composes must share one notion of
# "word", or selection picks targets that application cannot replace.
# ---------------------------------------------------------------------------

class TestCounterfactualSelectionApplicationAgreement:

    def test_candidates_are_longest_first_and_only_real_words(self):
        """Breaking this starves the Low/High pair search of its richest
        variant spaces, regressing Method A's item yield."""
        import tokenization

        candidate_words = tokenization.ordered_counterfactual_candidate_words(
            "the cat drove to france for finance",
            semantic_regimes.make_is_word({"cat", "the", "france", "finance"}))
        assert candidate_words[0] == "finance"
        assert tokenization.ordered_counterfactual_candidate_words(
            "some words here", lambda token: False) == []

    def test_selection_and_application_agree_on_letter_digit_boundaries(self):
        """'Python' extracted from 'Python3' by the [A-Za-z]+ tokenizer must
        be replaceable: \\b-style boundaries fail between 'n' and '3' and
        crashed the first Llama pilot on an MMLU item."""
        import tokenization

        content_text = "code written in Python3 syntax"
        candidate_words = tokenization.ordered_counterfactual_candidate_words(
            content_text, lambda token: token.lower() == "python")
        assert candidate_words == ["Python"]
        perturbed_text, _character_index = tokenization.apply_counterfactual_variant(
            content_text, candidate_words[0], "Pythxn")
        assert perturbed_text == "code written in Pythxn3 syntax"


# ---------------------------------------------------------------------------
# Confirmatory revision-pin guard: the run script's precondition for any
# real (non-dummy) end-to-end run.
# ---------------------------------------------------------------------------

class TestConfirmatoryRevisionPinGuard:

    def test_unpinned_revisions_block_a_confirmatory_run(self):
        """Breaking this lets a confirmatory run generate against floating
        model weights, making the pre-registered results irreproducible."""
        from inference import MODEL_ROSTER, assert_revisions_pinned

        with pytest.raises(RuntimeError):
            assert_revisions_pinned(list(MODEL_ROSTER.values()))

    def test_pinned_revisions_pass_the_guard(self):
        """The guard must not over-fire: breaking this blocks every legitimate
        pinned confirmatory run."""
        import dataclasses

        from inference import MODEL_ROSTER, assert_revisions_pinned

        full_length_commit_sha = "a" * 40
        pinned_specifications = [
            dataclasses.replace(specification, revision=full_length_commit_sha)
            for specification in MODEL_ROSTER.values()]
        assert_revisions_pinned(pinned_specifications)   # must not raise
