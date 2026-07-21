"""The idempotent, resumable generation runner (src/pipeline/runner.py) and
the run tool's --fresh cleanup: deterministic row-ID algebra, idempotent
resume, mid-shard crash recovery, the stale-manifest schema guard, the
scoped --fresh deletion, and exclusion-record content for unperturbable items.

Consolidates test_generation_runner.py.
"""

from __future__ import annotations

import importlib.util
import json

from pathlib import Path

import pytest

from enums import (
    FinishReason, Operation, Scope, SelectionPolicy, SemanticClass, TaskFamily)
from inference.engines import StreamedGeneration
from pipeline.runner import (
    DeterministicDummyEngine,
    GenerationRequest,
    SCHEMA_VERSION,
    ShardManifest,
    deterministic_row_id,
    load_generation_rows,
    run_shard,
)


_BASE_STATE_VECTOR = {
    "semantic_class": SemanticClass.A,
    "operation": Operation.SUBSTITUTE,
    "selection_policy": SelectionPolicy.KEYBOARD_NEIGHBOR,
    "scope": Scope.ANYWHERE,
    "edit_budget": 2,
}

_CLEAN_STATE_VECTOR = {
    "semantic_class": SemanticClass.CLEAN,
    "operation": Operation.NONE,
    "selection_policy": SelectionPolicy.NONE,
    "scope": Scope.NONE,
    "edit_budget": 0,
}


def _make_request(task_id: str, is_clean: bool = True) -> GenerationRequest:
    return GenerationRequest(
        task_id=task_id,
        task_family=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
        prompt=f"Solve: item {task_id}",
        gold_answer=0,
        is_clean=is_clean,
        perturbation_state_vector=(
            dict(_CLEAN_STATE_VECTOR) if is_clean else dict(_BASE_STATE_VECTOR)),
        seed=1,
    )


def _run_default_shard(tmp_path, requests, engine=None,
                       shard_id="shard1", manifest_name="manifest.json"):
    manifest = ShardManifest(tmp_path / manifest_name)
    written = run_shard(
        shard_id, requests, engine or DeterministicDummyEngine(),
        tmp_path / "out.jsonl", manifest, model_id="m", model_revision="rev1")
    return written, manifest


# ---------------------------------------------------------------------------
# Deterministic row-ID algebra.  The row ID is the sole dedup/resume key: it
# must ignore representation noise (dict key order) and respond to every
# component that defines a distinct generation.
# ---------------------------------------------------------------------------

class TestDeterministicRowIdAlgebra:

    def test_id_is_independent_of_state_vector_key_order(self):
        """Breaking this makes resume representation-dependent: the same
        logical row would get a fresh ID and be generated twice."""
        reordered_vector = dict(reversed(list(_BASE_STATE_VECTOR.items())))
        assert (deterministic_row_id("rev1", "t1", _BASE_STATE_VECTOR, 7, False)
                == deterministic_row_id("rev1", "t1", reordered_vector, 7, False))

    @pytest.mark.parametrize("variant_arguments", [
        ("rev2", "t1", _BASE_STATE_VECTOR, 7, False),
        ("rev1", "t2", _BASE_STATE_VECTOR, 7, False),
        ("rev1", "t1", {**_BASE_STATE_VECTOR, "edit_budget": 3}, 7, False),
        ("rev1", "t1", _BASE_STATE_VECTOR, 8, False),
        ("rev1", "t1", _BASE_STATE_VECTOR, 7, True),
    ], ids=["model_revision", "task_id", "state_vector_value", "seed", "is_clean"])
    def test_changing_any_single_component_changes_the_id(self, variant_arguments):
        """Breaking this collides two distinct generations onto one ID, so the
        second is silently skipped as 'already generated'."""
        base_id = deterministic_row_id("rev1", "t1", _BASE_STATE_VECTOR, 7, False)
        assert base_id != deterministic_row_id(*variant_arguments)
        assert isinstance(base_id, str) and base_id


# ---------------------------------------------------------------------------
# Idempotent resume and schema completeness.
# ---------------------------------------------------------------------------

class TestIdempotentResume:

    def test_rerunning_a_completed_shard_writes_zero_new_rows(self, tmp_path):
        """Breaking this duplicates rows on every resume, corrupting matched-
        pair counts downstream."""
        requests = [_make_request("t1"), _make_request("t2")]
        first_written, _ = _run_default_shard(tmp_path, requests)
        rerun_written, _ = _run_default_shard(
            tmp_path, requests, manifest_name="fresh_manifest.json")

        assert first_written == 2
        assert rerun_written == 0
        assert len(load_generation_rows([tmp_path / "out.jsonl"])) == 2

    def test_a_shard_marked_complete_in_the_manifest_is_skipped_entirely(self, tmp_path):
        """Breaking this re-submits finished shards to the engine: wasted GPU
        hours at best, duplicate rows at worst."""
        manifest = ShardManifest(tmp_path / "manifest.json")
        manifest.mark_shard_complete("shard1")
        written = run_shard(
            "shard1", [_make_request("t1")], DeterministicDummyEngine(),
            tmp_path / "out.jsonl", manifest, model_id="m", model_revision="rev1")
        assert written == 0

    def test_every_row_carries_the_complete_required_schema(self, tmp_path):
        """Breaking this ships rows the analysis layer cannot join or audit
        (missing IDs, scores, provenance, or throughput fields)."""
        _run_default_shard(tmp_path, [_make_request("t1", is_clean=False)])
        required_fields = {
            "row_id", "schema", "model_revision", "task_id", "is_clean",
            "expected_answer", "parsed_answer", "is_correct", "parse_status",
            "extraction_tier", "decoding", "r_edit_budget",
            "output_token_count", "finish_reason", "request_wall_seconds"}
        row = load_generation_rows([tmp_path / "out.jsonl"])[0]
        assert required_fields <= set(row)
        assert row["model_revision"] == "rev1"
        assert row["r_edit_budget"] == _BASE_STATE_VECTOR["edit_budget"]


# ---------------------------------------------------------------------------
# Crash recovery: per-row streaming means a dead engine loses only the
# in-flight requests, and a resume completes exactly the missing rows.
# ---------------------------------------------------------------------------

class _CrashAfterNEngine:
    """Streams real results for the first N requests, then raises, simulating
    an engine dying mid-shard (a real incident this test locks in against)."""

    revision = "crash-test"

    def __init__(self, crash_after: int):
        self.crash_after = crash_after

    def generate_streaming(self, prompts, max_new_tokens):
        for index, _prompt in enumerate(prompts):
            if index == self.crash_after:
                raise RuntimeError("simulated engine death")
            yield StreamedGeneration(
                prompt_index=index, text="#### 0", output_token_count=2,
                finish_reason=FinishReason.STOPPED, request_wall_seconds=0.0)


class TestCrashRecovery:

    _REQUEST_COUNT = 5
    _CRASH_AFTER = 3

    def _crash_then_inspect(self, tmp_path):
        requests = [_make_request(f"t{index}") for index in range(self._REQUEST_COUNT)]
        manifest = ShardManifest(tmp_path / "manifest.json")
        with pytest.raises(RuntimeError):
            run_shard("shard1", requests, _CrashAfterNEngine(self._CRASH_AFTER),
                      tmp_path / "out.jsonl", manifest,
                      model_id="m", model_revision="rev1")
        return requests, manifest

    def test_rows_finished_before_the_crash_are_already_on_disk(self, tmp_path):
        """Breaking this (e.g. buffering a whole batch) loses every completed
        generation in the batch when the process dies."""
        _requests, manifest = self._crash_then_inspect(tmp_path)
        assert len(load_generation_rows([tmp_path / "out.jsonl"])) == self._CRASH_AFTER
        assert not manifest.is_shard_complete("shard1")

    def test_resume_completes_exactly_the_missing_rows_with_no_duplicates(self, tmp_path):
        """Breaking this either regenerates finished rows (duplicates) or
        skips crashed ones (holes). Both corrupt the matched-pair join."""
        requests, manifest = self._crash_then_inspect(tmp_path)
        written_on_resume = run_shard(
            "shard1", requests, DeterministicDummyEngine(), tmp_path / "out.jsonl",
            manifest, model_id="m", model_revision="rev1")

        rows = load_generation_rows([tmp_path / "out.jsonl"])
        assert written_on_resume == self._REQUEST_COUNT - self._CRASH_AFTER
        assert len(rows) == self._REQUEST_COUNT
        assert len({row["row_id"] for row in rows}) == self._REQUEST_COUNT
        assert manifest.is_shard_complete("shard1")


# ---------------------------------------------------------------------------
# Stale-manifest schema guard.  A committed manifest from an older schema
# once made a fresh clone generate nothing; the guard must ignore it.
# ---------------------------------------------------------------------------

class TestManifestStaleSchemaGuard:

    def test_a_manifest_from_another_schema_version_is_ignored(self, tmp_path):
        """Breaking this lets an old-schema manifest mark shards complete, so
        a fresh clone silently generates zero rows (a real prior incident)."""
        stale_schema_version = "0.0-stale"
        assert stale_schema_version != SCHEMA_VERSION
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schema": stale_schema_version, "completed_shards": ["shard1"]}))

        manifest = ShardManifest(manifest_path)
        assert not manifest.is_shard_complete("shard1")

        written = run_shard(
            "shard1", [_make_request("t1")], DeterministicDummyEngine(),
            tmp_path / "out.jsonl", manifest, model_id="m", model_revision="rev1")
        assert written == 1

    def test_a_current_schema_manifest_is_honoured(self, tmp_path):
        """The guard must not over-fire: breaking this direction discards real
        progress and regenerates every shard on every resume."""
        _written, _manifest = _run_default_shard(tmp_path, [_make_request("t1")])
        reloaded = ShardManifest(tmp_path / "manifest.json")
        assert reloaded.is_shard_complete("shard1")


# ---------------------------------------------------------------------------
# --fresh deletion scope.  An earlier prefix glob deleted unrelated files
# that merely shared the run prefix (e.g. pilot_results.zip beside run_id
# "pilot"); --fresh must touch only the known pipeline output suffixes.
# ---------------------------------------------------------------------------

def _load_run_generation_tool_module():
    tool_path = Path(__file__).resolve().parent.parent / "tools" / "run_generation.py"
    module_spec = importlib.util.spec_from_file_location("run_generation", tool_path)
    assert module_spec is not None and module_spec.loader is not None
    tool_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(tool_module)
    return tool_module


class TestFreshDeletionScope:

    _RUN_OUTPUT_FILE_NAMES = (
        "pilot_generations.jsonl",
        "pilot_manifest.json",
        "pilot_exclusions.jsonl",
        "pilot_w0of2_generations.jsonl",
    )
    _UNRELATED_FILE_NAMES = ("pilot_results.zip", "pilot_notes.txt")

    def test_fresh_deletes_exactly_the_run_outputs_and_spares_the_rest(self, tmp_path):
        """Breaking this regresses to the prefix-glob bug: --fresh would
        destroy a user's unrelated files that happen to share the run prefix,
        or leave stale run outputs behind to poison the rerun."""
        run_generation = _load_run_generation_tool_module()
        for file_name in self._RUN_OUTPUT_FILE_NAMES + self._UNRELATED_FILE_NAMES:
            (tmp_path / file_name).write_text("{}")

        deleted_paths = run_generation._delete_previous_run_outputs(tmp_path, "pilot")

        assert {path.name for path in deleted_paths} == set(self._RUN_OUTPUT_FILE_NAMES)
        for run_output_name in self._RUN_OUTPUT_FILE_NAMES:
            assert not (tmp_path / run_output_name).exists()
        for unrelated_name in self._UNRELATED_FILE_NAMES:
            assert (tmp_path / unrelated_name).exists()

    def test_a_sibling_runs_outputs_survive(self, tmp_path):
        """Breaking this lets --fresh for one run delete another run's data in
        the same output directory."""
        run_generation = _load_run_generation_tool_module()
        for file_name in ("pilot_generations.jsonl", "main_generations.jsonl"):
            (tmp_path / file_name).write_text("{}")

        deleted_paths = run_generation._delete_previous_run_outputs(tmp_path, "pilot")

        assert {path.name for path in deleted_paths} == {"pilot_generations.jsonl"}
        assert (tmp_path / "main_generations.jsonl").exists()


# ---------------------------------------------------------------------------
# Exclusion-record content.  A dropped item must be reconstructable from its
# sidecar record alone; empty fields would make exclusions unauditable.
# ---------------------------------------------------------------------------

class TestExclusionRecordContent:

    _EDIT_BUDGET = 1

    def test_an_unperturbable_item_produces_a_fully_populated_record(self, is_word):
        """Breaking this leaves exclusions untraceable: the audit could no
        longer say which item, condition, budget, or stage dropped a row."""
        from pipeline.experiment import PerturbationCondition, _build_requests_for_item_slice
        from tasks.reasoning import REASONING_INSTRUCTION, ReasoningItem

        try:
            from tests.conftest import FakeTokenizer
        except ImportError:
            from conftest import FakeTokenizer

        # A synthetic-family item without a template cannot support the
        # Regime C operand swap, so this condition must fail into the sidecar.
        template_less_item = ReasoningItem(
            task_id="templateless_item",
            task_family=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
            source=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
            question_text="Ava has 5 apples and eats 2. How many remain?",
            instruction=REASONING_INSTRUCTION,
            gold_answer=3,
        )
        regime_c_condition = PerturbationCondition(
            "operand_swap_C", SemanticClass.C, Operation.WORD_SUBSTITUTE,
            SelectionPolicy.INFORMATIVE_WORD, Scope.ANSWER_CRITICAL,
            [self._EDIT_BUDGET])

        requests, exclusion_records = _build_requests_for_item_slice(
            [template_less_item], [regime_c_condition], is_word, FakeTokenizer(), seed=1)

        assert [request.is_clean for request in requests] == [True]
        assert len(exclusion_records) == 1
        record = exclusion_records[0]
        assert record["task_id"] == "templateless_item"
        assert record["condition_name"] == "operand_swap_C"
        assert record["edit_budget"] == self._EDIT_BUDGET
        assert record["failure_stage"]
        assert record["failure_reason"]
