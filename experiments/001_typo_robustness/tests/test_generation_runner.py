"""The idempotent, resumable generation runner (src/pipeline/runner.py):
deterministic row-ID stability and sensitivity, schema completeness,
idempotent resume, crash recovery (per-row streaming survives a mid-shard
crash), chat-template application, and multi-file row loading.
"""

from __future__ import annotations

import json

import pytest

from enums import SemanticClass, Operation, SelectionPolicy, Scope, TaskFamily
from pipeline.runner import (
    DeterministicDummyEngine,
    GenerationRequest,
    ShardManifest,
    deterministic_row_id,
    load_generation_rows,
    run_shard,
)


def _make_request(task_id, is_clean=True, edit_budget=0,
                  task_family=TaskFamily.GSM_SYMBOLIC_SYNTHETIC):
    return GenerationRequest(
        task_id=task_id,
        task_family=task_family,
        prompt=f"Solve: item {task_id}\n#### 0",
        gold_answer=0,
        is_clean=is_clean,
        perturbation_state_vector={
            "semantic_class": SemanticClass.CLEAN if is_clean else SemanticClass.A,
            "operation": Operation.NONE if is_clean else Operation.SUBSTITUTE,
            "selection_policy": SelectionPolicy.NONE if is_clean else SelectionPolicy.KEYBOARD_NEIGHBOR,
            "scope": Scope.ANYWHERE,
            "edit_budget": edit_budget,
        },
        seed=1,
    )


_BASE_STATE_VECTOR = {
    "semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
    "selection_policy": SelectionPolicy.KEYBOARD_NEIGHBOR, "scope": Scope.ANYWHERE, "edit_budget": 2,
}


# ---------------------------------------------------------------------------
# Deterministic row ID: dict-order independence, and sensitivity to every
# component that defines a unique generation.
# ---------------------------------------------------------------------------

class TestDeterministicRowId:

    def test_is_independent_of_state_vector_key_order(self):
        first = deterministic_row_id("rev1", "t1", _BASE_STATE_VECTOR, 7, False)
        reordered = dict(reversed(list(_BASE_STATE_VECTOR.items())))
        assert first == deterministic_row_id("rev1", "t1", reordered, 7, False)

    def test_is_a_nonempty_string(self):
        row_id = deterministic_row_id("rev1", "t1", _BASE_STATE_VECTOR, 1, True)
        assert isinstance(row_id, str) and len(row_id) > 0

    @pytest.mark.parametrize("make_variant", [
        lambda: ("rev2", "t1", _BASE_STATE_VECTOR, 7, False),                       # model_revision
        lambda: ("rev1", "t2", _BASE_STATE_VECTOR, 7, False),                       # task_id
        lambda: ("rev1", "t1", _BASE_STATE_VECTOR, 8, False),                       # seed
        lambda: ("rev1", "t1", _BASE_STATE_VECTOR, 7, True),                        # is_clean
        lambda: ("rev1", "t1", {**_BASE_STATE_VECTOR, "edit_budget": 3}, 7, False),  # a vector value
    ], ids=["model_revision", "task_id", "seed", "is_clean", "state_vector_value"])
    def test_changing_any_single_component_changes_the_id(self, make_variant):
        base = deterministic_row_id("rev1", "t1", _BASE_STATE_VECTOR, 7, False)
        assert base != deterministic_row_id(*make_variant())


# ---------------------------------------------------------------------------
# run_shard: schema completeness and idempotence.
# ---------------------------------------------------------------------------

class TestRunShardSchemaAndIdempotence:

    def test_every_row_has_the_complete_required_schema(self, tmp_path):
        requests = [_make_request("t1"), _make_request("t2")]
        manifest = ShardManifest(tmp_path / "manifest.json")
        output = tmp_path / "out.jsonl"

        written = run_shard("shard1", requests, DeterministicDummyEngine(), output, manifest,
                            model_id="m", model_revision="rev1")
        assert written == 2

        required_fields = {"row_id", "model_revision", "task_id", "is_clean",
                           "expected_answer", "parsed_answer", "is_correct",
                           "parse_status", "decoding", "r_edit_budget", "schema"}
        for row in load_generation_rows([output]):
            assert required_fields <= set(row)

    @pytest.mark.parametrize("request_count", [1, 3, 5])
    def test_row_count_equals_request_count(self, tmp_path, request_count):
        output_path = tmp_path / f"out_{request_count}.jsonl"
        manifest = ShardManifest(tmp_path / f"m_{request_count}.json")
        requests = [_make_request(f"t{i}") for i in range(request_count)]
        written = run_shard(f"s{request_count}", requests, DeterministicDummyEngine(),
                            output_path, manifest, model_id="m", model_revision="rev")
        assert written == request_count
        assert len(load_generation_rows([output_path])) == request_count

    def test_rerunning_an_already_written_shard_writes_nothing_new(self, tmp_path):
        requests = [_make_request("t1"), _make_request("t2")]
        output = tmp_path / "out.jsonl"

        run_shard("shard1", requests, DeterministicDummyEngine(), output,
                  ShardManifest(tmp_path / "m.json"), model_id="m", model_revision="rev1")
        written_again = run_shard("shard1", requests, DeterministicDummyEngine(), output,
                                  ShardManifest(tmp_path / "m2.json"), model_id="m", model_revision="rev1")

        assert written_again == 0
        assert len(load_generation_rows([output])) == 2

    def test_a_manifest_marked_complete_is_skipped_entirely(self, tmp_path):
        manifest = ShardManifest(tmp_path / "m.json")
        manifest.mark_shard_complete("shard1")
        written = run_shard("shard1", [_make_request("t1")], DeterministicDummyEngine(),
                            tmp_path / "out.jsonl", manifest, model_id="m", model_revision="rev1")
        assert written == 0

    def test_rows_carry_the_pinned_model_revision(self, tmp_path):
        output = tmp_path / "out.jsonl"
        run_shard("s1", [_make_request("t1")], DeterministicDummyEngine(), output,
                  ShardManifest(tmp_path / "m.json"), model_id="m", model_revision="pinned-sha")
        assert load_generation_rows([output])[0]["model_revision"] == "pinned-sha"

    def test_perturbed_request_rows_carry_state_vector_fields(self, tmp_path):
        output = tmp_path / "out.jsonl"
        request = _make_request("t1", is_clean=False, edit_budget=2)
        run_shard("s1", [request], DeterministicDummyEngine(), output,
                  ShardManifest(tmp_path / "m.json"), model_id="m", model_revision="rev1")
        row = load_generation_rows([output])[0]
        assert row["r_edit_budget"] == 2
        assert not row["is_clean"]


# ---------------------------------------------------------------------------
# Chat template application.
# ---------------------------------------------------------------------------

class _ChatTemplateEngine:
    revision = "chat-test"

    def apply_chat_template(self, user_message):
        return f"<|user|>{user_message}<|assistant|>"

    def generate_streaming(self, prompts, max_new_tokens):
        assert all(prompt.startswith("<|user|>") for prompt in prompts)
        for index, _prompt in enumerate(prompts):
            yield index, "#### 0"


def test_runner_applies_the_engines_chat_template(tmp_path):
    written = run_shard("shard_ct", [_make_request("t1")], _ChatTemplateEngine(),
                        tmp_path / "out.jsonl", ShardManifest(tmp_path / "m.json"),
                        model_id="m", model_revision="rev")
    assert written == 1


def test_state_vector_with_enum_members_serializes_to_plain_strings():
    vector = {**_BASE_STATE_VECTOR}
    reloaded = json.loads(json.dumps(vector))
    assert reloaded["semantic_class"] == "A"
    assert reloaded["operation"] == "substitute"
    assert reloaded["edit_budget"] == 2


# ---------------------------------------------------------------------------
# Per-row streaming: a mid-shard crash loses only the in-flight requests,
# and resuming completes only what's missing.
# ---------------------------------------------------------------------------

class _CrashAfterNEngine:
    """Streams real results for the first N requests, then raises — simulating
    an engine dying mid-shard (a real incident this test locks in against)."""
    revision = "crash-test"

    def __init__(self, crash_after: int):
        self.crash_after = crash_after

    def generate_streaming(self, prompts, max_new_tokens):
        for index, _prompt in enumerate(prompts):
            if index == self.crash_after:
                raise RuntimeError("simulated engine death")
            yield index, "#### 0"


class TestCrashRecovery:

    def test_rows_finished_before_a_crash_are_already_on_disk(self, tmp_path):
        requests = [_make_request(f"t{i}") for i in range(5)]
        manifest = ShardManifest(tmp_path / "manifest.json")
        output = tmp_path / "out.jsonl"

        with pytest.raises(RuntimeError):
            run_shard("shard1", requests, _CrashAfterNEngine(crash_after=3), output,
                      manifest, model_id="m", model_revision="rev1")

        rows = load_generation_rows([output])
        assert len(rows) == 3
        assert not manifest.is_shard_complete("shard1")

    def test_resuming_completes_only_the_rows_the_crash_never_reached(self, tmp_path):
        requests = [_make_request(f"t{i}") for i in range(5)]
        manifest = ShardManifest(tmp_path / "manifest.json")
        output = tmp_path / "out.jsonl"

        with pytest.raises(RuntimeError):
            run_shard("shard1", requests, _CrashAfterNEngine(crash_after=3), output,
                      manifest, model_id="m", model_revision="rev1")

        written_on_resume = run_shard("shard1", requests, DeterministicDummyEngine(),
                                      output, manifest, model_id="m", model_revision="rev1")

        assert written_on_resume == 2
        rows = load_generation_rows([output])
        assert len(rows) == 5
        assert len({row["row_id"] for row in rows}) == 5
        assert manifest.is_shard_complete("shard1")

    def test_progress_callback_fires_once_per_completed_row(self, tmp_path):
        requests = [_make_request(f"t{i}") for i in range(4)]
        calls = []
        run_shard("shard1", requests, DeterministicDummyEngine(), tmp_path / "out.jsonl",
                  ShardManifest(tmp_path / "manifest.json"), model_id="m", model_revision="rev1",
                  progress_callback=calls.append)
        assert calls == [1, 1, 1, 1]


def test_load_generation_rows_merges_rows_from_every_provided_file(tmp_path):
    out1, out2 = tmp_path / "out1.jsonl", tmp_path / "out2.jsonl"
    run_shard("s1", [_make_request("t1")], DeterministicDummyEngine(), out1,
              ShardManifest(tmp_path / "m1.json"), model_id="m", model_revision="rev")
    run_shard("s2", [_make_request("t2"), _make_request("t3")], DeterministicDummyEngine(), out2,
              ShardManifest(tmp_path / "m2.json"), model_id="m", model_revision="rev")
    assert len(load_generation_rows([out1, out2])) == 3
