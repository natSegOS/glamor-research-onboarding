"""Adversarial tests for the generation runner.

Covers: deterministic row-ID stability and sensitivity, idempotent resume,
schema completeness, chat-template application, enum coercion in state vectors,
and varied request types.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from enums import SemanticClass, Operation, SelectionPolicy, Scope, TaskFamily, Precision
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


# ---------------------------------------------------------------------------
# Deterministic row ID
# ---------------------------------------------------------------------------

def test_deterministic_row_id_is_stable():
    """Row ID is independent of dict key order."""
    vector = {"semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
              "selection_policy": SelectionPolicy.UNIFORM, "scope": Scope.ANYWHERE, "edit_budget": 2}
    first = deterministic_row_id("rev1", "t1", vector, 7, False)
    second = deterministic_row_id("rev1", "t1", dict(reversed(list(vector.items()))), 7, False)
    assert first == second


def test_row_id_changes_with_model_revision():
    vector = {"semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
              "selection_policy": SelectionPolicy.UNIFORM, "scope": Scope.ANYWHERE, "edit_budget": 2}
    base = deterministic_row_id("rev1", "t1", vector, 7, False)
    assert base != deterministic_row_id("rev2", "t1", vector, 7, False)


def test_row_id_changes_with_task_id():
    vector = {"semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
              "selection_policy": SelectionPolicy.UNIFORM, "scope": Scope.ANYWHERE, "edit_budget": 2}
    base = deterministic_row_id("rev1", "t1", vector, 7, False)
    assert base != deterministic_row_id("rev1", "t2", vector, 7, False)


def test_row_id_changes_with_seed():
    vector = {"semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
              "selection_policy": SelectionPolicy.UNIFORM, "scope": Scope.ANYWHERE, "edit_budget": 2}
    base = deterministic_row_id("rev1", "t1", vector, 7, False)
    assert base != deterministic_row_id("rev1", "t1", vector, 8, False)


def test_row_id_changes_with_is_clean():
    vector = {"semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
              "selection_policy": SelectionPolicy.UNIFORM, "scope": Scope.ANYWHERE, "edit_budget": 2}
    dirty = deterministic_row_id("rev1", "t1", vector, 7, False)
    clean = deterministic_row_id("rev1", "t1", vector, 7, True)
    assert dirty != clean


def test_row_id_changes_with_vector_value():
    """Changing a single value in the state vector must change the ID."""
    v1 = {"semantic_class": SemanticClass.A, "operation": Operation.SUBSTITUTE,
          "selection_policy": SelectionPolicy.UNIFORM, "scope": Scope.ANYWHERE, "edit_budget": 2}
    v2 = {**v1, "edit_budget": 3}
    assert deterministic_row_id("rev1", "t1", v1, 7, False) != deterministic_row_id("rev1", "t1", v2, 7, False)


def test_row_id_is_string():
    vector = {"semantic_class": SemanticClass.CLEAN, "operation": Operation.NONE,
              "selection_policy": SelectionPolicy.NONE, "scope": Scope.ANYWHERE, "edit_budget": 0}
    row_id = deterministic_row_id("rev1", "t1", vector, 1, True)
    assert isinstance(row_id, str)
    assert len(row_id) > 0


# ---------------------------------------------------------------------------
# run_shard — schema and idempotence
# ---------------------------------------------------------------------------

def test_run_shard_writes_complete_schema(tmp_path):
    requests = [_make_request("t1"), _make_request("t2")]
    manifest = ShardManifest(tmp_path / "manifest.json")
    output = tmp_path / "out.jsonl"

    written = run_shard("shard1", requests, DeterministicDummyEngine(), output, manifest,
                        model_id="m", model_revision="rev1")
    assert written == 2

    rows = load_generation_rows([output])
    required_fields = {"row_id", "model_revision", "task_id", "is_clean",
                       "expected_answer", "parsed_answer", "is_correct",
                       "parse_status", "decoding", "r_edit_budget", "schema"}
    for row in rows:
        assert required_fields <= set(row)


def test_run_shard_row_count_equals_request_count(tmp_path):
    for n in [1, 3, 5]:
        out = tmp_path / f"out_{n}.jsonl"
        m = ShardManifest(tmp_path / f"m_{n}.json")
        requests = [_make_request(f"t{i}") for i in range(n)]
        written = run_shard(f"s{n}", requests, DeterministicDummyEngine(), out, m,
                            model_id="m", model_revision="rev")
        assert written == n
        assert len(load_generation_rows([out])) == n


def test_run_shard_is_idempotent(tmp_path):
    requests = [_make_request("t1"), _make_request("t2")]
    output = tmp_path / "out.jsonl"

    first_manifest = ShardManifest(tmp_path / "m.json")
    run_shard("shard1", requests, DeterministicDummyEngine(), output, first_manifest,
              model_id="m", model_revision="rev1")

    second_manifest = ShardManifest(tmp_path / "m2.json")
    written_again = run_shard("shard1", requests, DeterministicDummyEngine(), output,
                              second_manifest, model_id="m", model_revision="rev1")
    assert written_again == 0
    assert len(load_generation_rows([output])) == 2


def test_completed_shard_is_skipped(tmp_path):
    manifest = ShardManifest(tmp_path / "m.json")
    manifest.mark_shard_complete("shard1")
    written = run_shard("shard1", [_make_request("t1")], DeterministicDummyEngine(),
                        tmp_path / "out.jsonl", manifest, model_id="m", model_revision="rev1")
    assert written == 0


def test_run_shard_model_revision_in_rows(tmp_path):
    output = tmp_path / "out.jsonl"
    manifest = ShardManifest(tmp_path / "m.json")
    run_shard("s1", [_make_request("t1")], DeterministicDummyEngine(), output, manifest,
              model_id="m", model_revision="pinned-sha")
    rows = load_generation_rows([output])
    assert rows[0]["model_revision"] == "pinned-sha"


def test_run_shard_perturbed_request_has_state_vector_fields(tmp_path):
    output = tmp_path / "out.jsonl"
    manifest = ShardManifest(tmp_path / "m.json")
    req = _make_request("t1", is_clean=False, edit_budget=2)
    run_shard("s1", [req], DeterministicDummyEngine(), output, manifest,
              model_id="m", model_revision="rev1")
    rows = load_generation_rows([output])
    row = rows[0]
    assert row["r_edit_budget"] == 2
    assert not row["is_clean"]


# ---------------------------------------------------------------------------
# Chat template application
# ---------------------------------------------------------------------------

class _ChatTemplateEngine:
    revision = "chat-test"

    def apply_chat_template(self, user_message):
        return f"<|user|>{user_message}<|assistant|>"

    def generate(self, prompts, max_new_tokens):
        assert all(prompt.startswith("<|user|>") for prompt in prompts)
        return ["#### 0" for _ in prompts]


def test_runner_applies_chat_template(tmp_path):
    manifest = ShardManifest(tmp_path / "m.json")
    written = run_shard("shard_ct", [_make_request("t1")], _ChatTemplateEngine(),
                        tmp_path / "out.jsonl", manifest, model_id="m", model_revision="rev")
    assert written == 1


# ---------------------------------------------------------------------------
# Enum coercion in state vectors (plain strings must serialize correctly)
# ---------------------------------------------------------------------------

def test_state_vector_with_plain_strings_still_serializable(tmp_path):
    """A state vector containing (str, Enum) members must serialize to JSON cleanly."""
    vector = {
        "semantic_class": SemanticClass.A,
        "operation": Operation.SUBSTITUTE,
        "selection_policy": SelectionPolicy.KEYBOARD_NEIGHBOR,
        "scope": Scope.ANYWHERE,
        "edit_budget": 2,
    }
    serialized = json.dumps(vector)
    reloaded = json.loads(serialized)
    assert reloaded["semantic_class"] == "A"
    assert reloaded["operation"] == "substitute"
    assert reloaded["edit_budget"] == 2


def test_load_generation_rows_from_multiple_files(tmp_path):
    """load_generation_rows must merge rows from all provided files."""
    out1 = tmp_path / "out1.jsonl"
    out2 = tmp_path / "out2.jsonl"
    m1 = ShardManifest(tmp_path / "m1.json")
    m2 = ShardManifest(tmp_path / "m2.json")
    run_shard("s1", [_make_request("t1")], DeterministicDummyEngine(), out1, m1,
              model_id="m", model_revision="rev")
    run_shard("s2", [_make_request("t2"), _make_request("t3")],
              DeterministicDummyEngine(), out2, m2, model_id="m", model_revision="rev")
    all_rows = load_generation_rows([out1, out2])
    assert len(all_rows) == 3
