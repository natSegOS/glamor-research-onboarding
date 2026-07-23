"""tools/run_pipeline.py: the single entrypoint's profile-loading override
chain, committed-data reuse detection, and generation-path discovery.

Only the pure, hermetic logic is tested here; subprocess-shelling commands
(setup/data-rebuild/generate/analyze/report) are exercised manually per
RUNBOOK.md.
"""

from __future__ import annotations

import importlib.util

from pathlib import Path

import pytest
import yaml

from pipeline import ExperimentConfiguration


def _load_run_pipeline_tool_module():
    tool_path = Path(__file__).resolve().parent.parent / "tools" / "run_pipeline.py"
    module_spec = importlib.util.spec_from_file_location("run_pipeline", tool_path)
    assert module_spec is not None and module_spec.loader is not None
    tool_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(tool_module)
    return tool_module


_MINIMAL_PROFILE = {
    "experiment_config": "configs/pilot.yaml",
    "models": ["qwen_1b5_pilot"],
    "output_root": "results/pilot",
    "analysis_dir": "analysis/pilot",
    "rebuild_data": False,
    "skip_if_complete": True,
}


def _write_profile(path: Path, overrides: dict | None = None) -> Path:
    profile = {**_MINIMAL_PROFILE, **(overrides or {})}
    path.write_text(yaml.safe_dump(profile))
    return path


# ---------------------------------------------------------------------------
# load_run_profile: the override chain (--models beats $TYPO_MODELS beats the
# file) is the whole point of letting one account run one model.
# ---------------------------------------------------------------------------

class TestLoadRunProfileOverrideChain:

    def test_the_file_alone_is_used_when_nothing_overrides_it(self, tmp_path):
        run_pipeline = _load_run_pipeline_tool_module()
        profile_path = _write_profile(tmp_path / "profile.yaml", {"models": ["llama_1b"]})
        profile = run_pipeline.load_run_profile(profile_path)
        assert profile["models"] == ["llama_1b"]

    def test_cli_models_override_beats_the_file(self, tmp_path):
        run_pipeline = _load_run_pipeline_tool_module()
        profile_path = _write_profile(tmp_path / "profile.yaml", {"models": ["llama_1b"]})
        profile = run_pipeline.load_run_profile(profile_path, models_override="qwen_7b, mistral_7b")
        assert profile["models"] == ["qwen_7b", "mistral_7b"]

    def test_env_var_override_beats_the_file(self, tmp_path, monkeypatch):
        run_pipeline = _load_run_pipeline_tool_module()
        profile_path = _write_profile(tmp_path / "profile.yaml", {"models": ["llama_1b"]})
        monkeypatch.setenv(run_pipeline._ENV_MODELS_OVERRIDE, "qwen_7b_awq")
        profile = run_pipeline.load_run_profile(profile_path)
        assert profile["models"] == ["qwen_7b_awq"]

    def test_cli_override_beats_the_env_var(self, tmp_path, monkeypatch):
        """Breaking this makes multi-account parallelization unreliable: a
        stray env var would silently override an explicit --models flag."""
        run_pipeline = _load_run_pipeline_tool_module()
        profile_path = _write_profile(tmp_path / "profile.yaml", {"models": ["llama_1b"]})
        monkeypatch.setenv(run_pipeline._ENV_MODELS_OVERRIDE, "qwen_7b_awq")
        profile = run_pipeline.load_run_profile(profile_path, models_override="mistral_7b")
        assert profile["models"] == ["mistral_7b"]

    def test_a_missing_required_field_raises(self, tmp_path):
        run_pipeline = _load_run_pipeline_tool_module()
        profile_path = tmp_path / "profile.yaml"
        incomplete = {key: value for key, value in _MINIMAL_PROFILE.items() if key != "output_root"}
        profile_path.write_text(yaml.safe_dump(incomplete))
        with pytest.raises(ValueError, match="output_root"):
            run_pipeline.load_run_profile(profile_path)

    def test_an_empty_model_list_raises(self, tmp_path):
        """Breaking this lets 'generate' silently do nothing instead of
        telling the user no model was selected."""
        run_pipeline = _load_run_pipeline_tool_module()
        profile_path = _write_profile(tmp_path / "profile.yaml", {"models": []})
        with pytest.raises(ValueError, match="no models selected"):
            run_pipeline.load_run_profile(profile_path)


# ---------------------------------------------------------------------------
# _committed_data_is_present: the reuse-vs-rebuild gate for the data step.
# ---------------------------------------------------------------------------

class TestCommittedDataIsPresent:

    def _isolate_paths(self, run_pipeline, monkeypatch, tmp_path):
        items_directory = tmp_path / "items"
        wordlists_directory = tmp_path / "wordlists"
        items_directory.mkdir()
        wordlists_directory.mkdir()
        monkeypatch.setattr(run_pipeline, "_ITEMS_DIRECTORY", items_directory)
        monkeypatch.setattr(
            run_pipeline, "_ANNOTATION_PROVENANCE_PATH",
            items_directory / "annotation_PROVENANCE.json")
        monkeypatch.setattr(
            run_pipeline, "_DEFAULT_DICTIONARY_PATH", wordlists_directory / "en_us_pinned.txt")
        return items_directory, wordlists_directory

    def _write_complete_items(self, items_directory, run_pipeline):
        (items_directory / "annotation_PROVENANCE.json").write_text("{}")
        for name in run_pipeline._ITEM_JSONL_NAMES:
            (items_directory / name).write_text("")

    def test_false_when_nothing_exists(self, tmp_path, monkeypatch):
        run_pipeline = _load_run_pipeline_tool_module()
        self._isolate_paths(run_pipeline, monkeypatch, tmp_path)
        assert run_pipeline._committed_data_is_present() is False

    def test_true_when_every_item_file_and_the_dictionary_exist(self, tmp_path, monkeypatch):
        run_pipeline = _load_run_pipeline_tool_module()
        items_directory, wordlists_directory = self._isolate_paths(run_pipeline, monkeypatch, tmp_path)
        self._write_complete_items(items_directory, run_pipeline)
        (wordlists_directory / "en_us_pinned.txt").write_text("")
        assert run_pipeline._committed_data_is_present() is True

    def test_false_when_the_dictionary_is_missing(self, tmp_path, monkeypatch):
        """Breaking this reuses a data/ directory with no dictionary at all,
        crashing generation instead of rebuilding it."""
        run_pipeline = _load_run_pipeline_tool_module()
        items_directory, _wordlists_directory = self._isolate_paths(run_pipeline, monkeypatch, tmp_path)
        self._write_complete_items(items_directory, run_pipeline)
        assert run_pipeline._committed_data_is_present() is False

    def test_false_when_one_item_file_is_missing(self, tmp_path, monkeypatch):
        run_pipeline = _load_run_pipeline_tool_module()
        items_directory, wordlists_directory = self._isolate_paths(run_pipeline, monkeypatch, tmp_path)
        self._write_complete_items(items_directory, run_pipeline)
        (items_directory / run_pipeline._ITEM_JSONL_NAMES[0]).unlink()
        (wordlists_directory / "en_us_pinned.txt").write_text("")
        assert run_pipeline._committed_data_is_present() is False


# ---------------------------------------------------------------------------
# _existing_generation_paths: globs {run_id}*_generations.jsonl under each
# model's subdirectory and warns (without raising) about missing models.
# ---------------------------------------------------------------------------

class TestExistingGenerationPaths:

    def test_raises_when_nothing_is_found(self, tmp_path):
        run_pipeline = _load_run_pipeline_tool_module()
        profile = {**_MINIMAL_PROFILE, "output_root": str(tmp_path), "models": ["llama_1b"]}
        configuration = ExperimentConfiguration(run_id="pilot", seed=1, conditions=[])
        with pytest.raises(FileNotFoundError, match="run 'generate' first"):
            run_pipeline._existing_generation_paths(profile, configuration)

    def test_finds_generations_and_warns_about_missing_models(self, tmp_path, capsys):
        run_pipeline = _load_run_pipeline_tool_module()
        present_model_directory = tmp_path / "llama_1b"
        present_model_directory.mkdir()
        (present_model_directory / "pilot_generations.jsonl").write_text("")

        profile = {
            **_MINIMAL_PROFILE, "output_root": str(tmp_path),
            "models": ["llama_1b", "llama_3b"]}
        configuration = ExperimentConfiguration(run_id="pilot", seed=1, conditions=[])

        paths = run_pipeline._existing_generation_paths(profile, configuration)

        assert [path.parent.name for path in paths] == ["llama_1b"]
        assert "llama_3b" in capsys.readouterr().err
