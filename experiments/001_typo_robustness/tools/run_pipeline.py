"""The single entrypoint: reads configs/run_profile.yaml and drives the whole
pipeline (deps, data prep, generation, analysis, report) with the same
commands on Google Colab and on a local GPU box.

    python tools/run_pipeline.py list-models
    python tools/run_pipeline.py setup
    python tools/run_pipeline.py all

Every subcommand reads configs/run_profile.yaml by default (--profile to
point elsewhere). The profile is the one place a user edits: which
experiment config, which models, where output goes, whether to reuse
committed data, whether to skip already-complete models. See
configs/run_profile.yaml for the full field list and RUNBOOK.md for the
end-to-end walkthrough.

Individual pipeline stages remain plain scripts under tools/ (build_task_
items.py, run_generation.py, run_analysis.py, ...); this tool only
sequences them according to the profile. Nothing here changes what those
scripts do.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# src imports (inference, pipeline) are deferred into the commands that need
# them: `setup` must run on a bare interpreter before any dependency exists.

_DEFAULT_PROFILE_PATH = _REPO_ROOT / "configs" / "run_profile.yaml"
_DEFAULT_DICTIONARY_PATH = _REPO_ROOT / "data" / "wordlists" / "en_us_pinned.txt"
_ITEMS_DIRECTORY = _REPO_ROOT / "data" / "items"
_ITEM_JSONL_NAMES = ("gsm_symbolic.jsonl", "gsm8k.jsonl", "mmlu_pro.jsonl", "mmlu.jsonl")
_ANNOTATION_PROVENANCE_PATH = _ITEMS_DIRECTORY / "annotation_PROVENANCE.json"
_SCOWL_VERSION = "2020.12.07"
_SCOWL_URL = (
    "https://sourceforge.net/projects/wordlist/files/SCOWL/"
    f"{_SCOWL_VERSION}/scowl-{_SCOWL_VERSION}.tar.gz/download"
)
_SPACY_MODEL_NAME = "en_core_web_trf"
_ENV_MODELS_OVERRIDE = "TYPO_MODELS"


# ---------------------------------------------------------------------------
# Profile loading.
# ---------------------------------------------------------------------------

def load_run_profile(profile_path: Path, models_override: str | None = None) -> dict:
    """Load configs/run_profile.yaml and apply the model-list override chain:
    CLI --models beats $TYPO_MODELS beats the file. Every other field comes
    from the file alone (edit it, or point --profile at a copy)."""

    profile = yaml.safe_load(Path(profile_path).read_text())

    override = models_override or os.environ.get(_ENV_MODELS_OVERRIDE)
    if override:
        profile["models"] = [key.strip() for key in override.split(",") if key.strip()]

    required_fields = (
        "experiment_config", "models", "output_root", "analysis_dir",
        "rebuild_data", "skip_if_complete")
    missing = [field for field in required_fields if field not in profile]
    if missing:
        raise ValueError(f"{profile_path} is missing required field(s): {missing}")
    if not profile["models"]:
        raise ValueError(
            "no models selected: set 'models' in the profile, or pass "
            f"--models, or set ${_ENV_MODELS_OVERRIDE}")
    return profile


def _experiment_configuration(profile: dict):
    from pipeline import ExperimentConfiguration
    return ExperimentConfiguration.from_yaml(_REPO_ROOT / profile["experiment_config"])


# ---------------------------------------------------------------------------
# Shared subprocess helper. Streams output live, exactly like the notebook's
# PTY-wrapped run(): subprocess.run with no capture already streams to the
# same terminal/notebook cell, so no PTY plumbing is needed off Colab.
# ---------------------------------------------------------------------------

def _run(command: list, **kwargs) -> None:
    print(f"[run_pipeline] $ {' '.join(str(part) for part in command)}")
    subprocess.run(command, check=True, cwd=_REPO_ROOT, **kwargs)


def _python(*args) -> list:
    return [sys.executable, *args]


# ---------------------------------------------------------------------------
# setup: install dependencies, fast.
# ---------------------------------------------------------------------------

def _running_on_colab() -> bool:
    return Path("/content").is_dir()


def cmd_setup(_profile: dict) -> None:
    _run(_python("-m", "pip", "install", "--upgrade", "-q", "uv"))

    environment = dict(os.environ)
    if _running_on_colab():
        cache_root = Path("/content/drive/MyDrive/glamor/cache")
        environment["PIP_CACHE_DIR"] = str(cache_root / "wheels")
        environment["HF_HOME"] = str(cache_root / "hf")
        for path in (environment["PIP_CACHE_DIR"], environment["HF_HOME"]):
            Path(path).mkdir(parents=True, exist_ok=True)
        print(f"[run_pipeline] Colab detected: caching wheels/weights under {cache_root}")

    _run(["uv", "pip", "install", "--system", "-e", "."], env=environment)
    for requirements_file in ("requirements.txt", "requirements-gpu.txt", "requirements-stats.txt"):
        _run(["uv", "pip", "install", "--system", "-r", requirements_file], env=environment)

    _run(_python("-m", "spacy", "download", _SPACY_MODEL_NAME, "-q"), env=environment)

    if _running_on_colab():
        _run([
            "R", "-e",
            "if (!requireNamespace('lme4', quietly=TRUE)) "
            "install.packages('lme4', repos='https://cloud.r-project.org')",
        ])


# ---------------------------------------------------------------------------
# data: build (or reuse) task items, annotations, and the dictionary.
# ---------------------------------------------------------------------------

def _committed_data_is_present() -> bool:
    items_present = (
        _ANNOTATION_PROVENANCE_PATH.exists()
        and all((_ITEMS_DIRECTORY / name).exists() for name in _ITEM_JSONL_NAMES))
    return items_present and _DEFAULT_DICTIONARY_PATH.exists()


def _download_scowl(destination: Path) -> Path:
    archive_path = destination / "scowl.tar.gz"
    print(f"[run_pipeline] downloading SCOWL {_SCOWL_VERSION} ...")
    urllib.request.urlretrieve(_SCOWL_URL, archive_path)
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination)
    return destination / f"scowl-{_SCOWL_VERSION}" / "final"


def _clone_gsm_symbolic_templates(destination: Path) -> Path:
    clone_path = destination / "ml-gsm-symbolic"
    _run(["git", "clone", "--depth", "1",
          "https://github.com/apple/ml-gsm-symbolic.git", str(clone_path)])
    return clone_path


def cmd_data(profile: dict) -> None:
    configuration = _experiment_configuration(profile)
    force_rebuild = profile["rebuild_data"] or configuration.is_confirmatory

    if not force_rebuild and _committed_data_is_present():
        print(
            "[run_pipeline] reusing committed data/items + "
            f"{_DEFAULT_DICTIONARY_PATH.relative_to(_REPO_ROOT)} "
            "(set rebuild_data: true to force a rebuild)")
        return

    reason = "is_confirmatory" if force_rebuild and configuration.is_confirmatory else (
        "rebuild_data: true" if force_rebuild else "committed data not found")
    print(f"[run_pipeline] rebuilding data ({reason}) ...")

    with tempfile.TemporaryDirectory() as work_dir:
        work_dir = Path(work_dir)
        templates_dir = _clone_gsm_symbolic_templates(work_dir)
        _run(_python(
            "tools/build_task_items.py",
            "--reasoning-items", "100", "--mcq-items", "100",
            "--gsm-config", "p1", "--seed", "1729",
            "--output-directory", str(_ITEMS_DIRECTORY),
            "--gsm-templates-dir", str(templates_dir)))

        scowl_final_dir = _download_scowl(work_dir)
        _run(_python(
            "tools/build_dictionary.py",
            "--scowl-path", str(scowl_final_dir),
            "--scowl-max-size", "60", "--scowl-dialect", "american",
            "--output-directory", str(_DEFAULT_DICTIONARY_PATH.parent)))

    _run(_python(
        "tools/build_annotated_dataset.py",
        "--model-name", _SPACY_MODEL_NAME, "--items-dir", str(_ITEMS_DIRECTORY)))


# ---------------------------------------------------------------------------
# generate: run every selected model, skipping already-complete ones cheaply.
# ---------------------------------------------------------------------------

def _git_commit_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            check=True, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001 (a missing/dirty git tree is survivable here)
        return "unpinned"


def _output_root(profile: dict) -> Path:
    output_root = Path(profile["output_root"])
    return output_root if output_root.is_absolute() else _REPO_ROOT / output_root


def cmd_generate(profile: dict) -> None:
    from pipeline import run_is_complete
    configuration = _experiment_configuration(profile)
    output_root = _output_root(profile)
    output_root.mkdir(parents=True, exist_ok=True)
    git_commit = _git_commit_sha()

    completed, failed = [], []
    for roster_key in profile["models"]:
        model_output_directory = output_root / roster_key
        already_complete = profile["skip_if_complete"] and run_is_complete(
            model_output_directory, configuration)
        if already_complete:
            print(f"[run_pipeline] {roster_key}: already complete, skipping")
            completed.append(roster_key)
            continue

        command = _python(
            "tools/run_generation.py",
            "--config", profile["experiment_config"],
            "--model", roster_key,
            "--output-directory", str(model_output_directory),
            "--dictionary", str(_DEFAULT_DICTIONARY_PATH),
            "--git-commit", git_commit,
            "--skip-if-complete" if profile["skip_if_complete"] else "--no-skip-if-complete")
        try:
            _run(command)
            completed.append(roster_key)
        except subprocess.CalledProcessError as error:
            print(f"[run_pipeline] WARNING: {roster_key} failed ({error}); continuing",
                  file=sys.stderr)
            failed.append(roster_key)

    print(f"[run_pipeline] generate: {len(completed)} completed, {len(failed)} failed"
          + (f" ({failed})" if failed else ""))


# ---------------------------------------------------------------------------
# analyze / report.
# ---------------------------------------------------------------------------

def _existing_generation_paths(profile: dict, configuration) -> list:
    output_root = _output_root(profile)
    paths = sorted(output_root.glob(f"*/{configuration.run_id}*_generations.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no generations found under {output_root} for run_id "
            f"{configuration.run_id!r}; run 'generate' first")
    present_models = {path.parent.name for path in paths}
    missing = set(profile["models"]) - present_models
    if missing:
        print(f"[run_pipeline] WARNING: no generations for {sorted(missing)}", file=sys.stderr)
    return paths


def cmd_analyze(profile: dict) -> None:
    configuration = _experiment_configuration(profile)
    generation_paths = _existing_generation_paths(profile, configuration)
    analysis_dir = _REPO_ROOT / profile["analysis_dir"]
    _run(_python(
        "tools/run_analysis.py",
        "--generations", *(str(path) for path in generation_paths),
        "--output-directory", str(analysis_dir),
        "--config", profile["experiment_config"]))


def cmd_report(profile: dict) -> None:
    configuration = _experiment_configuration(profile)
    generation_paths = _existing_generation_paths(profile, configuration)
    analysis_dir = _REPO_ROOT / profile["analysis_dir"]
    report_path = _output_root(profile) / "report.html"
    _run(_python(
        "tools/build_report.py",
        "--generations", *(str(path) for path in generation_paths),
        "--output", str(report_path),
        "--config", profile["experiment_config"],
        "--analysis-directory", str(analysis_dir)))
    print(f"[run_pipeline] report written to {report_path}")


# ---------------------------------------------------------------------------
# all: the whole pipeline, with an up-front plan summary so nothing runs as
# a surprise before any GPU time is spent.
# ---------------------------------------------------------------------------

def _print_plan_summary(profile: dict) -> None:
    from pipeline import run_is_complete
    configuration = _experiment_configuration(profile)
    output_root = _output_root(profile)
    data_status = "reuse committed" if (
        not (profile["rebuild_data"] or configuration.is_confirmatory)
        and _committed_data_is_present()) else "rebuild"

    print("[run_pipeline] plan:")
    print(f"  experiment_config : {profile['experiment_config']} (run_id={configuration.run_id!r})")
    print(f"  data              : {data_status}")
    print(f"  output_root       : {output_root}")
    print(f"  analysis_dir      : {profile['analysis_dir']}")
    for roster_key in profile["models"]:
        status = "already complete, will skip" if (
            profile["skip_if_complete"]
            and run_is_complete(output_root / roster_key, configuration)) else "will run"
        print(f"  model {roster_key:<18}: {status}")


def cmd_all(profile: dict) -> None:
    _print_plan_summary(profile)
    cmd_data(profile)
    cmd_generate(profile)
    cmd_analyze(profile)
    cmd_report(profile)


# ---------------------------------------------------------------------------
# list-models.
# ---------------------------------------------------------------------------

def cmd_list_models(_profile: dict) -> None:
    from inference import list_models
    for specification in list_models():
        print(f"{specification.roster_key:<18} {specification.huggingface_identifier:<48} "
              f"{specification.precision}")


_COMMANDS = {
    "setup": cmd_setup,
    "data": cmd_data,
    "generate": cmd_generate,
    "analyze": cmd_analyze,
    "report": cmd_report,
    "all": cmd_all,
    "list-models": cmd_list_models,
}


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("--profile", type=Path, default=_DEFAULT_PROFILE_PATH,
                         help=f"path to the run profile (default: {_DEFAULT_PROFILE_PATH})")
    parser.add_argument("--models", default=None,
                         help="comma-separated roster keys, overriding the profile's "
                              f"'models' list (and ${_ENV_MODELS_OVERRIDE})")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    arguments = parse_arguments(argv)
    if arguments.command == "list-models":
        cmd_list_models({})
        return
    profile = load_run_profile(arguments.profile, models_override=arguments.models)
    _COMMANDS[arguments.command](profile)


if __name__ == "__main__":
    main()
