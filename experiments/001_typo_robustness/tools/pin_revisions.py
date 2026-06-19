"""Resolve and record the current HEAD commit SHA for every model in the roster.

This is a **one-time pre-registration step** (design/10 §10.5, docs/PROVENANCE.md §4).
The confirmatory gate ``inference.assert_revisions_pinned`` refuses to start a
run while any model's ``revision`` field is still the placeholder ``"PIN_ME"``.
This script resolves the SHA from HuggingFace and writes it to
``configs/pinned_revisions.yaml`` in a format that is copy-pasteable into
``src/inference/roster.py``.

**Gated models** (``meta-llama/*``, ``mistralai/*``) require the HuggingFace
account to have accepted the model license. The script will print a clear error
for any model that cannot be reached and continue with the others.

Usage:

    python tools/pin_revisions.py

    # Write to a different location:
    python tools/pin_revisions.py --output configs/pinned_revisions.yaml

    # Resolve a single roster key (useful for re-pinning after a model update):
    python tools/pin_revisions.py --model llama_8b_awq

Outputs:
    configs/pinned_revisions.yaml    {roster_key: sha, ...} + fetch metadata
    stdout                           copy-pasteable roster.py patch instructions
"""

from __future__ import annotations

import argparse
import sys

from datetime import datetime, timezone
from pathlib import Path

import yaml

# Resolve paths relative to the repo root so the script works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "configs" / "pinned_revisions.yaml"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"path to write the YAML pin file (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="resolve only this roster key; default: resolve all models",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    # Import the roster after fixing the sys.path so the package resolves
    # without `pip install -e .` (the same trick tools/run_tests.py uses).
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from inference import MODEL_ROSTER, resolve_current_revision

    roster_items = list(MODEL_ROSTER.items())
    if arguments.model is not None:
        if arguments.model not in MODEL_ROSTER:
            print(f"ERROR: roster key {arguments.model!r} not found.", file=sys.stderr)
            print(f"Available keys: {list(MODEL_ROSTER)}", file=sys.stderr)
            sys.exit(1)
        roster_items = [(arguments.model, MODEL_ROSTER[arguments.model])]

    resolved: dict[str, str] = {}
    errors: list[str] = []

    print(f"resolving {len(roster_items)} model revision(s) from HuggingFace ...")

    for roster_key, specification in roster_items:
        hf_id = specification.huggingface_identifier
        print(f"  {roster_key}  ({hf_id})", end="  ", flush=True)
        try:
            sha = resolve_current_revision(hf_id)
            resolved[roster_key] = sha
            print(sha[:12] + "...")
        except Exception as fetch_error:
            error_message = str(fetch_error)
            errors.append(f"{roster_key}: {error_message}")
            print(f"FAILED — {error_message}")

    # Write the YAML output even if some models failed; the file records what
    # was successfully resolved so partial progress is not lost.
    output_data = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Paste the SHA values into the `revision` fields of "
            "src/inference/roster.py to satisfy "
            "assert_revisions_pinned."
        ),
        "revisions": resolved,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(yaml.dump(output_data, default_flow_style=False, allow_unicode=True))
    print(f"\nwrote {len(resolved)} resolved SHA(s) to {arguments.output}")

    if resolved:
        print("\n--- copy-pasteable roster.py patch ---")
        for roster_key, sha in resolved.items():
            print(f"  {roster_key!r}: revision={sha!r}")
        print("--------------------------------------")

    if errors:
        print(f"\n{len(errors)} model(s) could not be resolved:")
        for error in errors:
            print(f"  {error}")
        print("\nGated models require HuggingFace access:")
        print("  huggingface-cli login")
        print("  huggingface-cli download <model_id> --revision main --quiet")
        sys.exit(1)


if __name__ == "__main__":
    main()
