"""Run the test suite. Prefers real pytest if installed; otherwise falls back to
the offline shim in minipytest.py (the sandbox has no network to install
pytest). On a normal machine just run `pytest`."""

from __future__ import annotations

import subprocess
import sys

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIRECTORY = REPOSITORY_ROOT / "tests"


def _pytest_is_available() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    if _pytest_is_available():
        print("running with real pytest")
        return subprocess.call([sys.executable, "-m", "pytest", str(TESTS_DIRECTORY), "-q"])

    print("pytest unavailable; running offline shim (tools/minipytest.py)")
    # Inject src/ so the package resolves without `pip install -e .` — mirrors
    # what the root conftest.py does for pytest.  Must happen before importing
    # the shim (which in turn imports conftest.py, which imports the package).
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from tools.minipytest import run_all_tests
    return run_all_tests(TESTS_DIRECTORY)


if __name__ == "__main__":
    sys.exit(main())
