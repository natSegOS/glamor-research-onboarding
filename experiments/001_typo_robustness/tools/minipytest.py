"""A small offline stand-in for the slice of pytest these tests use: fixtures,
@pytest.mark.parametrize, and pytest.raises. It exists because the sandbox has
no network access to install pytest; on any normal machine, `pip install pytest`
and run `pytest` instead — the test files are ordinary pytest and need no
changes.

Supported:
  - test functions discovered by name (test_*) in tests/test_*.py
  - fixtures defined in tests/conftest.py via @pytest.fixture (function scope),
    resolved by parameter name, including fixtures that depend on fixtures
  - @pytest.mark.parametrize("a,b", [...]) including stacked parametrize
  - pytest.raises(ExceptionType) as a context manager
  - math/return-value assertions via plain assert

Unsupported (not used by these tests): yield fixtures, session scope, plugins,
parametrized fixtures, markers other than parametrize.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback

from pathlib import Path


# ---------------------------------------------------------------------------
# A minimal `pytest` module injected into sys.modules before tests import it.
# ---------------------------------------------------------------------------

class _FixtureMarker:
    def __init__(self, function):
        self.function = function
        self.__wrapped__ = function
        self.__name__ = function.__name__


def _fixture(function=None, **_kwargs):
    if function is None:
        return lambda f: _FixtureMarker(f)
    return _FixtureMarker(function)


class _ParametrizeMarker:
    def __init__(self, argument_names, argument_values):
        self.argument_names = [name.strip() for name in argument_names.split(",")]
        self.argument_values = argument_values


def _parametrize(argument_names, argument_values):
    def decorator(function):
        markers = getattr(function, "_parametrize_markers", [])
        markers = markers + [_ParametrizeMarker(argument_names, argument_values)]
        function._parametrize_markers = markers
        return function
    return decorator


class _RaisesContext:
    def __init__(self, expected_exception):
        self.expected_exception = expected_exception
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, _traceback):
        if exc_type is None:
            raise AssertionError(
                f"DID NOT RAISE {self.expected_exception.__name__}")
        if not issubclass(exc_type, self.expected_exception):
            return False
        self.value = exc_value
        return True


def _raises(expected_exception):
    return _RaisesContext(expected_exception)


def _approx(expected, abs_tol=1e-9, rel_tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - expected) <= max(abs_tol, rel_tol * abs(expected))
    return _Approx()


def _build_pytest_module():
    import types
    module = types.ModuleType("pytest")
    module.fixture = _fixture
    marks = types.SimpleNamespace(parametrize=_parametrize)
    module.mark = marks
    module.raises = _raises
    module.approx = _approx
    return module


sys.modules["pytest"] = _build_pytest_module()


# ---------------------------------------------------------------------------
# Discovery and fixture resolution.
# ---------------------------------------------------------------------------

def _load_module_from_path(path: Path):
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _collect_fixtures(*modules) -> dict:
    fixtures: dict = {}
    for module in modules:
        for name, value in vars(module).items():
            if isinstance(value, _FixtureMarker):
                fixtures[name] = value.function
    return fixtures


def _resolve_fixture(name, fixtures, cache):
    if name in cache:
        return cache[name]

    # Built-in fixtures the shim provides directly.
    if name == "tmp_path":
        import tempfile
        value = Path(tempfile.mkdtemp())
        cache[name] = value
        return value

    if name not in fixtures:
        raise KeyError(f"unknown fixture {name!r}")
    fixture_function = fixtures[name]
    arguments = {
        parameter: _resolve_fixture(parameter, fixtures, cache)
        for parameter in inspect.signature(fixture_function).parameters
    }
    value = fixture_function(**arguments)
    cache[name] = value
    return value


def _expand_parametrizations(test_function):
    """Yield (id_suffix, fixed_kwargs) for each parametrized combination, or a
    single empty combination if the function is not parametrized."""
    markers = getattr(test_function, "_parametrize_markers", [])
    if not markers:
        yield "", {}
        return

    combinations = [({}, "")]
    for marker in markers:
        expanded = []
        for fixed_kwargs, id_suffix in combinations:
            for value_row in marker.argument_values:
                row = value_row if isinstance(value_row, tuple) else (value_row,)
                new_kwargs = dict(fixed_kwargs)
                new_kwargs.update(dict(zip(marker.argument_names, row)))
                new_id = id_suffix + "[" + "-".join(str(v) for v in row) + "]"
                expanded.append((new_kwargs, new_id))
        combinations = expanded

    for fixed_kwargs, id_suffix in combinations:
        yield id_suffix, fixed_kwargs


def run_all_tests(tests_directory: Path) -> int:
    tests_directory = Path(tests_directory)

    conftest_path = tests_directory / "conftest.py"
    conftest_module = _load_module_from_path(conftest_path) if conftest_path.exists() else None

    passed = failed = 0
    failures: list = []

    for test_path in sorted(tests_directory.glob("test_*.py")):
        module = _load_module_from_path(test_path)
        fixtures = _collect_fixtures(*([conftest_module] if conftest_module else []), module)

        test_functions = [
            (name, value) for name, value in vars(module).items()
            if name.startswith("test_") and callable(value)
        ]

        for test_name, test_function in test_functions:
            parameters = inspect.signature(test_function).parameters

            for id_suffix, fixed_kwargs in _expand_parametrizations(test_function):
                fixture_cache: dict = {}
                call_kwargs = dict(fixed_kwargs)
                try:
                    for parameter_name in parameters:
                        if parameter_name not in call_kwargs:
                            call_kwargs[parameter_name] = _resolve_fixture(
                                parameter_name, fixtures, fixture_cache)

                    test_function(**call_kwargs)
                    passed += 1
                except Exception:
                    failed += 1
                    failures.append(
                        (f"{test_path.name}::{test_name}{id_suffix}", traceback.format_exc()))

    print(f"\n{'='*60}")
    print(f"PASSED: {passed}    FAILED: {failed}")
    print(f"{'='*60}")
    for test_identifier, error_text in failures:
        print(f"\nFAILED {test_identifier}")
        print(error_text)

    return 1 if failed else 0


if __name__ == "__main__":
    tests_directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests"
    sys.exit(run_all_tests(tests_directory))
