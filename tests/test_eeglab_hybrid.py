"""Execution and transport contracts for MATLAB-validated Python tests."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.eeglab_tests.backend import call_matlab, call_python


pytest_plugins = ("pytester",)


def test_python_backend_resolves_functions_at_call_time():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
    np.testing.assert_array_equal(call_python("rmbase", data), [[-1, 0, 1], [-4, 0, 4]])
    with pytest.raises(AttributeError, match="eegprep_test_transport"):
        call_python("eegprep_test_transport", "fixture")


def test_python_backend_selects_requested_outputs():
    arguments = ([0.0, 0.5], [], 100.0, [0.0, 1.0])
    np.testing.assert_array_equal(call_python("eeg_lat2point", *arguments), [1.0, 51.0])
    points, flag = call_python("eeg_lat2point", *arguments, nargout=2)
    np.testing.assert_array_equal(points, [1.0, 51.0])
    assert not flag
    with pytest.raises(AssertionError, match="3 outputs"):
        call_python("eeg_lat2point", *arguments, nargout=3)


def _isolated_suite(pytester, source):
    root = str(Path(__file__).resolve().parents[1])
    pytester.makeconftest(f'import sys\nsys.path.insert(0, {root!r})\npytest_plugins = ("tests.conftest",)')
    pytester.makepyfile(source)


def test_contracts_require_explicit_backend_selection(pytester):
    _isolated_suite(pytester, "def test_contract(eeglab_backend):\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(deselected=1)


def test_matlab_mode_rejects_unconverted_provenance_test(pytester):
    _isolated_suite(
        pytester,
        """
from tests.eeglab_tests import eeglab_test
@eeglab_test('test_reference.m', 'test_direct')
def test_direct():
    raise AssertionError('Python must not execute in the MATLAB lane')
""",
    )
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*still call Python directly*"])


def test_required_matlab_mode_fails_without_reference(pytester, tmp_path):
    _isolated_suite(pytester, "def test_contract(eeglab_backend):\n    eeglab_backend('eeglab_missing_function')")
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab", f"--eeglab-root={tmp_path / 'absent'}")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*MATLAB contracts require --eeglab-root*"])


def test_reference_datasets_require_an_explicit_checkout(pytester, monkeypatch):
    monkeypatch.delenv("EEGPREP_EEGLAB_ROOT", raising=False)
    _isolated_suite(pytester, "def test_source_data(eeglab_suite_root):\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*Reference datasets require --eeglab-suite-root or --eeglab-root*"])


def test_reference_datasets_reject_missing_checkout(pytester, tmp_path):
    _isolated_suite(pytester, "def test_source_data(eeglab_suite_root):\n    raise AssertionError('must not run')")
    result = pytester.runpytest_subprocess(f"--eeglab-suite-root={tmp_path / 'absent'}")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*EEGLAB test checkout does not exist*"])


def test_matlab_gate_honors_explicit_test_selection(pytester):
    _isolated_suite(
        pytester,
        """
from tests.eeglab_tests import eeglab_test
@eeglab_test('test_reference.m', 'test_direct')
def test_direct():
    raise AssertionError('must not run')
def test_converted(eeglab_backend):
    raise AssertionError('collect only')
""",
    )
    result = pytester.runpytest_subprocess("--eeglab-backend=matlab", "-k", "converted", "--collect-only")
    assert result.ret == pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*1/2 tests collected (1 deselected)*"])


def test_python_mode_collects_then_fails_missing_capability(pytester):
    _isolated_suite(pytester, "def test_contract(eeglab_backend):\n    eeglab_backend('eeglab_missing_function')")
    result = pytester.runpytest_subprocess("--eeglab-backend=python")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*AttributeError*eeglab_missing_function*"])


@pytest.mark.parametrize(
    ("dtype", "matlab_class"),
    [
        (np.float64, "double"),
        (np.float32, "single"),
        (np.int16, "int16"),
        (np.uint32, "uint32"),
        (np.complex64, "single"),
        (np.complex128, "double"),
        (np.bool_, "logical"),
    ],
)
@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2, 3), (1, 2, 3), (0, 3), (0, 0)])
def test_matlab_transport_preserves_dtype_and_dimensions(eeglab_matlab_engine, dtype, matlab_class, shape):
    expected = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        expected += (expected + 1) * 1j
    actual = call_matlab(
        eeglab_matlab_engine, "eegprep_test_transport", "echo", expected, matlab_class, np.array([shape], dtype=float)
    )
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_matlab_generated_struct_cells_logicals_and_complex(eeglab_matlab_engine):
    value = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "fixture")
    np.testing.assert_array_equal(value["numeric"], [[1, 3, 5], [2, 4, 6]])
    assert value["numeric"].dtype == np.float32
    np.testing.assert_array_equal(value["complex"], [[1 + 3j, 2 - 4j]])
    assert value["complex"].dtype == np.complex64
    np.testing.assert_array_equal(value["logical"], [[True, False]])
    assert value["logical"].dtype == np.bool_
    assert value["empty"].shape == (0, 3)
    assert value["empty"].dtype == np.uint16
    assert value["nested"]["label"] == "nested"
    np.testing.assert_array_equal(value["nested"]["column"], [[7], [8]])
    assert value["nested"]["column"].dtype == np.int16
    assert value["cells"].shape == (2, 2)
    assert value["cells"][0, 0].dtype == np.int8
    assert value["cells"][0, 0].item() == 1
    assert value["cells"][0, 1] == "two"
    assert value["cells"][1, 0].shape == (0, 0)
    assert value["cells"][1, 1]["last"].dtype == np.uint32
    assert value["cells"][1, 1]["last"].item() == 4
    returned = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "nested", value)
    assert returned["structs"].dtype.names == ("index",)
    assert returned["cell_structs"].dtype == object


@pytest.mark.parametrize("trials", [1, 3])
def test_matlab_transport_eeg_without_dataset_serializers(eeglab_matlab_engine, trials):
    shape = (2, 5) if trials == 1 else (2, 5, trials)
    events = np.array([("start", 1.0), ("end", float(5 * trials))], dtype=[("type", object), ("latency", object)])
    eeg = {
        "data": np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape, order="F"),
        "nbchan": 2.0,
        "pnts": 5.0,
        "trials": float(trials),
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.04,
        "times": np.arange(5, dtype=float)[None, :] * 10,
        "event": events,
        "urevent": events.copy(),
        "chanlocs": np.array([("Cz",), ("Pz",)], dtype=[("labels", object)]),
        "icaweights": np.empty((0, 0)),
        "etc": {"subject": "transport", "indices": np.array([[1, 5]], dtype=np.int32)},
    }
    actual = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "eeg", eeg)
    # A workflow passes returned EEG structs to the next MATLAB call unchanged.
    actual = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "eeg", actual)
    assert set(actual) == set(eeg)
    assert actual["data"].shape == shape
    assert actual["data"].dtype == np.float32
    np.testing.assert_array_equal(actual["data"], eeg["data"])
    np.testing.assert_array_equal(actual["times"], eeg["times"])
    assert actual["event"].shape == (1, 2)
    assert actual["event"][0, 0]["type"] == "start"
    assert actual["event"][0, 1]["latency"].item() == 5 * trials
    assert actual["urevent"][0, 0]["latency"].item() == 1
    assert actual["icaweights"].shape == (0, 0)
    np.testing.assert_array_equal(actual["etc"]["indices"], [[1, 5]])
    assert actual["etc"]["indices"].dtype == np.int32


def test_matlab_transport_equal_shape_arguments_and_multiple_outputs(eeglab_matlab_engine):
    first = np.array([[1, 2], [3, 4]], dtype=np.int16)
    second = np.array([[5, 6], [7, 8]], dtype=np.float32)
    outputs = call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "pair", first, second, nargout=2)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 2
    for actual, expected in zip(outputs, (first, second), strict=True):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_matlab_transport_paths_and_zero_outputs(eeglab_matlab_engine, tmp_path):
    path = tmp_path / "researcher's data.txt"
    path.write_text("EEGLAB reference path", encoding="utf-8")
    assert call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "path", str(path)) == "EEGLAB reference path"
    assert call_matlab(eeglab_matlab_engine, "eegprep_test_transport", "no_output", nargout=0) is None


def test_matlab_function_errors_are_not_skipped_or_retried_on_python(eeglab_matlab_engine, eeglab_backend):
    # This helper exists only in MATLAB, so a successful call cannot come from Python.
    assert eeglab_matlab_engine.which("eegprep_test_transport")
    assert eeglab_backend("eegprep_test_transport", "fixture")["numeric"].shape == (2, 3)
    matlab_engine: Any = importlib.import_module("matlab.engine")
    with pytest.raises(matlab_engine.MatlabExecutionError, match="Unknown mode"):
        eeglab_backend("eegprep_test_transport", "invalid")


def test_backend_dispatch_executes_real_reference_function(eeglab_backend):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]])
    np.testing.assert_array_equal(eeglab_backend("rmbase", data), [[-1, 0, 1], [-4, 0, 4]])
