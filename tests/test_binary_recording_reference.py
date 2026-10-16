"""Compare binary readers with EEGLAB on the pinned suite's real recordings.

These tests use ``eeglab_backend``, so default CI does not collect them.
Run them on MATLAB with the pinned checkouts::

    uv run --no-sync pytest tests/test_binary_recording_reference.py \\
        --eeglab-backend=matlab \\
        --eeglab-root=src/eegprep/eeglab \\
        --eeglab-suite-root=eeglab_tests -q

Plugin readers skip when ``pop_loadbv``, ``loadcnt``, ``pop_read_erpss``, or
Biosig ``sopen`` is not on the MATLAB path. A Git LFS pointer fails the test
instead of comparing the pointer text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib
import pytest
from pyedflib import highlevel

from eegprep.functions.popfunc.pop_biosig import pop_biosig
from eegprep.functions.popfunc.pop_importegimat import pop_importegimat
from eegprep.functions.popfunc.pop_loadbv import pop_loadbv
from eegprep.functions.popfunc.pop_loadcnt import pop_loadcnt
from eegprep.functions.popfunc.pop_read_erpss import pop_read_erpss
from eegprep.functions.popfunc.pop_readegi import pop_readegi
from eegprep.functions.popfunc.pop_readsegegi import pop_readsegegi
from eegprep.functions.sigprocfunc.loadcnt import loadcnt
from eegprep.functions.sigprocfunc.openbdf import openbdf
from eegprep.functions.sigprocfunc.writegdf import writegdf
from eegprep.functions.sigprocfunc.readbdf import readbdf
from eegprep.functions.sigprocfunc.readegi import readegi
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.backend import skip_without_matlab_plugin
from tests.eeglab_tests.recordings import (
    _events,
    _labels,
    _number,
    assert_same_bdf_header,
    assert_same_cnt,
    assert_same_eeg,
    assert_same_samples,
    require_matlab_reference,
    require_real_recording,
)


_FILES = "unittesting_binary/testfiles"
_WIDE_CHANNELS = 150
_WIDE_RATE = 128
_EDF_QUANTIZATION_ATOL = 2 / 65535
_BVA_CALLS = (
    ("brainvision_genericdataformat_binarymultiplexed_int16.vhdr", 1, list(range(1, 33))),
    ("BVA_withchanlocs.vhdr", None, None),
    ("brainvision_genericdataformat_binarymultiplexed_ieee32.vhdr", None, None),
    ("brainvision_genericdataformat_binarymultiplexed_int16.vhdr", None, None),
    ("brainvision_genericdataformat_binaryvectorized_ieee.vhdr", None, None),
    ("brainvision_genericdataformat_binaryvectorized_int16.vhdr", None, None),
    ("brainvision_recorder_acquisitiondataformat.vhdr", None, None),
    ("EEGLAB_export.vhdr", None, None),
)


def _recording(suite_root: Path, relative: str) -> Path:
    return require_real_recording(suite_root / _FILES / relative)


def _skip_plugin(request, function_name: str, plugin: str) -> None:
    require_matlab_reference(request)
    skip_without_matlab_plugin(request.getfixturevalue("eeglab_matlab_engine"), function_name, plugin)


def _number_options(options: dict) -> list:
    arguments = []
    for key, value in options.items():
        arguments.extend((key, value))
    return arguments


def test_recording_comparison_accepts_matlab_struct_transport() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    python = {
        "data": data,
        "srate": 250,
        "nbchan": 2,
        "pnts": 2,
        "trials": 1,
        "xmin": 0.0,
        "xmax": 1 / 250,
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "event": [{"type": "S  1", "latency": 2}],
    }
    events = np.array([("S  1", 2.0)], dtype=[("type", object), ("latency", float)])
    channels = np.array([("Cz",), ("Pz",)], dtype=[("labels", object)])
    matlab = {
        "data": np.array(data, dtype=np.float32),
        "srate": np.array([[250.0]]),
        "nbchan": np.array([[2.0]]),
        "pnts": np.array([[2.0]]),
        "trials": np.array([[1.0]]),
        "xmin": np.array([[0.0]]),
        "xmax": np.array([[1 / 250]]),
        "chanlocs": channels.reshape(2, 1),
        "event": events.reshape(1, 1),
    }

    assert_same_eeg(python, matlab, label="synthetic")

    matlab["data"] = matlab["data"] + 0.5
    with pytest.raises(AssertionError, match="max abs diff 0.5"):
        assert_same_eeg(python, matlab, label="synthetic")

    assert_same_samples(np.array([1.0 + 1e-8]), np.array([1.0], dtype=np.float32), label="float32")
    with pytest.raises(AssertionError, match="max abs diff"):
        assert_same_samples(np.array([2.0]), np.array([1.0], dtype=np.float32), label="float32")


def test_bdf_header_comparison_accepts_a_character_matrix() -> None:
    shared = {
        "NRec": 2,
        "NS": 2,
        "Dur": 1.0,
        "SPR": np.array([4.0, 4.0]),
        "Cal": np.array([0.1, 0.2]),
        "Off": np.zeros(2),
        "DigMin": np.full(2, -8388608.0),
        "DigMax": np.full(2, 8388607.0),
        "PhysMin": np.array([-1.0, -1.0]),
        "PhysMax": np.array([1.0, 1.0]),
        "SampleRate": np.array([4.0, 4.0]),
        "ChanTyp": "NN",
    }
    actual = {**shared, "Label": ["Cz", "Pz"]}
    expected = {**shared, "Label": np.array([["C", "z", " "], ["P", "z", " "]]), "ChanTyp": np.array(["NN"])}

    assert_same_bdf_header(actual, expected, label="synthetic")


@eeglab_test("unittesting_binary/pop_loadbv/binary_pop_loadbv_wrapperTest.m", "test_test_pop_loadbv")
def test_reference_pop_loadbv_original_recordings(eeglab_backend, eeglab_suite_root, request):
    _skip_plugin(request, "pop_loadbv", "bva-io")
    directory = str(eeglab_suite_root / _FILES / "BVA")
    for filename, srange, channels in _BVA_CALLS:
        _recording(eeglab_suite_root, f"BVA/{filename}")
        if srange is None:
            matlab = eeglab_backend("pop_loadbv", directory, filename)
            python = pop_loadbv(directory, filename)
        else:
            selected = np.asarray(channels, dtype=float)[None, :]
            matlab = eeglab_backend("pop_loadbv", directory, filename, float(srange), selected)
            python = pop_loadbv(directory, filename, srange, channels)
        assert_same_eeg(python, matlab, label=filename)


@eeglab_test("unittesting_binary/pop_loadcnt/binary_pop_loadcnt_wrapperTest.m", "test_test_pop_loadcnt")
@eeglab_test("unittesting_binary/loadcnt/binary_loadcnt_wrapperTest.m", "test_test_loadcnt")
def test_reference_cnt_original_recordings(eeglab_backend, eeglab_suite_root, request):
    _skip_plugin(request, "loadcnt", "neuroscanio")
    _skip_plugin(request, "pop_loadcnt", "neuroscanio")
    int32 = str(_recording(eeglab_suite_root, "neuroscan/TEST32BIT_WITHEVENT.CNT"))
    int16 = str(_recording(eeglab_suite_root, "neuroscan/TEST.CNT"))
    cases = (
        ("int32", int32, {"dataformat": "int32"}),
        ("int32-crop", int32, {"dataformat": "int32", "t1": 0.0, "lddur": 301.0}),
        ("int16", int16, {"dataformat": "int16"}),
    )
    for label, filename, options in cases:
        arguments = _number_options(options)
        matlab = eeglab_backend("loadcnt", filename, *arguments)
        python = loadcnt(filename, **options)
        assert_same_cnt(python, matlab, label=f"loadcnt {label}")
        matlab_eeg = eeglab_backend("pop_loadcnt", filename, *arguments)
        python_eeg = pop_loadcnt(filename, **options)
        assert_same_eeg(python_eeg, matlab_eeg, label=f"pop_loadcnt {label}")

    matlab_keys = eeglab_backend("pop_loadcnt", int32, "keystroke", "on", "dataformat", "int32")
    python_keys = pop_loadcnt(int32, keystroke="on", dataformat="int32")
    assert_same_eeg(python_keys, matlab_keys, label="pop_loadcnt keystroke")


@eeglab_test("unittesting_binary/readegi/binary_readegi_wrapperTest.m", "test_test_readegi")
def test_reference_readegi_original_recording(eeglab_backend, eeglab_suite_root, request):
    require_matlab_reference(request)
    filename = str(_recording(eeglab_suite_root, "EGI/TESTEGI.RAW"))
    selected = np.arange(1, 31, dtype=float)[None, :]
    matlab_header, matlab_data, matlab_events, matlab_categories = eeglab_backend(
        "readegi", filename, selected, nargout=4
    )
    python_header, python_data, python_events, python_categories = readegi(filename, list(range(1, 31)))

    assert_same_samples(python_data, matlab_data, label="readegi data")
    assert_same_samples(python_events, matlab_events, label="readegi events")
    assert_same_samples(np.ravel(python_categories), np.ravel(matlab_categories), label="readegi categories")
    assert python_header["nchan"] == int(np.asarray(matlab_header["nchan"]).reshape(-1)[0])
    assert python_header["samp_rate"] == int(np.asarray(matlab_header["samp_rate"]).reshape(-1)[0])


@eeglab_test("unittesting_binary/pop_readegi/binary_pop_readegi_wrapperTest.m", "test_test_pop_readegi")
def test_reference_pop_readegi_original_recording(eeglab_backend, eeglab_suite_root, request):
    require_matlab_reference(request)
    filename = str(_recording(eeglab_suite_root, "EGI/TESTEGI.RAW"))
    selected = np.arange(1, 31, dtype=float)[None, :]
    matlab = eeglab_backend("pop_readegi", filename, selected)
    python = pop_readegi(filename, list(range(1, 31)))
    assert_same_eeg(python, matlab, label="pop_readegi")


@eeglab_test("unittesting_binary/pop_readsegegi/binary_pop_readsegegi_wrapperTest.m", "test_test_pop_readsegegi")
def test_reference_pop_readsegegi_original_series(eeglab_backend, eeglab_suite_root, request):
    require_matlab_reference(request)
    filename = str(_recording(eeglab_suite_root, "EGI/TEST_001.RAW"))
    _recording(eeglab_suite_root, "EGI/TEST_002.RAW")
    matlab = eeglab_backend("pop_readsegegi", filename)
    python = pop_readsegegi(filename)
    assert_same_eeg(python, matlab, label="pop_readsegegi")


@eeglab_test("unittesting_binary/pop_importegimat/binary_pop_importegimat_wrapperTest.m", "test_test_pop_importegimat")
def test_reference_pop_importegimat_original_export(eeglab_backend, eeglab_suite_root, request):
    _skip_plugin(request, "pop_importegimat", "EGI")
    filename = str(_recording(eeglab_suite_root, "EGI/segmented_matlab.mat"))
    matlab = eeglab_backend("pop_importegimat", filename)
    python = pop_importegimat(filename)
    assert_same_eeg(python, matlab, label="pop_importegimat")


@eeglab_test("unittesting_binary/pop_read_erpss/binary_pop_read_erpss_wrapperTest.m", "test_test_pop_read_erpss")
def test_reference_pop_read_erpss_original_recordings(eeglab_backend, eeglab_suite_root, request):
    _skip_plugin(request, "pop_read_erpss", "ERPSS")
    for name in ("ERPSSTESTCOMP.RAW", "ERPSSCOMPRESSED.RAW"):
        filename = str(_recording(eeglab_suite_root, f"ERPSS/{name}"))
        matlab = eeglab_backend("pop_read_erpss", filename, 500.0)
        python = pop_read_erpss(filename, 500)
        assert_same_eeg(python, matlab, label=name)


@eeglab_test("unittesting_binary/pop_biosig/binary_pop_biosig_wrapperTest.m", "test_test_pop_biosig")
def test_reference_pop_biosig_original_bdf(eeglab_backend, eeglab_suite_root, request):
    _skip_plugin(request, "sopen", "Biosig")
    filename = str(_recording(eeglab_suite_root, "BDF/jo1_3.bdf"))
    blockrange = np.array([[1.0, 10.0]])
    matlab = eeglab_backend("pop_biosig", filename, "blockrange", blockrange)
    python = pop_biosig(filename, blockrange=[1, 10])
    assert_same_eeg(python, matlab, label="pop_biosig")


@eeglab_test("unittesting_binary/pop_biosig/binary_pop_biosig_wrapperTest.m", "test_test_pop_biosig")
def test_reference_pop_biosig_short_blockrange(eeglab_backend, eeglab_suite_root, request):
    """A one-second jo1_3 window is square after Status is removed.

    EEGLAB's biosig2eeglab transposes sread output only when the row count
    differs from the file channel count by more than one. sopen has already
    dropped Status, so 256 samples against 257 file channels stay sample-major.
    ``[0, 1]`` contains no triggers. ``[3, 4]`` contains trigger 34 at sample 96.
    """
    _skip_plugin(request, "sopen", "Biosig")
    filename = str(_recording(eeglab_suite_root, "BDF/jo1_3.bdf"))
    expected_events = {(0.0, 1.0): [], (3.0, 4.0): [(34, 96.0)]}
    for start, stop in ((0.0, 1.0), (3.0, 4.0)):
        matlab = eeglab_backend("pop_biosig", filename, "blockrange", np.array([[start, stop]]))
        python = pop_biosig(filename, blockrange=[start, stop])
        _assert_short_biosig_window(
            python,
            matlab,
            label=f"pop_biosig [{start:g} {stop:g}]",
            expected_events=expected_events[(start, stop)],
        )


@eeglab_test("unittesting_binary/pop_biosig/binary_pop_biosig_wrapperTest.m", "test_test_pop_biosig")
def test_reference_pop_biosig_short_edf_and_gdf(eeglab_backend, tmp_path: Path, request):
    """Compare EDF and GDF windows that EEGLAB can import.

    150 channels at 128 Hz make ``blockrange=[0.5, 1.5]`` and a one-second
    recording 128 samples, which is fewer than the channel count and at least
    100 samples. ``biosig2eeglab.m`` line 198 can index ``EEG.data(end, 1:100)``.
    The sample count and the channel count differ by more than one, so EEGLAB
    transposes the read and both sides stay channel-major. EDF values agree
    within one digital step.
    """
    _skip_plugin(request, "sopen", "Biosig")
    cases = (
        ("window", _WIDE_RATE * 2, [0.5, 1.5], _WIDE_RATE),
        ("short", _WIDE_RATE, None, _WIDE_RATE),
    )
    for name, samples, blockrange, expected_points in cases:
        data = _wide_data(samples)
        for suffix, write in (("edf", _write_wide_edf), ("gdf", _write_wide_gdf)):
            path = tmp_path / f"{name}.{suffix}"
            write(path, data)
            if blockrange is None:
                matlab = eeglab_backend("pop_biosig", str(path))
                python = pop_biosig(path)
            else:
                matlab = eeglab_backend("pop_biosig", str(path), "blockrange", np.array([blockrange], dtype=float))
                python = pop_biosig(path, blockrange=blockrange)
            _assert_quantized_biosig_window(
                python,
                matlab,
                label=f"pop_biosig {suffix} {name}",
                expected_shape=(_WIDE_CHANNELS, expected_points),
            )


def test_reference_readbdf_original_bdf_recording(eeglab_backend, eeglab_suite_root, request):
    require_matlab_reference(request)
    found = request.getfixturevalue("eeglab_matlab_engine").which("openbdf")
    if not (isinstance(found, str) and found.strip()):
        pytest.skip("openbdf is not on the MATLAB path")
    filename = str(_recording(eeglab_suite_root, "BDF/jo1_3.bdf"))
    python_opened = openbdf(filename)
    matlab_opened = eeglab_backend("openbdf", filename)
    assert_same_bdf_header(python_opened["Head"], matlab_opened["Head"], label="openbdf")
    record_count = min(3, int(np.asarray(python_opened["Head"]["NRec"]).reshape(-1)[0]))
    records = np.arange(1, record_count + 1, dtype=float)[None, :]
    python_data, python_signal = readbdf(python_opened, records.reshape(-1).astype(int))
    matlab_data, matlab_signal = eeglab_backend("readbdf", matlab_opened, records, nargout=2)
    assert_same_samples(python_data["Record"], matlab_data["Record"], label="readbdf Record")
    assert_same_samples(np.ravel(python_data["Valid"]), np.ravel(matlab_data["Valid"]), label="readbdf Valid")
    assert_same_samples(np.ravel(python_data["Idx"]), np.ravel(matlab_data["Idx"]), label="readbdf Idx")
    assert_same_samples(python_signal, matlab_signal, label="readbdf signal")


def _same_values(actual, expected) -> bool:
    """Return whether two arrays contain the same values, ignoring storage dtype."""
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if actual_array.shape != expected_array.shape:
        return False
    compare_dtype = expected_array.dtype if expected_array.dtype.kind == "f" else np.dtype(float)
    return np.array_equal(
        actual_array.astype(compare_dtype, copy=False),
        expected_array.astype(compare_dtype, copy=False),
        equal_nan=True,
    )


def _wide_data(samples: int) -> np.ndarray:
    """Return a 150-channel recording whose rows are distinct and inside (-1, 1)."""
    time = np.arange(samples) / _WIDE_RATE
    channel = np.arange(1, _WIDE_CHANNELS + 1)[:, None]
    return 0.5 * np.sin(2 * np.pi * channel * time)


def _write_wide_edf(path: Path, data: np.ndarray) -> None:
    """Write one plain EDF recording with a physical range of -1 to 1."""
    labels = [f"E{index}" for index in range(1, data.shape[0] + 1)]
    headers = highlevel.make_signal_headers(
        labels,
        sample_frequency=_WIDE_RATE,
        physical_min=-1,
        physical_max=1,
    )
    assert highlevel.write_edf(str(path), data, headers, file_type=pyedflib.FILETYPE_EDF)


def _write_wide_gdf(path: Path, data: np.ndarray) -> None:
    """Write one float64 GDF recording."""
    labels = [f"E{index}" for index in range(1, data.shape[0] + 1)]
    writegdf(path, data, _WIDE_RATE, labels=labels)


def _assert_short_biosig_window(
    python: dict,
    matlab: dict,
    *,
    label: str,
    expected_events: list[tuple],
) -> None:
    """Compare one short BioSemi window with EEGLAB's sample-major result.

    Metadata and labels match. The sample matrix is the transpose of EEGLAB's
    because ``biosig2eeglab`` did not orient it. ``expected_events`` is the
    trigger list both sides return.
    """
    if tuple(np.asarray(python["data"]).shape) != (256, 256):
        raise AssertionError(f"{label} shape {np.asarray(python['data']).shape} != (256, 256)")
    matlab_data = np.asarray(matlab["data"])
    python_data = np.asarray(python["data"])
    if python_data.shape != matlab_data.shape:
        raise AssertionError(f"{label} shape {python_data.shape} != {matlab_data.shape}")
    if _same_values(python_data, matlab_data):
        raise AssertionError(f"{label} data matches EEGLAB's untransposed matrix")
    assert_same_samples(python_data.T, matlab_data, label=f"{label} data")
    _assert_same_biosig_metadata(python, matlab, label=label)
    python_events = _events(python.get("event"))
    matlab_events = _events(matlab.get("event"))
    if python_events != expected_events or matlab_events != expected_events:
        raise AssertionError(f"{label} events {python_events} != {matlab_events}, expected {expected_events}")


def _assert_quantized_biosig_window(
    python: dict,
    matlab: dict,
    *,
    label: str,
    expected_shape: tuple[int, int],
) -> None:
    """Compare one EDF or GDF window that both readers keep channel-major."""
    python_data = np.asarray(python["data"], dtype=float)
    matlab_data = np.asarray(matlab["data"], dtype=float)
    if python_data.shape != expected_shape or matlab_data.shape != expected_shape:
        raise AssertionError(f"{label} shape {python_data.shape} != {matlab_data.shape}, expected {expected_shape}")
    if python_data.shape[0] <= python_data.shape[1]:
        raise AssertionError(f"{label} has at least as many samples as channels")
    difference = np.nanmax(np.abs(python_data - matlab_data))
    if difference > _EDF_QUANTIZATION_ATOL:
        raise AssertionError(f"{label} values differ; max abs diff {difference}")
    _assert_same_biosig_metadata(python, matlab, label=label)
    python_events = _events(python.get("event"))
    matlab_events = _events(matlab.get("event"))
    if python_events != matlab_events:
        raise AssertionError(f"{label} events {python_events} != {matlab_events}")


def _assert_same_biosig_metadata(python: dict, matlab: dict, *, label: str) -> None:
    """Compare rate, dimensions, time bounds, and channel labels."""
    for field in ("srate", "nbchan", "pnts", "trials", "xmin", "xmax"):
        actual = _number(python[field])
        expected = _number(matlab[field])
        if actual != expected:
            raise AssertionError(f"{label} {field} {actual} != {expected}")
    actual_labels = _labels(python.get("chanlocs"))
    expected_labels = _labels(matlab.get("chanlocs"))
    if actual_labels != expected_labels:
        raise AssertionError(f"{label} channels {actual_labels} != {expected_labels}")
