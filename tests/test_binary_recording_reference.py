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
import pytest

from eegprep.functions.popfunc.pop_biosig import pop_biosig
from eegprep.functions.popfunc.pop_importegimat import pop_importegimat
from eegprep.functions.popfunc.pop_loadbv import pop_loadbv
from eegprep.functions.popfunc.pop_loadcnt import pop_loadcnt
from eegprep.functions.popfunc.pop_read_erpss import pop_read_erpss
from eegprep.functions.popfunc.pop_readegi import pop_readegi
from eegprep.functions.popfunc.pop_readsegegi import pop_readsegegi
from eegprep.functions.sigprocfunc.loadcnt import loadcnt
from eegprep.functions.sigprocfunc.openbdf import openbdf
from eegprep.functions.sigprocfunc.readbdf import readbdf
from eegprep.functions.sigprocfunc.readegi import readegi
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.backend import skip_without_matlab_plugin
from tests.eeglab_tests.recordings import (
    assert_same_bdf_header,
    assert_same_cnt,
    assert_same_eeg,
    assert_same_samples,
    require_matlab_reference,
    require_real_recording,
)


_FILES = "unittesting_binary/testfiles"
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
