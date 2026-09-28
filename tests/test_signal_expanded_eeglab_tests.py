"""Faithful translations of the new native signal-processing contracts."""

from copy import deepcopy

import numpy as np
from numpy.lib.recfunctions import drop_fields
import pytest

from tests.eeglab_tests import assert_matlab_equal, expanded_matlab_test
from tests.eeglab_tests.assertions import matlab_field_concat


NATIVE_SOURCE = "tests/matlab/expanded/test_eegprep_signal_expanded.m"
NATIVE_SHA256 = "fd1cfc765199db5b67a0b6108bf5aa8d122c06f2751f4b93f7c895814f7aefd3"


@pytest.fixture(autouse=True)
def _double_precision(eeglab_backend, eeglab_options_directory):
    eeglab_backend("pop_editoptions", "option_single", 0.0, nargout=0)


@expanded_matlab_test(NATIVE_SOURCE, "test_fir_symmetric_dc_edges", NATIVE_SHA256)
def test_fir_symmetric_dc_edges(eeglab_backend):
    data = np.array([[2, 4], [4, 8], [8, 16], [16, 32]], dtype=float)
    actual = eeglab_backend("fir_filterdcpadded", np.array([[0.25, 0.5, 0.25]]), 1.0, data)
    _assert_close(actual, [[2.5, 5], [4.5, 9], [9, 18], [14, 28]], 1e-12)
    assert actual.shape == data.shape


@expanded_matlab_test(NATIVE_SOURCE, "test_fir_causal_asymmetric_dc_padding", NATIVE_SHA256)
def test_fir_causal_asymmetric_dc_padding(eeglab_backend):
    data = np.array([[2, -2], [4, -4], [8, -8], [16, -16]], dtype=float)
    expected = np.array([[2], [14 / 6], [22 / 6], [44 / 6]]) @ np.array([[1, -1]])
    actual = eeglab_backend("fir_filterdcpadded", np.array([[1, 2, 3]]) / 6, 1.0, data, True, False)
    _assert_close(actual, expected, 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_fir_antisymmetric_linear_ramp", NATIVE_SHA256)
def test_fir_antisymmetric_linear_ramp(eeglab_backend):
    actual = eeglab_backend(
        "fir_filterdcpadded", np.array([[-0.5, 0, 0.5]]), 1.0, np.arange(-2.0, 3.0)[:, None], False, False
    )
    _assert_close(actual, np.array([[-0.5, -1, -1, -1, -0.5]]).T, 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_fir_fft_single_precision", NATIVE_SHA256)
def test_fir_fft_single_precision(eeglab_backend):
    data = np.arange(1, 19, dtype=np.float32).reshape(6, 3, order="F")
    expected = data.copy()
    expected[0] += 0.25
    expected[-1] -= 0.25
    actual = eeglab_backend("fir_filterdcpadded", np.array([[0.25, 0.5, 0.25]]), 1.0, data, False, True)
    assert actual.dtype == np.float32
    _assert_close(actual, expected, np.float32(1e-5))


@expanded_matlab_test(NATIVE_SOURCE, "test_fir_fft_complex_signal", NATIVE_SHA256)
def test_fir_fft_complex_signal(eeglab_backend):
    factors = np.array([[1 + 2j, -3 + 1j]])
    data = np.arange(1.0, 7.0)[:, None] @ factors
    expected = np.array([[1.25, 2, 3, 4, 5, 5.75]]).T @ factors
    actual = eeglab_backend("fir_filterdcpadded", np.array([[0.25, 0.5, 0.25]]), 1.0, data, False, True)
    _assert_close(actual, expected, 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_fir_boundaries_do_not_mix_segments", NATIVE_SHA256)
def test_fir_boundaries_do_not_mix_segments(eeglab_backend):
    data = np.vstack((np.repeat([1.0, 100.0], 9), np.repeat([2.0, -50.0], 9), np.arange(1.0, 19.0)))
    eeg = _signal_eeg(eeglab_backend, data, 100.0)
    eeg["event"] = {"type": "boundary", "latency": np.array([[9.5]]), "duration": np.array([[20.0]])}
    actual = eeglab_backend("firfilt", eeg, np.array([[0.25, 0.5, 0.25]]), 7.0, np.array([[1.0, 2.0]]))
    _assert_close(actual["data"], data, 1e-12)
    assert_matlab_equal(actual["event"], eeg["event"])


@expanded_matlab_test(NATIVE_SOURCE, "test_reref_multiple_references_excludes_auxiliary_channel", NATIVE_SHA256)
def test_reref_multiple_references_excludes_auxiliary_channel(eeglab_backend):
    data = np.arange(1.0, 25.0).reshape(4, 3, 2, order="F")
    expected = data.copy()
    expected[[0, 1, 3]] -= data[[1, 3]].mean(axis=0, keepdims=True)
    actual = eeglab_backend("reref", data, np.array([[2.0, 4.0]]), "exclude", 3.0, "keepref", "on")
    _assert_close(actual, expected, 1e-12)
    _assert_close(actual[2], data[2], 0)
    _assert_close(actual[[1, 3]].mean(axis=0, keepdims=True), np.zeros((1, 3, 2)), 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_reref_removes_reference_and_updates_locations", NATIVE_SHA256)
def test_reref_removes_reference_and_updates_locations(eeglab_backend):
    data = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=float)
    locs = np.array([[("A",), ("B",), ("C",)]], dtype=[("labels", object)])
    actual, locations, removed = eeglab_backend("reref", data, 2.0, "elocs", locs, nargout=3)
    _assert_close(actual, [[-1, -1, -1], [1, 1, 1]], 0)
    assert list(locations["labels"].flat) == ["A", "C"]
    assert list(locations["ref"].flat) == ["B", "B"]
    assert removed["labels"] == "B"


@expanded_matlab_test(NATIVE_SOURCE, "test_reref_reconstructs_original_reference", NATIVE_SHA256)
def test_reref_reconstructs_original_reference(eeglab_backend):
    data = np.array([[2.0, 4.0], [6.0, 8.0]])
    locs = np.array([[("A", 0.0, 0.5), ("B", 90.0, 0.5)]], dtype=[(k, object) for k in ("labels", "theta", "radius")])
    actual, locations = eeglab_backend(
        "reref", data, np.empty((0, 0)), "elocs", locs, "refloc", np.array([["REF", 0.0, 0.0]], dtype=object), nargout=2
    )
    _assert_close(actual, [[-2 / 3, 0], [10 / 3, 4], [-8 / 3, -4]], 1e-12)
    _assert_close(actual.sum(axis=0, keepdims=True), [[0, 0]], 1e-12)
    assert list(locations["labels"].flat) == ["A", "B", "REF"]
    assert list(locations["ref"].flat) == ["average"] * 3


@expanded_matlab_test(NATIVE_SOURCE, "test_huber_reference_resists_offset_outlier", NATIVE_SHA256)
def test_huber_reference_resists_offset_outlier(eeglab_backend):
    data = np.array([[0.0], [0.0], [0.0], [1000.0]]) + np.array([[-2.0, 0.0, 3.0]])
    actual = eeglab_backend("reref", data, np.empty((0, 0)), "huber", 10.0)
    expected = np.repeat(np.array([[-10 / 3], [-10 / 3], [-10 / 3], [1000 - 10 / 3]]), 3, axis=1)
    _assert_close(actual, expected, 1e-5)
    _assert_close(actual[3:4] - actual[0:1], [[1000, 1000, 1000]], 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_huber_reference_without_outlier_is_average", NATIVE_SHA256)
def test_huber_reference_without_outlier_is_average(eeglab_backend):
    data = np.array([[-1.0], [0.0], [1.0]]) + np.array([[-2.0, 0.0, 3.0]])
    actual = eeglab_backend("reref", data, np.empty((0, 0)), "huber", 10.0)
    _assert_close(actual, np.repeat([[-1.0], [0.0], [1.0]], 3, axis=1), 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_interpolation_kang_preserves_constant_scalp_field", NATIVE_SHA256)
def test_interpolation_kang_preserves_constant_scalp_field(eeglab_backend):
    eeg = _interpolation_eeg(eeglab_backend)
    actual = eeglab_backend("eeg_interp", eeg, 3.0, "sphericalKang")
    _assert_interpolation(actual, eeg)


@expanded_matlab_test(NATIVE_SOURCE, "test_interpolation_explicit_spline_parameters", NATIVE_SHA256)
def test_interpolation_explicit_spline_parameters(eeglab_backend):
    eeg = _interpolation_eeg(eeglab_backend)
    actual = eeglab_backend("eeg_interp", eeg, 3.0, "spherical", np.empty((0, 0)), np.array([[1e-5, 4.0, 50.0]]))
    _assert_interpolation(actual, eeg)


@expanded_matlab_test(NATIVE_SOURCE, "test_resampling_discontinuities_preserves_dc_and_event_time", NATIVE_SHA256)
def test_resampling_discontinuities_preserves_dc_and_event_time(eeglab_backend):
    data = np.array([[1, 1, 1, 100, 100, 100, 100, 100], [-2, -2, -2, -200, -200, -200, -200, -200]], dtype=float)
    eeg = _signal_eeg(eeglab_backend, data, 100.0)
    eeg["event"] = np.array(
        [[("start", 1.0, 2.0, 1.0), ("boundary", 3.5, 10.0, 2.0), ("stim", 5.0, 4.0, 3.0)]],
        dtype=[(k, object) for k in ("type", "latency", "duration", "urevent")],
    )
    eeg["urevent"] = drop_fields(eeg["event"], "urevent", usemask=False)
    actual = eeglab_backend("pop_resample", eeg, 50.0, 0.8, 0.4)
    assert actual["data"].shape == (2, 5)
    _assert_close(actual["data"], [[1, 1, 100, 100, 100], [-2, -2, -200, -200, -200]], 1e-10)
    for field in ("event", "urevent"):
        _assert_close(matlab_field_concat(actual[field], "latency"), [[1, 2.5, 3.5]], 1e-12)
    _assert_close(matlab_field_concat(actual["event"], "duration"), [[1, 5, 2]], 0)
    _assert_close(actual["times"], [[0, 20, 40, 60, 80]], 1e-12)
    _assert_close(actual["srate"], [[50]], 0)


@expanded_matlab_test(NATIVE_SOURCE, "test_resampling_epoched_events_and_ica_cache", NATIVE_SHA256)
def test_resampling_epoched_events_and_ica_cache(eeglab_backend):
    eeg = _signal_eeg(eeglab_backend, np.arange(1.0, 41.0).reshape(2, 10, 2, order="F"), 100.0)
    eeg["event"] = np.array(
        [[("stim", 3.0, 2.0, 1.0), ("stim", 15.0, 4.0, 2.0)]],
        dtype=[(k, object) for k in ("type", "latency", "duration", "epoch")],
    )
    eeg["epoch"] = np.array([[(1.0,), (2.0,)]], dtype=[("event", object)])
    eeg["urevent"] = drop_fields(eeg["event"], "epoch", usemask=False)
    eeg["icaact"] = np.ones((2, 10, 2))
    actual = eeglab_backend("pop_resample", eeg, 50.0)
    assert actual["data"].shape == (2, 5, 2)
    _assert_close(matlab_field_concat(actual["event"], "latency"), [[2, 8]], 0)
    _assert_close(matlab_field_concat(actual["event"], "duration"), [[1, 2]], 0)
    assert actual["urevent"].size == 0
    assert actual["icaact"].size == 0
    _assert_close(actual["trials"], [[2]], 0)
    _assert_close(actual["xmax"], [[0.08]], 1e-12)


@expanded_matlab_test(NATIVE_SOURCE, "test_asr_custom_calibration_reconstructs_burst", NATIVE_SHA256)
def test_asr_custom_calibration_reconstructs_burst(eeglab_backend):
    eeg, calibration, burst = _asr_eeg(eeglab_backend)
    actual = eeglab_backend("clean_asr", eeg, 20.0, np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), calibration)
    _assert_reconstruction(actual, eeg, burst)


@expanded_matlab_test(NATIVE_SOURCE, "test_clean_artifacts_reconstruction_without_rejection", NATIVE_SHA256)
def test_clean_artifacts_reconstruction_without_rejection(eeglab_backend):
    eeg, calibration, burst = _asr_eeg(eeglab_backend)
    actual = eeglab_backend(
        "clean_artifacts",
        eeg,
        "FlatlineCriterion",
        "off",
        "Highpass",
        "off",
        "ChannelCriterion",
        "off",
        "LineNoiseCriterion",
        "off",
        "WindowCriterion",
        "off",
        "BurstCriterion",
        20.0,
        "BurstCriterionRefMaxBadChns",
        calibration,
        "BurstRejection",
        "off",
    )
    _assert_reconstruction(actual, eeg, burst)


def _signal_eeg(backend, data, srate):
    eeg = backend("eeg_emptyset")
    eeg.update(
        data=np.asarray(data, dtype=float),
        nbchan=float(data.shape[0]),
        pnts=float(data.shape[1]),
        trials=float(data.shape[2] if data.ndim == 3 else 1),
        srate=srate,
        xmin=0.0,
        xmax=(data.shape[1] - 1) / srate,
        times=np.arange(data.shape[1], dtype=float)[None, :] * 1000 / srate,
        chanlocs=np.array([[(f"C{i + 1}",) for i in range(data.shape[0])]], dtype=[("labels", object)]),
    )
    return eeg


def _interpolation_eeg(backend):
    eeg = _signal_eeg(backend, np.repeat([[-2.0, 1, 0, 4, -1, 3]], 6, axis=0), 100.0)
    coordinates = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    eeg["chanlocs"] = np.array(
        [[(f"C{i + 1}", *xyz, float(i * 60), 0.5) for i, xyz in enumerate(coordinates)]],
        dtype=[(k, object) for k in ("labels", "X", "Y", "Z", "theta", "radius")],
    )
    eeg["data"][2, :] = 10000
    return eeg


def _assert_interpolation(actual, eeg):
    _assert_close(actual["data"], np.repeat(eeg["data"][:1], 6, axis=0), 1e-10)
    _assert_close(actual["data"][[0, 1, 3, 4, 5]], eeg["data"][[0, 1, 3, 4, 5]], 0)
    assert list(actual["chanlocs"]["labels"].flat) == list(eeg["chanlocs"]["labels"].flat)
    _assert_close(actual["nbchan"], [[6]], 0)


def _asr_eeg(backend):
    time = np.arange(4096.0) / 128
    data = np.empty((8, 4096))
    for channel in range(1, 9):
        data[channel - 1] = 10 * np.sin(2 * np.pi * (5 + channel) * time) + 5 * np.cos(
            2 * np.pi * (23 + channel) * time
        )
    eeg = _signal_eeg(backend, data, 128.0)
    eeg["event"] = {"type": "stim", "latency": np.array([[2048.0]]), "duration": np.array([[1.0]])}
    calibration = deepcopy(eeg)
    burst = slice(1536, 1792)
    eeg["data"][0, burst] += 1000 * np.sin(2 * np.pi * 10 * np.arange(256) / float(np.asarray(eeg["srate"]).item()))
    return eeg, calibration, burst


def _assert_reconstruction(actual, eeg, burst):
    assert actual["data"].shape == eeg["data"].shape
    assert np.isfinite(actual["data"]).all()
    assert np.linalg.norm(actual["data"] - eeg["data"]) > 1
    assert np.linalg.norm(actual["data"][:, burst]) < 0.5 * np.linalg.norm(eeg["data"][:, burst])
    assert np.linalg.norm(actual["data"][:, :1024] - eeg["data"][:, :1024]) < 0.01 * np.linalg.norm(
        eeg["data"][:, :1024]
    )
    for field in ("pnts", "nbchan", "srate"):
        _assert_close(actual[field], np.asarray(eeg[field]).reshape(1, 1), 0)
    assert_matlab_equal(actual["event"], eeg["event"])
    assert list(actual["chanlocs"]["labels"].flat) == list(eeg["chanlocs"]["labels"].flat)


def _assert_close(actual, expected, atol):
    # Native verifyEqual checks numeric class and shape as well as values.
    expected = expected if isinstance(expected, np.ndarray) else np.asarray(expected, dtype=float)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, strict=True)
