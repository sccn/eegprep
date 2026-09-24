"""Ports of the current EEGLAB clean_rawdata regression tests."""

from copy import deepcopy

import numpy as np
import pytest

from eegprep import pop_clean_rawdata
from tests.eeglab_tests import eeglab_test
from tests.fixtures import create_test_eeg


UPSTREAM_SUITE = "unittesting_clean_rawdata/clean_rawdata_wrapperTest.m"
_SAMPLE_RATE = 128.0
_N_CHANNELS = 8
_N_SAMPLES = 4096
_BURST = slice(1800, 2050)
_ALL_OFF = {
    "FlatlineCriterion": "off",
    "ChannelCriterion": "off",
    "LineNoiseCriterion": "off",
    "Highpass": "off",
    "BurstCriterion": "off",
    "WindowCriterion": "off",
    "BurstRejection": "off",
    "Distance": "Euclidian",
}


def _continuous_eeg(*, with_locations: bool = True) -> dict:
    time = np.arange(_N_SAMPLES) / _SAMPLE_RATE
    rng = np.random.default_rng(20260317)
    common_signal = (
        np.sin(2 * np.pi * 10 * time)
        + 0.35 * np.sin(2 * np.pi * 6 * time)
        + 0.15 * np.sin(2 * np.pi * 3 * time)
        + 0.4 * np.sin(2 * np.pi * 0.15 * time)
    )
    data = np.vstack(
        [(1 + 0.04 * channel) * common_signal + 0.04 * rng.standard_normal(_N_SAMPLES) for channel in range(8)]
    )
    eeg = create_test_eeg(
        n_channels=_N_CHANNELS,
        n_samples=_N_SAMPLES,
        srate=_SAMPLE_RATE,
        n_trials=1,
    )
    eeg["data"] = data
    if not with_locations:
        eeg["chanlocs"] = []
    return eeg


def _with_bad_channel(*, with_locations: bool) -> dict:
    eeg = _continuous_eeg(with_locations=with_locations)
    eeg["data"][-1] = 3 * np.random.default_rng(42).standard_normal(_N_SAMPLES)
    return eeg


def _with_burst() -> dict:
    eeg = _continuous_eeg()
    eeg["data"][:, _BURST] += 80 * np.random.default_rng(4).standard_normal((_N_CHANNELS, 250))
    return eeg


def _clean_twice(eeg: dict, **options) -> dict:
    original = deepcopy(eeg)
    first = pop_clean_rawdata(eeg, gui=False, **options)
    np.testing.assert_equal(eeg, original)
    second = pop_clean_rawdata(eeg, gui=False, **options)
    np.testing.assert_equal(first, second)
    return first


def test_python_regression_clean_rawdata_highpass_is_deterministic_and_removes_drift():
    eeg = _continuous_eeg()
    cleaned = _clean_twice(eeg, **(_ALL_OFF | {"Highpass": [0.25, 0.75]}))

    time = np.arange(_N_SAMPLES) / _SAMPLE_RATE
    drift = np.sin(2 * np.pi * 0.15 * time)
    input_amplitude = abs(2 * np.dot(eeg["data"][0], drift) / _N_SAMPLES)
    output_amplitude = abs(2 * np.dot(cleaned["data"][0], drift) / _N_SAMPLES)

    assert cleaned["data"].shape == eeg["data"].shape
    assert output_amplitude < 0.1 * input_amplitude


def test_python_regression_clean_rawdata_without_locations_is_deterministic_and_removes_bad_channel():
    eeg = _with_bad_channel(with_locations=False)
    cleaned = _clean_twice(
        eeg,
        **(
            _ALL_OFF
            | {
                "FlatlineCriterion": 5,
                "ChannelCriterion": 0.8,
                "LineNoiseCriterion": 4,
            }
        ),
    )

    assert cleaned["nbchan"] == _N_CHANNELS - 1
    np.testing.assert_array_equal(
        cleaned["etc"]["clean_channel_mask"],
        [True, True, True, True, True, True, True, False],
    )


def test_python_regression_clean_rawdata_with_locations_is_deterministic_and_removes_bad_channel():
    eeg = _with_bad_channel(with_locations=True)
    cleaned = _clean_twice(
        eeg,
        **(
            _ALL_OFF
            | {
                "FlatlineCriterion": 5,
                "ChannelCriterion": 0.8,
                "LineNoiseCriterion": 4,
            }
        ),
    )

    assert cleaned["nbchan"] == _N_CHANNELS - 1
    assert [channel["labels"] for channel in cleaned["chanlocs"]] == [f"Ch{index}" for index in range(1, 8)]
    np.testing.assert_array_equal(
        cleaned["etc"]["clean_channel_mask"],
        [True, True, True, True, True, True, True, False],
    )


def test_python_regression_clean_rawdata_asr_repair_is_deterministic_and_reduces_burst_energy():
    eeg = _with_burst()
    cleaned = _clean_twice(eeg, **(_ALL_OFF | {"BurstCriterion": 20}))

    input_energy = np.linalg.norm(eeg["data"][:, _BURST])
    output_energy = np.linalg.norm(cleaned["data"][:, _BURST])
    assert cleaned["data"].shape == eeg["data"].shape
    assert np.all(np.isfinite(cleaned["data"]))
    assert output_energy < 0.95 * input_energy


def test_python_regression_clean_rawdata_asr_rejection_is_deterministic_and_removes_burst_samples():
    eeg = _with_burst()
    cleaned = _clean_twice(
        eeg,
        **(_ALL_OFF | {"BurstCriterion": 20, "BurstRejection": "on"}),
    )

    sample_mask = cleaned["etc"]["clean_sample_mask"]
    assert sample_mask.dtype == np.bool_
    assert sample_mask.shape == (_N_SAMPLES,)
    assert cleaned["pnts"] == np.count_nonzero(sample_mask)
    assert cleaned["pnts"] < eeg["pnts"]
    assert np.all(np.isfinite(cleaned["data"]))


def _assert_isequal(first, second):
    """MATLAB isequal: exact values and shapes, with NaNs unequal."""
    if isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for field in first:
            _assert_isequal(first[field], second[field])
        return
    if isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray)
        assert first.shape == second.shape
        assert first.dtype.names == second.dtype.names
        assert (first.dtype == object) == (second.dtype == object)
        if first.dtype.names:
            for field in first.dtype.names:
                _assert_isequal(first[field], second[field])
        elif first.dtype == object:
            for index in np.ndindex(first.shape):
                _assert_isequal(first[index], second[index])
        else:
            assert np.array_equal(first, second, equal_nan=False)
        return
    assert first == second


def test_isequal_assertion_matches_matlab(eeglab_matlab_engine, eeglab_backend):
    cells = np.empty((1, 1), dtype=object)
    cells[0, 0] = {"value": np.array([[1.0]])}
    comparisons = [
        (np.array([[1.0]]), np.array([[1]], dtype=np.int8)),
        (np.array([[np.nan]]), np.array([[np.nan]])),
        (np.array([[1, 2]]), np.array([[1], [2]])),
        (np.empty((0, 1)), np.empty((0, 2))),
        (np.array([[True]]), np.array([[1.0]])),
        ({"value": np.array([[1.0]])}, cells),
        ({"value": np.array([[1.0]])}, {"value": np.array([[1.0]])}),
    ]
    for first, second in comparisons:
        equal = bool(eeglab_backend("isequal", first, second).item())
        if equal:
            _assert_isequal(first, second)
        else:
            with pytest.raises(AssertionError):
                _assert_isequal(first, second)


def _reference_clean_twice(eeglab_backend, eeglab_suite_root, *, without_locations=False, **options):
    # readcontsamplefile.m loads this recording; do not substitute synthetic EEG.
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    if without_locations:
        eeg["chanlocs"] = np.empty((0, 0))
    parameters = _ALL_OFF | options
    first = eeglab_backend("pop_clean_rawdata", eeg, **parameters)
    second = eeglab_backend("pop_clean_rawdata", eeg, **parameters)
    _assert_isequal(first, second)


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_filtering_test")
def test_reference_clean_rawdata_filtering(eeglab_backend, eeglab_suite_root):
    _reference_clean_twice(eeglab_backend, eeglab_suite_root, Highpass=np.array([[0.25, 0.75]]))


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_chan_test")
def test_reference_clean_rawdata_channels_without_locations(eeglab_backend, eeglab_suite_root):
    _reference_clean_twice(
        eeglab_backend,
        eeglab_suite_root,
        without_locations=True,
        FlatlineCriterion=5.0,
        ChannelCriterion=0.8,
        LineNoiseCriterion=4.0,
    )


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_chanloc_test")
def test_reference_clean_rawdata_channels_with_locations(eeglab_backend, eeglab_suite_root):
    _reference_clean_twice(
        eeglab_backend, eeglab_suite_root, FlatlineCriterion=5.0, ChannelCriterion=0.8, LineNoiseCriterion=4.0
    )


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_asr_test")
def test_reference_clean_rawdata_asr(eeglab_backend, eeglab_suite_root):
    _reference_clean_twice(eeglab_backend, eeglab_suite_root, BurstCriterion=20.0)


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_rej_test")
def test_reference_clean_rawdata_rejection(eeglab_backend, eeglab_suite_root):
    _reference_clean_twice(eeglab_backend, eeglab_suite_root, BurstCriterion=20.0, BurstRejection="on")


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_finalrej_test")
def test_reference_clean_rawdata_final_rejection(eeglab_backend, eeglab_suite_root):
    _reference_clean_twice(
        eeglab_backend,
        eeglab_suite_root,
        WindowCriterion=0.25,
        WindowCriterionTolerances=np.array([[-np.inf, 7.0]]),
        channels=np.empty((0, 0)),
    )


def test_python_regression_clean_rawdata_final_window_rejection_is_deterministic_and_removes_bad_windows():
    eeg = _continuous_eeg()
    eeg["data"][:5, 1600:2200] += 35 * np.random.default_rng(1).standard_normal((5, 600))
    cleaned = _clean_twice(
        eeg,
        **(
            _ALL_OFF
            | {
                "WindowCriterion": 0.25,
                "WindowCriterionTolerances": [-np.inf, 7],
                "Channels": [],
            }
        ),
    )

    sample_mask = cleaned["etc"]["clean_sample_mask"]
    assert sample_mask.shape == (_N_SAMPLES,)
    assert np.any(~sample_mask)
    assert cleaned["pnts"] == np.count_nonzero(sample_mask)
