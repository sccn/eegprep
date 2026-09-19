"""Ports of the current EEGLAB clean_rawdata regression tests."""

from copy import deepcopy

import numpy as np

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


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_filtering_test")
def test_clean_rawdata_highpass_is_deterministic_and_removes_drift():
    eeg = _continuous_eeg()
    cleaned = _clean_twice(eeg, **(_ALL_OFF | {"Highpass": [0.25, 0.75]}))

    time = np.arange(_N_SAMPLES) / _SAMPLE_RATE
    drift = np.sin(2 * np.pi * 0.15 * time)
    input_amplitude = abs(2 * np.dot(eeg["data"][0], drift) / _N_SAMPLES)
    output_amplitude = abs(2 * np.dot(cleaned["data"][0], drift) / _N_SAMPLES)

    assert cleaned["data"].shape == eeg["data"].shape
    assert output_amplitude < 0.1 * input_amplitude


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_chan_test")
def test_clean_rawdata_without_locations_is_deterministic_and_removes_bad_channel():
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


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_chanloc_test")
def test_clean_rawdata_with_locations_is_deterministic_and_removes_bad_channel():
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


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_asr_test")
def test_clean_rawdata_asr_repair_is_deterministic_and_reduces_burst_energy():
    eeg = _with_burst()
    cleaned = _clean_twice(eeg, **(_ALL_OFF | {"BurstCriterion": 20}))

    input_energy = np.linalg.norm(eeg["data"][:, _BURST])
    output_energy = np.linalg.norm(cleaned["data"][:, _BURST])
    assert cleaned["data"].shape == eeg["data"].shape
    assert np.all(np.isfinite(cleaned["data"]))
    assert output_energy < 0.95 * input_energy


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_rej_test")
def test_clean_rawdata_asr_rejection_is_deterministic_and_removes_burst_samples():
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


@eeglab_test(UPSTREAM_SUITE, "test_clean_rawdata_finalrej_test")
def test_clean_rawdata_final_window_rejection_is_deterministic_and_removes_bad_windows():
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
