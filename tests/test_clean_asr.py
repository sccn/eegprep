"""Tests for clean_asr module.

This module tests the Artifact Subspace Reconstruction (ASR) functionality
including parameter validation, calibration data selection, and various
processing switches.
"""

import hashlib
import math
import unittest
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.miscfunc.misc import round_mat
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.clean_rawdata.asr_calibrate import ASRCalibrationError, asr_calibrate
from eegprep.plugins.clean_rawdata.private.stats import fit_eeg_distribution
from eegprep.plugins.clean_rawdata.private.windows import matlab_window_offsets
from eegprep.plugins.clean_rawdata.asr_process import asr_process
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.clean_windows import clean_windows
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from tests.fuzz.harness import compare_data, load_tolerances

# Nightly fuzz crop clean_asr|numeric|1e1, seed 20261031.
_NAN_TAIL_SHA256 = "f98111327da99e56c5c9bf58d5ca02364023868af91ce44d4354caa79065e4d6"


class TestCleanASRParameters(unittest.TestCase):
    """Test clean_asr parameter handling and validation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_channels = 4  # Smaller for faster testing
        self.n_samples = 500  # Shorter for faster testing
        self.srate = 250.0

        self.test_eeg = {
            'data': np.random.randn(self.n_channels, self.n_samples) * 0.5,
            'srate': self.srate,
            'nbchan': self.n_channels,
            'pnts': self.n_samples,
            'trials': 1,
            'xmin': 0.0,
            'xmax': (self.n_samples - 1) / self.srate,
            'times': np.arange(self.n_samples) / self.srate,
            'event': [],
        }

    def test_clean_asr_full_riemannian_keeps_shape_and_input(self):
        """useriemannian=True runs Riemannian ASR instead of refusing."""
        original = self.test_eeg['data'].copy()
        cleaned = clean_asr(self.test_eeg, ref_maxbadchannels='off', useriemannian=True)
        np.testing.assert_array_equal(self.test_eeg['data'], original)
        self.assertEqual(cleaned['data'].shape, original.shape)
        self.assertTrue(np.isfinite(cleaned['data']).all())


def test_short_recording_calibration_falls_back_to_the_whole_recording():
    """Three windows cannot be indexed, so ASR calibrates on the whole recording."""
    source = pop_loadset(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set")
    data = np.asarray(source["data"][:11, 8:264], dtype=np.float64)
    pnts = int(data.shape[1])
    eeg = {
        "data": data,
        "srate": float(source["srate"]),
        "nbchan": int(data.shape[0]),
        "pnts": pnts,
        "trials": 1,
        "xmin": 0.0,
        "xmax": (pnts - 1) / float(source["srate"]),
        "event": [],
    }
    windowed = clean_asr(eeg, cutoff=5.0, ref_maxbadchannels=0.075, ref_tolerances=(-3.5, 5.5), ref_wndlen=1.0)
    whole = clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")
    assert windowed["data"].shape == data.shape
    np.testing.assert_allclose(windowed["data"], whole["data"])


def _reference_shorter_than_one_rms_window():
    """Three isolated clean windows. Their surviving samples are under 0.5 s."""
    srate = 128.0
    samples = 1024
    window = int(srate)
    offsets = matlab_window_offsets(samples, float(window), window * 0.34).astype(int)
    quiet = np.zeros(samples, dtype=bool)
    for index in (0, 7, 14):
        quiet[int(offsets[index]) : int(offsets[index]) + window] = True
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, samples)) * 5.0
    data[:, quiet] = rng.standard_normal((4, int(np.count_nonzero(quiet)))) * 0.5
    return {
        "data": data,
        "srate": srate,
        "nbchan": 4,
        "pnts": samples,
        "trials": 1,
        "xmin": 0.0,
        "xmax": (samples - 1) / srate,
        "event": [],
    }


def test_short_selected_reference_is_not_replaced_by_the_recording():
    """asr_calibrate.m:191 errors when the reference has fewer samples than one 0.5 s window."""
    eeg = _reference_shorter_than_one_rms_window()
    cleaned, _mask = clean_windows(eeg, 0.075, (-3.5, 5.5), 1.0)
    kept = int(cleaned["pnts"])
    needed = int(round_mat(0.5 * float(eeg["srate"])))
    assert 0 < kept < needed
    assert int(eeg["pnts"]) > needed
    with pytest.raises(ASRCalibrationError, match=rf"has {kept} samples.*needs {needed}") as raised:
        clean_asr(eeg, cutoff=5.0, ref_maxbadchannels=0.075, ref_tolerances=(-3.5, 5.5), ref_wndlen=1.0)
    assert "Not enough calibration data" in str(raised.value)
    whole = clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")
    assert whole["data"].shape == eeg["data"].shape


def test_empty_calibration_names_the_missing_window():
    with pytest.raises(ASRCalibrationError, match=r"has 0 samples.*needs 64"):
        asr_calibrate(np.ones((2, 0)), 128.0)


def test_very_short_recording_still_calibrates_with_reference_off():
    """24 channels, 62 samples at 64 Hz. EEGLAB's fit indexes with width 0."""
    source = pop_loadset(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set")
    sample_index = np.arange(90, 90 + 62 * 2, 2)
    data = np.array(np.asarray(source["data"])[np.ix_(range(24), sample_index)], dtype=np.float64)
    srate = float(source["srate"]) / 2.0
    eeg = {
        "data": data,
        "srate": srate,
        "nbchan": int(data.shape[0]),
        "pnts": int(data.shape[1]),
        "trials": 1,
        "xmin": 0.0,
        "xmax": (data.shape[1] - 1) / srate,
        "event": [],
    }
    cleaned = clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")
    assert cleaned["data"].shape == data.shape
    assert np.max(np.abs(cleaned["data"] - data)) > 1.0


_ISSUE_526_SHA256 = "5f90669b03649cfc1fbd82a6cdabd4cda1b3dc43d7adb22db5bbaba07fd19569"


def test_issue_526_two_rms_windows_are_still_cleaned():
    """21 channels, 89 samples at 128 Hz. EEGLAB indexes width 0 and returns the input."""
    source = pop_loadset(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set")
    data = np.array(np.asarray(source["data"])[np.ix_(range(10, 31), np.arange(49, 49 + 89))], dtype=np.float64)
    digest = hashlib.sha256(np.ascontiguousarray(data, dtype=np.float64).tobytes()).hexdigest()
    assert digest == _ISSUE_526_SHA256
    srate = float(source["srate"])
    eeg = {
        "data": data,
        "srate": srate,
        "nbchan": int(data.shape[0]),
        "pnts": int(data.shape[1]),
        "trials": 1,
        "xmin": 0.0,
        "xmax": (data.shape[1] - 1) / srate,
        "event": [],
    }
    cleaned = clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")
    popped = pop_clean_rawdata(
        eeg,
        gui=False,
        FlatlineCriterion="off",
        Highpass="off",
        ChannelCriterion="off",
        LineNoiseCriterion="off",
        WindowCriterion="off",
        BurstCriterion=5.0,
        BurstCriterionRefMaxBadChns="off",
        BurstRejection="off",
    )
    np.testing.assert_array_equal(popped["data"], cleaned["data"])
    assert cleaned["data"].shape == (21, 89)
    gap, error_class, *_rest = compare_data(cleaned["data"], data, srate, srate, load_tolerances()["clean_asr"])
    assert gap is True
    assert error_class == "numeric"


def test_low_sampling_rate_calibration_names_the_spectral_filter():
    eeg = {
        "data": np.ones((1, 256), dtype=np.float64),
        "srate": 32.0,
        "nbchan": 1,
        "pnts": 256,
        "trials": 1,
        "xmin": 0.0,
        "event": [],
    }
    with pytest.raises(ASRCalibrationError, match="32 Hz") as raised:
        clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")
    message = str(raised.value)
    assert "spectral" in message.lower()
    assert "ff must be non-decreasing" not in message


def _nan_tail_recording():
    """Channels 24-30, samples 158:414, with channel index 1 samples 245-246 set to NaN."""
    source = pop_loadset(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set")
    sample_index = np.arange(158, 158 + 256)
    data = np.array(np.asarray(source["data"])[np.ix_(range(24, 31), sample_index)], dtype=np.float64)
    data[1, 245:247] = np.nan
    digest = hashlib.sha256(np.ascontiguousarray(data, dtype=np.float64).tobytes()).hexdigest()
    assert digest == _NAN_TAIL_SHA256
    srate = float(source["srate"])
    return {
        "data": data,
        "srate": srate,
        "nbchan": int(data.shape[0]),
        "pnts": int(data.shape[1]),
        "trials": 1,
        "xmin": 0.0,
        "xmax": (data.shape[1] - 1) / srate,
        "event": [],
    }


def _reflect_tail(data, srate):
    """clean_asr.m:189 reflection of the original samples, NaNs included."""
    window_len = max(0.5, 1.5 * data.shape[0] / srate)
    n_extrap = int(round_mat(window_len / 2 * srate))
    extrap_indices = np.arange(data.shape[1] - 2, data.shape[1] - n_extrap - 2, -1)
    reflected = 2 * data[:, [-1]] - data[:, extrap_indices]
    return window_len, n_extrap, extrap_indices, reflected


def test_trailing_nans_are_reflected_before_processing_zeros_them():
    """A NaN in the last half-window is reflected as NaN, then processed as 0.

    Calibration must not zero the shared array first. That turned samples
    245-246 into 0 and the lookahead into twice the final sample (133.655).
    """
    eeg = _nan_tail_recording()
    raw = eeg["data"]
    srate = float(eeg["srate"])
    window_len, n_extrap, extrap_indices, reflected = _reflect_tail(raw, srate)
    assert n_extrap == 32
    for sample in (245, 246):
        column = list(extrap_indices).index(sample)
        assert np.isnan(reflected[1, column])

    sig = np.concatenate((raw, reflected), axis=1)
    state = asr_calibrate(raw.copy(), srate, cutoff=5.0)
    step = int(math.floor(srate * window_len / 2))
    processed, _state = asr_process(
        sig,
        srate,
        state,
        window_len=window_len,
        lookahead=window_len / 2,
        step_size=step,
        max_dims=0.66,
        max_mem=64,
    )
    tail = processed[:, raw.shape[1] :]
    for sample in (245, 246):
        column = list(extrap_indices).index(sample)
        assert tail[1, column] == 0.0

    cleaned = clean_asr(eeg, cutoff=5.0, ref_maxbadchannels="off")
    np.testing.assert_array_equal(cleaned["data"], processed[:, : raw.shape[1]])
    np.testing.assert_array_equal(eeg["data"], raw)

    zeroed = raw.copy()
    zeroed[~np.isfinite(zeroed)] = 0.0
    zeroed_eeg = dict(eeg)
    zeroed_eeg["data"] = zeroed
    cleaned_zeroed = clean_asr(zeroed_eeg, cutoff=5.0, ref_maxbadchannels="off")
    finite = np.isfinite(raw)
    gap = np.abs(cleaned["data"] - cleaned_zeroed["data"])[finite]
    # Same gap the nightly fuzzer measured against EEGLAB on this crop.
    np.testing.assert_allclose(gap.max(), 11.59255144724768, rtol=0, atol=1e-9)
    window = slice(raw.shape[1] - n_extrap, None)
    signal_scale = np.max(np.abs(cleaned["data"][:, window][finite[:, window]]))
    np.testing.assert_allclose(gap.max() / signal_scale, 0.15200149020594309, rtol=0, atol=1e-9)


def test_three_rms_windows_skip_zero_fit_widths():
    """A 124-sample reference at 128 Hz has three RMS windows and a zero fit width.

    ``asr_calibrate.m`` line 392 rounds ``3*(0.578:-0.01:0.1445)`` and line 395
    indexes that width. EEGPrep skips the non-positive widths.
    """
    srate = 128.0
    samples = 124
    window = int(round_mat(0.5 * srate))
    offsets = matlab_window_offsets(samples, window, window * (1.0 - 0.66))
    assert offsets.tolist() == [0, 22, 44]
    max_width = 0.6 - 0.022
    widths = np.int32(round_mat(3 * np.arange(max_width, 0.25 * max_width, -0.01)))
    assert 0 in widths
    mu, sig, alpha, beta = fit_eeg_distribution(np.array([1.0, 1.1, 0.9]))
    assert np.isfinite([mu, sig, alpha, beta]).all()
    time = np.arange(samples) / srate
    signal = np.vstack([np.sin(2 * np.pi * 8 * time), np.cos(2 * np.pi * 10 * time)])
    state = asr_calibrate(signal, srate, cutoff=5.0)
    assert state["M"].shape == (2, 2)
    assert np.isfinite(state["M"]).all()
    assert np.isfinite(state["T"]).all()


if __name__ == '__main__':
    unittest.main()
