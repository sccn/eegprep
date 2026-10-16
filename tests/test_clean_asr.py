"""Tests for clean_asr module.

This module tests the Artifact Subspace Reconstruction (ASR) functionality
including parameter validation, calibration data selection, and various
processing switches.
"""

import unittest
from pathlib import Path

import numpy as np

import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.clean_rawdata.asr_calibrate import ASRCalibrationError
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr


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


def test_short_recording_calibration_is_not_the_whole_recording():
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
    assert np.max(np.abs(windowed["data"] - whole["data"])) > 50.0


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


if __name__ == '__main__':
    unittest.main()
