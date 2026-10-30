"""
Test suite for clean_artifacts.py - All-in-one artifact removal.

This module tests the clean_artifacts function that provides comprehensive
artifact removal including flatline channels, drifts, noisy channels, bursts, and windows.
"""

import hashlib
import logging
import unittest
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.plugins.clean_rawdata.asr_calibrate import ASRCalibrationError
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.utils.testing import DebuggableTestCase

from tests.fixtures import create_test_eeg as _create_test_eeg


def create_test_eeg():
    """Continuous (2D) EEG fixture sized for clean_artifacts (20 s at 500 Hz)."""
    return _create_test_eeg(n_channels=32, n_samples=10000, srate=500.0, n_trials=1)


class TestCleanArtifactsChannelSelection(DebuggableTestCase):
    """Test cases for channel selection."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_channels_include(self):
        """Test channel inclusion."""
        channels_to_include = ['Ch1', 'Ch2', 'Ch3']

        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Channels=channels_to_include,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have only the specified channels
        self.assertEqual(EEG['nbchan'], len(channels_to_include))


class TestCleanArtifactsHpSnapshot(DebuggableTestCase):
    """Regression test for the high-pass snapshot point-in-time contract."""

    def setUp(self):
        np.random.seed(11)
        self.test_eeg = create_test_eeg()

    def test_hp_snapshot_is_point_in_time(self):
        """HP must not carry the sample mask written by the later window stage."""
        EEG, HP, _BUR, _removed = clean_artifacts(
            self.test_eeg,
            Highpass='off',
            ChannelCriterion=0.8,
            LineNoiseCriterion=4.0,
            BurstCriterion='off',
            WindowCriterion=0.25,
        )

        # The window stage populates clean_sample_mask on the final EEG dataset...
        self.assertIn('clean_sample_mask', EEG['etc'])
        # ...but the high-pass snapshot predates that stage, so it must not share
        # the same etc object or carry the later mask.
        self.assertIsNot(HP['etc'], EEG['etc'])
        self.assertNotIn('clean_sample_mask', HP['etc'])


def _low_rate_eeg():
    return {
        "data": np.ones((1, 256), dtype=np.float64),
        "srate": 32.0,
        "nbchan": 1,
        "pnts": 256,
        "trials": 1,
        "xmin": 0.0,
        "event": [],
    }


def test_low_rate_burst_calibration_leaves_the_recording_unchanged(caplog):
    eeg = _low_rate_eeg()
    with caplog.at_level(logging.WARNING):
        cleaned, _hp, burst, _removed = clean_artifacts(
            eeg,
            FlatlineCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            BurstCriterion=5.0,
            BurstCriterionRefMaxBadChns="off",
            WindowCriterion="off",
        )
    np.testing.assert_array_equal(cleaned["data"], eeg["data"])
    np.testing.assert_array_equal(burst["data"], eeg["data"])
    assert any("32" in record.message and "spectral" in record.message.lower() for record in caplog.records)


def test_short_recording_without_a_spectral_filter_failure_leaves_the_burst_unchanged(caplog):
    eeg = _low_rate_eeg()
    eeg["data"] = np.ones((1, 10), dtype=np.float64)
    eeg["pnts"] = 10
    eeg["srate"] = 128.0
    with caplog.at_level(logging.WARNING):
        cleaned, _hp, burst, _removed = clean_artifacts(
            eeg,
            FlatlineCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            BurstCriterion=5.0,
            BurstCriterionRefMaxBadChns="off",
            WindowCriterion="off",
        )
    np.testing.assert_array_equal(cleaned["data"], eeg["data"])
    np.testing.assert_array_equal(burst["data"], eeg["data"])
    assert any(
        "Not enough calibration data" in record.message and "got 10" in record.message for record in caplog.records
    )


# Input hash from the issue 524 fuzzer repro, seed 20261027.
_ISSUE_524_SHA256 = "f07c280e79686ad9ca9d0e778957731cd6e953eb142f0964f4022c41029bd20e"


def _issue_524_eeg():
    source = pop_loadset(Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set")
    sample_index = np.arange(4474, 4474 + 44)
    data = np.array(np.asarray(source["data"])[np.ix_(range(11, 27), sample_index)], dtype=np.float64)
    data[12, 13:15] = np.nan
    digest = hashlib.sha256(np.ascontiguousarray(data, dtype=np.float64).tobytes()).hexdigest()
    assert digest == _ISSUE_524_SHA256
    srate = float(source["srate"])
    pnts = int(data.shape[1])
    return {
        "data": data,
        "srate": srate,
        "nbchan": int(data.shape[0]),
        "pnts": pnts,
        "trials": 1,
        "xmin": 0.0,
        "xmax": (pnts - 1) / srate,
        "event": [],
    }


def _issue_524_options():
    return {
        "FlatlineCriterion": "off",
        "Highpass": "off",
        "ChannelCriterion": "off",
        "LineNoiseCriterion": "off",
        "WindowCriterion": "off",
        "BurstCriterion": 20.0,
        "BurstCriterionRefMaxBadChns": 0.2,
        "BurstRejection": "off",
    }


def test_issue_524_short_calibration_keeps_every_sample(caplog):
    eeg = _issue_524_eeg()
    original = np.array(eeg["data"], copy=True)
    with caplog.at_level(logging.WARNING):
        cleaned = pop_clean_rawdata(eeg, gui=False, **_issue_524_options())
    np.testing.assert_allclose(cleaned["data"], original, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_allclose(eeg["data"], original, rtol=0.0, atol=0.0, equal_nan=True)
    assert cleaned["data"].shape == (16, 44)
    assert cleaned["srate"] == 128.0
    assert any("Need at least 64 samples, got 44" in record.message for record in caplog.records)


def test_issue_524_clean_asr_still_names_the_short_calibration():
    eeg = _issue_524_eeg()
    with pytest.raises(ASRCalibrationError, match=r"Need at least 64 samples, got 44"):
        clean_asr(
            eeg,
            cutoff=20.0,
            ref_maxbadchannels=0.2,
            ref_tolerances=(-np.inf, 5.5),
            ref_wndlen=1.0,
        )


def test_one_calibration_window_skips_the_burst_and_keeps_samples(caplog):
    rng = np.random.default_rng(524)
    data = rng.standard_normal((4, 71))
    eeg = {
        "data": data,
        "srate": 128.0,
        "nbchan": 4,
        "pnts": 71,
        "trials": 1,
        "xmin": 0.0,
        "event": [],
    }
    with caplog.at_level(logging.WARNING):
        cleaned, _hp, burst, _removed = clean_artifacts(
            eeg,
            FlatlineCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            BurstCriterion=20.0,
            BurstCriterionRefMaxBadChns=0.2,
            WindowCriterion="off",
        )
    np.testing.assert_array_equal(cleaned["data"], data)
    np.testing.assert_array_equal(burst["data"], data)
    assert any(
        "Fewer than 2 calibration windows" in record.message and "got 1" in record.message for record in caplog.records
    )


def test_short_calibration_still_runs_window_removal():
    eeg = _issue_524_eeg()
    with pytest.raises(ValueError, match="Not enough data for even a single window"):
        clean_artifacts(
            eeg,
            FlatlineCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            BurstCriterion=20.0,
            BurstCriterionRefMaxBadChns=0.2,
            WindowCriterion=0.25,
        )


class TestCleanArtifactsErrorSurfacing(DebuggableTestCase):
    """Errors inside the selection / channel-cleaning paths must surface, not be masked."""

    def setUp(self):
        self.test_eeg = create_test_eeg()

    def test_channels_ignore_preserves_events(self):
        """Restricting channels must not wipe the dataset's events."""
        eeg = create_test_eeg()
        eeg['event'] = [{'type': 'mark', 'latency': 100.0}, {'type': 'mark', 'latency': 5000.0}]
        original_events = list(eeg['event'])

        EEG, _HP, _BUR, _removed = clean_artifacts(
            eeg,
            Channels_ignore=['EEG001'],
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        self.assertEqual(len(EEG['event']), len(original_events))


if __name__ == '__main__':
    unittest.main()
