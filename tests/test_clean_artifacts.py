"""
Test suite for clean_artifacts.py - All-in-one artifact removal.

This module tests the clean_artifacts function that provides comprehensive
artifact removal including flatline channels, drifts, noisy channels, bursts, and windows.
"""

import logging
import unittest
import sys

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts
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


def test_short_recording_without_a_spectral_filter_failure_still_raises():
    eeg = _low_rate_eeg()
    eeg["data"] = np.ones((1, 10), dtype=np.float64)
    eeg["pnts"] = 10
    eeg["srate"] = 128.0
    with pytest.raises(ValueError, match="Not enough calibration data"):
        clean_artifacts(
            eeg,
            FlatlineCriterion="off",
            Highpass="off",
            ChannelCriterion="off",
            LineNoiseCriterion="off",
            BurstCriterion=5.0,
            BurstCriterionRefMaxBadChns="off",
            WindowCriterion="off",
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
