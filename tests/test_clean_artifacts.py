"""
Test suite for clean_artifacts.py - All-in-one artifact removal.

This module tests the clean_artifacts function that provides comprehensive
artifact removal including flatline channels, drifts, noisy channels, bursts, and windows.
"""

import unittest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.plugins.clean_rawdata.clean_artifacts import clean_artifacts
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata
from eegprep.utils.testing import DebuggableTestCase

from tests.fixtures import create_test_eeg as _create_test_eeg


def create_test_eeg():
    """Continuous (2D) EEG fixture sized for clean_artifacts (20 s at 500 Hz)."""
    return _create_test_eeg(n_channels=32, n_samples=10000, srate=500.0, n_trials=1)


class TestCleanArtifactsBasic(DebuggableTestCase):
    """Basic test cases for clean_artifacts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_basic_functionality(self):
        """Test basic clean_artifacts functionality with default parameters."""
        EEG, HP, BUR, removed_channels = clean_artifacts(self.test_eeg)

        # Check that all return values are present
        self.assertIsInstance(EEG, dict)
        self.assertIsInstance(HP, dict)
        self.assertIsInstance(BUR, dict)
        self.assertIsInstance(removed_channels, np.ndarray)

        # Check that EEG structure is preserved
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('pnts', EEG)

        # Check that data dimensions are reasonable
        self.assertEqual(EEG['srate'], self.test_eeg['srate'])
        self.assertGreaterEqual(EEG['nbchan'], 1)  # At least one channel should remain
        self.assertLessEqual(EEG['nbchan'], self.test_eeg['nbchan'])

    def test_clean_artifacts_all_off(self):
        """Test clean_artifacts with all criteria disabled."""
        self.test_eeg.pop('etc')
        original_keys = set(self.test_eeg)
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # With all criteria off, data should be unchanged
        self.assertEqual(EEG['nbchan'], self.test_eeg['nbchan'])
        self.assertEqual(EEG['pnts'], self.test_eeg['pnts'])
        np.testing.assert_array_equal(EEG['data'], self.test_eeg['data'])
        self.assertEqual(set(self.test_eeg), original_keys)

    def test_clean_artifacts_invalid_highpass_string(self):
        """Test clean_artifacts with invalid highpass string parameter."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(self.test_eeg, Highpass='invalid')
        self.assertIn('Highpass must be a (low, high) tuple or None/"off"', str(cm.exception))

    def test_clean_artifacts_invalid_highpass_single_value(self):
        """Test clean_artifacts with single value instead of tuple."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(self.test_eeg, Highpass=0.5)
        self.assertIn('Highpass must be a (low, high) tuple or None/"off"', str(cm.exception))

    def test_clean_artifacts_invalid_highpass_too_many_values(self):
        """Test clean_artifacts with too many values in highpass tuple."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(self.test_eeg, Highpass=(0.1, 0.5, 1.0))
        self.assertIn('Highpass must be a (low, high) tuple or None/"off"', str(cm.exception))

    def test_clean_artifacts_invalid_highpass_empty_tuple(self):
        """Test clean_artifacts with empty highpass tuple."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(self.test_eeg, Highpass=())
        self.assertIn('Highpass must be a (low, high) tuple or None/"off"', str(cm.exception))

    def test_clean_artifacts_invalid_highpass_list_single(self):
        """Test clean_artifacts with single-element list."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(self.test_eeg, Highpass=[0.5])
        self.assertIn('Highpass must be a (low, high) tuple or None/"off"', str(cm.exception))

    def test_clean_artifacts_valid_highpass_list(self):
        """Test clean_artifacts with valid highpass list (should work like tuple)."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Highpass=[0.25, 0.75],  # List instead of tuple
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            FlatlineCriterion='off',
        )
        # Should work - list is acceptable
        self.assertIsInstance(EEG, dict)

    def test_clean_artifacts_mutually_exclusive_channels(self):
        """Test clean_artifacts with mutually exclusive channel parameters."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(self.test_eeg, Channels=['Ch1', 'Ch2'], Channels_ignore=['Ch3'])
        self.assertIn('mutually exclusive', str(cm.exception))

    def test_clean_artifacts_mutually_exclusive_channels_both_empty(self):
        """Test clean_artifacts with both channel parameters empty (should work)."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Channels=[],  # Empty list
            Channels_ignore=[],  # Empty list
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )
        # Should work - empty lists are not mutually exclusive
        self.assertIsInstance(EEG, dict)

    def test_clean_artifacts_mutually_exclusive_channels_none_and_list(self):
        """Test clean_artifacts with None and non-empty list (should work)."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Channels=None,  # None
            Channels_ignore=['Ch1'],  # Non-empty list
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )
        # Should work - None and list is not mutually exclusive
        self.assertIsInstance(EEG, dict)

    def test_clean_artifacts_mutually_exclusive_channels_both_none(self):
        """Test clean_artifacts with both channel parameters as None (should work)."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Channels=None,  # None
            Channels_ignore=None,  # None
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )
        # Should work - both None is not mutually exclusive
        self.assertIsInstance(EEG, dict)

    def test_clean_artifacts_mutually_exclusive_channels_overlapping(self):
        """Test clean_artifacts with overlapping channel lists (error expected)."""
        with self.assertRaises(ValueError) as cm:
            clean_artifacts(
                self.test_eeg,
                Channels=['Ch1', 'Ch2', 'Ch3'],
                Channels_ignore=['Ch2', 'Ch4'],  # Ch2 overlaps
            )
        self.assertIn('mutually exclusive', str(cm.exception))


class TestCleanArtifactsFlatline(DebuggableTestCase):
    """Test cases for flatline channel removal."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_flatline_removal(self):
        """Test flatline channel removal."""
        # Create some flatline channels
        eeg_with_flatlines = self.test_eeg.copy()
        eeg_with_flatlines['data'] = self.test_eeg['data'].copy()
        eeg_with_flatlines['data'][5, :] = 0.0  # Flatline channel (2D data)
        eeg_with_flatlines['data'][10, :] = 1.0  # Another flatline channel
        original_nbchan = eeg_with_flatlines['nbchan']

        EEG, HP, BUR, removed_channels = clean_artifacts(
            eeg_with_flatlines,
            FlatlineCriterion=1.0,  # Short flatline duration
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
        )

        # Should have removed some channels
        self.assertLess(EEG['nbchan'], original_nbchan)

    def test_clean_artifacts_flatline_off(self):
        """Test flatline removal disabled."""
        # Create some flatline channels
        eeg_with_flatlines = self.test_eeg.copy()
        eeg_with_flatlines['data'] = self.test_eeg['data'].copy()
        eeg_with_flatlines['data'][5, :] = 0.0  # Flatline channel (2D data)

        EEG, HP, BUR, removed_channels = clean_artifacts(
            eeg_with_flatlines,
            FlatlineCriterion='off',
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
        )

        # Should not have removed any channels
        self.assertEqual(EEG['nbchan'], eeg_with_flatlines['nbchan'])


class TestCleanArtifactsHighpass(DebuggableTestCase):
    """Test cases for highpass filtering (drift removal)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_highpass_filtering(self):
        """Test highpass filtering."""
        original_data = self.test_eeg['data'].copy()  # Save before call

        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Highpass=(0.5, 1.0),
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            FlatlineCriterion='off',
        )

        # HP should contain the highpass filtered data
        self.assertIsInstance(HP, dict)
        self.assertIn('data', HP)

        # Data should be different after filtering
        self.assertFalse(np.array_equal(HP['data'], original_data))

    def test_clean_artifacts_highpass_off(self):
        """Test highpass filtering disabled."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Highpass='off',
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            FlatlineCriterion='off',
        )

        # Data should be unchanged
        np.testing.assert_array_equal(HP['data'], self.test_eeg['data'])


class TestCleanArtifactsChannelCleaning(DebuggableTestCase):
    """Test cases for channel cleaning."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_channel_criterion(self):
        """Test channel correlation criterion."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion=0.9,  # High threshold
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have removed some channels with high threshold
        self.assertLessEqual(EEG['nbchan'], self.test_eeg['nbchan'])

    def test_clean_artifacts_line_noise_criterion(self):
        """Test line noise criterion."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion=2.0,  # Low threshold
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have removed some channels with low threshold
        self.assertLessEqual(EEG['nbchan'], self.test_eeg['nbchan'])

    def test_clean_artifacts_both_channel_criteria(self):
        """Test both channel and line noise criteria."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion=0.8,
            LineNoiseCriterion=4.0,
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have removed some channels
        self.assertLessEqual(EEG['nbchan'], self.test_eeg['nbchan'])


class TestCleanArtifactsBurstCleaning(DebuggableTestCase):
    """Test cases for burst cleaning (ASR)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_burst_criterion(self):
        """Test burst criterion."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion=5.0,
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # BUR should contain the burst repaired data
        self.assertIsInstance(BUR, dict)
        self.assertIn('data', BUR)

    def test_clean_artifacts_burst_rejection(self):
        """Test burst rejection mode."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion=5.0,
            BurstRejection='on',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have removed some samples
        self.assertLessEqual(EEG['pnts'], self.test_eeg['pnts'])

    def test_clean_artifacts_burst_off(self):
        """Test burst cleaning disabled."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Data should be unchanged
        np.testing.assert_array_equal(BUR['data'], self.test_eeg['data'])


class TestCleanArtifactsWindowCleaning(DebuggableTestCase):
    """Test cases for window cleaning."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_window_criterion(self):
        """Test window criterion."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion=0.5,  # Allow 50% bad channels per window
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have removed some samples
        self.assertLessEqual(EEG['pnts'], self.test_eeg['pnts'])

    def test_clean_artifacts_window_off(self):
        """Test window cleaning disabled."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Data should be unchanged
        self.assertEqual(EEG['pnts'], self.test_eeg['pnts'])


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

    def test_clean_artifacts_channels_ignore(self):
        """Test channel exclusion."""
        channels_to_ignore = ['Ch1', 'Ch2']
        original_nbchan = self.test_eeg['nbchan']  # Save before call

        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Channels_ignore=channels_to_ignore,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should have fewer channels
        self.assertEqual(EEG['nbchan'], original_nbchan - len(channels_to_ignore))


class TestCleanArtifactsParameterValidation(DebuggableTestCase):
    """Test cases for parameter validation and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_invalid_channel_criterion_type(self):
        """Test clean_artifacts with invalid ChannelCriterion type."""
        # Should accept numeric values and 'off'
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion=0.8,
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

    def test_clean_artifacts_invalid_line_noise_criterion_type(self):
        """Test clean_artifacts with invalid LineNoiseCriterion type."""
        # Should accept numeric values and 'off'
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion=4.0,
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

    def test_clean_artifacts_invalid_burst_criterion_type(self):
        """Test clean_artifacts with invalid BurstCriterion type."""
        # Should accept numeric values and 'off'
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion=5.0,
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

    def test_clean_artifacts_invalid_window_criterion_type(self):
        """Test clean_artifacts with invalid WindowCriterion type."""
        # Should accept numeric values and 'off'
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion=0.25,
            Highpass='off',
            FlatlineCriterion='off',
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

    def test_clean_artifacts_invalid_flatline_criterion_type(self):
        """Test clean_artifacts with invalid FlatlineCriterion type."""
        # Should accept numeric values and 'off'
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion=5.0,
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

    def test_clean_artifacts_invalid_burst_rejection_type(self):
        """Test clean_artifacts with invalid BurstRejection type."""
        # Should accept 'on' and 'off' strings
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
            BurstRejection='on',
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
            BurstRejection='off',
        )

    def test_clean_artifacts_documented_distance_metrics_with_asr_disabled(self):
        """Test clean_artifacts accepts documented Distance spellings when ASR is disabled."""
        # Valid cases
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
            Distance='euclidian',
        )
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
            Distance='riemannian',
        )

    def test_clean_artifacts_rejects_unknown_distance_metric(self):
        """Test clean_artifacts rejects unknown Distance spellings before cleaning."""
        with self.assertRaisesRegex(ValueError, "Distance must be"):
            clean_artifacts(
                self.test_eeg,
                ChannelCriterion='off',
                LineNoiseCriterion='off',
                BurstCriterion='off',
                WindowCriterion='off',
                Highpass='off',
                FlatlineCriterion='off',
                Distance='riemann',
            )

    def test_clean_artifacts_negative_values(self):
        """Test clean_artifacts with negative parameter values."""
        # Some parameters should handle negative values gracefully
        try:
            clean_artifacts(
                self.test_eeg,
                ChannelCriterion='off',
                LineNoiseCriterion='off',
                BurstCriterion='off',
                WindowCriterion='off',
                Highpass='off',
                FlatlineCriterion='off',
                MaxMem=-1,
            )  # Negative MaxMem should be handled
        except Exception:
            # Negative values may cause errors - this is acceptable
            pass

    def test_clean_artifacts_zero_values(self):
        """Test clean_artifacts with zero parameter values."""
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion=0.0,  # Zero correlation threshold
            LineNoiseCriterion=0.0,
            BurstCriterion='off',
            WindowCriterion=0.0,
            Highpass='off',
            FlatlineCriterion=0.0,
        )

    def test_clean_artifacts_extreme_values(self):
        """Test clean_artifacts with extreme parameter values."""
        clean_artifacts(
            self.test_eeg,
            ChannelCriterion=1.0,  # Perfect correlation required
            LineNoiseCriterion=100.0,
            BurstCriterion='off',
            WindowCriterion=1.0,
            Highpass='off',
            FlatlineCriterion=1000.0,
        )


class TestCleanArtifactsParameters(DebuggableTestCase):
    """Test cases for various parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_available_ram(self):
        """Test available RAM parameter."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            availableRAM_GB=2.0,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should complete without error
        self.assertIsInstance(EEG, dict)

    def test_clean_artifacts_distance_metric(self):
        """Test distance metric parameter."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            Distance='euclidian',
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should complete without error
        self.assertIsInstance(EEG, dict)

    def test_clean_artifacts_max_mem(self):
        """Test max memory parameter."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            MaxMem=128,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Should complete without error
        self.assertIsInstance(EEG, dict)


class TestCleanArtifactsIntegration(DebuggableTestCase):
    """Integration test cases for clean_artifacts."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_artifacts_full_pipeline(self):
        """Test the full clean_artifacts pipeline."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            FlatlineCriterion=5.0,
            Highpass=(0.25, 0.75),
            ChannelCriterion=0.8,
            LineNoiseCriterion=4.0,
            BurstCriterion=5.0,
            WindowCriterion=0.25,
        )

        # Check all return values
        self.assertIsInstance(EEG, dict)
        self.assertIsInstance(HP, dict)
        self.assertIsInstance(BUR, dict)
        self.assertIsInstance(removed_channels, np.ndarray)

        # Check data integrity
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('pnts', EEG)

        # Check that some processing occurred
        self.assertLessEqual(EEG['nbchan'], self.test_eeg['nbchan'])

    def test_clean_artifacts_return_values(self):
        """Test that all return values have correct structure."""
        EEG, HP, BUR, removed_channels = clean_artifacts(
            self.test_eeg,
            ChannelCriterion='off',
            LineNoiseCriterion='off',
            BurstCriterion='off',
            WindowCriterion='off',
            Highpass='off',
            FlatlineCriterion='off',
        )

        # Check EEG structure
        self.assertIn('data', EEG)
        self.assertIn('srate', EEG)
        self.assertIn('nbchan', EEG)
        self.assertIn('pnts', EEG)
        self.assertIn('etc', EEG)

        # Check HP structure (should be same as EEG when no highpass)
        self.assertIn('data', HP)
        self.assertIn('srate', HP)
        self.assertIn('nbchan', HP)
        self.assertIn('pnts', HP)

        # Check BUR structure (should be same as EEG when no burst cleaning)
        self.assertIn('data', BUR)
        self.assertIn('srate', BUR)
        self.assertIn('nbchan', BUR)
        self.assertIn('pnts', BUR)

        # Check removed_channels array
        self.assertEqual(len(removed_channels), self.test_eeg['nbchan'])
        self.assertTrue(np.issubdtype(removed_channels.dtype, np.bool_))


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


class TestPopCleanRawdataNoMutation(DebuggableTestCase):
    """Regression test that pop_clean_rawdata never mutates the caller's EEG."""

    def setUp(self):
        np.random.seed(13)
        self.test_eeg = create_test_eeg()

    def test_does_not_mutate_input(self):
        """The wrapper must deep-copy so the caller's dataset is untouched."""
        EEG_in = self.test_eeg
        original_data = EEG_in['data'].copy()
        original_nbchan = EEG_in['nbchan']

        cleaned = pop_clean_rawdata(
            EEG_in,
            gui=False,
            ChannelCriterion=0.8,
            LineNoiseCriterion=4.0,
            BurstCriterion='off',
            WindowCriterion=0.25,
        )

        # Caller's data, channel count, and etc are all unchanged.
        self.assertTrue(np.array_equal(original_data, EEG_in['data']))
        self.assertEqual(EEG_in['nbchan'], original_nbchan)
        self.assertNotIn('clean_channel_mask', EEG_in.get('etc', {}))
        self.assertNotIn('clean_sample_mask', EEG_in.get('etc', {}))
        # The returned dataset is a distinct object.
        self.assertIsNot(cleaned, EEG_in)


class TestCleanArtifactsErrorSurfacing(DebuggableTestCase):
    """Errors inside the selection / channel-cleaning paths must surface, not be masked."""

    def setUp(self):
        self.test_eeg = create_test_eeg()

    def test_pop_select_internal_error_propagates(self):
        """A non-ImportError raised inside pop_select must propagate, not silently
        fall back to manual label selection (which could select different channels).
        """
        import eegprep

        original = eegprep.pop_select

        def failing_pop_select(*args, **kwargs):
            raise RuntimeError("simulated pop_select bug")

        eegprep.pop_select = failing_pop_select
        try:
            with self.assertRaises(RuntimeError):
                clean_artifacts(
                    self.test_eeg,
                    Channels_ignore=['EEG001', 'EEG002'],
                    ChannelCriterion='off',
                    LineNoiseCriterion='off',
                    BurstCriterion='off',
                    WindowCriterion='off',
                    Highpass='off',
                    FlatlineCriterion='off',
                )
        finally:
            eegprep.pop_select = original

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

    def test_clean_channels_unexpected_value_error_propagates(self):
        """A ValueError from clean_channels unrelated to missing locations must
        propagate rather than silently switching to the no-locs algorithm.
        """
        import eegprep.plugins.clean_rawdata.clean_artifacts as ca_mod

        original = ca_mod.clean_channels

        def boom(*args, **kwargs):
            raise ValueError("totally unrelated bug")

        ca_mod.clean_channels = boom
        try:
            with self.assertRaises(ValueError) as cm:
                clean_artifacts(
                    self.test_eeg,
                    ChannelCriterion=0.8,
                    LineNoiseCriterion='off',
                    BurstCriterion='off',
                    WindowCriterion='off',
                    Highpass='off',
                    FlatlineCriterion='off',
                )
            self.assertIn('totally unrelated bug', str(cm.exception))
        finally:
            ca_mod.clean_channels = original

    def test_clean_channels_location_error_falls_back(self):
        """A missing-locations ValueError still triggers the no-locs fallback."""
        import eegprep.plugins.clean_rawdata.clean_artifacts as ca_mod

        original_cc = ca_mod.clean_channels
        original_nolocs = ca_mod.clean_channels_nolocs

        def locs_error(*args, **kwargs):
            raise ValueError('To use this function most of your channels should have X,Y,Z location measurements.')

        called = {'nolocs': False}

        def fake_nolocs(EEG, **kwargs):
            called['nolocs'] = True
            return EEG, np.zeros(EEG['nbchan'], dtype=bool)

        ca_mod.clean_channels = locs_error
        ca_mod.clean_channels_nolocs = fake_nolocs
        try:
            clean_artifacts(
                self.test_eeg,
                ChannelCriterion=0.8,
                LineNoiseCriterion='off',
                BurstCriterion='off',
                WindowCriterion='off',
                Highpass='off',
                FlatlineCriterion='off',
            )
            self.assertTrue(called['nolocs'])
        finally:
            ca_mod.clean_channels = original_cc
            ca_mod.clean_channels_nolocs = original_nolocs


if __name__ == '__main__':
    unittest.main()
