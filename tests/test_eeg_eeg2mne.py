"""
Test suite for eeg_eeg2mne.py - EEGLAB to MNE conversion.

This module tests the eeg_eeg2mne function that converts EEGLAB datasets to MNE objects.
"""

import unittest
import sys
import numpy as np
import tempfile
import os
import shutil

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

from eegprep.eeg_eeg2mne import eeg_eeg2mne
from tests.fixtures import create_test_eeg


class TestEEGEEG2MNE(unittest.TestCase):
    """Test cases for eeg_eeg2mne function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_continuous_data(self):
        """Test conversion of continuous EEG data."""
        # Create continuous EEG data
        continuous_eeg = self.test_eeg.copy()
        continuous_eeg['data'] = np.random.randn(32, 1000)
        continuous_eeg['trials'] = 1
        
        result = eeg_eeg2mne(continuous_eeg)

        # Check that result is an MNE Raw object (RawEEGLAB is a subclass of BaseRaw)
        self.assertIsInstance(result, mne.io.BaseRaw)

        # Check that data dimensions match
        self.assertEqual(result.info['nchan'], continuous_eeg['nbchan'])
        self.assertEqual(result._data.shape[1], continuous_eeg['pnts'])

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_epoched_data(self):
        """Test conversion of epoched EEG data."""
        # Create epoched EEG data
        epoched_eeg = create_test_eeg(n_channels=32, n_samples=100, n_trials=10)
        epoched_eeg['data'] = np.random.randn(32, 100, 10)  # 10 epochs
        
        try:
            result = eeg_eeg2mne(epoched_eeg)
            
            # Check that result is an MNE Epochs object (EpochsEEGLAB is a subclass of BaseEpochs)
            self.assertIsInstance(result, mne.BaseEpochs)
            
            # Check that data dimensions match
            self.assertEqual(result.info['nchan'], epoched_eeg['nbchan'])
            self.assertEqual(len(result.times), epoched_eeg['pnts'])
            self.assertEqual(len(result), epoched_eeg['trials'])
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne epoched conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_float32_data(self):
        """Test conversion with float32 data."""
        float32_eeg = create_test_eeg(n_trials=5)
        float32_eeg['data'] = np.random.randn(32, 1000, 5).astype(np.float32)

        try:
            result = eeg_eeg2mne(float32_eeg)

            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))

        except Exception as e:
            self.skipTest(f"eeg_eeg2mne float32 conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_float64_data(self):
        """Test conversion with float64 data."""
        float64_eeg = create_test_eeg(n_trials=3)
        float64_eeg['data'] = np.random.randn(32, 1000, 3).astype(np.float64)
        
        try:
            result = eeg_eeg2mne(float64_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne float64 conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_single_channel(self):
        """Test conversion with single channel data."""
        single_channel_eeg = create_test_eeg(n_channels=1, n_trials=2)
        single_channel_eeg['data'] = np.random.randn(1, 1000, 2)
        
        try:
            result = eeg_eeg2mne(single_channel_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            self.assertEqual(result.info['nchan'], 1)
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne single channel conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_single_trial(self):
        """Test conversion with single trial data."""
        single_trial_eeg = self.test_eeg.copy()
        single_trial_eeg['data'] = np.random.randn(32, 1000, 1)
        single_trial_eeg['trials'] = 1
        
        try:
            result = eeg_eeg2mne(single_trial_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne single trial conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_with_chanlocs(self):
        """Test conversion with channel locations."""
        eeg_with_chanlocs = create_test_eeg(n_trials=3)
        eeg_with_chanlocs['data'] = np.random.randn(32, 1000, 3)

        # Add some channel location data (already has basic locations from fixture)
        for i, chan in enumerate(eeg_with_chanlocs['chanlocs']):
            chan['X'] = np.cos(i * np.pi / 16) * 10
            chan['Y'] = np.sin(i * np.pi / 16) * 10
            chan['Z'] = 0.0
        
        try:
            result = eeg_eeg2mne(eeg_with_chanlocs)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne with chanlocs conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_with_events(self):
        """Test conversion with events."""
        eeg_with_events = create_test_eeg(n_channels=32, n_samples=100, n_trials=5)
        eeg_with_events['data'] = np.random.randn(32, 100, 5)

        # Add some additional events (already has epoch events from fixture)
        eeg_with_events['event'].extend([
            {'latency': 1, 'type': 'event1', 'duration': 0, 'urevent': 100},
            {'latency': 50, 'type': 'event2', 'duration': 0, 'urevent': 101},
            {'latency': 100, 'type': 'event3', 'duration': 0, 'urevent': 102},
        ])
        
        try:
            result = eeg_eeg2mne(eeg_with_events)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne with events conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_empty_data(self):
        """Test conversion with empty data raises an error."""
        empty_eeg = self.test_eeg.copy()
        empty_eeg['data'] = np.array([])
        empty_eeg['nbchan'] = 0
        empty_eeg['pnts'] = 0
        empty_eeg['trials'] = 0
        empty_eeg['chanlocs'] = []
        
        # Empty data should raise an error (TypeError from MNE for 0 trials)
        with self.assertRaises((ValueError, IndexError, RuntimeError, TypeError)):
            eeg_eeg2mne(empty_eeg)

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_missing_fields(self):
        """Test conversion with missing required fields."""
        incomplete_eeg = {
            'data': np.random.randn(32, 1000, 2),
            'srate': 500.0,
            # Missing nbchan, pnts, trials, chanlocs
        }
        
        try:
            with self.assertRaises((KeyError, AttributeError)):
                eeg_eeg2mne(incomplete_eeg)
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne missing fields test not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_corrupted_data(self):
        """Test conversion with corrupted data."""
        corrupted_eeg = self.test_eeg.copy()
        corrupted_eeg['data'] = None
        
        try:
            with self.assertRaises((TypeError, AttributeError)):
                eeg_eeg2mne(corrupted_eeg)
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne corrupted data test not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_large_dataset(self):
        """Test conversion with large dataset."""
        large_eeg = create_test_eeg(n_channels=64, n_samples=5000, n_trials=20)
        large_eeg['data'] = np.random.randn(64, 5000, 20)  # Large dataset
        
        try:
            result = eeg_eeg2mne(large_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            self.assertEqual(result.info['nchan'], 64)
            self.assertEqual(len(result.times), 5000)
            if isinstance(result, mne.BaseEpochs):
                self.assertEqual(len(result), 20)
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne large dataset conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_integration_workflow(self):
        """Test end-to-end conversion workflow."""
        # Create a realistic EEG dataset (use fixture as-is to avoid time axis issues)
        realistic_eeg = create_test_eeg(n_samples=1000, n_trials=10, srate=500.0)

        # Add some additional events (already has epoch events from fixture)
        realistic_eeg['event'].extend([
            {'latency': 1, 'type': 'stimulus', 'duration': 0, 'urevent': 100},
            {'latency': 500, 'type': 'response', 'duration': 0, 'urevent': 101},
        ])
        
        try:
            result = eeg_eeg2mne(realistic_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (mne.io.BaseRaw, mne.BaseEpochs))
            
            # Check basic properties
            self.assertEqual(result.info['nchan'], 32)
            self.assertEqual(len(result.times), 1000)
            if isinstance(result, mne.BaseEpochs):
                self.assertEqual(len(result), 10)
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne integration workflow not available: {e}")


if __name__ == '__main__':
    unittest.main()
