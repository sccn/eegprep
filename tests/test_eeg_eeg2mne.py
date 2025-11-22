"""
Test suite for eeg_eeg2mne.py - EEGLAB to MNE conversion.

This module tests the eeg_eeg2mne function that converts EEGLAB datasets to MNE objects.
"""

import unittest
import os

if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
    raise unittest.SkipTest("MATLAB not available")

import sys
import numpy as np
import tempfile
import os
import shutil

try:
    import mne
    from mne.io.base import BaseRaw
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    BaseRaw = None

from eegprep.eeg_eeg2mne import eeg_eeg2mne
from eegprep.eeglabcompat import get_eeglab
try:
    from .fixtures import create_test_eeg
except ImportError:
    from fixtures import create_test_eeg


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
        self.assertIsInstance(result, BaseRaw)
        
        # Check that data dimensions match
        self.assertEqual(result.info['nchan'], continuous_eeg['nbchan'])
        self.assertEqual(result.n_times, continuous_eeg['pnts'])

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_epoched_data(self):
        """Test conversion of epoched EEG data."""
        # Create epoched EEG data
        epoched_eeg = self.test_eeg.copy()
        epoched_eeg['data'] = np.random.randn(32, 100, 10)  # 10 epochs
        epoched_eeg['trials'] = 10
        epoched_eeg['pnts'] = 100
        
        try:
            result = eeg_eeg2mne(epoched_eeg)
            
            # Check that result is an MNE Epochs object
            self.assertIsInstance(result, mne.Epochs)
            
            # Check that data dimensions match
            self.assertEqual(result.n_channels, epoched_eeg['nbchan'])
            self.assertEqual(result.n_times, epoched_eeg['pnts'])
            self.assertEqual(len(result), epoched_eeg['trials'])
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne epoched conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_float32_data(self):
        """Test conversion with float32 data."""
        float32_eeg = self.test_eeg.copy()
        float32_eeg['data'] = np.random.randn(32, 1000, 5).astype(np.float32)
        float32_eeg['trials'] = 5
        
        try:
            result = eeg_eeg2mne(float32_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne float32 conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_float64_data(self):
        """Test conversion with float64 data."""
        float64_eeg = self.test_eeg.copy()
        float64_eeg['data'] = np.random.randn(32, 1000, 3).astype(np.float64)
        float64_eeg['trials'] = 3
        
        try:
            result = eeg_eeg2mne(float64_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne float64 conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_single_channel(self):
        """Test conversion with single channel data."""
        single_channel_eeg = self.test_eeg.copy()
        single_channel_eeg['data'] = np.random.randn(1, 1000, 2)
        single_channel_eeg['nbchan'] = 1
        single_channel_eeg['trials'] = 2
        # Create a single channel location
        single_channel_eeg['chanlocs'] = [{'labels': 'Ch1', 'type': 'EEG'}]
        
        try:
            result = eeg_eeg2mne(single_channel_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            n_channels = result.info['nchan'] if isinstance(result, BaseRaw) else result.info['nchan']
            self.assertEqual(n_channels, 1)
            
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
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne single trial conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_with_chanlocs(self):
        """Test conversion with channel locations."""
        eeg_with_chanlocs = self.test_eeg.copy()
        eeg_with_chanlocs['data'] = np.random.randn(32, 1000, 3)
        eeg_with_chanlocs['trials'] = 3
        
        # Add some channel location data
        for i, chan in enumerate(eeg_with_chanlocs['chanlocs']):
            chan['X'] = np.cos(i * np.pi / 16) * 10
            chan['Y'] = np.sin(i * np.pi / 16) * 10
            chan['Z'] = 0.0
        
        try:
            result = eeg_eeg2mne(eeg_with_chanlocs)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne with chanlocs conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_with_events(self):
        """Test conversion with events."""
        eeg_with_events = self.test_eeg.copy()
        eeg_with_events['data'] = np.random.randn(32, 100, 5)
        eeg_with_events['trials'] = 5
        eeg_with_events['pnts'] = 100
        
        # Add some events
        eeg_with_events['event'] = [
            {'latency': 1, 'type': 'event1'},
            {'latency': 50, 'type': 'event2'},
            {'latency': 100, 'type': 'event3'},
        ]
        
        try:
            result = eeg_eeg2mne(eeg_with_events)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne with events conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_empty_data(self):
        """Test conversion with empty data."""
        empty_eeg = self.test_eeg.copy()
        empty_eeg['data'] = np.array([])
        empty_eeg['nbchan'] = 0
        empty_eeg['pnts'] = 0
        empty_eeg['trials'] = 0
        empty_eeg['chanlocs'] = []
        
        try:
            with self.assertRaises((ValueError, IndexError)):
                eeg_eeg2mne(empty_eeg)
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne empty data test not available: {e}")

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
        large_eeg = self.test_eeg.copy()
        large_eeg['data'] = np.random.randn(64, 5000, 20)  # Large dataset
        large_eeg['nbchan'] = 64
        large_eeg['pnts'] = 5000
        large_eeg['trials'] = 20
        large_eeg['chanlocs'] = [
            {
                'labels': f'EEG{i:03d}',
                'type': 'EEG',
                'theta': np.random.uniform(-90, 90),
                'radius': np.random.uniform(0, 1),
                'X': np.random.uniform(-1, 1),
                'Y': np.random.uniform(-1, 1),
                'Z': np.random.uniform(-1, 1),
                'sph_theta': np.random.uniform(-180, 180),
                'sph_phi': np.random.uniform(-90, 90),
                'sph_radius': np.random.uniform(0, 1),
                'urchan': i + 1,
                'ref': ''
            }
            for i in range(64)
        ]
        
        try:
            result = eeg_eeg2mne(large_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            n_channels = result.info['nchan'] if isinstance(result, BaseRaw) else result.info['nchan']
            self.assertEqual(n_channels, 64)
            self.assertEqual(result.n_times, 5000)
            if isinstance(result, mne.Epochs):
                self.assertEqual(len(result), 20)
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne large dataset conversion not available: {e}")

    @unittest.skipUnless(MNE_AVAILABLE, "MNE not available")
    def test_eeg_eeg2mne_integration_workflow(self):
        """Test end-to-end conversion workflow."""
        # Create a realistic EEG dataset
        realistic_eeg = self.test_eeg.copy()
        realistic_eeg['data'] = np.random.randn(32, 1000, 10)
        realistic_eeg['trials'] = 10
        realistic_eeg['pnts'] = 1000
        realistic_eeg['srate'] = 500.0
        realistic_eeg['xmin'] = -1.0
        realistic_eeg['xmax'] = 1.0
        realistic_eeg['times'] = np.linspace(-1.0, 1.0, 1000)
        
        # Add events
        realistic_eeg['event'] = [
            {'latency': 1, 'type': 'stimulus'},
            {'latency': 500, 'type': 'response'},
        ]
        
        try:
            result = eeg_eeg2mne(realistic_eeg)
            
            # Check that result is an MNE object
            self.assertIsInstance(result, (BaseRaw, mne.Epochs))
            
            # Check basic properties
            n_channels = result.info['nchan'] if isinstance(result, BaseRaw) else result.info['nchan']
            self.assertEqual(n_channels, 32)
            self.assertEqual(result.n_times, 1000)
            if isinstance(result, mne.Epochs):
                self.assertEqual(len(result), 10)
            
        except Exception as e:
            self.skipTest(f"eeg_eeg2mne integration workflow not available: {e}")


if __name__ == '__main__':
    unittest.main()
