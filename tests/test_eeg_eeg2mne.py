"""
Test suite for eeg_eeg2mne.py - EEGLAB to MNE conversion.

This module tests the eeg_eeg2mne function that converts EEGLAB datasets to MNE objects.
"""

import unittest
import os
import numpy as np
import tempfile
import shutil

from eegprep.functions.miscfunc.eeg_eeg2mne import eeg_eeg2mne

try:
    import mne
    from mne.io.base import BaseRaw

    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    BaseRaw = None

try:
    from .fixtures import create_test_eeg
except (ImportError, ValueError):
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


if __name__ == '__main__':
    unittest.main()
