"""
Test suite for clean_drifts.py - Drift removal filtering.

This module tests the clean_drifts function that removes low-frequency
drifts from EEG data using high-pass filtering.
"""

import unittest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.plugins.clean_rawdata.clean_drifts import clean_drifts
from eegprep.utils.testing import DebuggableTestCase

from tests.fixtures import create_test_eeg as _create_test_eeg


def create_test_eeg():
    """Continuous (2D) EEG fixture sized for clean_drifts (20 s at 500 Hz)."""
    return _create_test_eeg(n_channels=32, n_samples=10000, srate=500.0, n_trials=1)


class TestCleanDriftsBasic(DebuggableTestCase):
    """Basic test cases for clean_drifts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_drifts_basic_functionality(self):
        """Test basic clean_drifts functionality with default parameters."""
        result = clean_drifts(self.test_eeg.copy())

        # Check that EEG structure is preserved
        self.assertIn('data', result)
        self.assertIn('srate', result)
        self.assertIn('nbchan', result)
        self.assertIn('pnts', result)
        self.assertIn('etc', result)

        # Check that data dimensions are preserved
        self.assertEqual(result['srate'], self.test_eeg['srate'])
        self.assertEqual(result['nbchan'], self.test_eeg['nbchan'])
        self.assertEqual(result['pnts'], self.test_eeg['pnts'])
        self.assertEqual(result['trials'], self.test_eeg['trials'])

        # Check that data type is float64
        self.assertEqual(result['data'].dtype, np.float64)

        # Check that filter kernel is stored
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_default_parameters(self):
        """Test clean_drifts with default parameters."""
        result = clean_drifts(self.test_eeg.copy())

        # Should work with default parameters
        self.assertIn('data', result)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_custom_transition(self):
        """Test clean_drifts with custom transition band."""
        result = clean_drifts(self.test_eeg.copy(), transition=(1.0, 2.0))

        # Should work with custom transition band
        self.assertIn('data', result)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_custom_attenuation(self):
        """Test clean_drifts with custom attenuation."""
        result = clean_drifts(self.test_eeg.copy(), attenuation=60.0)

        # Should work with custom attenuation
        self.assertIn('data', result)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_fir_method(self):
        """Test clean_drifts with FIR method."""
        result = clean_drifts(self.test_eeg.copy(), method='fir')

        # Should work with FIR method
        self.assertIn('data', result)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_fft_method(self):
        """Test clean_drifts with FFT method."""
        result = clean_drifts(self.test_eeg.copy(), method='fft')

        # Should work with FFT method
        self.assertIn('data', result)
        self.assertIn('clean_drifts_kernel', result['etc'])


class TestCleanDriftsEdgeCases(DebuggableTestCase):
    """Edge case test cases for clean_drifts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_drifts_single_channel(self):
        """Test clean_drifts with single channel data."""
        # Create single channel data (2D continuous)
        single_channel_eeg = self.test_eeg.copy()
        single_channel_eeg['data'] = np.random.randn(1, 10000)
        single_channel_eeg['nbchan'] = 1
        single_channel_eeg['chanlocs'] = [single_channel_eeg['chanlocs'][0]]

        result = clean_drifts(single_channel_eeg)

        # Should work with single channel
        self.assertEqual(result['nbchan'], 1)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_single_trial(self):
        """Test clean_drifts with continuous (single trial) data."""
        # Create continuous data (2D - single trial is the normal case)
        single_trial_eeg = self.test_eeg.copy()
        single_trial_eeg['data'] = np.random.randn(32, 10000)
        single_trial_eeg['trials'] = 1

        result = clean_drifts(single_trial_eeg)

        # Should work with single trial
        self.assertEqual(result['trials'], 1)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_continuous_data(self):
        """Test clean_drifts with continuous (2D) data."""
        # Create continuous data (2D)
        continuous_eeg = self.test_eeg.copy()
        continuous_eeg['data'] = np.random.randn(32, 1000)
        continuous_eeg['trials'] = 1

        result = clean_drifts(continuous_eeg)

        # Should work with continuous data
        self.assertIn('data', result)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_float32_data(self):
        """Test clean_drifts with float32 data."""
        # Create float32 data
        float32_eeg = self.test_eeg.copy()
        float32_eeg['data'] = np.random.randn(32, 10000).astype(np.float32)

        result = clean_drifts(float32_eeg)

        # Should convert to float64
        self.assertEqual(result['data'].dtype, np.float64)
        self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_float64_data(self):
        """Test clean_drifts with float64 data."""
        # Create float64 data
        float64_eeg = self.test_eeg.copy()
        float64_eeg['data'] = np.random.randn(32, 10000).astype(np.float64)

        result = clean_drifts(float64_eeg)

        # Should remain float64
        self.assertEqual(result['data'].dtype, np.float64)
        self.assertIn('clean_drifts_kernel', result['etc'])


class TestCleanDriftsParameters(DebuggableTestCase):
    """Parameter test cases for clean_drifts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_drifts_different_transition_bands(self):
        """Test clean_drifts with different transition bands."""
        # Test different transition bands
        transitions = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]

        for transition in transitions:
            result = clean_drifts(self.test_eeg.copy(), transition=transition)
            self.assertIn('data', result)
            self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_different_attenuations(self):
        """Test clean_drifts with different attenuation values."""
        # Test different attenuation values
        attenuations = [40.0, 60.0, 80.0, 100.0]

        for attenuation in attenuations:
            result = clean_drifts(self.test_eeg.copy(), attenuation=attenuation)
            self.assertIn('data', result)
            self.assertIn('clean_drifts_kernel', result['etc'])

    def test_clean_drifts_both_methods(self):
        """Test clean_drifts with both FIR and FFT methods."""
        # Test both methods
        methods = ['fir', 'fft']

        for method in methods:
            result = clean_drifts(self.test_eeg.copy(), method=method)
            self.assertIn('data', result)
            self.assertIn('clean_drifts_kernel', result['etc'])


class TestCleanDriftsIntegration(DebuggableTestCase):
    """Integration test cases for clean_drifts function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_drifts_preserves_structure(self):
        """Test that clean_drifts preserves EEG structure."""
        result = clean_drifts(self.test_eeg.copy())

        # Check that all essential fields are preserved
        essential_fields = ['data', 'srate', 'nbchan', 'pnts', 'trials', 'xmin', 'xmax', 'times', 'chanlocs']
        for field in essential_fields:
            self.assertIn(field, result)

        # Check that data integrity is maintained
        self.assertEqual(result['srate'], self.test_eeg['srate'])
        self.assertEqual(result['nbchan'], self.test_eeg['nbchan'])
        self.assertEqual(result['pnts'], self.test_eeg['pnts'])
        self.assertEqual(result['trials'], self.test_eeg['trials'])

    def test_clean_drifts_data_modification(self):
        """Test that clean_drifts actually modifies the data."""
        original_data = self.test_eeg['data'].copy()
        result = clean_drifts(self.test_eeg.copy())

        # Data should be modified (filtered)
        self.assertFalse(np.array_equal(original_data, result['data']))

        # But shape should be preserved
        self.assertEqual(original_data.shape, result['data'].shape)

    def test_clean_drifts_kernel_properties(self):
        """Test properties of the filter kernel."""
        result = clean_drifts(self.test_eeg.copy())

        kernel = result['etc']['clean_drifts_kernel']

        # Kernel should be a numpy array
        self.assertIsInstance(kernel, np.ndarray)

        # Kernel should not be empty
        self.assertGreater(len(kernel), 0)

        # Kernel should be 1D
        self.assertEqual(kernel.ndim, 1)


if __name__ == '__main__':
    unittest.main()
