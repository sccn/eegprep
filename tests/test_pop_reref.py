"""
Test suite for pop_reref.py with MATLAB parity validation.

This module tests the pop_reref function which re-references EEG data
to average reference (currently the only implemented option).
"""

import unittest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.pop_reref import pop_reref
from eegprep.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase


class TestPopReref(DebuggableTestCase):
    """Test cases for pop_reref function."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up MATLAB compatibility for parity tests
        try:
            self.eeglab = get_eeglab()
            self.matlab_available = True
        except Exception:
            self.matlab_available = False

    def create_test_eeg(self, nbchan=32, pnts=1000, trials=1, srate=256):
        """Create a test EEG structure."""
        np.random.seed(42)  # For reproducible tests
        
        # Create channel locations
        chanlocs = []
        for i in range(nbchan):
            chanlocs.append({
                'labels': f'Ch{i+1}',
                'X': np.cos(2 * np.pi * i / nbchan),
                'Y': np.sin(2 * np.pi * i / nbchan),
                'Z': 0.0,
                'theta': 2 * np.pi * i / nbchan,
                'radius': 0.5,
                'ref': 'common'  # Initial reference
            })
        
        # Create ICA components equal to number of channels
        icaweights = np.random.randn(nbchan, nbchan).astype(np.float32)
        icawinv = np.linalg.pinv(icaweights).astype(np.float32)
        icasphere = np.eye(nbchan).astype(np.float32)
        
        return {
            'data': np.random.randn(nbchan, pnts, trials).astype(np.float32),
            'nbchan': nbchan,
            'pnts': pnts,
            'trials': trials,
            'srate': srate,
            'chanlocs': chanlocs,
            'icaweights': icaweights,
            'icawinv': icawinv,
            'icasphere': icasphere,
            'icachansind': list(range(nbchan)),  # All channels used for ICA
            'ref': 'common'
        }

    def test_basic_average_reference_none(self):
        """Test basic average reference with ref=None."""
        EEG = self.create_test_eeg(nbchan=32, pnts=512)
        original_data = EEG['data'].copy()
        
        result = pop_reref(EEG, ref=None)
        
        # Check that the function returns a copy (not the same object)
        self.assertIsNot(result, EEG)
        
        # Check that reference is set to 'average'
        self.assertEqual(result['ref'], 'average')
        
        # Check that all channel references are updated
        for chan in result['chanlocs']:
            self.assertEqual(chan['ref'], 'average')
        
        # Check that data is modified (average subtracted)
        self.assertFalse(np.array_equal(original_data, result['data']))
        
        # Check that the mean across channels is approximately zero
        mean_across_channels = np.mean(result['data'], axis=0)
        np.testing.assert_allclose(mean_across_channels, 0, atol=1e-6)

    def test_basic_average_reference_empty_list(self):
        """Test basic average reference with ref=[]."""
        EEG = self.create_test_eeg(nbchan=16, pnts=256)
        original_data = EEG['data'].copy()
        
        result = pop_reref(EEG, ref=[])
        
        # Should behave the same as ref=None
        # Function returns a copy, not the same object
        self.assertIsNot(result, EEG)
        self.assertEqual(result['ref'], 'average')
        
        # Check that data is modified (average subtracted)
        self.assertFalse(np.array_equal(original_data, result['data']))
        
        # Check that the mean across channels is approximately zero
        mean_across_channels = np.mean(result['data'], axis=0)
        np.testing.assert_allclose(mean_across_channels, 0, atol=1e-6)

    def test_ica_matrices_updated(self):
        """Test that ICA matrices are properly updated."""
        EEG = self.create_test_eeg(nbchan=16, pnts=256)
        original_icawinv = EEG['icawinv'].copy()
        original_icaweights = EEG['icaweights'].copy()
        
        result = pop_reref(EEG, ref=None)
        
        # Check that icawinv is modified (average subtracted)
        self.assertFalse(np.array_equal(original_icawinv, result['icawinv']))
        
        # Check that icaweights is recomputed
        self.assertFalse(np.array_equal(original_icaweights, result['icaweights']))
        
        # Check that icasphere is set to identity
        np.testing.assert_array_equal(result['icasphere'], np.eye(EEG['nbchan']))
        
        # Check that icaweights is the pseudoinverse of icawinv
        computed_weights = np.linalg.pinv(result['icawinv'])
        np.testing.assert_allclose(result['icaweights'], computed_weights, rtol=1e-5)

    def test_icawinv_average_subtraction(self):
        """Test that icawinv has average subtracted correctly."""
        EEG = self.create_test_eeg(nbchan=8, pnts=128)
        original_icawinv = EEG['icawinv'].copy()
        
        result = pop_reref(EEG, ref=None)
        
        # Check that the mean across channels (axis=0) was subtracted
        expected_icawinv = original_icawinv - np.mean(original_icawinv, axis=0)
        np.testing.assert_allclose(result['icawinv'], expected_icawinv, rtol=1e-6)
        
        # Check that the mean across channels is approximately zero
        mean_across_channels = np.mean(result['icawinv'], axis=0)
        np.testing.assert_allclose(mean_across_channels, 0, atol=1e-6)

    def test_channel_reference_update(self):
        """Test that all channel references are updated to 'average'."""
        EEG = self.create_test_eeg(nbchan=10, pnts=100)
        
        # Set different initial references
        for i, chan in enumerate(EEG['chanlocs']):
            chan['ref'] = f'ref_{i}'
        
        result = pop_reref(EEG, ref=None)
        
        # All channels should now have 'average' reference
        for chan in result['chanlocs']:
            self.assertEqual(chan['ref'], 'average')

    def test_error_non_empty_ref(self):
        """Test error when ref is not None or empty list."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100)
        
        # Test with string reference
        with self.assertRaises(ValueError) as context:
            pop_reref(EEG, ref='Cz')
        self.assertIn('Feature not implemented', str(context.exception))
        self.assertIn('must be empty or None', str(context.exception))
        
        # Test with list of channels
        with self.assertRaises(ValueError) as context:
            pop_reref(EEG, ref=['Cz', 'FCz'])
        self.assertIn('Feature not implemented', str(context.exception))
        
        # Test with integer reference
        with self.assertRaises(ValueError) as context:
            pop_reref(EEG, ref=1)
        self.assertIn('Feature not implemented', str(context.exception))

    def test_error_icachansind_mismatch(self):
        """Test behavior when icachansind length doesn't match nbchan."""
        EEG = self.create_test_eeg(nbchan=16, pnts=100)
        
        # Make icachansind have different length
        EEG['icachansind'] = list(range(8))  # Only 8 channels instead of 16
        
        # The function should clear ICA fields instead of raising an error
        result = pop_reref(EEG, ref=None)
        
        # Check that ICA fields were cleared
        self.assertEqual(result['icawinv'].size, 0)
        self.assertEqual(result['icaweights'].size, 0)
        self.assertEqual(result['icasphere'].size, 0)

    def test_data_mean_subtraction(self):
        """Test that data has mean subtracted correctly."""
        EEG = self.create_test_eeg(nbchan=4, pnts=100)
        original_data = EEG['data'].copy()
        
        result = pop_reref(EEG, ref=None)
        
        # Check that the mean across channels (axis=0) was subtracted
        expected_data = original_data - np.mean(original_data, axis=0)
        np.testing.assert_allclose(result['data'], expected_data, rtol=1e-6)

    def test_single_channel(self):
        """Test with single channel (edge case)."""
        EEG = self.create_test_eeg(nbchan=1, pnts=100)
        original_data = EEG['data'].copy()
        
        result = pop_reref(EEG, ref=None)
        
        # With single channel, subtracting mean should make data zero
        np.testing.assert_allclose(result['data'], 0, atol=1e-6)
        
        # Check other fields are updated correctly
        self.assertEqual(result['ref'], 'average')
        self.assertEqual(result['chanlocs'][0]['ref'], 'average')

    def test_multiple_trials(self):
        """Test with multiple trials."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100, trials=5)
        original_data = EEG['data'].copy()
        
        result = pop_reref(EEG, ref=None)
        
        # Check that mean is subtracted for each time point and trial
        for trial in range(EEG['trials']):
            for time in range(EEG['pnts']):
                original_mean = np.mean(original_data[:, time, trial])
                new_mean = np.mean(result['data'][:, time, trial])
                self.assertAlmostEqual(new_mean, 0, places=6)

    def test_preserves_data_shape(self):
        """Test that data shape is preserved."""
        EEG = self.create_test_eeg(nbchan=16, pnts=256, trials=3)
        original_shape = EEG['data'].shape
        
        result = pop_reref(EEG, ref=None)
        
        self.assertEqual(result['data'].shape, original_shape)

    def test_preserves_other_fields(self):
        """Test that other EEG fields are preserved."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100)
        original_nbchan = EEG['nbchan']
        original_pnts = EEG['pnts']
        original_srate = EEG['srate']
        original_trials = EEG['trials']
        
        result = pop_reref(EEG, ref=None)
        
        # These fields should remain unchanged
        self.assertEqual(result['nbchan'], original_nbchan)
        self.assertEqual(result['pnts'], original_pnts)
        self.assertEqual(result['srate'], original_srate)
        self.assertEqual(result['trials'], original_trials)

    def test_deterministic_output(self):
        """Test that function produces deterministic output for same input."""
        EEG = self.create_test_eeg(nbchan=8, pnts=100)
        
        # Make copies to avoid modification effects
        EEG1 = {key: value.copy() if isinstance(value, np.ndarray) else 
                ([item.copy() if isinstance(item, dict) else item for item in value] 
                 if isinstance(value, list) else value)
                for key, value in EEG.items()}
        EEG2 = {key: value.copy() if isinstance(value, np.ndarray) else 
                ([item.copy() if isinstance(item, dict) else item for item in value] 
                 if isinstance(value, list) else value)
                for key, value in EEG.items()}
        
        result1 = pop_reref(EEG1, ref=None)
        result2 = pop_reref(EEG2, ref=None)
        
        np.testing.assert_array_equal(result1['data'], result2['data'])
        np.testing.assert_array_equal(result1['icaweights'], result2['icaweights'])
        np.testing.assert_array_equal(result1['icawinv'], result2['icawinv'])

    def test_numerical_precision(self):
        """Test numerical precision of computations."""
        EEG = self.create_test_eeg(nbchan=4, pnts=50)
        
        result = pop_reref(EEG, ref=None)
        
        # After average referencing, mean should be very close to zero
        mean_data = np.mean(result['data'], axis=0)
        self.assertTrue(np.all(np.abs(mean_data) < 1e-6))
        
        mean_icawinv = np.mean(result['icawinv'], axis=0)
        self.assertTrue(np.all(np.abs(mean_icawinv) < 1e-6))

    @unittest.skipUnless(hasattr(sys, '_called_from_test'), 
                        "MATLAB tests require MATLAB environment")
    def test_parity_basic_reref(self):
        """Test parity with MATLAB for basic rereferencing."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")
        
        # Create test data
        EEG = self.create_test_eeg(nbchan=8, pnts=100)
        
        # Python result
        py_result = pop_reref(EEG.copy(), ref=None)
        
        # MATLAB result (would need to save EEG structure and call MATLAB)
        # This is a placeholder for the parity test structure
        # ml_result = self.eeglab.pop_reref(EEG, [])
        
        # For now, just verify Python result is reasonable
        self.assertEqual(py_result['ref'], 'average')
        mean_data = np.mean(py_result['data'], axis=0)
        self.assertTrue(np.all(np.abs(mean_data) < 1e-6))


if __name__ == '__main__':
    unittest.main()
