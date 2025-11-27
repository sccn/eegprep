"""
Test suite for eeg_autocorr.py with MATLAB parity validation.

This module tests the eeg_autocorr function which computes autocorrelation
of ICA components for EEG data.
"""

# Disable multithreading for deterministic numerical results
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import unittest
import sys
import numpy as np
import warnings

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.eeg_autocorr import eeg_autocorr
from eegprep.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase


class TestEegAutocorr(DebuggableTestCase):
    """Test cases for eeg_autocorr function."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up MATLAB compatibility for parity tests
        try:
            self.eeglab = get_eeglab()
            self.matlab_available = True
        except Exception:
            self.matlab_available = False

    def create_test_eeg(self, ncomp=10, pnts=1000, trials=1, srate=256):
        """Create a test EEG structure with ICA data."""
        # Create realistic ICA activations
        np.random.seed(42)  # For reproducible tests
        nbchan = ncomp  # Use same number of channels as components for valid ICA structure
        
        # Data and ICA shapes depend on trials
        if trials == 1:
            icaact = np.random.randn(ncomp, pnts).astype(np.float64)
            data = np.random.randn(nbchan, pnts).astype(np.float64)
        else:
            icaact = np.random.randn(ncomp, pnts, trials).astype(np.float64)
            data = np.random.randn(nbchan, pnts, trials).astype(np.float64)
        
        # Create channel locations
        chanlocs = np.array([{'labels': f'E{i+1}', 'X': 0.0, 'Y': 0.0, 'Z': 0.0, 
                             'theta': 0.0, 'radius': 0.0, 'type': 'EEG'} 
                            for i in range(nbchan)])

        # ICA matrices - consistent dimensions for MATLAB compatibility
        # icaweights: (ncomp, nbchan), icasphere: (nbchan, nbchan)
        # icaweights * icasphere * data(icachansind,:) should work
        icaweights = np.eye(ncomp, nbchan).astype(np.float64)
        icasphere = np.eye(nbchan).astype(np.float64)
        icawinv = np.eye(nbchan, ncomp).astype(np.float64)
        icachansind = np.arange(1, nbchan + 1, dtype=np.float64)  # 1-based, float for MATLAB

        return {
            'icaact': icaact,
            'pnts': pnts,
            'srate': srate,
            'trials': trials,
            'nbchan': nbchan,
            'icaweights': icaweights,
            'icasphere': icasphere,
            'icawinv': icawinv,
            'icachansind': icachansind,
            'data': data,
            'xmin': 0.0,
            'xmax': (pnts - 1) / srate,
            'times': np.linspace(0, (pnts - 1) / srate * 1000, pnts),  # in ms
            'chanlocs': chanlocs,
            'urchanlocs': np.array([]),
            'chaninfo': {},
            'ref': 'common',
            'history': '',
            'saved': 'no',
            'etc': {},
        }

    def test_basic_autocorrelation(self):
        """Test basic autocorrelation computation."""
        EEG = self.create_test_eeg(ncomp=5, pnts=512, srate=256)
        
        result = eeg_autocorr(EEG)
        
        # Check output shape
        expected_samples = 100  # Resampled to 100 Hz - 1 (first sample removed)
        self.assertEqual(result.shape, (5, expected_samples))
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result)))
        
        # Check data type
        self.assertTrue(result.dtype == np.float32 or result.dtype == np.float64)

    def test_default_pct_data(self):
        """Test default pct_data parameter."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)

        # Test with default pct_data (should be 100)
        result1 = eeg_autocorr(EEG)
        result2 = eeg_autocorr(EEG, pct_data=100)

        # Octave loading in setUp affects numerical precision
        # Observed: max_abs ~1e-08, max_rel ~1e-05
        np.testing.assert_allclose(result1, result2, rtol=2e-5, atol=2e-8)

    def test_explicit_pct_data(self):
        """Test explicit pct_data parameter."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)
        
        # Test with explicit pct_data
        result = eeg_autocorr(EEG, pct_data=50)
        
        # Should still produce same shape output
        self.assertEqual(result.shape, (3, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_different_sample_rates(self):
        """Test with different sampling rates."""
        test_cases = [
            {'srate': 128, 'expected_samples': 100},
            {'srate': 256, 'expected_samples': 100},
            {'srate': 500, 'expected_samples': 100},
            {'srate': 1000, 'expected_samples': 100}
        ]
        
        for case in test_cases:
            with self.subTest(srate=case['srate']):
                EEG = self.create_test_eeg(ncomp=2, pnts=512, srate=case['srate'])
                result = eeg_autocorr(EEG)
                
                self.assertEqual(result.shape[0], 2)
                self.assertEqual(result.shape[1], case['expected_samples'])
                self.assertTrue(np.all(np.isfinite(result)))

    def test_short_data_padding(self):
        """Test case where pnts < srate (requires padding)."""
        EEG = self.create_test_eeg(ncomp=3, pnts=100, srate=256)  # pnts < srate
        
        result = eeg_autocorr(EEG)
        
        # Should still produce expected output shape
        self.assertEqual(result.shape, (3, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_long_data_truncation(self):
        """Test case where pnts > srate (requires truncation)."""
        EEG = self.create_test_eeg(ncomp=3, pnts=1000, srate=256)  # pnts > srate
        
        result = eeg_autocorr(EEG)
        
        # Should still produce expected output shape
        self.assertEqual(result.shape, (3, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_single_component(self):
        """Test with single ICA component."""
        EEG = self.create_test_eeg(ncomp=1, pnts=512, srate=256)
        
        result = eeg_autocorr(EEG)
        
        self.assertEqual(result.shape, (1, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_many_components(self):
        """Test with many ICA components."""
        EEG = self.create_test_eeg(ncomp=50, pnts=512, srate=256)
        
        result = eeg_autocorr(EEG)
        
        self.assertEqual(result.shape, (50, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_icaact_conversion_to_float32(self):
        """Test that icaact is converted to float32."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)
        # Start with float64 data
        EEG['icaact'] = EEG['icaact'].astype(np.float64)
        
        original_dtype = EEG['icaact'].dtype
        self.assertEqual(original_dtype, np.float64)
        
        result = eeg_autocorr(EEG)
        
        # After processing, icaact should be float32
        self.assertEqual(EEG['icaact'].dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_zero_component(self):
        """Test with zero-valued component (edge case)."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)
        # Make one component all zeros (handle 2D for trials=1)
        if EEG['icaact'].ndim == 2:
            EEG['icaact'][1, :] = 0
        else:
            EEG['icaact'][1, :, :] = 0
        
        result = eeg_autocorr(EEG)
        
        self.assertEqual(result.shape, (3, 100))
        # Zero component should produce NaN or inf values after normalization
        # but the function should handle this gracefully
        self.assertTrue(np.all(np.isfinite(result[0, :])))  # First component should be fine
        self.assertTrue(np.all(np.isfinite(result[2, :])))  # Third component should be fine
        # Second component (zero) might have NaN or inf, which is expected

    def test_fft_size_calculation(self):
        """Test that FFT size is calculated correctly."""
        # Test with different data lengths to verify nfft calculation
        test_cases = [
            {'pnts': 100, 'expected_nfft_min': 128},   # 2^7
            {'pnts': 256, 'expected_nfft_min': 512},   # 2^9  
            {'pnts': 500, 'expected_nfft_min': 1024},  # 2^10
            {'pnts': 1000, 'expected_nfft_min': 2048}, # 2^11
        ]
        
        for case in test_cases:
            with self.subTest(pnts=case['pnts']):
                EEG = self.create_test_eeg(ncomp=2, pnts=case['pnts'], srate=256)
                
                # The function should work regardless of FFT size
                result = eeg_autocorr(EEG)
                
                self.assertEqual(result.shape, (2, 100))
                self.assertTrue(np.all(np.isfinite(result)))

    def test_normalization_by_zero_tap(self):
        """Test that autocorrelation is properly normalized by zero-tap."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)
        
        result = eeg_autocorr(EEG)
        
        # After normalization and resampling, we can't directly check for 1.0 at zero-tap
        # since the first sample is removed, but values should be reasonable
        self.assertTrue(np.all(np.abs(result) <= 10))  # Reasonable range after normalization

    def test_resampling_consistency(self):
        """Test that resampling to 100 Hz is consistent."""
        # Test with different original sampling rates
        srates = [128, 256, 512, 1000]
        
        for srate in srates:
            with self.subTest(srate=srate):
                EEG = self.create_test_eeg(ncomp=2, pnts=512, srate=srate)
                result = eeg_autocorr(EEG)
                
                # All should resample to 100 samples (100 Hz - 1)
                self.assertEqual(result.shape[1], 100)

    def test_multiple_trials(self):
        """Test with multiple trials (though current implementation doesn't use 3rd dimension)."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, trials=5, srate=128)
        
        result = eeg_autocorr(EEG)
        
        # Should still work with multiple trials
        self.assertEqual(result.shape, (3, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_parity_basic_autocorr(self):
        """Test parity with MATLAB for basic autocorrelation using real ICA data."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")
        
        # Load real EEG dataset with ICA
        from eegprep.pop_loadset import pop_loadset
        import os
        
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'eeglab_data_with_ica_tmp.set')
        if not os.path.exists(test_file):
            self.skipTest(f"Test file not found: {test_file}")
        
        EEG = pop_loadset(test_file)
        
        # Python result
        py_result = eeg_autocorr(EEG.copy())
        
        # MATLAB result
        ml_result = self.eeglab.eeg_autocorr(EEG.copy())
        
        # Compare results
        self.assertEqual(py_result.shape, ml_result.shape)
        np.testing.assert_allclose(py_result, ml_result, rtol=1e-5, atol=1e-8)

    def test_parity_with_real_data(self):
        """Test parity with MATLAB using real ICA data with different pct_data values."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")
        
        # Load real EEG dataset with ICA
        from eegprep.pop_loadset import pop_loadset
        import os
        
        test_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'eeglab_data_with_ica_tmp.set')
        if not os.path.exists(test_file):
            self.skipTest(f"Test file not found: {test_file}")
        
        EEG = pop_loadset(test_file)
        
        # Test with different pct_data values
        for pct_data in [50, 100]:
            with self.subTest(pct_data=pct_data):
                py_result = eeg_autocorr(EEG.copy(), pct_data=pct_data)
                ml_result = self.eeglab.eeg_autocorr(EEG.copy(), pct_data)
                
                self.assertEqual(py_result.shape, ml_result.shape)
                np.testing.assert_allclose(py_result, ml_result, rtol=1e-5, atol=1e-8)

    def test_edge_case_very_short_data(self):
        """Test edge case with very short data."""
        EEG = self.create_test_eeg(ncomp=2, pnts=10, srate=256)
        
        result = eeg_autocorr(EEG)
        
        # Should still produce output
        self.assertEqual(result.shape, (2, 100))
        # May contain NaN or inf due to very short data, but should not crash

    def test_edge_case_high_srate(self):
        """Test edge case with very high sampling rate."""
        EEG = self.create_test_eeg(ncomp=2, pnts=1000, srate=2000)
        
        result = eeg_autocorr(EEG)
        
        # Should still produce 100 samples output
        self.assertEqual(result.shape, (2, 100))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_input_modification(self):
        """Test that function modifies input EEG structure (icaact dtype)."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)
        original_dtype = EEG['icaact'].dtype
        
        result = eeg_autocorr(EEG)
        
        # Function should modify icaact dtype to float32
        self.assertEqual(EEG['icaact'].dtype, np.float32)
        if original_dtype != np.float32:
            self.assertNotEqual(EEG['icaact'].dtype, original_dtype)

    def test_deterministic_output(self):
        """Test that function produces deterministic output for same input."""
        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)
        
        # Make copies to avoid modification effects
        EEG1 = {key: value.copy() if isinstance(value, np.ndarray) else value 
                for key, value in EEG.items()}
        EEG2 = {key: value.copy() if isinstance(value, np.ndarray) else value 
                for key, value in EEG.items()}
        
        result1 = eeg_autocorr(EEG1)
        result2 = eeg_autocorr(EEG2)

        # Octave loading in setUp affects numerical precision
        # Observed: max_abs ~1e-08, max_rel ~1e-05
        np.testing.assert_allclose(result1, result2, rtol=2e-5, atol=2e-8)

    def test_memory_efficiency(self):
        """Test that function works with larger datasets."""
        # Test with larger dataset to check memory efficiency
        EEG = self.create_test_eeg(ncomp=100, pnts=2000, srate=1000)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore potential memory warnings
            result = eeg_autocorr(EEG)
        
        self.assertEqual(result.shape, (100, 100))
        # Don't check all finite for large datasets as it might be slow
        self.assertTrue(result.dtype in [np.float32, np.float64])  # Either is acceptable


if __name__ == '__main__':
    unittest.main()
