"""
Cross-check eeg_autocorr_fftw (epoched ICLabel autocorrelation) against eeg_autocorr.
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

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.plugins.ICLabel.eeg_autocorr_fftw import eeg_autocorr_fftw
from eegprep.utils.testing import DebuggableTestCase


class TestEegAutocorrFftw(DebuggableTestCase):
    """Test cases for eeg_autocorr_fftw function."""

    def create_test_eeg(self, ncomp=10, pnts=1000, trials=1, srate=256):
        """Create a test EEG structure with ICA data."""
        # Create realistic ICA activations
        np.random.seed(42)  # For reproducible tests
        icaact = np.random.randn(ncomp, pnts, trials).astype(
            np.float64
        )  # Use float64 to match MATLAB default precision

        return {
            'icaact': icaact,
            'pnts': pnts,
            'srate': srate,
            'trials': trials,
            'nbchan': 64,  # Original channels before ICA
            'icaweights': np.random.randn(ncomp, 64).astype(np.float64),
            'icasphere': np.random.randn(64, 64).astype(np.float64),
        }
        # Second component (zero) might have NaN or inf, which is expected
        # May contain NaN or inf due to very short data, but should not crash

    def test_comparison_with_regular_autocorr(self):
        """Test that FFTW version produces similar results to regular version."""
        # Import the regular autocorr function for comparison
        from eegprep.plugins.ICLabel.eeg_autocorr import eeg_autocorr

        EEG = self.create_test_eeg(ncomp=3, pnts=256, srate=128)

        # Make copies to avoid modification effects
        EEG_fftw = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in EEG.items()}
        EEG_regular = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in EEG.items()}

        result_fftw = eeg_autocorr_fftw(EEG_fftw)
        result_regular = eeg_autocorr(EEG_regular)

        # Both should have same shape
        self.assertEqual(result_fftw.shape, result_regular.shape)

        # Both compute the same autocorrelation; the only divergence is that
        # eeg_autocorr casts its FFT to single precision (complex64) for MATLAB
        # parity while eeg_autocorr_fftw stays double precision. Observed max
        # relative difference is ~3e-5, so a float-realistic tolerance still
        # catches any several-percent port regression.
        self.assertTrue(np.allclose(result_fftw, result_regular, rtol=1e-4, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
