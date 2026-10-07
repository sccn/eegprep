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

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.plugins.ICLabel.eeg_autocorr import eeg_autocorr
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
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
        chanlocs = np.array(
            [
                {'labels': f'E{i + 1}', 'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'theta': 0.0, 'radius': 0.0, 'type': 'EEG'}
                for i in range(nbchan)
            ]
        )

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
        # Second component (zero) might have NaN or inf, which is expected

    def test_parity_with_real_data(self):
        """Test parity with MATLAB using real ICA data with different pct_data values."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Load real EEG dataset with ICA
        from eegprep.functions.popfunc.pop_loadset import pop_loadset
        import os

        test_file = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'eeglab_data_with_ica_tmp.set')
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
        # May contain NaN or inf due to very short data, but should not crash


if __name__ == '__main__':
    unittest.main()
