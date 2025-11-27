"""
Test parity between Python and MATLAB implementations of eeg_rpsd.

This test compares the Python implementation against the MATLAB/EEGLAB reference.
Multithreading is disabled for deterministic numerical results.
"""

# Disable multithreading for deterministic numerical results
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import unittest
import numpy as np
import tempfile
import scipy.io
from eegprep import pop_loadset, pop_saveset, eeg_rpsd
from eegprep.eeglabcompat import get_eeglab

local_url = os.path.join(os.path.dirname(__file__), '../data/')


class TestEegRpsdParity(unittest.TestCase):
    """Test parity between Python and MATLAB eeg_rpsd implementations."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to get MATLAB engine
        try:
            self.eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.matlab_available = True
        except Exception as e:
            self.matlab_available = False
            self.skipTest(f"MATLAB not available: {e}")

        # Load real EEG dataset with ICA
        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(test_file)

    def test_parity_default_nfreqs(self):
        """Test parity with MATLAB using default nfreqs parameter."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        py_result = eeg_rpsd(self.EEG.copy())

        # MATLAB result - use file roundtrip
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        psdmed = eeg_rpsd(EEG);
        save('{temp_file}.mat', 'psdmed');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        ml_result = mat_data['psdmed']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare results
        # Max absolute diff: 0.00017, Mismatched: 1/2048 (0.05%)
        # Max relative diff: 1.14e-05
        np.testing.assert_allclose(py_result, ml_result, rtol=2e-5, atol=1e-8,
                                   err_msg="eeg_rpsd results differ beyond tolerance (default nfreqs)")

    def test_parity_custom_nfreqs_100(self):
        """Test parity with MATLAB using nfreqs=100."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        py_result = eeg_rpsd(self.EEG.copy(), nfreqs=100)

        # MATLAB result - use file roundtrip
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        psdmed = eeg_rpsd(EEG, 100);
        save('{temp_file}.mat', 'psdmed');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        ml_result = mat_data['psdmed']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare results
        # Max absolute diff: 0.00017, Mismatched: 1/2048 (0.05%)
        # Max relative diff: 1.14e-05
        np.testing.assert_allclose(py_result, ml_result, rtol=2e-5, atol=1e-8,
                                   err_msg="eeg_rpsd results differ beyond tolerance (nfreqs=100)")

    def test_parity_custom_nfreqs_50(self):
        """Test parity with MATLAB using nfreqs=50."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        py_result = eeg_rpsd(self.EEG.copy(), nfreqs=50)

        # MATLAB result - use file roundtrip
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        psdmed = eeg_rpsd(EEG, 50);
        save('{temp_file}.mat', 'psdmed');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        ml_result = mat_data['psdmed']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare results
        # Max absolute diff: ~0.0002, Mismatched: ~1/1024 (0.1%)
        # Max relative diff: ~1.14e-05
        np.testing.assert_allclose(py_result, ml_result, rtol=2e-5, atol=1e-8,
                                   err_msg="eeg_rpsd results differ beyond tolerance (nfreqs=50)")


if __name__ == '__main__':
    unittest.main()
