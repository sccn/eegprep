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
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


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

        test_file = os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')
        self.EEG = pop_loadset(test_file)

    def test_parity_default_nfreqs(self):
        """Test parity with MATLAB using default nfreqs parameter."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        # eeg_rpsd.m reads EEG.icaact and does not recompute it. pop_loadset fills
        # that field (eeg_checkset.m:1030-1032); both sides window that array.
        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        psdmed = eeg_rpsd(EEG);
        icaact = EEG.icaact;
        save('{temp_file}.mat', 'psdmed', 'icaact');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        mat_data = scipy.io.loadmat(temp_file + '.mat', mat_dtype=True)
        ml_result = mat_data['psdmed']
        compared = self.EEG.copy()
        compared['icaact'] = mat_data['icaact']
        py_result = eeg_rpsd(compared)

        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        np.testing.assert_allclose(
            py_result,
            ml_result,
            rtol=2e-5,
            atol=1e-8,
            err_msg="eeg_rpsd results differ beyond tolerance (default nfreqs)",
        )


if __name__ == '__main__':
    unittest.main()
