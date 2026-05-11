"""
Test parity between Python and MATLAB implementations of ICL_feature_extractor.

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
from eegprep import pop_loadset, pop_saveset, ICL_feature_extractor
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')

ICLABEL_PARITY_RTOL = 2e-5
ICLABEL_PARITY_ATOL = 1e-8
# MATLAB ICLabel casts PSD features to single precision before returning them.
ICLABEL_PSD_PARITY_ATOL = 5e-8


class TestICLFeatureExtractorParity(unittest.TestCase):
    """Test parity between Python and MATLAB ICL_feature_extractor implementations."""

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

    def test_parity_without_autocorr(self):
        """Test parity with MATLAB without autocorrelation (flag_autocorr=False)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        py_features = ICL_feature_extractor(self.EEG.copy(), flag_autocorr=False)

        # MATLAB result - use file roundtrip
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, false);
        topo = features{{1}};
        psd = features{{2}};
        save('{temp_file}.mat', 'topo', 'psd');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        ml_topo = mat_data['topo']
        ml_psd = mat_data['psd']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare topo feature
        py_topo = py_features[0]
        print(f"\nTopo comparison:")
        print(f"  Python shape: {py_topo.shape}")
        print(f"  MATLAB shape: {ml_topo.shape}")
        print(f"  Max absolute diff: {np.max(np.abs(py_topo - ml_topo)):.6f}")

        # Calculate mismatched elements
        mismatch_mask = ~np.isclose(py_topo, ml_topo, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PARITY_ATOL)
        n_mismatch = np.sum(mismatch_mask)
        n_total = py_topo.size
        print(f"  Mismatched elements: {n_mismatch} / {n_total} ({100*n_mismatch/n_total:.2f}%)")

        if n_mismatch > 0:
            max_rel_diff = np.max(np.abs((py_topo - ml_topo) / (ml_topo + 1e-10)))
            print(f"  Max relative diff: {max_rel_diff:.6f}")

        np.testing.assert_allclose(py_topo, ml_topo, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PARITY_ATOL,
                                   err_msg="Topo feature differs beyond tolerance")

        # Compare psd feature
        py_psd = py_features[1]
        print(f"\nPSD comparison:")
        print(f"  Python shape: {py_psd.shape}")
        print(f"  MATLAB shape: {ml_psd.shape}")
        print(f"  Max absolute diff: {np.max(np.abs(py_psd - ml_psd)):.6f}")

        # Calculate mismatched elements
        mismatch_mask = ~np.isclose(py_psd, ml_psd, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PSD_PARITY_ATOL)
        n_mismatch = np.sum(mismatch_mask)
        n_total = py_psd.size
        print(f"  Mismatched elements: {n_mismatch} / {n_total} ({100*n_mismatch/n_total:.2f}%)")

        if n_mismatch > 0:
            max_rel_diff = np.max(np.abs((py_psd - ml_psd) / (ml_psd + 1e-10)))
            print(f"  Max relative diff: {max_rel_diff:.6f}")

        np.testing.assert_allclose(py_psd, ml_psd, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PSD_PARITY_ATOL,
                                   err_msg="PSD feature differs beyond tolerance")

    def test_parity_with_autocorr(self):
        """Test parity with MATLAB with autocorrelation (flag_autocorr=True)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        py_features = ICL_feature_extractor(self.EEG.copy(), flag_autocorr=True)

        # MATLAB result - use file roundtrip
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        topo = features{{1}};
        psd = features{{2}};
        autocorr = features{{3}};
        save('{temp_file}.mat', 'topo', 'psd', 'autocorr');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        ml_topo = mat_data['topo']
        ml_psd = mat_data['psd']
        ml_autocorr = mat_data['autocorr']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare topo feature
        py_topo = py_features[0]
        print(f"\nTopo comparison:")
        print(f"  Python shape: {py_topo.shape}")
        print(f"  MATLAB shape: {ml_topo.shape}")
        print(f"  Max absolute diff: {np.max(np.abs(py_topo - ml_topo)):.6f}")

        # Calculate mismatched elements
        mismatch_mask = ~np.isclose(py_topo, ml_topo, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PARITY_ATOL)
        n_mismatch = np.sum(mismatch_mask)
        n_total = py_topo.size
        print(f"  Mismatched elements: {n_mismatch} / {n_total} ({100*n_mismatch/n_total:.2f}%)")

        if n_mismatch > 0:
            max_rel_diff = np.max(np.abs((py_topo - ml_topo) / (ml_topo + 1e-10)))
            print(f"  Max relative diff: {max_rel_diff:.6f}")

        np.testing.assert_allclose(py_topo, ml_topo, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PARITY_ATOL,
                                   err_msg="Topo feature differs beyond tolerance")

        # Compare psd feature
        py_psd = py_features[1]
        print(f"\nPSD comparison:")
        print(f"  Python shape: {py_psd.shape}")
        print(f"  MATLAB shape: {ml_psd.shape}")
        print(f"  Max absolute diff: {np.max(np.abs(py_psd - ml_psd)):.6f}")

        # Calculate mismatched elements
        mismatch_mask = ~np.isclose(py_psd, ml_psd, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PSD_PARITY_ATOL)
        n_mismatch = np.sum(mismatch_mask)
        n_total = py_psd.size
        print(f"  Mismatched elements: {n_mismatch} / {n_total} ({100*n_mismatch/n_total:.2f}%)")

        if n_mismatch > 0:
            max_rel_diff = np.max(np.abs((py_psd - ml_psd) / (ml_psd + 1e-10)))
            print(f"  Max relative diff: {max_rel_diff:.6f}")

        np.testing.assert_allclose(py_psd, ml_psd, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PSD_PARITY_ATOL,
                                   err_msg="PSD feature differs beyond tolerance")

        # Compare autocorr feature
        py_autocorr = py_features[2]
        print(f"\nAutocorr comparison:")
        print(f"  Python shape: {py_autocorr.shape}")
        print(f"  MATLAB shape: {ml_autocorr.shape}")
        print(f"  Max absolute diff: {np.max(np.abs(py_autocorr - ml_autocorr)):.6f}")

        # Calculate mismatched elements
        mismatch_mask = ~np.isclose(py_autocorr, ml_autocorr, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PARITY_ATOL)
        n_mismatch = np.sum(mismatch_mask)
        n_total = py_autocorr.size
        print(f"  Mismatched elements: {n_mismatch} / {n_total} ({100*n_mismatch/n_total:.2f}%)")

        if n_mismatch > 0:
            max_rel_diff = np.max(np.abs((py_autocorr - ml_autocorr) / (ml_autocorr + 1e-10)))
            print(f"  Max relative diff: {max_rel_diff:.6f}")

        np.testing.assert_allclose(py_autocorr, ml_autocorr, rtol=ICLABEL_PARITY_RTOL, atol=ICLABEL_PARITY_ATOL,
                                   err_msg="Autocorr feature differs beyond tolerance")


if __name__ == '__main__':
    unittest.main()
