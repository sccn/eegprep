"""
Test parity between Python and MATLAB implementations of ICL_feature_extractor.

This test compares the full feature extraction (topo, PSD, autocorr) against MATLAB/EEGLAB reference.
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
from eegprep import pop_loadset, ICL_feature_extractor
from eegprep.eeglabcompat import get_eeglab
import tempfile
import scipy.io

local_url = os.path.join(os.path.dirname(__file__), '../data/')


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

    def test_parity_full_feature_extraction(self):
        """Test parity with MATLAB for complete feature extraction."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        features_py = ICL_feature_extractor(self.EEG.copy(), True)

        # MATLAB result - use file roundtrip for cell array output
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        save('{temp_file}.mat', 'features');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB features
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        features_ml = [
            mat_data['features'][0, 0],
            mat_data['features'][0, 1],
            mat_data['features'][0, 2]
        ]

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare all three features
        feature_names = ['Topo', 'PSD', 'Autocorr']

        for i, name in enumerate(feature_names):
            py_feat = features_py[i]
            ml_feat = features_ml[i]

            # Verify shapes match
            self.assertEqual(py_feat.shape, ml_feat.shape,
                           f"{name} feature shape mismatch: {py_feat.shape} vs {ml_feat.shape}")

            # Compare values
            # Max absolute diff: TBD, Mean absolute diff: TBD
            # Max relative diff: TBD, Mean relative diff: TBD
            np.testing.assert_allclose(py_feat, ml_feat, rtol=1e-5, atol=1e-8,
                                       err_msg=f"{name} feature differs beyond tolerance")

    def test_parity_topo_feature_only(self):
        """Test parity specifically for topography feature."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        features_py = ICL_feature_extractor(self.EEG.copy(), True)
        topo_py = features_py[0]

        # MATLAB result
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        topo = features{{1}};
        save('{temp_file}.mat', 'topo');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        topo_ml = mat_data['topo']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare
        # Max absolute diff: TBD, Mean absolute diff: TBD
        # Max relative diff: TBD, Mean relative diff: TBD
        np.testing.assert_allclose(topo_py, topo_ml, rtol=1e-5, atol=1e-8,
                                   err_msg="Topo feature differs beyond tolerance")

    def test_parity_psd_feature_only(self):
        """Test parity specifically for PSD feature."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        features_py = ICL_feature_extractor(self.EEG.copy(), True)
        psd_py = features_py[1]

        # MATLAB result
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        psd = features{{2}};
        save('{temp_file}.mat', 'psd');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        psd_ml = mat_data['psd']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare
        # Max absolute diff: TBD, Mean absolute diff: TBD
        # Max relative diff: TBD, Mean relative diff: TBD
        np.testing.assert_allclose(psd_py, psd_ml, rtol=1e-5, atol=1e-8,
                                   err_msg="PSD feature differs beyond tolerance")

    def test_parity_autocorr_feature_only(self):
        """Test parity specifically for autocorrelation feature."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        # Python result
        features_py = ICL_feature_extractor(self.EEG.copy(), True)
        autocorr_py = features_py[2]

        # MATLAB result
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        autocorr = features{{3}};
        save('{temp_file}.mat', 'autocorr');
        """
        self.eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB result
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        autocorr_ml = mat_data['autocorr']

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        # Compare
        # Max absolute diff: TBD, Mean absolute diff: TBD
        # Max relative diff: TBD, Mean relative diff: TBD
        np.testing.assert_allclose(autocorr_py, autocorr_ml, rtol=1e-5, atol=1e-8,
                                   err_msg="Autocorr feature differs beyond tolerance")


if __name__ == '__main__':
    unittest.main()
