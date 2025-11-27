"""
Test to compare ICLabel features between Python and MATLAB implementations.
Tests both float32 and float64 precision.
"""
import os
import unittest
import numpy as np
from eegprep import pop_loadset, ICL_feature_extractor
from eegprep.eeglabcompat import get_eeglab
import tempfile
import scipy.io

local_url = os.path.join(os.path.dirname(__file__), '../data/')

class TestICLabelFeatureComparison(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

    def test_feature_comparison_float32(self):
        """Compare Python vs MATLAB features in float32."""
        print(f"\n{'='*70}")
        print("FEATURE COMPARISON: FLOAT32 (Default)")
        print(f"{'='*70}")

        # Extract Python features (float32)
        features_py = ICL_feature_extractor(self.EEG, True)

        # Extract MATLAB features using direct MATLAB call
        eeglab = get_eeglab('MAT', auto_file_roundtrip=False)

        # Save EEG to temp file for MATLAB
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        # Call MATLAB to extract features and save them
        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        save('{temp_file}.mat', 'features');
        """
        eeglab.eval(matlab_code, nargout=0)

        # Load MATLAB features
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        features_mat = [
            mat_data['features'][0, 0],
            mat_data['features'][0, 1],
            mat_data['features'][0, 2]
        ]

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        self._compare_features(features_py, features_mat, "FLOAT32")

    def test_feature_comparison_float64(self):
        """Compare Python vs MATLAB features in float64."""
        print(f"\n{'='*70}")
        print("FEATURE COMPARISON: FLOAT64 (Modified)")
        print(f"{'='*70}")

        # Extract Python features and convert to float64
        features_py32 = ICL_feature_extractor(self.EEG, True)
        features_py = [
            features_py32[0].astype(np.float64),
            features_py32[1].astype(np.float64),
            features_py32[2].astype(np.float64)
        ]

        # Extract MATLAB features
        eeglab = get_eeglab('MAT', auto_file_roundtrip=False)

        # Save EEG to temp file
        from eegprep import pop_saveset
        temp_file = tempfile.mktemp(suffix='.set')
        pop_saveset(self.EEG, temp_file)

        # Call MATLAB to extract features and save them
        matlab_code = f"""
        EEG = pop_loadset('{temp_file}');
        features = ICL_feature_extractor(EEG, true);
        save('{temp_file}.mat', 'features');
        """
        eeglab.eval(matlab_code, nargout=0)

        # Load and convert to float64
        mat_data = scipy.io.loadmat(temp_file + '.mat')
        features_mat = [
            mat_data['features'][0, 0].astype(np.float64),
            mat_data['features'][0, 1].astype(np.float64),
            mat_data['features'][0, 2].astype(np.float64)
        ]

        # Clean up
        os.remove(temp_file)
        os.remove(temp_file + '.mat')
        if os.path.exists(temp_file.replace('.set', '.fdt')):
            os.remove(temp_file.replace('.set', '.fdt'))

        self._compare_features(features_py, features_mat, "FLOAT64")

    def _compare_features(self, features_py, features_mat, precision):
        """Helper to compare and report feature differences."""
        feature_names = ['Topo', 'PSD', 'Autocorr']

        for i, name in enumerate(feature_names):
            py_feat = features_py[i]
            mat_feat = features_mat[i]

            print(f"\n{'-'*70}")
            print(f"{name} Feature ({precision})")
            print(f"{'-'*70}")

            # Shape and dtype
            print(f"Python shape: {py_feat.shape}, dtype: {py_feat.dtype}")
            print(f"MATLAB shape: {mat_feat.shape}, dtype: {mat_feat.dtype}")

            # Min/Max values
            print(f"\nPython  - min: {np.min(py_feat):+.10f}, max: {np.max(py_feat):+.10f}")
            print(f"MATLAB  - min: {np.min(mat_feat):+.10f}, max: {np.max(mat_feat):+.10f}")

            # Explain min/max
            print(f"\nMin/Max Explanation:")
            print(f"  - Features are scaled by 0.99 in ICL_feature_extractor")
            print(f"  - Expected range: [-0.99, +0.99] for topo, [0, 0.99] for others")
            if np.min(py_feat) < -0.99 or np.max(py_feat) > 0.99:
                print(f"  ⚠️  Python feature EXCEEDS expected range!")
            if np.min(mat_feat) < -0.99 or np.max(mat_feat) > 0.99:
                print(f"  ⚠️  MATLAB feature EXCEEDS expected range!")

            # Statistics
            print(f"\nPython  - mean: {np.mean(py_feat):+.10f}, std: {np.std(py_feat):.10f}")
            print(f"MATLAB  - mean: {np.mean(mat_feat):+.10f}, std: {np.std(mat_feat):.10f}")

            # Differences
            if py_feat.shape == mat_feat.shape:
                diff = py_feat - mat_feat
                abs_diff = np.abs(diff)
                rel_diff = np.abs(diff) / (np.abs(mat_feat) + 1e-10)

                print(f"\nDifference Statistics:")
                print(f"  Max absolute diff:  {np.max(abs_diff):.2e}")
                print(f"  Mean absolute diff: {np.mean(abs_diff):.2e}")
                print(f"  Max relative diff:  {np.max(rel_diff):.2e}")
                print(f"  Mean relative diff: {np.mean(rel_diff):.2e}")

                # Count values exceeding tolerances
                exceeds_abs_1e8 = np.sum(abs_diff > 1e-8)
                exceeds_abs_1e6 = np.sum(abs_diff > 1e-6)
                exceeds_rel_1e5 = np.sum(rel_diff > 1e-5)
                total = py_feat.size

                print(f"\nValues exceeding tolerances:")
                print(f"  |diff| > 1e-8:  {exceeds_abs_1e8:6d}/{total} ({100*exceeds_abs_1e8/total:.1f}%)")
                print(f"  |diff| > 1e-6:  {exceeds_abs_1e6:6d}/{total} ({100*exceeds_abs_1e6/total:.1f}%)")
                print(f"  rel diff > 1e-5: {exceeds_rel_1e5:6d}/{total} ({100*exceeds_rel_1e5/total:.1f}%)")

                # Are they close?
                is_close_1e5 = np.allclose(py_feat, mat_feat, rtol=1e-5, atol=1e-8)
                is_close_1e4 = np.allclose(py_feat, mat_feat, rtol=1e-4, atol=1e-6)
                is_close_1e3 = np.allclose(py_feat, mat_feat, rtol=1e-3, atol=1e-5)

                print(f"\nallclose() results:")
                print(f"  rtol=1e-5, atol=1e-8: {is_close_1e5}")
                print(f"  rtol=1e-4, atol=1e-6: {is_close_1e4}")
                print(f"  rtol=1e-3, atol=1e-5: {is_close_1e3}")
            else:
                print(f"\n⚠️  Shape mismatch! Cannot compare values.")

        print(f"\n{'='*70}\n")

if __name__ == '__main__':
    unittest.main()
