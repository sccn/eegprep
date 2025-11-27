import os
import unittest
from copy import deepcopy
import numpy as np
from eegprep import *
import tempfile
import scipy.io

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../data/')

class TestICLabelEngines(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')) 

    def test_basic(self):
        # First, extract features separately to compare
        from eegprep import ICL_feature_extractor
        features_python = ICL_feature_extractor(self.EEG, True)
        print(f"\n{'='*60}")
        print("FEATURE EXTRACTION COMPARISON")
        print(f"{'='*60}")
        print(f"Python features[0] (topo) shape: {features_python[0].shape}, dtype: {features_python[0].dtype}")
        print(f"Python features[1] (psd) shape: {features_python[1].shape}, dtype: {features_python[1].dtype}")
        print(f"Python features[2] (autocorr) shape: {features_python[2].shape}, dtype: {features_python[2].dtype}")
        print(f"Python topo max: {np.max(features_python[0]):.6f}, min: {np.min(features_python[0]):.6f}")
        print(f"Python psd max: {np.max(features_python[1]):.6f}, min: {np.min(features_python[1]):.6f}")
        print(f"Python autocorr max: {np.max(features_python[2]):.6f}, min: {np.min(features_python[2]):.6f}")
        print(f"{'='*60}\n")

        EEG_python = iclabel(self.EEG, algorithm='default', engine=None)
        EEG_matlab = iclabel(self.EEG, algorithm='default', engine='matlab')

        res1 = EEG_python['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        res2 = EEG_matlab['etc']['ic_classification']['ICLabel']['classifications'].flatten()

        # Diagnostic output
        print(f"\n{'='*60}")
        print("DIAGNOSTIC OUTPUT")
        print(f"{'='*60}")
        print(f"Python result dtype: {res1.dtype}")
        print(f"MATLAB result dtype: {res2.dtype}")
        print(f"Python result shape: {res1.shape}")
        print(f"MATLAB result shape: {res2.shape}")
        print(f"\nMax absolute difference: {np.max(np.abs(res1 - res2)):.2e}")
        print(f"Mean absolute difference: {np.mean(np.abs(res1 - res2)):.2e}")
        print(f"Max relative difference: {np.max(np.abs(res1 - res2) / (np.abs(res2) + 1e-10)):.2e}")
        print(f"Mean relative difference: {np.mean(np.abs(res1 - res2) / (np.abs(res2) + 1e-10)):.2e}")
        print(f"\nPython results (first 20 values):\n{res1[:20]}")
        print(f"\nMATLAB results (first 20 values):\n{res2[:20]}")
        print(f"\nDifferences (first 20 values):\n{(res1 - res2)[:20]}")
        print(f"\nRelative differences (first 20 values):\n{((res1 - res2) / (res2 + 1e-10))[:20]}")

        # Count how many values exceed tolerances
        abs_diffs = np.abs(res1 - res2)
        rel_diffs = np.abs(res1 - res2) / (np.abs(res2) + 1e-10)
        exceeds_abs = abs_diffs > 1e-8
        exceeds_rel = rel_diffs > 1e-5
        exceeds_both = exceeds_abs & exceeds_rel
        print(f"\nValues exceeding absolute tolerance (1e-8): {np.sum(exceeds_abs)}/{len(res1)}")
        print(f"Values exceeding relative tolerance (1e-5): {np.sum(exceeds_rel)}/{len(res1)}")
        print(f"Values exceeding BOTH tolerances: {np.sum(exceeds_both)}/{len(res1)}")
        print(f"{'='*60}\n")

        self.assertTrue(np.allclose(res1, res2, rtol=1e-5, atol=1e-8),
                       'ICLabel results differ beyond tolerance')

if __name__ == '__main__':
    unittest.main()
