import os
import unittest
import numpy as np
from eegprep import ICL_feature_extractor, iclabel, pop_loadset
from eegprep.utils.testing import has_optional_dependency

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestICLabelEngines(unittest.TestCase):
    def setUp(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

    def test_basic(self):
        if not has_optional_dependency('onnxruntime'):
            self.skipTest("onnxruntime is not installed; install eegprep[iclabel] to run ICLabel parity")

        features_python = ICL_feature_extractor(self.EEG, True)
        print(f"\n{'=' * 60}")
        print("FEATURE EXTRACTION COMPARISON")
        print(f"{'=' * 60}")
        print(f"Python features[0] (topo) shape: {features_python[0].shape}, dtype: {features_python[0].dtype}")
        print(f"Python features[1] (psd) shape: {features_python[1].shape}, dtype: {features_python[1].dtype}")
        print(f"Python features[2] (autocorr) shape: {features_python[2].shape}, dtype: {features_python[2].dtype}")
        print(f"Python topo max: {np.max(features_python[0]):.6f}, min: {np.min(features_python[0]):.6f}")
        print(f"Python psd max: {np.max(features_python[1]):.6f}, min: {np.min(features_python[1]):.6f}")
        print(f"Python autocorr max: {np.max(features_python[2]):.6f}, min: {np.min(features_python[2]):.6f}")
        print(f"{'=' * 60}\n")

        EEG_python = iclabel(self.EEG, algorithm='default', engine=None)
        EEG_matlab = iclabel(self.EEG, algorithm='default', engine='matlab')

        res1 = EEG_python['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        res2 = EEG_matlab['etc']['ic_classification']['ICLabel']['classifications'].flatten()

        # Diagnostic output
        print(f"\n{'=' * 60}")
        print("DIAGNOSTIC OUTPUT")
        print(f"{'=' * 60}")
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
        print(f"{'=' * 60}\n")

        # Max abs diff: 3.37e-06, Max rel diff: 4.25e-05
        self.assertTrue(np.allclose(res1, res2, rtol=1e-4, atol=1e-5), 'ICLabel results differ beyond tolerance')


class TestICLabelOnnxExport(unittest.TestCase):
    """Cross-check the packaged iclabel.onnx artifact against the torch network it was exported from."""

    def setUp(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

    def test_onnx_matches_torch_on_sample_data(self):
        if not has_optional_dependency('torch'):
            self.skipTest("PyTorch is not installed; install eegprep[torch] to run the ONNX-vs-torch cross-check")
        if not has_optional_dependency('onnxruntime'):
            self.skipTest("onnxruntime is not installed; install eegprep[iclabel] to run ICLabel classification")

        import torch

        from eegprep.plugins.ICLabel.iclabel_net import ICLabelNet

        mat_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'eegprep', 'plugins', 'ICLabel', 'netICL.mat')
        model = ICLabelNet(mat_path)
        model.eval()

        features = ICL_feature_extractor(self.EEG, True)
        features[0] = np.single(
            np.concatenate([features[0], -features[0], features[0][:, ::-1, :, :], -features[0][:, ::-1, :, :]], axis=3)
        )
        features[1] = np.single(np.tile(features[1], (1, 1, 1, 4)))
        features[2] = np.single(np.tile(features[2], (1, 1, 1, 4)))

        image = np.transpose(features[0], (3, 2, 0, 1))
        psdmed = np.transpose(features[1], (3, 2, 0, 1))
        autocorr = np.transpose(features[2], (3, 2, 0, 1))

        with torch.no_grad():
            torch_out = model(torch.from_numpy(image), torch.from_numpy(psdmed), torch.from_numpy(autocorr)).numpy()

        def postprocess(output):
            out = output.T
            out = np.reshape(out, (-1, 4), order='F')
            out = np.mean(out, axis=1)
            out = np.reshape(out, (7, -1), order='F')
            return out.T

        torch_final = postprocess(torch_out)

        EEG_onnx = iclabel(self.EEG, algorithm='default', engine=None)
        onnx_final = EEG_onnx['etc']['ic_classification']['ICLabel']['classifications']

        diff = np.abs(torch_final - onnx_final)
        print(f"\nONNX vs torch max abs diff: {diff.max():.2e}, mean abs diff: {diff.mean():.2e}")

        # Measured on this sample_data dataset while building the export
        # (tools/iclabel/export_iclabel_onnx.py): max abs diff 1.43e-06, max
        # rel diff 2.59e-05, from ordinary float32 op-ordering differences
        # between eager torch execution and onnxruntime's fused CPU kernels.
        # That is the same order of magnitude as the Python-vs-MATLAB gap
        # tolerated above (~3.4e-06 abs / ~4.3e-05 rel), so this test reuses
        # that already-established tolerance instead of introducing a new one.
        self.assertTrue(
            np.allclose(torch_final, onnx_final, rtol=1e-4, atol=1e-5),
            'ONNX and torch ICLabel outputs differ beyond tolerance',
        )


if __name__ == '__main__':
    unittest.main()
