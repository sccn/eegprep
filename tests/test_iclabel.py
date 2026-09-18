import os
import unittest
from unittest import mock

import numpy as np
from eegprep import ICL_feature_extractor, iclabel, pop_loadset
from eegprep.utils.testing import has_optional_dependency

import eegprep.plugins.ICLabel.iclabel as iclabel_module
import eegprep.plugins.ICLabel.iclabel_net_onnx as iclabel_onnx_module

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


def _async_eeg():
    return {
        'data': np.zeros((2, 20), dtype=np.float32),
        'nbchan': 2,
        'pnts': 20,
        'trials': 1,
        'srate': 100,
        'icaweights': np.eye(2),
        'icasphere': np.eye(2),
        'icawinv': np.eye(2),
        'icachansind': np.arange(2),
        'etc': {},
    }


class TestICLabelAsync(unittest.IsolatedAsyncioTestCase):
    async def test_native_async_entry_point_uses_shared_postprocessing(self):
        eeg = _async_eeg()
        network_output = np.arange(28, dtype=np.float32).reshape(4, 7, 1, 1)
        features = tuple(np.zeros((4, 1), dtype=np.float32) for _ in range(3))

        with (
            mock.patch.object(iclabel_module, '_prepare_features', return_value=features) as prepare,
            mock.patch(
                'eegprep.plugins.ICLabel.iclabel_net_onnx.run_iclabel_net_async',
                new=mock.AsyncMock(return_value=network_output),
            ) as run,
        ):
            output = await iclabel_module.iclabel_async(eeg)

        prepare.assert_called_once()
        run.assert_awaited_once_with(*features)
        classification = output['etc']['ic_classification']['ICLabel']['classifications']
        self.assertEqual(classification.shape, (1, 7))
        self.assertEqual(output['etc']['ic_classification']['ICLabel']['version'], 'default')

    def test_sync_entry_point_fails_fast_under_emscripten(self):
        with mock.patch.object(iclabel_module, '_IS_EMSCRIPTEN', True):
            with self.assertRaisesRegex(RuntimeError, r'use await iclabel_async\(\.\.\.\)'):
                iclabel_module.iclabel(None)

    def test_sync_onnx_backend_fails_fast_under_emscripten(self):
        with mock.patch.object(iclabel_onnx_module, '_IS_EMSCRIPTEN', True):
            with self.assertRaisesRegex(RuntimeError, r'use await run_iclabel_net_async\(\.\.\.\)'):
                iclabel_onnx_module.run_iclabel_net(None, None, None)


def _float32_classifications(eeg):
    image, psdmed, autocorr = _reference_network_inputs(eeg)
    import onnxruntime as ort

    float32_path = os.path.join(
        os.path.dirname(__file__), '..', 'tools', 'iclabel', 'artifacts', 'iclabel_float32.onnx'
    )
    (output,) = ort.InferenceSession(float32_path, providers=['CPUExecutionProvider']).run(
        ['output'], {'image': image, 'psdmed': psdmed, 'autocorr': autocorr}
    )
    return _reference_postprocess(output)


def _reference_network_inputs(eeg):
    """Build ICLabel inputs independently from the production helpers."""
    features = ICL_feature_extractor(eeg, True)
    features[0] = np.single(
        np.concatenate([features[0], -features[0], features[0][:, ::-1, :, :], -features[0][:, ::-1, :, :]], axis=3)
    )
    features[1] = np.single(np.tile(features[1], (1, 1, 1, 4)))
    features[2] = np.single(np.tile(features[2], (1, 1, 1, 4)))
    return tuple(np.transpose(feature, (3, 2, 0, 1)) for feature in features)


def _reference_postprocess(output):
    output = output.T
    output = np.reshape(output, (-1, 4), order='F')
    output = np.mean(output, axis=1)
    output = np.reshape(output, (7, -1), order='F')
    return output.T


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

        # Keep probability-level MATLAB parity against the preserved float32 reference;
        # the package default int8 artifact is covered by the semantic gate tests.
        EEG_matlab = iclabel(self.EEG, algorithm='default', engine='matlab')

        res1 = _float32_classifications(self.EEG).flatten()
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
    """Cross-check the preserved float32 ONNX reference against its torch source."""

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

        image, psdmed, autocorr = _reference_network_inputs(self.EEG)

        with torch.no_grad():
            torch_out = model(torch.from_numpy(image), torch.from_numpy(psdmed), torch.from_numpy(autocorr)).numpy()

        torch_final = _reference_postprocess(torch_out)

        onnx_final = _float32_classifications(self.EEG)

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


@unittest.skipUnless(has_optional_dependency('onnxruntime'), "install eegprep[iclabel] to run ICLabel runtime coverage")
class TestICLabelRuntime(unittest.TestCase):
    def test_default_entry_point_uses_packaged_onnx_runtime(self):
        eeg = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

        output = iclabel(eeg)
        classifications = output['etc']['ic_classification']['ICLabel']['classifications']
        reference = _float32_classifications(eeg)

        self.assertEqual(classifications.shape[1], 7)
        self.assertTrue(np.isfinite(classifications).all())
        self.assertEqual(output['etc']['ic_classification']['ICLabel']['version'], 'default')
        np.testing.assert_array_equal(classifications.argmax(axis=1), reference.argmax(axis=1))


if __name__ == '__main__':
    unittest.main()
