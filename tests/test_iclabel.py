import os
import unittest
from unittest import mock

import numpy as np
from eegprep import ICL_feature_extractor, iclabel, pop_loadset
from eegprep.utils.testing import has_optional_dependency

from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS
import eegprep.plugins.ICLabel.iclabel as iclabel_module
import eegprep.plugins.ICLabel.iclabel_net_onnx as iclabel_onnx_module

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
_NET_ICL_MAT = os.path.join(os.path.dirname(__file__), '..', 'src', 'eegprep', 'plugins', 'ICLabel', 'netICL.mat')


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


# Shipped int8 model versus the float32 network on eeglab_data_with_ica_tmp.set
# (32 components), measured with ONNX Runtime 1.30.0 and PyTorch 2.12.0: max
# abs diff 0.0079, mean 0.00061, identical argmax and default pop_icflag
# decisions, with the closest top-two margin at 0.019 and the closest Eye
# probability 0.062 away from its 0.9 threshold. The frozen gate accepts up to
# 0.015 on its 217 evaluation components (measured 0.0135, mean 0.0008). These
# bounds are tighter than the gate yet above the measured drift by far more
# than the ~1e-6 float32 differences between CPUs.
_INT8_MAX_ABS_DIFF = 1e-2
_INT8_MEAN_ABS_DIFF = 1e-3


def _rejection_flags(probabilities):
    labeled = {'etc': {'ic_classification': {'ICLabel': {'classifications': probabilities}}}}
    return eeg_icflag(labeled, DEFAULT_ICFLAG_THRESHOLDS)['reject']['gcompreject']


def _assert_within_int8_drift(actual, reference):
    difference = np.abs(actual - reference)
    assert float(difference.max()) < _INT8_MAX_ABS_DIFF
    assert float(difference.mean()) < _INT8_MEAN_ABS_DIFF
    np.testing.assert_array_equal(actual.argmax(axis=1), reference.argmax(axis=1))
    np.testing.assert_array_equal(_rejection_flags(actual), _rejection_flags(reference))


def _float32_classifications(eeg):
    """Classify with the float32 network from netICL.mat through eager PyTorch."""
    import torch

    from eegprep.plugins.ICLabel.iclabel_net import ICLabelNet

    model = ICLabelNet(str(_NET_ICL_MAT))
    model.eval()
    inputs = iclabel_module.prepare_network_inputs(ICL_feature_extractor(eeg, True))
    with torch.no_grad():
        output = model(*(torch.from_numpy(np.ascontiguousarray(array)) for array in inputs)).numpy()
    return iclabel_module.postprocess_network_output(output)


def test_postprocess_averages_the_four_augmentations_of_each_component():
    # Batch row ``augmentation * components + component`` holds one augmented
    # copy, matching prepare_network_inputs. The offsets cancel in the
    # four-way mean, and scaling them per component makes any mix-up of
    # components or augmentations change the result. All values are dyadic
    # rationals, so float32 arithmetic recovers the table exactly.
    components = 3
    table = np.arange(components * 7, dtype=np.float32).reshape(components, 7) / 8
    offsets = np.array([-3, -1, 1, 3], dtype=np.float32) / 8
    output = np.empty((4 * components, 7, 1, 1), dtype=np.float32)
    for augmentation, offset in enumerate(offsets):
        for component in range(components):
            output[augmentation * components + component, :, 0, 0] = table[component] + offset * (component + 1)

    np.testing.assert_array_equal(iclabel_module.postprocess_network_output(output), table)


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

        # MATLAB runs the float32 network and iclabel() the shipped int8 one.
        # MATLAB matched the Python float32 network within 3.37e-06 on this
        # dataset, so by the triangle inequality the int8 drift bounds above
        # still hold, and the class and keep-or-reject margins (0.019 and 0.062)
        # exceed twice that drift, so those decisions must agree exactly.
        EEG_matlab = iclabel(self.EEG, algorithm='default', engine='matlab')

        res1 = iclabel(self.EEG)['etc']['ic_classification']['ICLabel']['classifications']
        res2 = EEG_matlab['etc']['ic_classification']['ICLabel']['classifications']

        difference = np.abs(res1 - res2)
        print(f"\nPython int8 versus MATLAB: max abs diff {difference.max():.2e}, mean {difference.mean():.2e}")
        _assert_within_int8_drift(res1, res2)


@unittest.skipUnless(has_optional_dependency('onnxruntime'), "install eegprep[iclabel] to run ICLabel runtime coverage")
class TestICLabelRuntime(unittest.TestCase):
    def test_default_entry_point_stays_within_int8_drift_of_the_float32_network(self):
        if not has_optional_dependency('torch'):
            self.skipTest("install eegprep[torch] to rebuild the float32 ICLabel network from netICL.mat")
        eeg = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

        output = iclabel(eeg)
        classifications = output['etc']['ic_classification']['ICLabel']['classifications']

        self.assertEqual(classifications.shape, (32, 7))
        self.assertEqual(output['etc']['ic_classification']['ICLabel']['version'], 'default')
        _assert_within_int8_drift(classifications, _float32_classifications(eeg))


if __name__ == '__main__':
    unittest.main()
