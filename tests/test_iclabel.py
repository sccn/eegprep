import asyncio
import importlib
import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np
import pytest
from eegprep import ICL_feature_extractor, iclabel, pop_loadset
from eegprep.utils.testing import has_optional_dependency

from eegprep.plugins.ICLabel import iclabel_net_onnx_web
from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS
import eegprep.plugins.ICLabel.iclabel as iclabel_module
import eegprep.plugins.ICLabel.iclabel_net_onnx as iclabel_onnx_module
import eegprep.plugins.ICLabel.pop_iclabel as pop_iclabel_module

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')
_NET_ICL_MAT = os.path.join(os.path.dirname(__file__), '..', 'src', 'eegprep', 'plugins', 'ICLabel', 'netICL.mat')
_SAMPLE_ICA_SET = os.path.join(os.path.dirname(__file__), '..', 'sample_data', 'eeglab_data_with_ica_tmp.set')
_PLATFORM_SELECTED_MODULES = (iclabel_onnx_module, iclabel_module, pop_iclabel_module)


@pytest.fixture
def emscripten_iclabel(monkeypatch):
    """Reload the ICLabel modules as Pyodide imports them, then restore the native ones."""
    monkeypatch.setattr(sys, 'platform', 'emscripten')
    try:
        yield tuple(importlib.reload(module) for module in _PLATFORM_SELECTED_MODULES)
    finally:
        monkeypatch.undo()
        for module in _PLATFORM_SELECTED_MODULES:
            importlib.reload(module)


def _run_python(code):
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)], capture_output=True, text=True, check=False, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_iclabel_net_onnx_selects_web_backend_on_emscripten(emscripten_iclabel):
    onnx_module, _, _ = emscripten_iclabel

    assert onnx_module.IS_EMSCRIPTEN is True
    assert onnx_module._run_iclabel_net_async is iclabel_net_onnx_web.run_iclabel_net_async


def test_sync_entry_points_fail_fast_under_emscripten(emscripten_iclabel):
    onnx_module, emscripten_iclabel_module, emscripten_pop_module = emscripten_iclabel

    with pytest.raises(RuntimeError, match=r'use await iclabel_async\(\.\.\.\)'):
        emscripten_iclabel_module.iclabel(None)
    with pytest.raises(RuntimeError, match=r'use await iclabel_async\(\.\.\.\)'):
        emscripten_pop_module.pop_iclabel(None)
    with pytest.raises(RuntimeError, match=r'use await run_iclabel_net_async\(\.\.\.\)'):
        onnx_module.run_iclabel_net(None, None, None)


def test_emscripten_iclabel_never_imports_onnxruntime():
    # A fresh interpreter that imports ICLabel as Pyodide would. The probe has
    # no JavaScript host, so the browser bridge call itself fails; what matters
    # is that neither entry point reached for native onnxruntime first.
    output = _run_python(
        f"""
        import asyncio
        import os
        import sys
        import traceback

        from eegprep import pop_loadset

        eeg = pop_loadset({_SAMPLE_ICA_SET!r})
        sys.platform = "emscripten"
        from eegprep.plugins.ICLabel.iclabel import iclabel, iclabel_async

        for call in (lambda: iclabel(eeg), lambda: asyncio.run(iclabel_async(eeg))):
            try:
                call()
            except Exception as exc:
                failing_module = os.path.basename(traceback.extract_tb(exc.__traceback__)[-1].filename)
                print(type(exc).__name__, failing_module)
        print(sorted(name for name in sys.modules if name.split(".")[0] == "onnxruntime"))
        """
    )

    sync_failure, async_failure, onnxruntime_modules = output.splitlines()
    assert sync_failure.startswith('RuntimeError ')
    # The asynchronous call got through feature extraction to the browser bridge.
    assert async_failure.endswith(' iclabel_net_onnx_web.py')
    assert onnxruntime_modules == '[]'


@pytest.mark.skipif(not has_optional_dependency('onnxruntime'), reason='needs onnxruntime to prove the block matters')
def test_iclabel_without_onnxruntime_names_the_iclabel_extra():
    output = _run_python(
        f"""
        import sys

        sys.modules["onnxruntime"] = None
        from eegprep import iclabel, pop_loadset

        try:
            iclabel(pop_loadset({_SAMPLE_ICA_SET!r}))
        except ImportError as exc:
            print(exc)
        """
    )

    assert 'pip install "eegprep[iclabel]"' in output


@pytest.mark.skipif(not has_optional_dependency('onnxruntime'), reason='install eegprep[iclabel] to run ICLabel')
def test_iclabel_async_matches_sync_on_sample_ica_dataset():
    eeg = pop_loadset(_SAMPLE_ICA_SET)
    original_weights = eeg['icaweights'].copy()

    asynchronous = asyncio.run(iclabel_module.iclabel_async(eeg))
    synchronous = iclabel(eeg)

    labels = asynchronous['etc']['ic_classification']['ICLabel']
    assert labels['classifications'].shape == (32, 7)
    np.testing.assert_array_equal(
        labels['classifications'], synchronous['etc']['ic_classification']['ICLabel']['classifications']
    )
    assert labels['version'] == 'default'
    assert 'ic_classification' not in eeg['etc']
    np.testing.assert_array_equal(eeg['icaweights'], original_weights)


def test_web_bridge_output_must_hold_seven_probabilities_per_input():
    flat = np.arange(4 * 7, dtype=np.float32)

    np.testing.assert_array_equal(iclabel_net_onnx_web._network_output(flat, batch_size=4), flat.reshape(4, 7, 1, 1))
    # 28 values are a multiple of 7, but they describe 4 inputs, not 8.
    with pytest.raises(RuntimeError, match='returned 28 values for a batch of 8; expected 56'):
        iclabel_net_onnx_web._network_output(flat, batch_size=8)


@pytest.mark.parametrize('algorithm', ['lite', 'beta'])
def test_python_engine_only_ships_the_default_network(sample_ica_eeg, algorithm):
    with pytest.raises(NotImplementedError, match=r'iclabel\.onnx'):
        iclabel(sample_ica_eeg, algorithm=algorithm)
    with pytest.raises(NotImplementedError, match=r'iclabel\.onnx'):
        asyncio.run(iclabel_module.iclabel_async(sample_ica_eeg, algorithm=algorithm))


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [({'engine': 'torch'}, 'Unsupported engine'), ({'algorithm': 'huge'}, 'algorithm must be one of')],
)
def test_unsupported_engine_or_algorithm_is_rejected(sample_ica_eeg, arguments, message):
    with pytest.raises(ValueError, match=message):
        iclabel(sample_ica_eeg, **arguments)
    with pytest.raises(ValueError, match=message):
        asyncio.run(iclabel_module.iclabel_async(sample_ica_eeg, **arguments))


def test_pop_iclabel_requires_an_ica_decomposition_before_classifying(sample_ica_eeg):
    without_ica = dict(sample_ica_eeg, icaweights=np.empty((0, 0)))
    # Classifying this one fails with AttributeError, so the ValueError below
    # shows that every dataset is checked before any is classified.
    unclassifiable = dict(sample_ica_eeg, icawinv=None)

    for EEG in (without_ica, [unclassifiable, without_ica]):
        with pytest.raises(ValueError, match='requires an ICA decomposition'):
            pop_iclabel_module.pop_iclabel(EEG, 'default')
        with pytest.raises(ValueError, match='requires an ICA decomposition'):
            asyncio.run(pop_iclabel_module.pop_iclabel_async(EEG, 'default'))


@pytest.mark.skipif(not has_optional_dependency('onnxruntime'), reason='install eegprep[iclabel] to run ICLabel')
def test_pop_iclabel_sync_and_async_classify_alike_with_replayable_history(sample_ica_eeg):
    synchronous, sync_command = pop_iclabel_module.pop_iclabel(sample_ica_eeg, 'Default', return_com=True)
    asynchronous, async_command = asyncio.run(
        pop_iclabel_module.pop_iclabel_async([sample_ica_eeg, None], 'default', return_com=True)
    )

    assert sync_command == "EEG = pop_iclabel(EEG, 'default');"
    assert async_command == "EEG = await pop_iclabel_async(EEG, 'default');"
    np.testing.assert_array_equal(
        asynchronous[0]['etc']['ic_classification']['ICLabel']['classifications'],
        synchronous['etc']['ic_classification']['ICLabel']['classifications'],
    )
    assert asynchronous[1] is None
    assert pop_iclabel_module.pop_iclabel(None, return_com=True) == (None, '')
    assert asyncio.run(pop_iclabel_module.pop_iclabel_async(None, return_com=True)) == (None, '')


@pytest.mark.parametrize('engine', ['matlab', 'octave'])
def test_eeglab_engines_are_unavailable_under_emscripten(sample_ica_eeg, emscripten_iclabel, engine):
    _, emscripten_iclabel_module, emscripten_pop_module = emscripten_iclabel
    message = 'MATLAB and Octave engines are unavailable under Emscripten; use engine=None'

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(emscripten_iclabel_module.iclabel_async(sample_ica_eeg, engine=engine))
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(emscripten_pop_module.pop_iclabel_async(sample_ica_eeg, 'default', engine=engine))
    with pytest.raises(RuntimeError, match=message):
        emscripten_iclabel_module.iclabel(sample_ica_eeg, engine=engine)
    with pytest.raises(RuntimeError, match=message):
        emscripten_pop_module.pop_iclabel(None, engine=engine)


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
