import asyncio
import importlib
import os
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path

import numpy as np
import pytest
from eegprep import (
    ICL_feature_extractor,
    eeg_autocorr_fftw,
    eeg_autocorr_welch,
    eeg_rpsd,
    iclabel,
    pop_loadset,
    pop_saveset,
)
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
_ICLABEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'eegprep', 'plugins', 'ICLabel')
_FLOAT32_NETWORK = 'iclabel.onnx'
_INT8_NETWORK = 'iclabel_int8.onnx'


@pytest.fixture
def emscripten_iclabel():
    """Reload the ICLabel modules as Pyodide imports them, then restore the native ones.

    Only the reload sees the patched platform. The reloaded modules keep the
    Emscripten backend they selected at import, while the test body, including
    any asyncio event loop it starts, runs on the real platform. The patch is
    private to this fixture, so it cannot undo patches made by other fixtures.
    """
    try:
        with pytest.MonkeyPatch.context() as platform_patch:
            platform_patch.setattr(sys, 'platform', 'emscripten')
            modules = tuple(importlib.reload(module) for module in _PLATFORM_SELECTED_MODULES)
        yield modules
    finally:
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
    # The probe has no JavaScript host, so the bridge call itself fails; what matters
    # is that neither entry point imported native onnxruntime first.
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


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf], ids=['nan', 'inf', '-inf'])
def test_web_bridge_output_must_be_finite(bad_value):
    flat = np.full(4 * 7, 0.5, dtype=np.float32)
    flat[3] = bad_value

    # eeg_icflag compares with ">", so a NaN probability would silently keep that component.
    with pytest.raises(RuntimeError, match='returned 1 non-finite values out of 28'):
        iclabel_net_onnx_web._network_output(flat, batch_size=4)


def _icl(eeg):
    return eeg['etc']['ic_classification']['ICLabel']


@pytest.mark.skipif(not has_optional_dependency('onnxruntime'), reason='install eegprep[iclabel] to run ICLabel')
def test_native_iclabel_records_the_float32_network_that_ran(sample_ica_eeg):
    synchronous = iclabel(sample_ica_eeg)
    asynchronous = asyncio.run(iclabel_module.iclabel_async(sample_ica_eeg))

    for labeled in (synchronous, asynchronous):
        assert _icl(labeled)['version'] == 'default'
        assert _icl(labeled)['network'] == _FLOAT32_NETWORK


class _Buffer:
    """Stands in for the JsBuffer Pyodide makes of the bridge's ``modelBytes``."""

    def __init__(self, data):
        self._data = data

    def to_bytes(self):
        return self._data


class _Output:
    """Stands in for the typed array the bridge's ``run`` resolves to."""

    def __init__(self, values):
        self._values = values

    def to_py(self):
        return list(self._values)


class _Int8WebBridge:
    """Stands in for the host's bridge: the packaged 8-bit network, run by native onnxruntime."""

    def __init__(self):
        import onnxruntime

        model = Path(_ICLABEL_DIR, _INT8_NETWORK).read_bytes()
        self.modelBytes = _Buffer(model)
        self._session = onnxruntime.InferenceSession(model, providers=['CPUExecutionProvider'])

    async def run(self, image, image_shape, psdmed, psdmed_shape, autocorr, autocorr_shape):
        inputs = {
            'image': image.reshape(image_shape),
            'psdmed': psdmed.reshape(psdmed_shape),
            'autocorr': autocorr.reshape(autocorr_shape),
        }
        (output,) = self._session.run(['output'], inputs)
        return _Output(output.reshape(-1))


@pytest.mark.skipif(not has_optional_dependency('onnxruntime'), reason='install eegprep[iclabel] to run ICLabel')
def test_browser_iclabel_records_the_int8_network_that_ran(sample_ica_eeg, emscripten_iclabel, monkeypatch):
    _, emscripten_iclabel_module, _ = emscripten_iclabel
    ffi = types.ModuleType('pyodide.ffi')
    ffi.to_js = lambda value: value
    host = types.ModuleType('js')
    host.eegprep_iclabel_web = _Int8WebBridge()
    monkeypatch.setitem(sys.modules, 'js', host)
    monkeypatch.setitem(sys.modules, 'pyodide', types.ModuleType('pyodide'))
    monkeypatch.setitem(sys.modules, 'pyodide.ffi', ffi)

    labeled = asyncio.run(emscripten_iclabel_module.iclabel_async(sample_ica_eeg))

    assert _icl(labeled)['version'] == 'default'
    assert _icl(labeled)['network'] == _INT8_NETWORK
    assert _icl(labeled)['classifications'].shape == (32, 7)


class _EeglabStandIn:
    """Stands in for an EEGLAB engine, an external program: it labels the dataset without ``network``."""

    def iclabel(self, EEG, algorithm='default'):
        labeled = dict(EEG)
        labeled['etc'] = {
            'ic_classification': {'ICLabel': {**EEG['etc']['ic_classification']['ICLabel'], 'version': algorithm}}
        }
        return labeled


@pytest.mark.parametrize('algorithm', ['default', 'lite'])
def test_eeglab_engine_result_drops_a_network_recorded_by_an_earlier_eegprep_run(algorithm):
    classified = {
        'etc': {
            'ic_classification': {
                'ICLabel': {'classifications': np.eye(2, 7), 'version': 'default', 'network': _INT8_NETWORK}
            }
        }
    }

    labeled = iclabel_module._run_eeglab_engine(_EeglabStandIn(), classified, algorithm)

    assert 'network' not in _icl(labeled)
    assert _icl(labeled)['version'] == algorithm
    np.testing.assert_array_equal(_icl(labeled)['classifications'], np.eye(2, 7))


@pytest.mark.skipif(not has_optional_dependency('onnxruntime'), reason='install eegprep[iclabel] to run ICLabel')
def test_saved_dataset_keeps_the_recorded_network(sample_ica_eeg, tmp_path):
    labeled = iclabel(sample_ica_eeg)
    path = str(tmp_path / 'labeled.set')

    pop_saveset(labeled, path)
    reloaded = pop_loadset(path)

    assert str(_icl(reloaded)['network']) == _FLOAT32_NETWORK
    assert str(_icl(reloaded)['version']) == 'default'


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


def _rejection_flags(probabilities):
    labeled = {'etc': {'ic_classification': {'ICLabel': {'classifications': probabilities}}}}
    return eeg_icflag(labeled, DEFAULT_ICFLAG_THRESHOLDS)['reject']['gcompreject']


def _assert_matches_float32_network(actual, reference):
    """The native default is the float32 export of netICL.mat, so it follows PyTorch to float32 noise."""
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
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


@unittest.skipUnless(has_optional_dependency('onnxruntime'), "install eegprep[iclabel] to run ICLabel runtime coverage")
class TestICLabelRuntime(unittest.TestCase):
    def test_default_entry_point_matches_the_float32_network_from_netICL(self):
        if not has_optional_dependency('torch'):
            self.skipTest("install eegprep[torch] to rebuild the float32 ICLabel network from netICL.mat")
        eeg = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))

        output = iclabel(eeg)
        classifications = output['etc']['ic_classification']['ICLabel']['classifications']

        self.assertEqual(classifications.shape, (32, 7))
        self.assertEqual(output['etc']['ic_classification']['ICLabel']['version'], 'default')
        _assert_matches_float32_network(classifications, _float32_classifications(eeg))


def test_continuous_icaact_without_a_trial_axis_matches_a_singleton_trial():
    rng = np.random.default_rng(0)
    pnts = 256
    activations = rng.standard_normal((3, pnts)).astype(np.float32)
    eeg = {
        "srate": 128,
        "pnts": pnts,
        "trials": 1,
        "icaweights": np.eye(3),
        "icaact": activations,
    }
    with_trial = dict(eeg)
    with_trial["icaact"] = activations[:, :, None]

    np.testing.assert_array_equal(eeg_rpsd(eeg, nfreqs=8), eeg_rpsd(with_trial, nfreqs=8))
    np.testing.assert_array_equal(eeg_autocorr_welch(eeg), eeg_autocorr_welch(with_trial))
    np.testing.assert_array_equal(eeg_autocorr_fftw(eeg), eeg_autocorr_fftw(with_trial))


if __name__ == '__main__':
    unittest.main()
