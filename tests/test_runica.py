"""
Comprehensive test suite for runica.py

This test suite includes:
1. TestRunicaFunctionality - Basic functionality tests without MATLAB dependency
2. TestRunicaParity - MATLAB parity tests for numerical equivalence

IMPORTANT: For parity tests, use rndreset='off' to ensure deterministic behavior
with seed 5489 (MATLAB default). This allows exact comparison of results.
"""

import os
import unittest
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io

from eegprep.functions.sigprocfunc.runica import runica
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.eeglab_tests import eeglab_test


_RUNICA_SOURCE = "unittesting_sigprocfunc/runica/sigprocfunc_runica_wrapperTest.m"
_SOURCE_ICA_EPSILON = 0.0145  # unittesting_common/helpfunc/tc_icaepsilon.m


def _reference_runica_mixture(noise_db=None):
    sources = np.vstack((np.sin(np.linspace(0, 50, 1000)), np.sin(np.linspace(0, 37, 1000) + 5)))
    tolerance = _SOURCE_ICA_EPSILON
    if noise_db is not None:
        level = 10 ** (np.log10(np.ptp(sources, axis=1, keepdims=True)) - noise_db / 20)
        sources += level * (np.random.default_rng(1).random(sources.shape) - 0.5)
        tolerance += np.max(level) / 2
    data = np.vstack((sources[0] - 2 * sources[1], 1.73 * sources[0] + 3.41 * sources[1]))
    return sources, data, tolerance


def _reference_runica_comparison(eeglab_backend, sources, data, *options):
    weights, sphere = eeglab_backend("runica", data, *options, nargout=2)
    demixed = eeglab_backend("icaact", data, np.einsum("ij,jk->ik", weights, sphere), data.mean(axis=1, keepdims=True))
    sources = 2 * (sources - sources.min(axis=1, keepdims=True)) / np.ptp(sources, axis=1, keepdims=True) - 1
    demixed = 2 * (demixed - demixed.min(axis=1, keepdims=True)) / np.ptp(demixed, axis=1, keepdims=True) - 1
    transformations = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[-1, 0], [0, 1]],
            [[0, -1], [1, 0]],
            [[1, 0], [0, -1]],
            [[0, 1], [-1, 0]],
            [[-1, 0], [0, -1]],
            [[0, -1], [-1, 0]],
        ]
    )
    differences = np.max(np.abs(sources - np.einsum("kij,jt->kit", transformations, demixed)), axis=(1, 2))
    return np.min(differences), demixed


@eeglab_test(_RUNICA_SOURCE, "test_pass_extended")
@eeglab_test(_RUNICA_SOURCE, "test_pass_extended_nobias")
def test_reference_runica_extended(eeglab_backend):
    for options in (("extended", 1.0), ("extended", 1.0, "bias", "off")):
        sources, data, tolerance = _reference_runica_mixture()
        difference, _demixed = _reference_runica_comparison(eeglab_backend, sources, data, *options)
        # Preserve the source's single-run guard and strict ICA tolerance.
        if difference >= 0:
            assert difference < tolerance


@eeglab_test(_RUNICA_SOURCE, "test_pass_extended_noise10dB")
@eeglab_test(_RUNICA_SOURCE, "test_pass_extended_noise20dB")
def test_reference_runica_extended_noise(eeglab_backend):
    for noise_db in (10, 20):
        sources, data, tolerance = _reference_runica_mixture(noise_db)
        difference, _demixed = _reference_runica_comparison(eeglab_backend, sources, data, "extended", 1.0)
        if difference >= 0:
            assert difference < tolerance


@eeglab_test(_RUNICA_SOURCE, "test_pass_general")
@eeglab_test(_RUNICA_SOURCE, "test_pass_general_nobias")
@eeglab_test(_RUNICA_SOURCE, "test_pass_general_noise20dB")
@eeglab_test(_RUNICA_SOURCE, "test_pass_weights")
def test_reference_runica_general_smoke(eeglab_backend):
    # These source cases compute the sign/permutation error but comment out
    # their final assertions; do not substitute new pass criteria here.
    for options in ((), ("bias", "off"), ("weights", np.random.default_rng(1).random((2, 2)))):
        sources, data, _tolerance = _reference_runica_mixture()
        _reference_runica_comparison(eeglab_backend, sources, data, *options)
    sources, data, _tolerance = _reference_runica_mixture(20)
    _reference_runica_comparison(eeglab_backend, sources, data)


@eeglab_test(_RUNICA_SOURCE, "test_pass_pca")
def test_reference_runica_pca(eeglab_backend):
    sources, data, _tolerance = _reference_runica_mixture()
    data = np.vstack((data, -0.9 * sources[0] + 0.6 * sources[1]))
    _reference_runica_comparison(eeglab_backend, sources, data, "pca", 2.0)


@eeglab_test(_RUNICA_SOURCE, "test_pass_posact")
@pytest.mark.gui
def test_reference_runica_posact(eeglab_backend, request):
    sources, data, tolerance = _reference_runica_mixture()
    difference, demixed = _reference_runica_comparison(eeglab_backend, sources, data, "posact", "on", "extended", 1.0)
    matlab = request.config.getoption("--eeglab-backend") == "matlab"
    try:
        if matlab:
            eeglab_backend("plot", demixed.T, nargout=0)
        else:
            plt.plot(demixed.T)
        if difference >= 0:
            assert difference < tolerance
    finally:
        if matlab:
            eeglab_backend("close", "all", nargout=0)
        else:
            plt.close("all")


class TestRunicaFunctionality(unittest.TestCase):
    """Test runica functionality without MATLAB dependency."""

    def test_runica_does_not_mutate_input_array(self):
        """runica must not modify the caller's data array (mean subtraction)."""
        np.random.seed(42)
        data = np.random.randn(8, 500).astype(np.float64)
        original = data.copy()

        runica(data, maxsteps=5, verbose=False, rndreset='off')

        # The float64 input array passed by the caller must be untouched.
        self.assertTrue(np.array_equal(data, original))

    def test_posact_orients_largest_absolute_activation_positive(self):
        sources = np.vstack(
            [
                np.sin(np.linspace(0, 50, 1000)),
                np.sin(np.linspace(0, 37, 1000) + 5),
            ]
        )
        data = np.vstack([sources[0] - 2 * sources[1], 1.73 * sources[0] + 3.41 * sources[1]])

        weights, sphere, *_ = runica(data, posact="on", extended=1, verbose=False, rndreset="off")
        activations = weights @ sphere @ data
        peak_frames = np.argmax(np.abs(activations), axis=1)

        self.assertTrue(np.all(activations[np.arange(activations.shape[0]), peak_frames] >= 0))


class TestRunicaParity(unittest.TestCase):
    """Test parity with MATLAB runica."""

    def setUp(self):
        """Set up MATLAB engine."""
        try:
            self.eeglab = get_eeglab('MAT', auto_file_roundtrip=False)
            self.matlab_available = True
        except Exception as e:
            self.matlab_available = False
            self.skipTest(f"MATLAB not available: {e}")

    def test_parity_sample_data_extended_ica_initial_sphere(self):
        """Sample-data extended ICA should match EEGLAB's initial sphering."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        eeg = pop_loadset("sample_data/eeglab_data.set")
        data = eeg["data"].astype(np.float64).reshape(eeg["nbchan"], -1)

        w_py, s_py, _cv_py, b_py, sg_py, lr_py = runica(
            data.copy(), extended=1, rndreset="off", maxsteps=1, verbose=False
        )

        temp_file = tempfile.mktemp(suffix=".mat")
        scipy.io.savemat(temp_file, {"data": data})
        try:
            matlab_code = f"""
            load('{temp_file}');
            [w_ml, s_ml, ~, b_ml, sg_ml, lr_ml] = runica(data, 'extended', 1, 'maxsteps', 1, 'verbose', 'off', 'pythoncompat', 'on');
            save('{temp_file}_out.mat', 'w_ml', 's_ml', 'b_ml', 'sg_ml', 'lr_ml');
            """
            self.eeglab.eval(matlab_code, nargout=0)
            ml_data = scipy.io.loadmat(temp_file + "_out.mat")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(temp_file + "_out.mat"):
                os.remove(temp_file + "_out.mat")

        w_ml = ml_data["w_ml"]
        s_ml = ml_data["s_ml"]
        b_ml = ml_data["b_ml"]
        sg_ml = ml_data["sg_ml"].flatten()
        lr_ml = ml_data["lr_ml"].flatten()

        np.testing.assert_allclose(s_py, s_ml, rtol=1e-10, atol=1e-12)
        self.assertEqual(w_py.shape, w_ml.shape)
        self.assertEqual(b_py.shape, b_ml.shape)
        np.testing.assert_array_equal(sg_py, sg_ml)
        np.testing.assert_allclose(lr_py, lr_ml, rtol=1e-12, atol=1e-15)
        self.assertTrue(np.isfinite(w_py).all())
        self.assertTrue(np.isfinite(w_ml).all())
        self.assertTrue(np.isfinite(s_py).all())
        self.assertTrue(np.isfinite(s_ml).all())


if __name__ == '__main__':
    unittest.main()
