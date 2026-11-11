"""
Comprehensive test suite for runica.py

This test suite includes:
1. TestRunicaFunctionality - Basic functionality tests without MATLAB dependency
2. TestRunicaParity - MATLAB parity tests for numerical equivalence

IMPORTANT: For parity tests, use rndreset='off' to ensure deterministic behavior
with seed 5489 (MATLAB default). This allows exact comparison of results.
"""

import os
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.io
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

from eegprep.functions.sigprocfunc.runica import runica
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.eeg_runica import eeg_runica
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_runica import pop_runica
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


_SIGNED_PERMUTATIONS = np.array(
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


def _minimum_signed_permutation_error(sources, demixed):
    sources = np.asarray(sources, dtype=float)
    demixed = np.asarray(demixed, dtype=float)
    sources = 2 * (sources - sources.min(axis=1, keepdims=True)) / np.ptp(sources, axis=1, keepdims=True) - 1
    demixed = 2 * (demixed - demixed.min(axis=1, keepdims=True)) / np.ptp(demixed, axis=1, keepdims=True) - 1
    differences = np.max(np.abs(sources - np.einsum("kij,jt->kit", _SIGNED_PERMUTATIONS, demixed)), axis=(1, 2))
    return float(np.min(differences)), demixed


def _reference_runica_comparison(eeglab_backend, sources, data, *options):
    weights, sphere = eeglab_backend("runica", data, *options, nargout=2)
    demixed = eeglab_backend("icaact", data, np.einsum("ij,jk->ik", weights, sphere), data.mean(axis=1, keepdims=True))
    difference, demixed = _minimum_signed_permutation_error(sources, demixed)
    return difference, demixed


def _eeg_from(data):
    array = np.asarray(data)
    frames = int(array.shape[1])
    return {
        "data": array,
        "nbchan": int(array.shape[0]),
        "pnts": frames,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": (frames - 1) / 100.0,
        "chanlocs": [{} for _channel in range(array.shape[0])],
    }


def _aligned_mixing_columns(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    components = reference.shape[1]
    correlation = np.empty((components, components))
    for row in range(components):
        for column in range(components):
            correlation[row, column] = np.corrcoef(reference[:, row], candidate[:, column])[0, 1]
    matched_rows, matched_columns = linear_sum_assignment(-np.abs(correlation))
    aligned = np.empty_like(reference)
    for row, column in zip(matched_rows, matched_columns):
        sign = np.sign(correlation[row, column])
        aligned[:, row] = (1.0 if sign == 0 else sign) * candidate[:, column]
    return aligned, np.abs(correlation[matched_rows, matched_columns])


def _runica_pca_projection(data, component_count):
    """Project ``data`` onto the rank-``component_count`` subspace ``runica`` keeps."""
    centered = data - np.mean(data, axis=1, keepdims=True)
    second_moment = centered @ (centered.T / centered.shape[1])
    eigenvalues, eigenvectors = eig(second_moment)
    order = np.argsort(eigenvalues.real)[::-1]
    basis = eigenvectors[:, order[:component_count]].real
    projector = basis @ np.linalg.inv(basis.T @ basis) @ basis.T
    return projector @ data


def _float32_average_reference(channels, frames, seed):
    values = np.random.default_rng(seed).standard_normal((channels, frames)).astype(np.float32)
    return values - values.mean(axis=0, keepdims=True)


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

    def test_runica_name_value_pairs_match_keywords(self):
        sources = np.vstack(
            (
                np.sin(np.linspace(0, 50, 200)),
                np.sin(np.linspace(0, 37, 200) + 5),
            )
        )
        data = np.vstack((sources[0] - 2 * sources[1], 1.73 * sources[0] + 3.41 * sources[1]))
        keyword = runica(data, extended=1, maxsteps=1, verbose="off", rndreset="off")
        positional = runica(data, "extended", 1, "maxsteps", 1, "verbose", "off", "rndreset", "off")

        np.testing.assert_array_equal(positional[0], keyword[0])
        np.testing.assert_array_equal(positional[1], keyword[1])

    def test_omitted_rndreset_matches_off_and_repeats(self):
        sources = np.vstack(
            (
                np.sin(np.linspace(0, 50, 200)),
                np.sin(np.linspace(0, 37, 200) + 5),
            )
        )
        data = np.vstack((sources[0] - 2 * sources[1], 1.73 * sources[0] + 3.41 * sources[1]))
        first = runica(data, maxsteps=1, verbose="off")
        second = runica(data, maxsteps=1, verbose="off")
        explicit = runica(data, maxsteps=1, verbose="off", rndreset="off")

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[0], explicit[0])

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

    def test_parity_average_reference_pop_runica_rank(self):
        """Average-referenced pop_runica should keep EEGLAB's rank-sized decomposition."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")

        eeg = pop_reref(pop_loadset("sample_data/eeglab_data.set"), [])
        data = np.asarray(eeg["data"], dtype=np.float64)
        eeg["data"] = data
        python = pop_runica(
            eeg,
            "icatype",
            "runica",
            "extended",
            1,
            "maxsteps",
            1,
            "verbose",
            "off",
            "rndreset",
            "off",
            "lrate",
            0.001,
            "pythoncompat",
            "on",
        )
        sample = Path("sample_data/eeglab_data.set").resolve()
        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "data.mat")
            out_path = os.path.join(directory, "out.mat")
            scipy.io.savemat(data_path, {"data": data})
            self.eeglab.eval(
                f"""
                load('{data_path}');
                EEG = pop_loadset('filename', '{sample.name}', 'filepath', '{sample.parent.as_posix()}');
                EEG.data = data;
                EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 1, ...
                    'verbose', 'off', 'rndreset', 'off', 'lrate', 0.001, 'pythoncompat', 'on');
                w_ml = EEG.icaweights; s_ml = EEG.icasphere; a_ml = EEG.icawinv;
                save('{out_path}', 'w_ml', 's_ml', 'a_ml', '-v7');
                """,
                nargout=0,
            )
            matlab = scipy.io.loadmat(out_path)

        weights = np.asarray(matlab["w_ml"], dtype=float)
        sphere = np.asarray(matlab["s_ml"], dtype=float)
        maps = np.asarray(matlab["a_ml"], dtype=float)
        self.assertEqual(python["icaweights"].shape, (31, 32))
        self.assertEqual(weights.shape, python["icaweights"].shape)
        self.assertEqual(sphere.shape, python["icasphere"].shape)
        self.assertEqual(maps.shape, python["icawinv"].shape)
        np.testing.assert_allclose(python["icasphere"], sphere, rtol=1e-10, atol=1e-12)
        python_data = np.asarray(python["data"], dtype=float)
        reconstructed = python["icawinv"] @ python["icaweights"] @ python["icasphere"] @ python_data
        np.testing.assert_allclose(reconstructed, _runica_pca_projection(python_data, 31), rtol=1e-10, atol=1e-12)
        _aligned, correlations = _aligned_mixing_columns(maps, python["icawinv"])
        self.assertGreaterEqual(float(correlations.min()), 0.9999)


def test_pop_runica_uses_covariance_eigenvalue_rank_when_matrix_rank_is_full():
    data = _float32_average_reference(8, 400, 0)
    sample = np.asarray(data, dtype=np.float64)
    eigenvalue_rank = int(np.sum(np.linalg.eigvalsh(np.cov(sample, rowvar=True, ddof=0)) > 1e-7))
    assert np.linalg.matrix_rank(sample) > eigenvalue_rank

    output, command = pop_runica(_eeg_from(data), maxsteps=1, verbose="off", rndreset="off", return_com=True)

    assert output["icaweights"].shape == (eigenvalue_rank, sample.shape[0])
    assert output["icasphere"].shape == (sample.shape[0], sample.shape[0])
    assert f"'pca', {eigenvalue_rank}" in command
    reconstructed = output["icawinv"] @ output["icaweights"] @ output["icasphere"] @ sample
    np.testing.assert_allclose(reconstructed, sample, rtol=1e-5, atol=1e-5)


def test_pop_runica_average_referenced_sample_keeps_the_signal_subspace():
    eeg = pop_reref(pop_loadset("sample_data/eeglab_data.set"), [])
    data = np.asarray(eeg["data"], dtype=np.float64)
    eeg["data"] = data

    output, command = pop_runica(eeg, maxsteps=1, extended=1, verbose="off", rndreset="off", return_com=True)

    assert output["icaweights"].shape == (31, 32)
    assert output["icawinv"].shape == (32, 31)
    np.testing.assert_array_equal(output["icasphere"], np.eye(32))
    assert (
        command
        == "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 1, 'verbose', 'off', 'rndreset', 'off', 'pca', 31);"
    )
    reconstructed = output["icawinv"] @ output["icaweights"] @ output["icasphere"] @ data
    np.testing.assert_allclose(reconstructed, _runica_pca_projection(data, 31), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reconstructed, data, rtol=1e-5, atol=1e-5)
    unmixing = output["icaweights"] @ output["icasphere"]
    assert np.linalg.norm(unmixing @ np.ones(32)) < 1e-5 * np.linalg.norm(unmixing)


def test_pop_runica_keeps_an_explicit_pca_option():
    generator = np.random.default_rng(1)
    data = generator.standard_normal((4, 500))
    data[3] = data[0]

    reduced, reduced_command = pop_runica(
        _eeg_from(data), pca=2, maxsteps=1, verbose="off", rndreset="off", return_com=True
    )
    negative, negative_command = pop_runica(
        _eeg_from(data), pca=-1, maxsteps=1, verbose="off", rndreset="off", return_com=True
    )

    assert reduced["icaweights"].shape == (2, 4)
    assert reduced_command.count("'pca'") == 1
    assert "'pca', 2" in reduced_command
    assert negative["icaweights"].shape == (3, 4)
    assert "'pca', -1" in negative_command
    assert "'pca', 3" not in negative_command


def test_pop_runica_rank_estimate_uses_the_first_3000_frames():
    generator = np.random.default_rng(0)
    data = generator.standard_normal((4, 4000))
    data[3, :3000] = data[0, :3000]
    assert np.linalg.matrix_rank(data[:, :3000]) == 3
    assert np.linalg.matrix_rank(data) == 4

    output, command = pop_runica(_eeg_from(data), maxsteps=1, verbose="off", rndreset="off", return_com=True)

    assert output["icaweights"].shape == (3, 4)
    assert "'pca', 3" in command


def test_pop_runica_rank_reduction_recovers_sources_up_to_sign_and_order():
    sources, data, tolerance = _reference_runica_mixture()
    redundant = np.vstack((data, 0.4 * data[0] - 0.2 * data[1]))
    mixing = np.array([[1.0, -2.0], [1.73, 3.41]])
    mixing = np.vstack((mixing, 0.4 * mixing[0] - 0.2 * mixing[1]))

    output, command = pop_runica(
        _eeg_from(redundant), extended=1, maxsteps=512, verbose="off", rndreset="off", return_com=True
    )

    assert output["icaweights"].shape == (2, 3)
    assert "'pca', 2" in command
    activations = np.asarray(output["icaact"], dtype=float).reshape(2, -1, order="F")
    difference, _demixed = _minimum_signed_permutation_error(sources, activations)
    assert difference < tolerance
    reconstructed = output["icawinv"] @ output["icaweights"] @ output["icasphere"] @ redundant
    np.testing.assert_allclose(reconstructed, redundant, rtol=1e-8, atol=1e-8)
    _aligned, correlations = _aligned_mixing_columns(mixing, output["icawinv"])
    assert float(correlations.min()) >= 0.9999


def test_eeg_runica_omitted_pca_uses_the_estimated_rank():
    data = _float32_average_reference(8, 400, 0)

    output = eeg_runica(_eeg_from(data), maxsteps=1, verbose="off", rndreset="off")

    assert output["icaweights"].shape == (7, 8)
    assert output["icasphere"].shape == (8, 8)


def test_concatenated_pop_runica_records_the_shared_rank():
    data = _float32_average_reference(8, 400, 0)
    first = _eeg_from(data[:, :200])
    second = _eeg_from(data[:, 200:])

    output, command = pop_runica(
        [first, second], concatenate="on", maxsteps=1, verbose="off", rndreset="off", return_com=True
    )

    assert output[0]["icaweights"].shape == (7, 8)
    assert output[1]["icaweights"].shape == (7, 8)
    assert "'pca', 7" in command
    assert "'concatenate', 'on'" in command


def test_separate_datasets_keep_their_own_runica_ranks():
    generator = np.random.default_rng(2)
    deficient = generator.standard_normal((4, 80))
    deficient[3] = deficient[0]
    full = generator.standard_normal((4, 80))

    output, command = pop_runica(
        [_eeg_from(deficient), _eeg_from(full)],
        maxsteps=1,
        verbose="off",
        rndreset="off",
        return_com=True,
    )

    assert output[0]["icaweights"].shape == (3, 4)
    assert output[1]["icaweights"].shape == (4, 4)
    assert "'pca'" not in command


def test_bare_runica_average_reference_sphere_is_ill_conditioned_on_the_null_direction():
    """Direct runica keeps the full sphere; only its signal subspace is meaningful."""
    data = np.asarray(_float32_average_reference(8, 400, 0), dtype=np.float64)
    _weights, sphere, *_rest = runica(data, maxsteps=1, verbose="off", rndreset="off")
    _eigenvalues, vectors = np.linalg.eigh(np.cov(data))
    null_gain = np.linalg.norm(sphere @ vectors[:, 0])
    signal = vectors[:, 1:]
    signal_gain = np.linalg.norm(sphere @ signal, axis=0).max()
    projected = signal.T @ sphere @ signal

    assert sphere.shape == (8, 8)
    assert null_gain > 1e3 * signal_gain
    assert np.isfinite(projected).all()
    assert np.linalg.norm(projected, 2) < 100


if __name__ == '__main__':
    unittest.main()
