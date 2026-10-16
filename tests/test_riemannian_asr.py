"""Riemannian ASR geometry and processing."""

import numpy as np
import pytest

from eegprep.plugins.clean_rawdata.asr_calibrate_r import _block_outer_sums, asr_calibrate_r
from eegprep.plugins.clean_rawdata.asr_process_r import asr_process_r
from eegprep.plugins.clean_rawdata.clean_asr import clean_asr
from eegprep.plugins.clean_rawdata.private.riemann import (
    asr_subspace_reconstruction,
    rasr_nonlinear_eigenspace,
    spd_geodesic_midpoint,
)
from tests.fixtures import create_test_eeg


def test_spd_geodesic_midpoint_of_diagonal_matrices_is_geometric_mean():
    left = np.diag([1.0, 4.0, 9.0])
    right = np.diag([4.0, 4.0, 1.0])

    midpoint = spd_geodesic_midpoint(left, right)

    np.testing.assert_allclose(midpoint, np.diag([2.0, 4.0, 3.0]), atol=1e-8)
    np.testing.assert_allclose(spd_geodesic_midpoint(left, left), left, atol=1e-8)
    np.testing.assert_allclose(midpoint, spd_geodesic_midpoint(right, left), atol=1e-8)


def test_nonlinear_eigenspace_is_eigenbasis_of_shifted_covariance():
    rng = np.random.default_rng(1)
    factor = rng.normal(size=(5, 5))
    covariance = factor @ factor.T + np.eye(5)

    vectors, values = rasr_nonlinear_eigenspace(covariance, 5)
    solved = np.linalg.solve(covariance, np.ones(5))
    operator = covariance + np.diag(solved)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (operator + operator.T))

    np.testing.assert_allclose(np.sort(np.diag(values)), eigenvalues, atol=1e-8)
    for index in range(5):
        sign = np.sign(np.dot(vectors[:, index], eigenvectors[:, index])) or 1.0
        np.testing.assert_allclose(vectors[:, index], sign * eigenvectors[:, index], atol=1e-8)


def test_reconstruction_drops_a_singular_value_below_the_matlab_cutoff():
    channels = 62
    mixing = _spectrum_mixing(channels, 2e-13)
    observed = np.linalg.svd(mixing, compute_uv=False)
    cutoff = channels * np.spacing(observed[0])
    numpy_cutoff = 1e-15 * observed[0]
    assert observed[-1] < cutoff
    assert observed[-1] > numpy_cutoff

    reconstructed = asr_subspace_reconstruction(mixing, np.eye(channels), np.ones(channels, dtype=bool))

    # Inverting this component reproduces the identity and coefficients near 1e13.
    assert np.linalg.norm(reconstructed, 2) < 2.0
    assert np.linalg.norm(reconstructed - np.eye(channels), 2) > 0.5


def test_reconstruction_keeps_a_singular_value_above_the_matlab_cutoff():
    channels = 62
    mixing = _spectrum_mixing(channels, 5e-13)
    observed = np.linalg.svd(mixing, compute_uv=False)
    cutoff = channels * np.spacing(observed[0])
    scipy_cutoff = channels * np.finfo(float).eps * observed[0]
    assert observed[-1] > cutoff
    assert observed[-1] < scipy_cutoff

    reconstructed = asr_subspace_reconstruction(mixing, np.eye(channels), np.ones(channels, dtype=bool))

    # Keeping the component reproduces the identity up to the conditioning of sigma_min.
    assert np.linalg.norm(reconstructed - np.eye(channels), 2) < 0.2


def _spectrum_mixing(channels, smallest):
    singular_values = np.linspace(45.11, 1.42, channels)
    singular_values[-1] = smallest
    left, _ = np.linalg.qr(np.random.default_rng(306).standard_normal((channels, channels)))
    right, _ = np.linalg.qr(np.random.default_rng(457).standard_normal((channels, channels)))
    return left @ np.diag(singular_values) @ right


def test_reconstruction_is_invariant_to_eigenvector_sign_flips():
    rng = np.random.default_rng(2)
    mixing = rng.normal(size=(4, 4))
    basis, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    keep = np.array([True, True, False, False])

    original = asr_subspace_reconstruction(mixing, basis, keep)
    flipped = asr_subspace_reconstruction(mixing, basis @ np.diag([1.0, -1.0, -1.0, 1.0]), keep)

    np.testing.assert_allclose(original, flipped, atol=1e-8)


def test_block_outer_sums_repeat_the_tail_sample():
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )

    totals = _block_outer_sums(data, 2)

    expected = np.array(
        [
            [[5.0, 2.0], [2.0, 1.0]],
            [[25.0, 4.0], [4.0, 1.0]],
            [[50.0, 0.0], [0.0, 0.0]],
        ]
    )
    np.testing.assert_allclose(totals, expected)


def _bursty_eeg():
    eeg = create_test_eeg(n_channels=4, n_samples=2048, srate=128.0)
    time = np.arange(2048) / 128.0
    rng = np.random.default_rng(443)
    data = np.vstack([(1.0 + 0.05 * channel) * np.sin(2 * np.pi * 10 * time) for channel in range(4)])
    data += 0.05 * rng.standard_normal(data.shape)
    data[:, 800:1000] += 40.0
    eeg['data'] = data
    return eeg


def test_clean_asr_riemannian_repairs_burst_and_differs_from_euclidean_and_calib():
    eeg = _bursty_eeg()
    original = eeg['data'].copy()
    burst = slice(800, 1000)

    riemannian = clean_asr(eeg, cutoff=5, ref_maxbadchannels='off', useriemannian=True)
    euclidean = clean_asr(eeg, cutoff=5, ref_maxbadchannels='off', useriemannian=False)
    calib = clean_asr(eeg, cutoff=5, ref_maxbadchannels='off', useriemannian='calib')

    np.testing.assert_array_equal(eeg['data'], original)
    assert riemannian['data'].shape == original.shape
    assert np.isfinite(riemannian['data']).all()
    assert np.linalg.norm(riemannian['data'][:, burst]) < 0.5 * np.linalg.norm(original[:, burst])
    assert not np.allclose(riemannian['data'], euclidean['data'])
    assert not np.allclose(riemannian['data'], calib['data'])


def test_clean_asr_rejects_unknown_useriemannian_value():
    eeg = _bursty_eeg()
    with pytest.raises(ValueError, match='useriemannian'):
        clean_asr(eeg, ref_maxbadchannels='off', useriemannian='sideways')


def test_carried_covariance_changes_the_next_chunk():
    eeg = _bursty_eeg()
    state = asr_calibrate_r(eeg['data'], eeg['srate'], cutoff=5, maxmem=64)
    channels = eeg['data'].shape[0]
    near = dict(state)
    far = dict(state)
    near['cov'] = np.eye(channels) * 1e-2
    far['cov'] = np.eye(channels) * 1e3
    kwargs = {'window_len': 0.5, 'step_size': 16, 'max_dims': 1.0, 'max_mem': 64}

    cleaned_near, updated = asr_process_r(eeg['data'], eeg['srate'], near, **kwargs)
    cleaned_far, _ = asr_process_r(eeg['data'], eeg['srate'], far, **kwargs)

    assert updated['cov'].shape == (channels, channels)
    assert np.allclose(updated['cov'], updated['cov'].T)
    assert not np.allclose(cleaned_near, cleaned_far)
