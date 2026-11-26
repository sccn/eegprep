"""eeg_mir: the mutual information reduction of an ICA decomposition, against closed forms and pAMICA."""

from __future__ import annotations

import numpy as np
import pytest

from eegprep import eeg_mir
from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc.eeg_mir import MIR_NBINS
from tests.fixtures import create_test_eeg

ROTATION_45 = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
# Two independent unit-variance uniform sources rotated by 45 degrees: each mixture is triangular
# (entropy 1/2 + ln sqrt(6)) and each source uniform (ln sqrt(12)), so the rotation back removes
# 2 * (1/2 - ln sqrt(2)) = 1 - ln 2 nats.
UNIFORM_45_MIR = 1.0 - np.log(2.0)
# Bin k of a row holds round((nbins - 1) * (x - min) / (max - min)) == k, so the first and last bins
# are half as wide as the others and a uniform row reads ln 2 / (nbins - 1) nats too high, while the
# triangular mixtures have almost no mass at their ends. With 256 bins the estimate is therefore
# 2 ln 2 / 255 = 0.0054 nats below the closed form; with getMIR's round(3 * log2(1 + N / 10)), 43 bins
# at these 200000 samples, it would be 0.033 nats below. The other 0.004 is four standard deviations
# of the estimate over seeds at this length.
UNIFORM_45_TOLERANCE = 2.0 * np.log(2.0) / 255 + 0.004


def _ica_eeg(data, weights, sphere, icachansind=None):
    data = np.asarray(data, dtype=np.float64)
    eeg = create_test_eeg(n_channels=data.shape[0], n_samples=data.shape[1], n_trials=1)
    eeg["data"] = data
    if data.ndim == 3:
        eeg["pnts"], eeg["trials"] = data.shape[1], data.shape[2]
    eeg["icaweights"] = np.asarray(weights, dtype=np.float64)
    eeg["icasphere"] = np.asarray(sphere, dtype=np.float64)
    eeg["icachansind"] = np.arange(data.shape[0]) if icachansind is None else np.asarray(icachansind)
    return eeg


def _laplace_mixture(n_channels, n_samples, seed):
    rng = np.random.default_rng(seed)
    sources = rng.laplace(size=(n_channels, n_samples))
    mixing = rng.normal(size=(n_channels, n_channels))
    return finite_matmul(mixing, sources), rng


def test_two_uniform_sources_rotated_by_45_degrees_lose_one_minus_ln2_nats():
    rng = np.random.default_rng(0)
    sources = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(2, 200_000))
    eeg = _ica_eeg(finite_matmul(ROTATION_45, sources), weights=ROTATION_45.T, sphere=np.eye(2))

    value, _variance = eeg_mir(eeg)

    assert abs(value - UNIFORM_45_MIR) < UNIFORM_45_TOLERANCE


def test_identity_weights_reduce_nothing():
    data, rng = _laplace_mixture(4, 20_000, seed=1)
    sphere = rng.normal(size=(4, 4))
    eeg = _ica_eeg(data, weights=np.eye(4), sphere=sphere)

    value, _variance = eeg_mir(eeg)

    assert value == 0.0


def test_scaling_reordering_and_sign_flips_of_the_weights_leave_mir_unchanged():
    data, rng = _laplace_mixture(4, 20_000, seed=2)
    weights = rng.normal(size=(4, 4))
    sphere = rng.normal(size=(4, 4))
    rescale = np.diag([2.5, -0.3, 7.0, -1.0])
    reorder = np.eye(4)[[2, 0, 3, 1]]

    value, variance = eeg_mir(_ica_eeg(data, weights, sphere))
    moved, moved_variance = eeg_mir(_ica_eeg(data, reorder @ rescale @ weights, sphere))

    assert abs(moved - value) < 1e-12
    assert abs(moved_variance - variance) < 1e-12


def test_matches_pamica_on_epoched_data_a_channel_subset_and_a_reduced_sphere():
    metrics = pytest.importorskip("pamica.metrics")
    rng = np.random.default_rng(3)
    n_channels, pnts, trials, rank = 8, 2_000, 5, 4
    icachansind = np.array([0, 1, 2, 4, 5, 7])
    sources = rng.laplace(size=(n_channels, pnts * trials))
    mixed = finite_matmul(rng.normal(size=(n_channels, n_channels)), sources)
    data = mixed.reshape(n_channels, pnts, trials, order="F")
    weights = rng.normal(size=(rank, rank))
    sphere = rng.normal(size=(rank, icachansind.size))

    value, variance = eeg_mir(_ica_eeg(data, weights, sphere, icachansind=icachansind))

    expected = metrics.mir(weights, finite_matmul(sphere, mixed[icachansind]), nbins=MIR_NBINS)
    assert (value, variance) == expected


def _no_ica(eeg):
    eeg["icaweights"] = np.array([])
    eeg["icasphere"] = np.array([])


def _pca_reduced_runica(eeg):
    eeg["icaweights"] = eeg["icaweights"][:3]


def _duplicated_component(eeg):
    eeg["icaweights"][1] = eeg["icaweights"][0]


def _rank_deficient_sphere(eeg):
    eeg["icasphere"][3] = eeg["icasphere"][0] + eeg["icasphere"][1]


def _sphere_for_other_channels(eeg):
    eeg["icachansind"] = np.arange(3)


def _nan_sample(eeg):
    eeg["data"][2, 100] = np.nan


def _flat_channel(eeg):
    eeg["icasphere"] = np.eye(4)
    eeg["data"][1] = 5.0


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_no_ica, "no ICA decomposition"),
        (_pca_reduced_runica, "square"),
        (_duplicated_component, "icaweights is singular"),
        (_rank_deficient_sphere, "icasphere is singular"),
        (_sphere_for_other_channels, "icachansind"),
        (_nan_sample, "NaN or infinite"),
        (_flat_channel, "constant"),
    ],
)
def test_refuses_a_decomposition_it_cannot_measure(corrupt, message):
    data, rng = _laplace_mixture(4, 5_000, seed=4)
    eeg = _ica_eeg(data, weights=rng.normal(size=(4, 4)), sphere=rng.normal(size=(4, 4)))
    corrupt(eeg)

    with pytest.raises(ValueError, match=message):
        eeg_mir(eeg)
