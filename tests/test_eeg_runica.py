from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.pinv import pinv
from eegprep.functions.popfunc.eeg_runica import eeg_runica
from eegprep.functions.popfunc.pop_runica import pop_runica


def _epoched_eeg(offset=0):
    data = np.zeros((2, 3, 2), dtype=np.float64)
    data[0, :, 0] = np.array([1, 2, 3]) + offset
    data[0, :, 1] = np.array([4, 5, 6]) + offset
    data[1, :, 0] = np.array([11, 12, 13]) + offset
    data[1, :, 1] = np.array([14, 15, 16]) + offset
    return {
        "data": data,
        "nbchan": 2,
        "pnts": 3,
        "trials": 2,
        "srate": 100,
        "chanlocs": [],
    }


def _eeglab_flattened(data):
    return np.concatenate([data[:, :, trial] for trial in range(data.shape[2])], axis=1)


def test_eeg_runica_flattens_and_reshapes_epoched_data_like_eeglab(monkeypatch):
    eeg = _epoched_eeg()
    captured = {}

    def fake_runica(data, **_kwargs):
        captured["data"] = data.copy()
        return np.eye(2), np.eye(2), np.zeros(2), np.zeros((2, 1)), np.ones((2, 1)), []

    monkeypatch.setattr("eegprep.functions.popfunc.eeg_runica.runica", fake_runica)

    out = eeg_runica(eeg, extended=1, maxsteps=1)

    np.testing.assert_array_equal(captured["data"], _eeglab_flattened(eeg["data"]))
    np.testing.assert_array_equal(out["icaact"], eeg["data"])
    np.testing.assert_array_equal(out["icachansind"], np.array([0, 1]))


def _nonidentity_decomposition():
    """Weights and a non-identity sphere that make activations sign-flippable."""
    weights = np.array([[2.0, 1.0], [-1.0, 3.0]])
    sphere = np.array([[1.5, 0.5], [0.0, 2.0]])
    return weights, sphere


def _continuous_eeg():
    data = np.array([[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]])
    return {
        "data": data,
        "nbchan": 2,
        "pnts": 4,
        "trials": 1,
        "srate": 100,
        "chanlocs": [],
        "icaweights": np.eye(2),
        "icasphere": np.eye(2),
    }


def test_eeg_runica_does_not_mutate_caller(monkeypatch):
    eeg = _continuous_eeg()
    data_before = eeg["data"].copy()
    weights_before = eeg["icaweights"].copy()
    sphere_before = eeg["icasphere"].copy()

    weights, sphere = _nonidentity_decomposition()

    def fake_runica(data, **_kwargs):
        return weights.copy(), sphere.copy(), np.zeros(2), np.zeros((2, 1)), np.ones((2, 1)), []

    monkeypatch.setattr("eegprep.functions.popfunc.eeg_runica.runica", fake_runica)

    eeg_runica(eeg, posact=True)

    assert np.array_equal(eeg["data"], data_before)
    assert np.array_equal(eeg["icaweights"], weights_before)
    assert np.array_equal(eeg["icasphere"], sphere_before)


def test_eeg_runica_posact_preserves_ica_invariants(monkeypatch):
    eeg = _continuous_eeg()
    weights, sphere = _nonidentity_decomposition()
    weights_unflipped = weights.copy()

    def fake_runica(data, **_kwargs):
        return weights.copy(), sphere.copy(), np.zeros(2), np.zeros((2, 1)), np.ones((2, 1)), []

    monkeypatch.setattr("eegprep.functions.popfunc.eeg_runica.runica", fake_runica)

    out = eeg_runica(eeg, posact=True)

    data2d = out["data"].reshape(out["nbchan"], -1, order="F")
    icaact2d = out["icaact"].reshape(out["icaact"].shape[0], -1, order="F")

    # A posact flip must have occurred for this decomposition (otherwise the
    # test would not exercise the invariant-preserving branch).
    assert not np.array_equal(out["icaweights"], weights_unflipped)
    # Core EEGLAB ICA invariants must hold after sign normalization.
    np.testing.assert_allclose(out["icaweights"] @ out["icasphere"] @ data2d[out["icachansind"]], icaact2d)
    np.testing.assert_allclose(out["icawinv"], pinv(out["icaweights"] @ out["icasphere"]))
    # Every component's max-abs activation must be positive.
    ix = np.argmax(np.abs(icaact2d), axis=1)
    assert np.all(icaact2d[np.arange(icaact2d.shape[0]), ix] >= 0)


def test_finalize_ica_fields_shared_sort_and_sign_normalization():
    """Lock the K4 dedup: runica/AMICA/Picard share one finalize_ica_fields.

    The helper must sort components by descending activation variance, then
    sign-normalize while preserving the ICA factorization invariants.
    """
    from eegprep.functions.popfunc._ica_utils import finalize_ica_fields

    rng = np.random.default_rng(11)
    nbchan, pnts, trials = 4, 12, 3
    sphere = np.eye(nbchan)
    weights = rng.standard_normal((nbchan, nbchan))
    winv = pinv(weights @ sphere)
    icaact = (weights @ sphere) @ rng.standard_normal((nbchan, pnts * trials))
    icaact = icaact.reshape(nbchan, pnts, trials, order="F")
    eeg = {
        "icaweights": weights.copy(),
        "icasphere": sphere.copy(),
        "icawinv": winv.copy(),
        "icaact": icaact.copy(),
    }

    out = finalize_ica_fields(eeg, sortcomps=True, posact=True)
    icaact2d = out["icaact"].reshape(out["icaact"].shape[0], -1, order="F")

    variance_metric = np.sum(out["icawinv"] ** 2, axis=0) * np.sum(icaact2d**2, axis=1)
    assert np.all(np.diff(variance_metric) <= 1e-9)
    ix = np.argmax(np.abs(icaact2d), axis=1)
    assert np.all(icaact2d[np.arange(icaact2d.shape[0]), ix] >= 0)
    np.testing.assert_allclose(out["icawinv"], pinv(out["icaweights"] @ out["icasphere"]))


def test_pop_runica_concatenates_epoched_datasets_in_eeglab_order(monkeypatch):
    first = _epoched_eeg()
    second = _epoched_eeg(offset=100)
    captured = {}

    def fake_eeg_runica(eeg, sortcomps="off", **_kwargs):
        captured["data"] = np.asarray(eeg["data"]).copy()
        return dict(
            eeg,
            icasphere=np.eye(2),
            icaweights=np.eye(2),
            icawinv=np.eye(2),
            icaact=np.zeros((2, int(eeg["pnts"]), int(eeg["trials"]))),
            icachansind=np.array([0, 1]),
        )

    monkeypatch.setattr("eegprep.functions.popfunc.pop_runica.eeg_runica", fake_eeg_runica)

    out, command = pop_runica([first, second], concatenate="on", return_com=True)

    expected = np.concatenate([_eeglab_flattened(first["data"]), _eeglab_flattened(second["data"])], axis=1)
    np.testing.assert_array_equal(captured["data"], expected)
    assert len(out) == 2
    assert "'concatenate', 'on'" in command
