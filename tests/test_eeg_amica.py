"""
Tests for eeg_amica.py -- high-level AMICA wrapper for EEG structures.

Tests cover:
  - Reconstruction accuracy: icawinv @ (icaweights @ icasphere) @ data ~ data
  - EEGLAB epoch flattening, caller immutability, and posact invariants
  - Model switching via load_amica_model()
"""

import unittest
from unittest import mock

import numpy as np

from eegprep.functions.miscfunc.pinv import pinv
from eegprep.functions.popfunc import eeg_amica as eeg_amica_module
from eegprep.functions.popfunc.eeg_amica import eeg_amica, load_amica_model
from eegprep.functions.sigprocfunc.runamica import is_amica_available


def _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1, srate=250.0, seed=42):
    """Create a minimal EEG dict with mixed sinusoidal sources.

    The data is constructed as mixing @ sources so that ICA can recover
    meaningful independent components.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_samples)

    # Create independent sources (sinusoids at different frequencies)
    sources = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        freq = 1.0 + i * 3.0
        sources[i, :] = np.sin(freq * t + rng.uniform(0, 2 * np.pi))

    # Random mixing matrix
    mixing = rng.randn(n_channels, n_channels)
    data_2d = mixing @ sources

    # Reshape for trials
    if n_trials > 1:
        # Tile data to create trials
        total_samples = n_samples * n_trials
        t_long = np.linspace(0, 2 * np.pi * n_trials, total_samples)
        sources_long = np.zeros((n_channels, total_samples))
        for i in range(n_channels):
            freq = 1.0 + i * 3.0
            sources_long[i, :] = np.sin(freq * t_long + rng.uniform(0, 2 * np.pi))
        data_long = mixing @ sources_long
        data = data_long.reshape(n_channels, n_samples, n_trials)
    else:
        data = data_2d

    # Build EEG dict
    EEG = {
        'data': data.astype(np.float64),
        'srate': srate,
        'nbchan': n_channels,
        'pnts': n_samples,
        'trials': n_trials,
        'xmin': 0.0,
        'xmax': (n_samples - 1) / srate,
        'times': np.arange(n_samples) / srate,
        'chanlocs': [
            {
                'labels': f'Ch{i + 1}',
                'type': 'EEG',
                'X': rng.uniform(-1, 1),
                'Y': rng.uniform(-1, 1),
                'Z': rng.uniform(-1, 1),
            }
            for i in range(n_channels)
        ],
        'event': [],
        'epoch': [],
        'icaact': None,
        'icawinv': None,
        'icasphere': None,
        'icaweights': None,
        'icachansind': None,
        'etc': {},
        'setname': 'test_amica',
        'history': '',
    }
    return EEG


@unittest.skipUnless(is_amica_available(), "AMICA binary not functional on this platform")
class TestEegAmicaReconstruction(unittest.TestCase):
    """Test data reconstruction from ICA decomposition."""

    def test_eeg_amica_reconstruction(self):
        """Verify icawinv @ (icaweights @ icasphere) @ data ~ data.

        AMICA normalizes column norms of A and scales W accordingly, so
        the reconstruction should still be accurate but we use a tolerance
        that accounts for the normalization.
        """
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, max_iter=150, max_threads=2)

        data = result['data'].reshape(result['nbchan'], -1).astype(np.float64)
        W = result['icaweights']
        S = result['icasphere']
        A = result['icawinv']

        # A @ W @ S @ data should reconstruct data
        # W@S extracts components, A maps back to channels
        reconstructed = A @ (W @ S) @ data
        np.testing.assert_allclose(
            reconstructed, data, atol=0.01, err_msg="ICA reconstruction should approximate original data"
        )


def _eeglab_flattened(data):
    """Concatenate trials in EEGLAB column-major epoch order."""
    return np.concatenate([data[:, :, trial] for trial in range(data.shape[2])], axis=1)


def _epoched_eeg():
    data = np.zeros((2, 3, 2), dtype=np.float64)
    data[0, :, 0] = [1.0, -2.0, 3.0]
    data[0, :, 1] = [4.0, -5.0, 6.0]
    data[1, :, 0] = [-7.0, 8.0, -9.0]
    data[1, :, 1] = [10.0, -11.0, 12.0]
    return {
        'data': data,
        'nbchan': 2,
        'pnts': 3,
        'trials': 2,
        'srate': 100,
        'chanlocs': [],
        'icaweights': np.eye(2),
        'icasphere': np.eye(2),
        'etc': {},
    }


def _continuous_eeg():
    data = np.array([[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]])
    return {
        'data': data,
        'nbchan': 2,
        'pnts': 4,
        'trials': 1,
        'srate': 100,
        'chanlocs': [],
        'icaweights': np.eye(2),
        'icasphere': np.eye(2),
        'etc': {},
    }


# A weights/sphere pair with a non-identity sphere so that the posact sign-flip
# step exercises the icaweights @ icasphere @ data == icaact invariant.
_AMICA_WEIGHTS = np.array([[2.0, 1.0], [-1.0, 3.0]])
_AMICA_SPHERE = np.array([[1.5, 0.5], [0.0, 2.0]])


def _fake_runamica_factory(captured):
    winv = pinv(_AMICA_WEIGHTS @ _AMICA_SPHERE)

    def fake_runamica(data, **_kwargs):
        captured['data'] = np.asarray(data).copy()
        mods = {
            'num_pcs': 2,
            'num_models': 1,
            'W': _AMICA_WEIGHTS.copy()[:, :, None],
            'S': _AMICA_SPHERE.copy(),
            'A': winv.copy()[:, :, None],
        }
        return _AMICA_WEIGHTS.copy(), _AMICA_SPHERE.copy(), mods

    return fake_runamica


class TestEegAmicaRegressions(unittest.TestCase):
    """Regression tests that do not require the AMICA binary (runamica mocked)."""

    def test_eeg_amica_flattens_epoched_data_like_eeglab(self):
        eeg = _epoched_eeg()
        captured = {}
        with mock.patch.object(eeg_amica_module, 'runamica', _fake_runamica_factory(captured)):
            out = eeg_amica(eeg)

        # AMICA must receive data flattened in EEGLAB column-major epoch order.
        np.testing.assert_array_equal(captured['data'], _eeglab_flattened(eeg['data']))

        # icaact must reshape back to channel x point x trial via the same order,
        # so it equals icaweights @ icasphere @ data reshaped per-trial.
        expected_2d = (out['icaweights'] @ out['icasphere']) @ _eeglab_flattened(eeg['data'])
        expected_3d = expected_2d.reshape(expected_2d.shape[0], out['pnts'], out['trials'], order='F')
        np.testing.assert_allclose(out['icaact'], expected_3d)

    def test_eeg_amica_does_not_mutate_caller(self):
        eeg = _continuous_eeg()
        data_before = eeg['data'].copy()
        weights_before = eeg['icaweights'].copy()
        captured = {}
        with mock.patch.object(eeg_amica_module, 'runamica', _fake_runamica_factory(captured)):
            eeg_amica(eeg, posact=True)

        self.assertTrue(np.array_equal(eeg['data'], data_before))
        self.assertTrue(np.array_equal(eeg['icaweights'], weights_before))
        self.assertEqual(eeg['etc'], {})

    def test_eeg_amica_posact_preserves_ica_invariants(self):
        eeg = _continuous_eeg()
        captured = {}
        with mock.patch.object(eeg_amica_module, 'runamica', _fake_runamica_factory(captured)):
            out = eeg_amica(eeg, posact=True)

        data2d = out['data'].reshape(out['nbchan'], -1, order='F')
        icaact2d = out['icaact'].reshape(out['icaact'].shape[0], -1, order='F')

        # A posact flip must have occurred (non-identity sphere makes this the
        # case that previously folded the sphere into icaweights).
        self.assertFalse(np.array_equal(out['icaweights'], _AMICA_WEIGHTS))
        # Core EEGLAB ICA invariants must hold after sign normalization.
        np.testing.assert_allclose(out['icaweights'] @ out['icasphere'] @ data2d[out['icachansind']], icaact2d)
        np.testing.assert_allclose(out['icawinv'], pinv(out['icaweights'] @ out['icasphere']))
        # icasphere must be untouched by the sign-flip step.
        np.testing.assert_array_equal(out['icasphere'], _AMICA_SPHERE)
        ix = np.argmax(np.abs(icaact2d), axis=1)
        self.assertTrue(np.all(icaact2d[np.arange(icaact2d.shape[0]), ix] >= 0))

    def test_load_amica_model_does_not_mutate_caller(self):
        eeg = _continuous_eeg()
        captured = {}
        with mock.patch.object(eeg_amica_module, 'runamica', _fake_runamica_factory(captured)):
            decomposed = eeg_amica(eeg)

        mods = decomposed['etc']['amica']
        icaact_before = decomposed['icaact'].copy()
        reloaded = load_amica_model(decomposed, mods, model_num=0)

        # Reloading model 0 must reproduce the same activations and leave the
        # source EEG structure untouched.
        np.testing.assert_allclose(reloaded['icaact'], icaact_before)
        self.assertTrue(np.array_equal(decomposed['icaact'], icaact_before))


if __name__ == '__main__':
    unittest.main()
