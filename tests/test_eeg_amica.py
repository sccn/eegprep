"""
Tests for eeg_amica.py -- high-level AMICA wrapper for EEG structures.

Tests cover:
  - Basic ICA field population from eeg_amica()
  - Correct shapes for 2D (continuous) and 3D (epoched) data
  - Reconstruction accuracy: icawinv @ (icaweights @ icasphere) @ data ~ data
  - Storage of full AMICA output in EEG['etc']['amica']
  - Component sorting by variance (sortcomps)
  - Positive activation sign normalization (posact)
  - Model switching via load_amica_model()
  - Error handling for invalid model index
"""

import unittest

import numpy as np

from eegprep.eeg_amica import eeg_amica, load_amica_model
from eegprep.runamica import _find_amica_binary


def _amica_binary_available():
    """Return True if the AMICA binary is available and executable."""
    try:
        _find_amica_binary()
        return True
    except FileNotFoundError:
        return False


def _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1, srate=250.0,
                   seed=42):
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
            {'labels': f'Ch{i+1}', 'type': 'EEG',
             'X': rng.uniform(-1, 1), 'Y': rng.uniform(-1, 1),
             'Z': rng.uniform(-1, 1)}
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


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
class TestEegAmicaBasic(unittest.TestCase):
    """Basic eeg_amica tests with continuous data."""

    def test_eeg_amica_basic(self):
        """Run eeg_amica on 4-channel continuous data, verify ICA fields."""
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, max_iter=150, max_threads=2)

        n_chans = result['nbchan']
        n_pnts = result['pnts']
        n_trials = result['trials']

        # All ICA fields must exist and be ndarrays
        for key in ('icaweights', 'icasphere', 'icawinv', 'icaact', 'icachansind'):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], np.ndarray, f"{key} not ndarray")

        ncomps = result['icaweights'].shape[0]

        # icaweights: (ncomps, ncomps) since num_pcs defaults to nbchan
        self.assertEqual(result['icaweights'].shape, (ncomps, ncomps))

        # icasphere: (num_pcs, nbchan)
        self.assertEqual(result['icasphere'].shape[1], n_chans)

        # icawinv: (nbchan, ncomps)
        self.assertEqual(result['icawinv'].shape, (n_chans, ncomps))

        # icaact: (ncomps, pnts, trials)
        self.assertEqual(result['icaact'].shape, (ncomps, n_pnts, n_trials))

        # icachansind: (nbchan,)
        self.assertEqual(len(result['icachansind']), n_chans)

        # No NaN or Inf in outputs
        self.assertFalse(np.any(np.isnan(result['icaweights'])))
        self.assertFalse(np.any(np.isnan(result['icasphere'])))
        self.assertFalse(np.any(np.isnan(result['icawinv'])))
        self.assertFalse(np.any(np.isnan(result['icaact'])))
        self.assertTrue(np.all(np.isfinite(result['icaweights'])))
        self.assertTrue(np.all(np.isfinite(result['icaact'])))


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
class TestEegAmicaOutputShapes(unittest.TestCase):
    """Test output shapes with 3D (epoched) data."""

    def test_eeg_amica_output_shapes(self):
        """Run eeg_amica on epoched data (4 channels, 2 trials)."""
        EEG = _make_test_eeg(n_channels=4, n_samples=1000, n_trials=2)
        result = eeg_amica(EEG, max_iter=100, max_threads=2)

        ncomps = result['icaweights'].shape[0]
        n_pnts = result['pnts']
        n_trials = result['trials']

        self.assertEqual(n_trials, 2)
        self.assertEqual(result['icaact'].shape, (ncomps, n_pnts, n_trials))
        self.assertEqual(result['icawinv'].shape[0], result['nbchan'])
        self.assertEqual(result['icawinv'].shape[1], ncomps)


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
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
            reconstructed, data, atol=0.01,
            err_msg="ICA reconstruction should approximate original data"
        )


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
class TestEegAmicaEtcStored(unittest.TestCase):
    """Test that full AMICA output is stored in EEG['etc']['amica']."""

    def test_eeg_amica_etc_stored(self):
        """Verify EEG['etc']['amica'] is a dict with expected keys."""
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, max_iter=100, max_threads=2)

        self.assertIn('etc', result)
        self.assertIn('amica', result['etc'])
        mods = result['etc']['amica']
        self.assertIsInstance(mods, dict)

        # Key outputs from _load_amica_output
        for key in ('W', 'S', 'A', 'mean', 'gm', 'alpha', 'mu', 'sbeta',
                     'rho', 'LL', 'svar', 'origord', 'num_pcs', 'num_models'):
            self.assertIn(key, mods, f"Missing key in mods: {key}")

        # num_models and num_pcs should be correct
        self.assertEqual(mods['num_models'], 1)
        self.assertEqual(mods['num_pcs'], 4)


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
class TestEegAmicaSortcomps(unittest.TestCase):
    """Test component sorting by variance."""

    def test_eeg_amica_sortcomps(self):
        """Run with sortcomps=True, verify variance metric is descending."""
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, sortcomps=True, max_iter=150, max_threads=2)

        # Compute variance metric: sum(icawinv^2, axis=0) * sum(icaact^2, axis=1)
        icaact_2d = result['icaact'].reshape(result['icaact'].shape[0], -1)
        variance_metric = (
            np.sum(result['icawinv'] ** 2, axis=0)
            * np.sum(icaact_2d ** 2, axis=1)
        )

        # Should be in descending order
        for i in range(len(variance_metric) - 1):
            self.assertGreaterEqual(
                variance_metric[i], variance_metric[i + 1],
                f"Variance metric not descending at index {i}: "
                f"{variance_metric[i]} < {variance_metric[i + 1]}"
            )


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
class TestEegAmicaPosact(unittest.TestCase):
    """Test positive activation sign normalization."""

    def test_eeg_amica_posact(self):
        """Run with posact=True, verify max abs activation is positive."""
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, posact=True, max_iter=150, max_threads=2)

        icaact_2d = result['icaact'].reshape(result['icaact'].shape[0], -1)
        ncomps = icaact_2d.shape[0]

        for r in range(ncomps):
            ix = np.argmax(np.abs(icaact_2d[r, :]))
            self.assertGreaterEqual(
                icaact_2d[r, ix], 0,
                f"Component {r}: max abs activation value should be positive"
            )


@unittest.skipUnless(_amica_binary_available(),
                     "AMICA binary not available or not executable")
class TestLoadAmicaModel(unittest.TestCase):
    """Test load_amica_model() for model switching."""

    def test_load_amica_model(self):
        """Run AMICA, then load model 0 via load_amica_model, verify fields."""
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, num_models=1, max_iter=100, max_threads=2)

        mods = result['etc']['amica']

        # Create a copy and reload model 0
        EEG2 = dict(result)
        EEG2['data'] = result['data'].copy()
        EEG2 = load_amica_model(EEG2, mods, model_num=0)

        # Fields should match the original
        np.testing.assert_array_equal(
            EEG2['icaweights'], result['icaweights'],
            err_msg="icaweights should match after reloading model 0"
        )
        np.testing.assert_array_equal(
            EEG2['icasphere'], result['icasphere'],
            err_msg="icasphere should match after reloading model 0"
        )
        np.testing.assert_array_equal(
            EEG2['icawinv'], result['icawinv'],
            err_msg="icawinv should match after reloading model 0"
        )
        np.testing.assert_allclose(
            EEG2['icaact'], result['icaact'], atol=1e-10,
            err_msg="icaact should match after reloading model 0"
        )

    def test_load_amica_model_invalid(self):
        """Verify ValueError on out-of-range model_num."""
        EEG = _make_test_eeg(n_channels=4, n_samples=2000, n_trials=1)
        result = eeg_amica(EEG, num_models=1, max_iter=100, max_threads=2)
        mods = result['etc']['amica']

        with self.assertRaises(ValueError):
            load_amica_model(result, mods, model_num=5)

        with self.assertRaises(ValueError):
            load_amica_model(result, mods, model_num=-1)


if __name__ == '__main__':
    unittest.main()
