"""
Tests for runamica.py -- low-level AMICA binary wrapper.

Tests cover:
  - Parameter file writing and format
  - Data file round-trip (float32)
  - Loading of AMICA output directory (synthetic binary fixtures)
  - Binary discovery logic
  - Integration test with the actual AMICA binary (skipped if unavailable)
"""

import os
import platform
import shutil
import tempfile
import unittest

import numpy as np

from eegprep.functions.sigprocfunc.runamica import (
    _find_amica_binary,
    _load_amica_output,
    _write_data_file,
    _write_param_file,
    is_amica_available,
    runamica,
)


def _make_synthetic_sources(n_channels, n_samples, seed=42):
    """Create mixed sinusoidal sources for ICA testing."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_samples)
    sources = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        freq = 1.0 + i * 2.0
        sources[i, :] = np.sin(freq * t + rng.uniform(0, 2 * np.pi))
    mixing = rng.randn(n_channels, n_channels)
    data = mixing @ sources
    return data, mixing, sources


class TestWriteParamFile(unittest.TestCase):
    """Test _write_param_file() output format and content."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='amica_test_param_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_param_file(self):
        """Verify format: key value lines, required params present."""
        params = {
            'files': '/tmp/data.fdt',
            'outdir': self.tmpdir + os.sep,
            'data_dim': 4,
            'field_dim': 2000,
            'num_models': 1,
            'max_iter': 100,
            'num_mix_comps': 3,
            'pcakeep': 4,
            'max_threads': 2,
        }
        path = _write_param_file(self.tmpdir, params)
        self.assertTrue(os.path.isfile(path))

        with open(path) as f:
            lines = f.readlines()

        # Build a dict of key -> value strings
        kv = {}
        for line in lines:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                kv[parts[0]] = parts[1]

        # Required keys must be present
        for key in ('files', 'outdir', 'data_dim', 'field_dim',
                     'num_models', 'max_iter', 'num_mix_comps', 'pcakeep'):
            self.assertIn(key, kv, f"Missing required param: {key}")

        # Integer params should parse as int
        self.assertEqual(int(kv['data_dim']), 4)
        self.assertEqual(int(kv['field_dim']), 2000)
        self.assertEqual(int(kv['max_iter']), 100)

        # String params should be as-is
        self.assertEqual(kv['files'], '/tmp/data.fdt')

    def test_defaults_are_written(self):
        """Verify that default params are included even if not explicitly set."""
        params = {
            'files': '/tmp/data.fdt',
            'outdir': self.tmpdir + os.sep,
            'data_dim': 4,
            'field_dim': 2000,
        }
        path = _write_param_file(self.tmpdir, params)

        with open(path) as f:
            content = f.read()

        # Some defaults that should always appear
        self.assertIn('lrate', content)
        self.assertIn('block_size', content)
        self.assertIn('do_newton', content)


class TestWriteDataFile(unittest.TestCase):
    """Test _write_data_file() round-trip accuracy."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='amica_test_data_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_data_file(self):
        """Write ndarray, read back as float32, verify round-trip."""
        rng = np.random.RandomState(42)
        data = rng.randn(4, 500).astype(np.float64)

        path = os.path.join(self.tmpdir, 'data.fdt')
        _write_data_file(data, path)

        # Read back
        raw = np.fromfile(path, dtype='<f4')
        recovered = raw.reshape(data.shape)

        # Should match to float32 precision
        np.testing.assert_allclose(recovered, data.astype(np.float32), atol=0)


class TestLoadAmicaOutput(unittest.TestCase):
    """Test _load_amica_output() with synthetic binary fixture files.

    This validates the loadmodout15.m port WITHOUT requiring the AMICA binary.
    We create known arrays, write them as raw float64, and verify that
    _load_amica_output() returns the expected shapes and invariants.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='amica_test_load_')
        self.num_models = 1
        self.nw = 4          # num_pcs
        self.data_dim = 4    # number of channels
        self.num_mix = 3     # num_mix_comps
        self.max_iter = 50
        self.field_dim = 2000

        # Create known fixture arrays and write as float64
        rng = np.random.RandomState(123)

        # gm: model probabilities
        gm = np.array([1.0])
        gm.tofile(os.path.join(self.tmpdir, 'gm'))

        # W: unmixing weights (nw x nw x num_models), Fortran order
        self.W = rng.randn(self.nw, self.nw).astype(np.float64)
        W_3d = self.W.reshape(self.nw, self.nw, 1, order='F')
        W_3d.flatten(order='F').tofile(os.path.join(self.tmpdir, 'W'))

        # mean
        self.mn = rng.randn(self.data_dim).astype(np.float64)
        self.mn.tofile(os.path.join(self.tmpdir, 'mean'))

        # S: sphering matrix (data_dim x data_dim)
        self.S = np.eye(self.data_dim, dtype=np.float64)
        self.S.flatten(order='F').tofile(os.path.join(self.tmpdir, 'S'))

        # c: model centers (nw x num_models)
        c = np.zeros((self.nw, 1), dtype=np.float64)
        c.flatten(order='F').tofile(os.path.join(self.tmpdir, 'c'))

        # alpha: (num_mix x nw*num_models), Fortran order
        alpha = np.ones((self.num_mix, self.nw), dtype=np.float64) / self.num_mix
        alpha.flatten(order='F').tofile(os.path.join(self.tmpdir, 'alpha'))

        # mu: (num_mix x nw*num_models)
        mu = np.zeros((self.num_mix, self.nw), dtype=np.float64)
        mu.flatten(order='F').tofile(os.path.join(self.tmpdir, 'mu'))

        # sbeta: (num_mix x nw*num_models)
        sbeta = np.ones((self.num_mix, self.nw), dtype=np.float64)
        sbeta.flatten(order='F').tofile(os.path.join(self.tmpdir, 'sbeta'))

        # rho: (num_mix x nw*num_models)
        rho = np.full((self.num_mix, self.nw), 2.0, dtype=np.float64)
        rho.flatten(order='F').tofile(os.path.join(self.tmpdir, 'rho'))

        # LL: log-likelihood per iteration
        LL = rng.randn(self.max_iter).astype(np.float64)
        LL.tofile(os.path.join(self.tmpdir, 'LL'))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_amica_output(self):
        """Load synthetic AMICA output and verify shapes and values."""
        mods = _load_amica_output(
            self.tmpdir,
            num_models=self.num_models,
            num_pcs=self.nw,
            data_dim=self.data_dim,
            num_mix_comps=self.num_mix,
            max_iter=self.max_iter,
            field_dim=self.field_dim,
        )

        # W shape: (nw, nw, num_models)
        self.assertEqual(mods['W'].shape, (self.nw, self.nw, self.num_models))

        # S shape: (data_dim, data_dim) since S is identity and data_dim == nw
        self.assertEqual(mods['S'].shape, (self.data_dim, self.data_dim))

        # A shape: (data_dim, nw, num_models) -- pseudo-inverse of W@S
        self.assertEqual(mods['A'].shape, (self.data_dim, self.nw, self.num_models))

        # mean shape
        self.assertEqual(mods['mean'].shape, (self.data_dim,))

        # gm / mod_prob
        self.assertEqual(len(mods['gm']), self.num_models)
        np.testing.assert_array_equal(mods['gm'], mods['mod_prob'])

        # Mixture params shapes: (num_mix, nw, num_models)
        for key in ('alpha', 'mu', 'sbeta', 'rho'):
            self.assertEqual(mods[key].shape[1], self.nw)
            self.assertEqual(mods[key].shape[2], self.num_models)

        # LL
        self.assertEqual(mods['LL'].shape, (self.max_iter,))

        # svar and origord shapes: (nw, num_models)
        self.assertEqual(mods['svar'].shape, (self.nw, self.num_models))
        self.assertEqual(mods['origord'].shape, (self.nw, self.num_models))

        # num_pcs stored correctly
        self.assertEqual(mods['num_pcs'], self.nw)

        # No NaN in key matrices
        self.assertFalse(np.any(np.isnan(mods['W'])))
        self.assertFalse(np.any(np.isnan(mods['A'])))
        self.assertFalse(np.any(np.isnan(mods['S'])))


class TestFindAmicaBinary(unittest.TestCase):
    """Test binary discovery logic."""

    def test_find_amica_binary(self):
        """Verify that _find_amica_binary() finds the vendored binary."""
        if platform.system() not in ('Darwin', 'Linux', 'Windows'):
            self.skipTest("Unsupported platform for vendored binary")
        try:
            path = _find_amica_binary()
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.access(path, os.X_OK))
        except FileNotFoundError:
            self.skipTest("AMICA binary not available on this system")

    def test_find_amica_binary_not_found(self):
        """Verify FileNotFoundError for nonexistent explicit path."""
        with self.assertRaises(FileNotFoundError):
            _find_amica_binary('/nonexistent/path/amica15')


@unittest.skipUnless(is_amica_available(),
                     "AMICA binary not functional on this platform")
class TestRunamicaIntegration(unittest.TestCase):
    """Integration test: run AMICA binary on small synthetic data."""

    def test_runamica_integration(self):
        """Run runamica() on 4-channel mixed sinusoids, verify output."""
        n_channels = 4
        n_samples = 2000
        data, mixing, sources = _make_synthetic_sources(n_channels, n_samples)

        weights, sphere, mods = runamica(
            data,
            num_models=1,
            max_iter=150,
            num_mix_comps=3,
            max_threads=2,
        )

        # Verify output shapes
        num_pcs = mods['num_pcs']
        self.assertEqual(weights.shape, (num_pcs, num_pcs))
        self.assertEqual(sphere.shape, (num_pcs, n_channels))
        self.assertEqual(mods['W'].shape, (num_pcs, num_pcs, 1))
        self.assertEqual(mods['A'].shape, (n_channels, num_pcs, 1))
        self.assertEqual(mods['S'].shape[:1], (num_pcs,))

        # Matrices should be finite
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(np.isfinite(sphere)))

        # Reconstruction test: pinv(W@S) @ W @ S @ data ~ data
        WS = weights @ sphere
        from eegprep.functions.miscfunc.pinv import pinv
        reconstructed = pinv(WS) @ WS @ data
        np.testing.assert_allclose(
            reconstructed, data, atol=1e-6,
            err_msg="AMICA reconstruction should reproduce input data"
        )

    def test_runamica_custom_outdir(self):
        """Run runamica with explicit outdir, verify cleanup=False retains files."""
        n_channels = 4
        n_samples = 2000
        data, _, _ = _make_synthetic_sources(n_channels, n_samples, seed=99)

        outdir = tempfile.mkdtemp(prefix='amica_test_outdir_')
        try:
            weights, sphere, mods = runamica(
                data,
                num_models=1,
                max_iter=100,
                outdir=outdir,
                cleanup=False,
                max_threads=2,
            )
            # Output files should still exist
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'W')))
            self.assertTrue(os.path.isfile(os.path.join(outdir, 'S')))
        finally:
            shutil.rmtree(outdir, ignore_errors=True)

    def test_runamica_ll_decreasing(self):
        """Verify that log-likelihood generally increases over iterations."""
        n_channels = 4
        n_samples = 3000
        data, _, _ = _make_synthetic_sources(n_channels, n_samples, seed=7)

        _, _, mods = runamica(
            data,
            num_models=1,
            max_iter=200,
            max_threads=2,
        )

        LL = mods['LL']
        if len(LL) > 10:
            # LL should generally increase (AMICA maximizes LL).
            # Compare first 10% mean to last 10% mean.
            n = len(LL)
            early = np.mean(LL[:max(1, n // 10)])
            late = np.mean(LL[-(max(1, n // 10)):])
            self.assertGreater(late, early,
                               "Log-likelihood should increase over training")


if __name__ == '__main__':
    unittest.main()
