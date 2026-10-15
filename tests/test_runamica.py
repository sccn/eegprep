"""
Tests for runamica.py -- low-level AMICA binary wrapper.

Tests cover:
  - MATLAB-compatible input file layout
  - Temp-directory cleanup when the AMICA binary fails
  - Training progress with the actual AMICA binary (skipped if unavailable)
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.sigprocfunc.runamica import (
    _write_data_file,
    is_amica_available,
    runamica,
)


@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_write_data_file_matlab_layout(tmp_path, layout):
    data = np.array([[1.25, -2.5, 3.75], [10.5, 20.25, -30.75]], dtype='>f8')
    if layout == 'strided':
        padded = np.zeros((2, 6), dtype=data.dtype)
        padded[:, ::2] = data
        data = padded[:, ::2]
    else:
        data = data.copy(order=layout)

    path = tmp_path / 'data.fdt'
    _write_data_file(data, path)

    # MATLAB fwrite visits every channel of frame 1 before frame 2.
    expected = np.array([1.25, 10.5, -2.5, 20.25, 3.75, -30.75], dtype='<f4')
    assert path.read_bytes() == expected.tobytes()
    np.testing.assert_array_equal(np.fromfile(path, dtype='<f4').reshape(2, 3, order='F'), data)


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


class TestRunamicaTempCleanup(unittest.TestCase):
    """A failed run must not leak the auto-created temp directory."""

    def _amica_temp_dirs(self):
        root = tempfile.gettempdir()
        return {name for name in os.listdir(root) if name.startswith('amica_')}

    def test_failed_run_removes_temp_dir(self):
        data = np.random.RandomState(0).randn(4, 500)
        before = self._amica_temp_dirs()

        def _boom(binary, param_file):
            raise RuntimeError("amica binary failed")

        with mock.patch(
            'eegprep.functions.sigprocfunc.runamica._find_amica_binary',
            return_value='/dummy/amica',
        ):
            with mock.patch('eegprep.functions.sigprocfunc.runamica._run_amica', side_effect=_boom):
                with self.assertRaises(RuntimeError):
                    runamica(data, num_models=1, max_iter=10, max_threads=1)

        # No new amica_* temp directory should survive the failure.
        self.assertEqual(self._amica_temp_dirs() - before, set())


@unittest.skipUnless(is_amica_available(), "AMICA binary not functional on this platform")
class TestRunamicaIntegration(unittest.TestCase):
    """Integration test: run AMICA binary on small synthetic data."""

    def test_runamica_ll_decreasing(self):
        """Verify that log-likelihood generally increases over iterations."""
        n_channels = 4
        n_samples = 3000
        data, _, _ = _make_synthetic_sources(n_channels, n_samples, seed=7)
        data += np.arange(n_channels)[:, None] * 10.0

        _, _, mods = runamica(
            data,
            num_models=1,
            max_iter=200,
            max_threads=2,
        )

        # The binary must fit the intended channels, not a scrambled reshape.
        expected_mean = data.astype('<f4').mean(axis=1, dtype=np.float64)
        np.testing.assert_allclose(mods['mean'], expected_mean, rtol=1e-10, atol=1e-10)

        LL = mods['LL']
        if len(LL) > 10:
            # LL should generally increase (AMICA maximizes LL).
            # Compare first 10% mean to last 10% mean.
            n = len(LL)
            early = np.mean(LL[: max(1, n // 10)])
            late = np.mean(LL[-(max(1, n // 10)) :])
            self.assertGreater(late, early, "Log-likelihood should increase over training")


if __name__ == '__main__':
    unittest.main()
