"""Regression tests for the ICLabel PSD feature helper ``eeg_rpsd``."""

import unittest
import numpy as np

from eegprep.plugins.ICLabel.eeg_rpsd import eeg_rpsd

from tests.fixtures import create_test_eeg


class TestEegRpsdGlobalRng(unittest.TestCase):
    """Regression tests that eeg_rpsd never mutates the global numpy RNG."""

    def setUp(self):
        np.random.seed(42)
        n_channels = 16
        n_components = 4
        n_samples = 500
        srate = 250.0
        eeg = create_test_eeg(n_channels=n_channels, n_samples=n_samples, srate=srate, n_trials=1)
        eeg['icawinv'] = np.random.randn(n_channels, n_components) * 0.5
        eeg['icaweights'] = np.linalg.pinv(eeg['icawinv'])
        eeg['icasphere'] = np.eye(n_channels)
        eeg['icaact'] = np.random.randn(n_components, n_samples, 1) * 0.5
        eeg['icachansind'] = np.arange(n_channels)
        self.eeg = eeg

    def test_does_not_mutate_global_rng(self):
        """eeg_rpsd must use a local RNG, leaving np.random's global state intact."""
        np.random.seed(123)
        before = np.random.get_state()
        eeg_rpsd(self.eeg)
        after = np.random.get_state()

        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2:], after[2:])

    def test_output_is_deterministic(self):
        """eeg_rpsd must return identical output regardless of global RNG state."""
        np.random.seed(1)
        psd_a = eeg_rpsd(self.eeg)
        np.random.seed(999)
        np.random.rand(37)  # perturb the global RNG between calls
        psd_b = eeg_rpsd(self.eeg)

        self.assertTrue(np.array_equal(psd_a, psd_b))


if __name__ == '__main__':
    unittest.main()
