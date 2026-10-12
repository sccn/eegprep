"""Tests for clean_asr module.

This module tests the Artifact Subspace Reconstruction (ASR) functionality
including parameter validation, calibration data selection, and various
processing switches.
"""

import unittest
import numpy as np

from eegprep.plugins.clean_rawdata.clean_asr import clean_asr


class TestCleanASRParameters(unittest.TestCase):
    """Test clean_asr parameter handling and validation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_channels = 4  # Smaller for faster testing
        self.n_samples = 500  # Shorter for faster testing
        self.srate = 250.0

        self.test_eeg = {
            'data': np.random.randn(self.n_channels, self.n_samples) * 0.5,
            'srate': self.srate,
            'nbchan': self.n_channels,
            'pnts': self.n_samples,
            'trials': 1,
            'xmin': 0.0,
            'xmax': (self.n_samples - 1) / self.srate,
            'times': np.arange(self.n_samples) / self.srate,
            'event': [],
        }

    def test_clean_asr_rejects_full_riemannian_processing_request(self):
        """Test that unsupported full Riemannian ASR processing fails clearly."""
        with self.assertRaisesRegex(ValueError, "full Riemannian ASR processing is not ported"):
            clean_asr(self.test_eeg, useriemannian=True)


if __name__ == '__main__':
    unittest.main()
