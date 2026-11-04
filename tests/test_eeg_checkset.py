"""
Test suite for eeg_checkset.py - EEG structure validation and normalization.

This module tests the eeg_checkset function that validates and normalizes
EEG data structures, ensuring required fields exist and have correct types.
"""

import unittest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.utils.testing import DebuggableTestCase


def create_minimal_eeg():
    """Create a minimal EEG structure with only required fields."""
    return {
        'data': np.random.randn(32, 1000),  # 2D continuous data
        'srate': 250.0,
        'xmin': 0.0,
        'xmax': 4.0,
    }


class TestEegChecksetTypeConversion(DebuggableTestCase):
    """Test type conversion and validation."""

    def test_xmax_converted_to_float(self):
        """Test that xmax is converted to float and EEGLAB-consistent timing."""
        eeg = create_minimal_eeg()
        eeg['xmax'] = 4  # Int

        result = eeg_checkset(eeg)

        self.assertEqual(result['xmax'], 3.996)
        self.assertIsInstance(result['xmax'], float)

    def test_xmax_half_frame_uses_matlab_rounding(self):
        """100.5 frames snaps. Python round(100.5) is 100 and would keep 0.995 s."""
        eeg = {
            'data': np.zeros((1, 100)),
            'nbchan': 1,
            'pnts': 100,
            'trials': 1,
            'srate': 100.0,
            'xmin': 0.0,
            'xmax': 0.995,
        }
        result = eeg_checkset(eeg)
        self.assertAlmostEqual(result['xmax'], 0.99)


class TestEegChecksetDataSqueezing(DebuggableTestCase):
    """Test 3D data squeezing behavior."""

    def test_3d_data_single_trial_squeezed(self):
        """Test that 3D data with single trial is squeezed to 2D."""
        eeg = create_minimal_eeg()
        eeg['data'] = np.random.randn(32, 1000, 1)  # 3D with single trial

        result = eeg_checkset(eeg)

        # Should be squeezed to 2D
        self.assertEqual(result['data'].ndim, 2)
        self.assertEqual(result['data'].shape, (32, 1000))

    def test_continuous_icaact_drops_trailing_trial_axis(self):
        data = np.arange(8, dtype=np.float64).reshape(1, 8)
        eeg = {
            'data': data,
            'nbchan': 1,
            'pnts': 8,
            'trials': 1,
            'srate': 128.0,
            'xmin': 0.0,
            'xmax': 7 / 128,
            'icaweights': np.array([[0.5]]),
            'icasphere': np.array([[1.0]]),
            'icawinv': np.array([[2.0]]),
            'icachansind': np.array([0]),
            'icaact': np.array([]),
        }

        result = eeg_checkset(eeg)

        self.assertEqual(result['icaact'].shape, (1, 8))

    def test_icaact_product_uses_the_data_dtype(self):
        weights = np.array([[0.25, -0.5, 0.125, 0.125]], dtype=np.float64)
        sphere = np.eye(4, dtype=np.float64)
        data = np.arange(1, 33, dtype=np.float32).reshape(4, 8)
        eeg = {
            'data': data,
            'nbchan': 4,
            'pnts': 8,
            'trials': 1,
            'srate': 128.0,
            'xmin': 0.0,
            'xmax': 7 / 128,
            'icaweights': weights,
            'icasphere': sphere,
            'icawinv': np.linalg.pinv(weights @ sphere),
            'icachansind': np.arange(4),
            'icaact': np.array([]),
        }

        result = eeg_checkset(eeg)

        flat = np.asfortranarray(np.asarray(result['data']).reshape(result['data'].shape[0], -1, order='F'))
        unmixing = np.asfortranarray(np.asarray(result['icaweights'])) @ np.asfortranarray(
            np.asarray(result['icasphere'])
        )
        expected = np.asfortranarray(unmixing, dtype=np.float32) @ flat
        self.assertEqual(result['icaact'].dtype, np.float32)
        np.testing.assert_array_equal(result['icaact'], expected)

    def test_single_trial_clears_epoch_and_event_epoch_field(self):
        """Like eeg_checkset.m, trials == 1 drops the epoch struct and event.epoch."""
        eeg = pop_loadset('sample_data/eeglab_data_epochs_ica.set')
        eeg['data'] = eeg['data'][:, :, :1]
        eeg['trials'] = 1
        eeg['event'] = [ev for ev in eeg['event'] if ev['epoch'] == 1]
        eeg['epoch'] = eeg['epoch'][:1]
        self.assertEqual(len(eeg['epoch']), 1)

        result = eeg_checkset(eeg)

        self.assertEqual(len(result['epoch']), 0)
        self.assertEqual(len(result['event']), 3)
        self.assertFalse(any('epoch' in ev for ev in result['event']))


class TestEegChecksetStrictMode(DebuggableTestCase):
    """Test strict mode functionality."""

    def test_strict_mode_context_manager(self):
        """Test that strict mode context manager works correctly."""
        eeg1 = create_minimal_eeg()
        eeg2 = create_minimal_eeg()

        # Add invalid ICA setup
        eeg1['icaweights'] = np.random.randn(8, 32)
        eeg1['icasphere'] = np.random.randn(16, 16)  # Wrong size

        # Should work inside strict_mode(False)
        with strict_mode(False):
            result1 = eeg_checkset(eeg1)
            self.assertIsNotNone(result1)

        # Should revert to strict mode after context
        eeg2['icaweights'] = np.random.randn(8, 32)
        eeg2['icasphere'] = np.random.randn(16, 16)  # Wrong size

        with self.assertRaises(Exception):
            eeg_checkset(eeg2)


if __name__ == '__main__':
    unittest.main()
