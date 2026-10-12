"""
Test suite for clean_flatlines.py - Flatline channel removal.

This module tests the clean_flatlines function that removes channels with
prolonged flatline periods from EEG data.
"""

import unittest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.plugins.clean_rawdata.clean_flatlines import clean_flatlines
from eegprep.utils.testing import DebuggableTestCase

from tests.fixtures import create_test_eeg as _create_test_eeg


def create_test_eeg():
    """Epoched EEG fixture sized for clean_flatlines (32 ch, 1000 pnts, 10 trials)."""
    return _create_test_eeg(n_channels=32, n_samples=1000, srate=500.0, n_trials=10)


class TestCleanFlatlinesBasic(DebuggableTestCase):
    """Basic test cases for clean_flatlines function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    def test_clean_flatlines_all_flatlines(self):
        """Test clean_flatlines when all channels have flatlines."""
        # Create data where all channels have flatlines
        eeg_all_flatlines = self.test_eeg.copy()
        eeg_all_flatlines['data'] = np.zeros_like(eeg_all_flatlines['data'])

        result = clean_flatlines(eeg_all_flatlines, max_flatline_duration=1.0)

        # Should not remove all channels (warning should be logged)
        self.assertEqual(result['nbchan'], eeg_all_flatlines['nbchan'])


class TestCleanFlatlinesValidation(DebuggableTestCase):
    """Validation test cases for clean_flatlines function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_eeg = create_test_eeg()

    # def test_clean_flatlines_pop_select_fallback(self):
    #     """Test clean_flatlines fallback when pop_select is not available."""
    #     eeg_fallback = self.test_eeg.copy()
    #     eeg_fallback['data'][5, :, :] = 1.0  # Create flatline

    #     # Mock the import to fail
    #     import sys
    #     original_import = __builtins__['__import__']

    #     def mock_import(name, *args, **kwargs):
    #         if name == 'eegprep':
    #             raise ImportError("Mock import error")
    #         return original_import(name, *args, **kwargs)

    #     __builtins__['__import__'] = mock_import

    #     try:
    #         result = clean_flatlines(eeg_fallback, max_flatline_duration=1.0)
    #         # Should still work with fallback
    #         self.assertIsInstance(result, dict)
    #     finally:
    #         __builtins__['__import__'] = original_import

    def test_clean_flatlines_fallback_composites_existing_mask(self):
        """Fallback path with a prior clean_channel_mask must composite, not crash.

        Reproduces the walrus-precedence bug: when pop_select fails and a prior
        clean_channel_mask exists, the mask update must run ``mask[mask] = ~removed``
        rather than treating the mask as a bool. Uses continuous (2D) data so the
        composite indexing exercises the real fallback branch.
        """
        eeg = self.test_eeg.copy()
        eeg['data'] = np.random.randn(32, 1000)
        eeg['trials'] = 1
        eeg['data'][5, :] = 1.0  # flatline channel 5
        eeg['etc'] = {'clean_channel_mask': np.ones(32, dtype=bool)}
        # Empty chanlocs so the unrelated chanlocs-trim branch is skipped and the
        # test isolates the clean_channel_mask compositing branch.
        eeg['chanlocs'] = []

        # Force the no-pop_select fallback with a non-ImportError so the
        # mask-compositing branch runs (this is where the bug lived).
        import eegprep

        original = eegprep.pop_select

        def failing_pop_select(*args, **kwargs):
            raise RuntimeError("simulated pop_select failure")

        eegprep.pop_select = failing_pop_select
        try:
            result = clean_flatlines(eeg, max_flatline_duration=1.0)
        finally:
            eegprep.pop_select = original

        mask = result['etc']['clean_channel_mask']
        # Original mask had 32 True entries; after compositing exactly channel 5
        # (the flatline) must be False and the rest True.
        self.assertEqual(mask.shape[0], 32)
        self.assertFalse(mask[5])
        self.assertEqual(int(np.sum(~mask)), 1)


if __name__ == '__main__':
    unittest.main()
