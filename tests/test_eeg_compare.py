"""
Test suite for eeg_compare.py with MATLAB parity validation.

This module tests the eeg_compare function which compares two EEG datasets
and reports differences in structure and data.
"""

import logging
import os
import unittest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.utils.testing import DebuggableTestCase

EEG_COMPARE_LOGGER = 'eegprep.functions.popfunc.eeg_compare'


def run_compare(*args, **kwargs):
    """Call eeg_compare while capturing its log output.

    eeg_compare emits informational lines (section headers and "OK" results) at INFO and
    differences at WARNING. Return the result plus the WARNING-level text as ``stderr`` and the
    full text as ``stdout`` so callers can assert on either, mirroring the previous stream split.
    """
    logger = logging.getLogger(EEG_COMPARE_LOGGER)
    with _LogCapture(logger) as capture:
        result = eeg_compare(*args, **kwargs)
    return result, capture.text(logging.INFO), capture.text(logging.WARNING)


class _LogCapture(logging.Handler):
    def __init__(self, logger):
        super().__init__(level=logging.DEBUG)
        self._logger = logger
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self._prev_level = self._logger.level
        self._prev_propagate = self._logger.propagate
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self)
        self._logger.setLevel(self._prev_level)
        self._logger.propagate = self._prev_propagate
        return False

    def text(self, min_level):
        return '\n'.join(self.format(r) for r in self.records if r.levelno >= min_level)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEegCompare(DebuggableTestCase):
    """Test cases for eeg_compare function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create basic test EEG structures
        self.basic_eeg1 = self.create_test_eeg()
        self.basic_eeg2 = self.create_test_eeg()

        # Set up MATLAB compatibility for parity tests
        try:
            self.eeglab = get_eeglab()
            self.matlab_available = True
        except Exception:
            self.matlab_available = False

    def create_test_eeg(self, nbchan=32, pnts=1000, trials=1):
        """Create a test EEG structure."""
        srate = 250.0
        xmin = -0.2
        xmax = (pnts - 1) / srate + xmin

        return {
            'setname': 'test_dataset',
            'filename': 'test.set',
            'filepath': '/tmp/',
            'subject': 'S01',
            'condition': 'test',
            'session': 1,
            'run': 1,
            'task': 'test',
            'nbchan': nbchan,
            'trials': trials,
            'pnts': pnts,
            'srate': srate,
            'xmin': xmin,
            'xmax': xmax,
            'times': np.linspace(xmin * 1000, xmax * 1000, pnts),
            'data': np.random.randn(nbchan, pnts, trials).astype(np.float32),
            'icaact': np.array([]),
            'icawinv': np.array([]),
            'icasphere': np.array([]),
            'icaweights': np.array([]),
            'icachansind': np.array([]),
            'chanlocs': [
                {
                    'labels': f'Ch{i + 1}',
                    'X': np.cos(2 * np.pi * i / nbchan),
                    'Y': np.sin(2 * np.pi * i / nbchan),
                    'Z': 0.0,
                    'theta': 2 * np.pi * i / nbchan,
                    'radius': 0.5,
                    'sph_theta': 0.0,
                    'sph_phi': 0.0,
                    'sph_radius': 1.0,
                }
                for i in range(nbchan)
            ],
            'urchanlocs': np.array([]),
            'chaninfo': {'plotrad': [], 'shrink': [], 'nosedir': '+X'},
            'ref': 'common',
            'event': [
                {'type': 'stimulus', 'latency': 250, 'duration': 0, 'epoch': 1},
                {'type': 'response', 'latency': 500, 'duration': 0, 'epoch': 1},
                {'type': 'boundary', 'latency': 750, 'duration': 0, 'epoch': 1},
            ],
            'urevent': np.array([]),
            'eventdescription': ['stimulus', 'response', 'boundary'],
            'epoch': np.array([]),
            'epochdescription': np.array([]),
            'reject': {},
            'stats': {},
            'specdata': np.array([]),
            'specicaact': np.array([]),
            'splinefile': '',
            'icasplinefile': '',
            'dipfit': {},
            'history': 'EEG = create_test_eeg();',
            'saved': 'no',
            'etc': {},
            'datfile': '',
            'comments': 'Test dataset for eeg_compare',
        }

    def test_channel_coordinate_differences(self):
        """Test detection of channel coordinate differences."""
        eeg2 = self.create_test_eeg()
        # Modify some channel coordinates
        eeg2['chanlocs'][0]['X'] = 999.0
        eeg2['chanlocs'][1]['Y'] = 999.0
        eeg2['chanlocs'][2]['Z'] = 999.0

        result, _stdout_output, stderr_output = run_compare(self.basic_eeg1, eeg2)

        self.assertTrue(result)

        self.assertIn('channel coordinates differ', stderr_output)

    def test_event_latency_differences(self):
        """Test detection of event latency differences."""
        eeg2 = self.create_test_eeg()
        eeg2['event'][0]['latency'] = 300  # Different from 250
        eeg2['event'][1]['latency'] = 600  # Different from 500

        result, _stdout_output, stderr_output = run_compare(self.basic_eeg1, eeg2)

        self.assertTrue(result)

        self.assertIn('Event latency', stderr_output)
        self.assertIn('not OK', stderr_output)


class TestEegCompareReturnContract(unittest.TestCase):
    """Pin the documented return contract: eeg_compare returns a summary string."""

    def _eeg(self):
        return {
            'setname': 'ds',
            'subject': 'S01',
            'xmin': 0.0,
            'xmax': 1.0,
            'chanlocs': [],
            'event': [],
        }

    def test_identical_returns_match_summary_string(self):
        result, stdout_output, stderr_output = run_compare(self._eeg(), self._eeg())
        self.assertIsInstance(result, str)
        self.assertEqual(result, "All fields match (no differences found)")
        self.assertEqual(stderr_output, "")

    def test_differences_returned_as_string_not_bool(self):
        eeg2 = self._eeg()
        eeg2['subject'] = 'S02'
        result, _stdout_output, stderr_output = run_compare(self._eeg(), eeg2)
        self.assertIsInstance(result, str)
        self.assertNotIsInstance(result, bool)
        self.assertIn('differences', result.lower())
        self.assertIn('subject differs', stderr_output)

    def test_array_mismatch_returns_summary_string(self):
        result, _stdout_output, _stderr_output = run_compare(np.zeros((2, 3)), np.zeros((3, 2)))
        self.assertIsInstance(result, str)
        self.assertIn('Array shape mismatch', result)

    def test_trigger_error_raises_on_difference(self):
        eeg2 = self._eeg()
        eeg2['subject'] = 'S02'
        with self.assertRaises(ValueError):
            eeg_compare(self._eeg(), eeg2, trigger_error=True)


if __name__ == '__main__':
    unittest.main()
