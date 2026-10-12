import os
import tempfile
import unittest
import numpy as np

# Assume eeg_eegrej is defined as in your module that imports: from eegrej import eegrej
from eegprep import eeg_eegrej
from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset

# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


def ensure_file(fname: str) -> str:  # duplicate of test_clean_rawdata.py
    """Download a file if it does not exist and return the local path."""
    full_url = f"{web_root}{fname}"
    local_file = os.path.abspath(f"{local_url}{fname}")
    if not os.path.exists(local_file):
        from urllib.request import urlretrieve

        urlretrieve(full_url, local_file)
    return local_file


def _make_continuous_eeg():
    # 2 channels × 20 samples, 1-based event latencies
    data = np.arange(40, dtype=float).reshape(2, 20)
    EEG = {
        "data": data,
        "xmin": 0.0,
        "xmax": 2.0,
        "pnts": data.shape[1],
        "srate": 100,
        "trials": 1,
        "event": [
            {"type": "stim", "latency": 3.0},
            {"type": "boundary", "latency": 6.0, "duration": 0.0},
            {"type": "stim", "latency": 7.0},
            {"type": "resp", "latency": 12.0},
        ],
    }
    return EEG


def _make_continuous_eeg2():
    # 2 channels × 20 samples, 1-based event latencies
    data = np.arange(75350 * 4, dtype=float).reshape(4, 75350)
    timevals = np.array(range(data.shape[1])) / 100
    EEG = dict(
        {
            "data": data,
            "xmin": 0.0,
            "xmax": 753.4900,
            "pnts": data.shape[1],
            "srate": 100,
            "nbchan": 4,
            "trials": 1,
            "times": timevals,
            "event": [
                {"type": "stim", "latency": 3.0},
                {"type": "boundary", "latency": 6.0, "duration": 0.0},
                {"type": "stim", "latency": 7.0},
                {"type": "resp", "latency": 12.0},
            ],
        }
    )
    EEG = eeg_checkset(EEG)
    return EEG


def _save_eeg(path, EEG):
    # Save as a single file object array for simplicity
    np.save(path, EEG, allow_pickle=True)


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEEGEegrej(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fpath = os.path.join(self.tmpdir.name, "eeg.npy")
        self.fpath_eeglab = ensure_file('FlankerTest.set')
        self.eeglab = get_eeglab('MAT')

        _save_eeg(self.fpath, _make_continuous_eeg())

    def tearDown(self):
        self.tmpdir.cleanup()

    # def test_rmtime_continuous(self):
    #     EEG = _make_continuous_eeg2()

    #     xmin = float(EEG['xmin'])
    #     xmax = float(EEG['xmax'])
    #     span = xmax - xmin
    #     rm_seg = np.array([[xmin + 0.1 * span, xmin + 0.2 * span]], dtype=float)*EEG['srate']
    #     rm_seg = rm_seg.astype(int)
    #     EEG_py = eeg_eegrej(EEG, rm_seg)

    #     EEG_mat = self.eeglab.eeg_eegrej(EEG, rm_seg)

    #     self.assertEqual(EEG_py['pnts'], EEG_mat['pnts'])
    #     self.assertTrue(np.allclose(EEG_py['data'], EEG_mat['data'], atol=1e-7, equal_nan=True))

    def test_rmtime_continuous2(self):
        EEG = pop_loadset(self.fpath_eeglab)
        rm_seg = np.array([[7535.900000, 15070.800000]], dtype=float)
        EEG_py = eeg_eegrej(EEG, rm_seg)

        EEG_mat = self.eeglab.eeg_eegrej(EEG, rm_seg)
        self.assertEqual(EEG_py['pnts'], EEG_mat['pnts'])
        np.testing.assert_array_equal(EEG_py["data"], EEG_mat["data"])

    # def test_compare_to_eeglab(self):
    #     EEG = pop_loadset(self.fpath_eeglab)
    #     regions = np.array([[6, 10]], dtype=int)
    #     EEG_out = eeg_eegrej(EEG, regions)

    #     eeglab_outdata = self.eeglab.eeg_eegrej(EEG, [6, 10])
    #     np.testing.assert_array_equal(EEG_out["data"], eeglab_outdata["data"])


class TestEEGEegrejExtended(unittest.TestCase):
    """Extended tests for eeg_eegrej functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_eeg = _make_continuous_eeg()

    def test_eeg_eegrej_overlapping_regions(self):
        """Test eeg_eegrej with overlapping regions that should be merged."""
        EEG = self.base_eeg.copy()

        # Overlapping regions: [3, 7] and [5, 10] should merge to [3, 10]
        regions = np.array([[3, 7], [5, 10]])

        with self.assertLogs("eegprep.functions.popfunc.eeg_eegrej", level="WARNING") as captured:
            result = eeg_eegrej(EEG, regions)

        self.assertTrue(
            any("Overlapping regions detected and fixed in eeg_eegrej" in message for message in captured.output)
        )

        # Should have 20 - 8 = 12 samples remaining (removed samples 3-10)
        self.assertEqual(result['pnts'], 12)
        self.assertEqual(result['data'].shape[1], 12)

    def test_eeg_eegrej_eegplot_style_regions(self):
        """Test eeg_eegrej with eegplot-style regions (4 columns)."""
        EEG = self.base_eeg.copy()

        # eegplot-style: [channel1, channel2, start, end]
        regions = np.array([[1, 2, 5, 8], [1, 2, 15, 17]])
        result = eeg_eegrej(EEG, regions)

        # Should extract columns 2:4 (start, end) and process normally
        # Removing samples 5-8 and 15-17: 20 - 4 - 3 = 13 samples remaining
        self.assertEqual(result['pnts'], 13)
        self.assertEqual(result['data'].shape[1], 13)

    def test_eeg_eegrej_preserves_numeric_boundary99_when_enabled(self):
        """Numeric -99 boundaries use the canonical boundary contract."""
        EEG = self.base_eeg.copy()
        EEG["event"][1]["type"] = -99
        old = EEG_OPTIONS["option_boundary99"]
        EEG_OPTIONS["option_boundary99"] = 1
        try:
            result = eeg_eegrej(EEG, np.array([[6, 10]], dtype=int))
        finally:
            EEG_OPTIONS["option_boundary99"] = old

        preserved = [event for event in result["event"] if event.get("type") == -99]
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0]["latency"], 6.0)

    def test_eeg_eegrej_xmax_update(self):
        """Test eeg_eegrej correctly updates xmax."""
        EEG = self.base_eeg.copy()
        original_duration = EEG['xmax'] - EEG['xmin']
        original_pnts = EEG['pnts']

        # Remove samples 5-9 (1-based indexing)
        regions = np.array([[5, 9]])
        result = eeg_eegrej(EEG, regions)

        # Calculate expected new duration
        new_pnts = result['pnts']
        expected_duration = original_duration * (new_pnts / original_pnts)
        expected_xmax = EEG['xmin'] + expected_duration

        self.assertAlmostEqual(result['xmax'], expected_xmax, places=10)

    def test_eeg_eegrej_event_cleanup(self):
        """Test eeg_eegrej event cleanup logic."""
        EEG = self.base_eeg.copy()

        # Add problematic events that should be cleaned up
        EEG['event'] = [
            {"type": "boundary", "latency": 0.0},  # Should be removed (latency 0)
            {"type": "stim", "latency": 5.0},
            {"type": "boundary", "latency": 20.0},  # Should be removed (latency == pnts)
        ]

        regions = np.array([[10, 12]])
        result = eeg_eegrej(EEG, regions)

        # Check that problematic events are cleaned up
        latencies = [e.get('latency') for e in result['event']]

        # Should not have events at latency 0 or at pnts
        self.assertNotIn(0.0, latencies)
        self.assertNotIn(float(result['pnts']), latencies)

    def test_eeg_eegrej_keeps_nonboundary_events_at_edges(self):
        """Genuine (non-boundary) events at the first/last sample must not be dropped."""
        EEG = self.base_eeg.copy()
        # stim at the first sample (latency 0) and a stim that lands on the
        # final sample after rejection (latency 20 -> 17 once 3 samples removed)
        EEG['event'] = [
            {"type": "stim", "latency": 0.0},
            {"type": "stim", "latency": 20.0},
        ]

        regions = np.array([[3, 5]])
        result = eeg_eegrej(EEG, regions)

        self.assertEqual(result['pnts'], 17)
        stim_events = [e for e in result['event'] if e.get('type') == 'stim']
        stim_latencies = sorted(e['latency'] for e in stim_events)
        # both stim events survive: one at the first sample, one at the last sample
        self.assertEqual(stim_latencies, [0.0, 17.0])

    def test_eeg_eegrej_floating_point_regions(self):
        """Test eeg_eegrej with floating point regions (should be rounded)."""
        EEG = self.base_eeg.copy()

        # Provide floating point regions
        regions = np.array([[5.3, 8.7], [15.1, 17.9]])
        result = eeg_eegrej(EEG, regions)

        # Should round to [5, 9] and [15, 18]
        # 20 - (9-5+1) - (18-15+1) = 20 - 5 - 4 = 11
        self.assertEqual(result['pnts'], 11)
        self.assertEqual(result['data'].shape[1], 11)


if __name__ == "__main__":
    unittest.main()
