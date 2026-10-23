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

    def test_rmtime_continuous2(self):
        EEG = pop_loadset(self.fpath_eeglab)
        rm_seg = np.array([[7535.900000, 15070.800000]], dtype=float)
        EEG_py = eeg_eegrej(EEG, rm_seg)

        EEG_mat = self.eeglab.eeg_eegrej(EEG, rm_seg)
        self.assertEqual(EEG_py['pnts'], EEG_mat['pnts'])
        np.testing.assert_array_equal(EEG_py["data"], EEG_mat["data"])


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

        new_pnts = result['pnts']
        proportional = EEG['xmin'] + original_duration * (new_pnts / original_pnts)
        # eeg_checkset.m:849 snaps when round(srate*(xmax-xmin)+1) is not pnts.
        # 15 kept samples of a 2 s, 100 Hz recording are not on that grid.
        self.assertEqual(new_pnts, 15)
        self.assertAlmostEqual(proportional, 1.5, places=10)
        self.assertAlmostEqual(result['xmax'], 0.14, places=10)

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

        latencies = [e.get('latency') for e in result['event']]

        # Should not have events at latency 0 or at pnts
        self.assertNotIn(0.0, latencies)
        self.assertNotIn(float(result['pnts']), latencies)

    def test_eeg_eegrej_drops_events_before_half_a_sample(self):
        """eventconsistency removes latency < 0.5 and keeps a stim on the last sample."""
        EEG = self.base_eeg.copy()
        EEG['event'] = [
            {"type": "stim", "latency": 0.0},
            {"type": "stim", "latency": 20.0},
        ]

        regions = np.array([[3, 5]])
        result = eeg_eegrej(EEG, regions)

        self.assertEqual(result['pnts'], 17)
        stim_events = [e for e in result['event'] if e.get('type') == 'stim']
        stim_latencies = sorted(e['latency'] for e in stim_events)
        self.assertEqual(stim_latencies, [17.0])

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

    def test_eeg_eegrej_adjacent_regions_stay_separate(self):
        """combineregions closes on cumsum 0, so a one-sample gap is not a merge."""
        EEG = self.base_eeg.copy()
        regions = np.array([[1, 5], [6, 10]])
        with self.assertNoLogs("eegprep.functions.popfunc.eeg_eegrej", level="WARNING"):
            result = eeg_eegrej(EEG, regions)
        self.assertEqual(result["pnts"], 10)

    def test_eeg_eegrej_merges_reversed_fractional_regions_before_shifting(self):
        """Overlapping reversed bounds are one span before event latencies move."""
        EEG = self.base_eeg.copy()
        EEG["data"] = np.zeros((1, 400), dtype=float)
        EEG["pnts"] = 400
        EEG["nbchan"] = 1
        EEG["event"] = [{"type": "stim", "latency": 340.0}]
        regions = np.array(
            [
                [145.9771655965767, 123.72682490095676],
                [320.3381323213592, 26.319768749552097],
                [272.79633006697026, 151.05315648702828],
            ]
        )
        with self.assertLogs("eegprep.functions.popfunc.eeg_eegrej", level="WARNING"):
            result = eeg_eegrej(EEG, regions)
        stims = [event["latency"] for event in result["event"] if event.get("type") == "stim"]
        self.assertEqual(stims, [45.0])
        self.assertEqual(result["pnts"], 105)

    def test_eeg_eegrej_empty_result_xmax_is_minus_one_sample(self):
        EEG = self.base_eeg.copy()
        result = eeg_eegrej(EEG, np.array([[1, 20]]))
        self.assertEqual(result["pnts"], 0)
        self.assertAlmostEqual(result["xmax"], -1.0 / EEG["srate"])
        self.assertEqual(np.asarray(result["times"]).size, 0)

    def test_structured_events_are_kept_as_a_list(self):
        """A MATLAB struct array is converted; it is not replaced with an empty list."""
        EEG = self.base_eeg.copy()
        EEG["event"] = np.array(
            [[("mrk1", 5.0), ("mrk2", 12.0)]],
            dtype=[("type", object), ("latency", object)],
        )
        result = eeg_eegrej(EEG, np.array([[10.0, 12.0]]))
        self.assertIsInstance(result["event"], list)
        self.assertEqual([event["latency"] for event in result["event"] if event.get("type") == "mrk1"], [5.0])
        self.assertFalse(any(event.get("type") == "mrk2" for event in result["event"]))

    def test_non_event_array_raises(self):
        EEG = self.base_eeg.copy()
        EEG["event"] = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(TypeError):
            eeg_eegrej(EEG, np.array([[10.0, 12.0]]))

    def test_last_event_pop_follows_eventconsistency(self):
        """eeg_eegrej.m:144-147 drops the new last event after out-of-range events are gone."""
        EEG = self.base_eeg.copy()
        EEG["event"] = [
            {"type": "stim", "latency": 5.0},
            {"type": "stim", "latency": 20.6},
            {"type": "stim", "latency": 21.2},
        ]
        result = eeg_eegrej(EEG, np.array([[1.0, 3.0]]))
        stims = [event["latency"] for event in result["event"] if event.get("type") == "stim"]
        self.assertEqual(stims, [2.0])

    def test_zero_trials_does_not_drop_events_past_sample_one(self):
        EEG = self.base_eeg.copy()
        EEG["trials"] = 0
        EEG["event"] = [{"type": "stim", "latency": 5.0}]
        result = eeg_eegrej(EEG, np.array([[1, 2]]))
        self.assertEqual(result["trials"], 1)
        stims = [event["latency"] for event in result["event"] if event.get("type") == "stim"]
        self.assertEqual(stims, [3.0])


if __name__ == "__main__":
    unittest.main()
