import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

# Assume eeg_eegrej is defined as in your module that imports: from eegrej import eegrej
from eegprep import eeg_eegrej
from eegprep.eeglabcompat import get_eeglab
from eegprep.pop_loadset import pop_loadset

# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../data/')

def ensure_file(fname: str) -> str: # duplicate of test_clean_rawdata.py
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
        "trials": 1,
        "event": [
            {"type": "stim", "latency": 3.0},
            {"type": "boundary", "latency": 6.0, "duration": 0.0},
            {"type": "stim", "latency": 7.0},
            {"type": "resp", "latency": 12.0},
        ],
    }
    return EEG

def _save_eeg(path, EEG):
    # Save as a single file object array for simplicity
    np.save(path, EEG, allow_pickle=True)

def _load_eeg(path):
    return np.load(path, allow_pickle=True).item()

class TestEEGEegrej(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fpath = os.path.join(self.tmpdir.name, "eeg.npy")
        self.fpath_eeglab = ensure_file('FlankerTest.set')
        _save_eeg(self.fpath, _make_continuous_eeg())

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_eeg_eegrej_continuous_read_and_reject(self):
        EEG = _load_eeg(self.fpath)

        # Reject samples 6 to 10 inclusive (1-based)
        regions = np.array([[6, 10]], dtype=int)

        EEG_out = eeg_eegrej(EEG, regions)

        # After excision, length is 20 - 5 = 15
        self.assertEqual(EEG_out["pnts"], 15)
        self.assertEqual(EEG_out["data"].shape, (2, 15))

        # Event at latency 7 is removed because it is inside the rejected region
        # Boundary event at original 6 is preserved and a new boundary is inserted at new latency 6 with duration 5
        # Event at 12 shifts by 5 to latency 7. Event at 3 stays at 3.
        ev = EEG_out["event"]
        lats = [e.get("latency") for e in ev]
        types = [e.get("type") for e in ev]

        # Must contain exactly one boundary we inserted at new position 6 with duration 5
        self.assertIn(6.0, lats)
        bidx = lats.index(6.0)
        self.assertEqual(types[bidx], "boundary")
        self.assertEqual(ev[bidx].get("duration"), 5.0)

        # Stim at 3 remains
        self.assertIn(3.0, lats)
        sidx = lats.index(3.0)
        self.assertEqual(types[sidx], "stim")

        # Stim at 7 inside region removed, but resp at 12 shifts to 7
        self.assertIn(7.0, lats)
        ridx = lats.index(7.0)
        self.assertEqual(types[ridx], "resp")

        # xmax was updated correctly to reflect the new duration after rejection
        # new_duration = old_duration * (new_pnts / old_pnts)
        old_duration = EEG["xmax"] - EEG["xmin"]
        new_duration = old_duration * (EEG_out["pnts"] / EEG["pnts"])
        expected_xmax = EEG["xmin"] + new_duration
        self.assertAlmostEqual(EEG_out["xmax"], expected_xmax, places=7)

        # No event latencies exceed pnts
        self.assertTrue(all(0 < e["latency"] <= EEG_out["pnts"] for e in ev))
        
    def test_compare_to_eeglab(self):
        EEG = pop_loadset(self.fpath_eeglab)
        regions = np.array([[6, 10]], dtype=int)
        EEG_out = eeg_eegrej(EEG, regions)
        
        eeglab = get_eeglab('MAT')
        eeglab_outdata = eeglab.eeg_eegrej(EEG, [6, 10])
        np.testing.assert_array_equal(EEG_out["data"], eeglab_outdata["data"])

if __name__ == "__main__":
    unittest.main()