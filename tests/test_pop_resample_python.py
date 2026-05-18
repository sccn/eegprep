import unittest

import numpy as np

from eegprep.functions.popfunc.pop_resample import pop_resample


def _continuous_eeg():
    return {
        "data": np.arange(40, dtype=np.float32).reshape(2, 20),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "xmin": 0,
        "xmax": 0.19,
        "times": np.arange(20),
        "setname": "demo",
        "event": [
            {"type": "stim", "latency": 6.0, "duration": 2.0, "urevent": 1},
            {"type": "boundary", "latency": 10.5, "duration": 1.0, "urevent": 2},
            {"type": "resp", "latency": 16.0, "duration": 4.0, "urevent": 3},
        ],
        "urevent": [
            {"type": "stim", "latency": 6.0, "duration": 2.0},
            {"type": "boundary", "latency": 10.5, "duration": 1.0},
            {"type": "resp", "latency": 16.0, "duration": 4.0},
        ],
        "icaweights": np.eye(2),
        "icasphere": np.eye(2),
        "icawinv": np.eye(2),
        "icaact": np.ones((2, 20, 1), dtype=np.float32),
        "icachansind": np.arange(2),
    }


class PopResamplePythonTests(unittest.TestCase):
    def test_continuous_boundaries_split_segments_and_remap_events(self):
        out, command = pop_resample(_continuous_eeg(), 50, engine="scipy", return_com=True)

        self.assertEqual(out["data"].shape, (2, 10))
        self.assertEqual(out["pnts"], 10)
        self.assertEqual(out["srate"], 50)
        self.assertEqual(out["setname"], "demo resampled")
        self.assertEqual(command, "EEG = pop_resample( EEG, 50);")
        self.assertEqual(out["icaact"].size, 0)
        np.testing.assert_allclose([event["latency"] for event in out["event"]], [3.5, 5.5, 8.5])
        np.testing.assert_allclose([event["duration"] for event in out["event"]], [1.0, 0.5, 2.0])
        np.testing.assert_allclose([event["latency"] for event in out["urevent"]], [3.5, 5.5, 8.5])

    def test_epoched_data_resamples_each_epoch_and_clears_urevents(self):
        eeg = _continuous_eeg()
        eeg["data"] = np.arange(80, dtype=np.float32).reshape(2, 20, 2)
        eeg["trials"] = 2
        eeg["event"] = [{"type": "stim", "latency": 26.0, "duration": 2.0, "epoch": 2}]
        eeg["urevent"] = [{"type": "stim", "latency": 26.0, "duration": 2.0}]

        out = pop_resample(eeg, 50, engine="scipy")

        self.assertEqual(out["data"].shape, (2, 10, 2))
        self.assertEqual(out["pnts"], 10)
        self.assertEqual(out["trials"], 2)
        self.assertEqual(out["urevent"], [])
        self.assertAlmostEqual(out["event"][0]["latency"], 13.5)
        self.assertAlmostEqual(out["event"][0]["duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
