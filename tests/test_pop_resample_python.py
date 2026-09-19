import unittest

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.popfunc.pop_resample import pop_resample
from tests.eeglab_tests import eeglab_test


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

    def test_resample_preserves_numpy_event_containers(self):
        eeg = _continuous_eeg()
        eeg["event"] = np.asarray(eeg["event"], dtype=object)
        eeg["urevent"] = np.asarray(eeg["urevent"], dtype=object)

        out = pop_resample(eeg, 50, engine="scipy")

        self.assertIsInstance(out["event"], np.ndarray)
        self.assertIsInstance(out["urevent"], np.ndarray)
        np.testing.assert_allclose([event["latency"] for event in out["urevent"]], [3.5, 5.5, 8.5])

    def test_numeric_boundary99_splits_continuous_segments_when_enabled(self):
        eeg = _continuous_eeg()
        eeg["event"][1]["type"] = -99
        eeg["urevent"][1]["type"] = -99
        old = EEG_OPTIONS["option_boundary99"]
        EEG_OPTIONS["option_boundary99"] = 1
        try:
            out = pop_resample(eeg, 50, engine="scipy")
        finally:
            EEG_OPTIONS["option_boundary99"] = old

        np.testing.assert_allclose([event["latency"] for event in out["event"]], [3.5, 5.5, 8.5])
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


def _epoched_resample_eeg():
    eeg = _continuous_eeg()
    eeg["data"] = np.arange(80, dtype=np.float32).reshape(2, 20, 2)
    eeg["trials"] = 2
    eeg["event"] = [
        {"type": "stim", "latency": 6.0, "duration": 10.0, "epoch": 1},
        {"type": "resp", "latency": 26.0, "duration": 20.0, "epoch": 2},
    ]
    eeg["urevent"] = []
    return eeg


@eeglab_test("unittesting_popfunc/pop_resample/popfunc_pop_resample_wrapperTest.m", "test_test_pop_resample")
def test_pop_resample_current_suite_epoched_and_continuous_rates():
    for eeg in (_epoched_resample_eeg(), _continuous_eeg()):
        low_rate = pop_resample(eeg, 10)
        high_rate = pop_resample(eeg, 1000)

        assert low_rate["srate"] == 10
        assert high_rate["srate"] == 1000
        assert low_rate["trials"] == eeg["trials"]
        assert high_rate["trials"] == eeg["trials"]


@eeglab_test("unittesting_popfunc/pop_resample/popfunc_pop_resample_wrapperTest.m", "test_test_pop_resample2")
def test_pop_resample_current_suite_preserves_event_duration_seconds():
    epoched = _epoched_resample_eeg()
    continuous = _continuous_eeg()
    continuous["event"] = [
        {"type": "stim", "latency": 6.0, "duration": 10000.0},
        {"type": "resp", "latency": 16.0, "duration": 20000.0},
    ]
    continuous["urevent"] = []

    for eeg, expected_seconds in ((epoched, [0.1, 0.2]), (continuous, [100.0, 200.0])):
        for rate in (10, 1000):
            output = pop_resample(eeg, rate)
            durations = np.asarray([event["duration"] for event in output["event"]])
            np.testing.assert_allclose(durations / output["srate"], expected_seconds)


@eeglab_test("unittesting_popfunc/pop_resample/popfunc_pop_resample_wrapperTest.m", "test_testcase_boundary")
def test_pop_resample_current_suite_preserves_half_sample_boundaries():
    eeg = {
        "data": np.zeros((1, 10000), dtype=np.float32),
        "nbchan": 1,
        "pnts": 10000,
        "trials": 1,
        "srate": 500.0,
        "xmin": 0.0,
        "xmax": 19.998,
        "times": np.arange(10000, dtype=float) / 500 * 1000,
        "setname": "boundary resampling",
        "event": [
            {"type": "boundary", "latency": 0.5},
            {"type": "boundary", "latency": 500.5},
        ],
        "urevent": [],
        "epoch": [],
        "chanlocs": [],
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([], dtype=int),
    }

    for rate in (200, 250, 300, 350, 450, 550, 600, 650, 700):
        output = pop_resample(eeg, rate)
        np.testing.assert_allclose(
            [output["event"][0]["latency"], output["event"][1]["latency"]],
            [0.5, rate + 0.5],
        )


if __name__ == "__main__":
    unittest.main()
