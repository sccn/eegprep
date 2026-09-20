import math
import unittest

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.popfunc.pop_resample import (
    RATIO_TOLERANCE,
    _resample_ratio,
    pop_resample,
)
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


class ResampleRatioTests(unittest.TestCase):
    """The ratio that drives both the resampling and the event latencies.

    EEGLAB computes it as ``rat(freq/EEG.srate, 1e-12)`` (pop_resample.m line
    119). This used to be ported as ``sympy.nsimplify``, which is not the same
    function: it looks for a simple symbolic expression, not a rational one,
    and returns an irrational when no simple fraction is close enough. The
    numerator of such an expression truncates silently under ``int()``, so the
    ratio came back small, wrong, and unflagged.
    """

    # Sampling rates that are not integers are the whole point: every pair here
    # with a whole-number source rate was already handled correctly, and every
    # pair with a fractional one was not.
    INTEGER_RATES = {
        (250, 1000): (1, 4),
        (256, 512): (1, 2),
        (500, 2048): (125, 512),
        (128, 1000): (16, 125),
        (1000, 1024): (125, 128),
        (44100, 48000): (147, 160),
        (96, 128): (3, 4),
    }
    FRACTIONAL_RATES = {
        (999.9, 2000): (9999, 20000),
        (1023.4, 2000): (5117, 10000),
        (250, 999.9): (2500, 9999),
        (128, 512.03): (12800, 51203),
    }

    def test_whole_number_rates_are_unchanged(self):
        """The fix must not move the ratios that were already right."""
        for (freq, srate), expected in self.INTEGER_RATES.items():
            with self.subTest(freq=freq, srate=srate):
                self.assertEqual(_resample_ratio(freq, srate), expected)

    def test_fractional_rates_are_exact(self):
        """Each of these came back truncated, by up to 16.7 percent."""
        for (freq, srate), expected in self.FRACTIONAL_RATES.items():
            with self.subTest(freq=freq, srate=srate):
                self.assertEqual(_resample_ratio(freq, srate), expected)

    def test_every_ratio_lands_inside_the_tolerance(self):
        """The property the pairs above are examples of.

        Stated as a property rather than a table so that a future change of
        method has to hold the invariant for rates nobody wrote down. An
        implementation that returns something other than a ratio fails here for
        the same reason it failed in the field: the number it returns is not
        the number it was asked for.
        """
        sources = [250, 256, 500, 512, 1000, 1024, 2000, 2048, 5000, 999.9, 1023.4, 512.03]
        targets = [100, 125, 128, 200, 250, 256, 500, 512]

        for srate in sources:
            for freq in targets:
                if freq >= srate:
                    continue
                with self.subTest(freq=freq, srate=srate):
                    p, q = _resample_ratio(freq, srate)
                    self.assertGreater(q, 0)
                    self.assertAlmostEqual(p / q, freq / srate, delta=RATIO_TOLERANCE)

    def test_a_fractional_rate_resamples_through_the_whole_pipeline(self):
        """The end-to-end test that the defect actually fails.

        The test below deliberately uses whole-number rates, and so passes
        against the old implementation too: it says nothing about this fix. This
        one drives `pop_resample` itself at a fractional source rate, where the
        old ratio was 21/84 and the correct one is 2560/9999, and asserts the
        sample count that follows from the correct one.

        It runs through the `scipy` engine on purpose. The `poly` engine builds
        an anti-aliasing filter whose length scales with the denominator, and an
        exact ratio for a fractional rate has a large one, so the faithful
        engine would make this test cost seconds. The `scipy` engine consumes
        the same p and q through `ceil(n * p / q)`, which is what is being
        checked here.
        """
        eeg = _continuous_eeg()
        eeg["srate"] = 999.9
        eeg["data"] = np.zeros((2, 2000), dtype=np.float32)
        eeg["pnts"] = 2000
        eeg["event"] = []
        eeg["urevent"] = []
        eeg["icaact"] = np.zeros((2, 2000, 1), dtype=np.float32)

        output = pop_resample(eeg, 256, engine="scipy")

        p, q = _resample_ratio(256, 999.9)
        self.assertEqual((p, q), (2560, 9999))
        self.assertEqual(output["pnts"], math.ceil(2000 * p / q))
        self.assertEqual(output["srate"], 256)

    def test_the_resampled_rate_matches_the_data(self):
        """The invariant the defect broke: the label and the samples agree.

        pop_resample stamps ``srate`` with the rate it was asked for whatever
        the ratio turned out to be, so a wrong ratio produced a recording whose
        declared rate and sample count disagreed, with nothing to notice it.
        Whole-number rates keep this cheap; the ratio tests above are where the
        fractional ones are covered, because an exact fractional ratio means a
        denominator in the thousands and an anti-aliasing filter to match.
        """
        eeg = _continuous_eeg()
        eeg["event"] = []
        duration = eeg["pnts"] / eeg["srate"]

        for rate in (50, 200, 250):
            with self.subTest(rate=rate):
                output = pop_resample(dict(eeg), rate)

                self.assertEqual(output["srate"], rate)
                self.assertAlmostEqual(output["pnts"] / output["srate"], duration, delta=1 / rate)


if __name__ == "__main__":
    unittest.main()
