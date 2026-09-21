"""Tests for the low-level signal-processing eegrej (sigprocfunc.eegrej)."""

import unittest

import numpy as np

from eegprep import eeg_eegrej
from eegprep.functions.sigprocfunc.eegrej import eegrej
from tests.eeglab_tests import eeglab_test


class TestSigprocEegrej(unittest.TestCase):
    def setUp(self):
        # 1 channel, 30 samples with values equal to their 1-based sample index
        self.data = np.arange(1, 31, dtype=float).reshape(1, 30)
        self.timelength = 30.0

    def test_boundary_shift_uses_base_span_not_nested_duration(self):
        # The first removed region already contains a boundary event with a large
        # duration. EEGLAB shifts later boundary latencies by the prior regions'
        # base spans only; the nested duration must NOT pull later boundaries left.
        events = [
            {"type": "boundary", "latency": 7.0, "duration": 100.0},  # inside [5, 10]
            {"type": "stim", "latency": 25.0},
        ]
        _, _, newevents, boundevents = eegrej(self.data, [[5, 10], [20, 22]], self.timelength, events)

        # Region1 boundary at start-1 = 4 -> 4.5.
        # Region2 boundary at start-1 = 19, shifted by region1 base span (6) -> 13 -> 13.5.
        # The augmented duration (106) must not be used for the shift.
        np.testing.assert_array_equal(boundevents, [4.5, 13.5])

        bnd = {ev["latency"]: ev["duration"] for ev in newevents if ev.get("type") == "boundary"}
        # The first boundary's .duration carries the nested duration (base 6 + 100).
        self.assertEqual(bnd[4.5], 106.0)
        # The second boundary's .duration is region2's base span (22 - 20 + 1 = 3).
        self.assertEqual(bnd[13.5], 3.0)

    def test_multiple_regions_without_nested_boundaries(self):
        # With no nested boundary durations, base span == duration, so the shift is
        # unchanged: region2 boundary at 11 shifted by region1 base span 4 -> 7.5.
        _, _, _, boundevents = eegrej(self.data, [[5, 8], [12, 14]], self.timelength)
        np.testing.assert_array_equal(boundevents, [4.5, 7.5])

    @eeglab_test(
        "unittesting_sigprocfunc/eegrej/sigprocfunc_eegrej_wrapperTest.m",
        "test_pass_general",
    )
    @eeglab_test(
        "unittesting_sigprocfunc/eegrej/sigprocfunc_eegrej_wrapperTest.m",
        "test_pass_events",
    )
    def test_upstream_disjoint_regions_data_duration_and_boundaries(self):
        data = np.arange(1, 16, dtype=float).reshape(3, 5)

        outdata, newtime, events, boundaries = eegrej(data, [[1, 1], [3, 4]], 5, [])

        np.testing.assert_array_equal(outdata, [[2, 5], [7, 10], [12, 15]])
        self.assertEqual(newtime, 2)
        self.assertEqual(events, [])
        np.testing.assert_array_equal(boundaries, [0.5, 1.5])

    def test_adjacent_regions_merge_to_single_boundary(self):
        # Adjacent regions excise a contiguous block; the two boundaries collapse
        # to one latency after the base-span shift.
        _, _, _, boundevents = eegrej(self.data, [[5, 8], [9, 12]], self.timelength)
        np.testing.assert_array_equal(boundevents, [4.5])

    @eeglab_test(
        "unittesting_sigprocfunc/eegrej/sigprocfunc_eegrej_wrapperTest.m",
        "test_pass_overlap",
    )
    def test_overlapping_regions_merge_to_single_boundary(self):
        # Overlapping regions are de-overlapped then excised as one contiguous block.
        _, _, _, boundevents = eegrej(self.data, [[5, 10], [8, 12]], self.timelength)
        np.testing.assert_array_equal(boundevents, [4.5])

    @eeglab_test(
        "unittesting_sigprocfunc/eegrej/sigprocfunc_eegrej_wrapperTest.m",
        "test_passboundary",
    )
    def test_eeg_eegrej_boundaries_round_trip_to_original_intervals(self):
        eeg = {
            "data": np.arange(2000, dtype=float).reshape(1, 2000),
            "nbchan": 1,
            "pnts": 2000,
            "trials": 1,
            "srate": 100.0,
            "xmin": 0.0,
            "xmax": 19.99,
            "times": np.arange(2000) / 100,
            "event": [],
            "urevent": [],
            "epoch": [],
        }
        for intervals in (
            np.asarray([[1, 200], [1000, 1200]]),
            np.asarray([[100, 150], [200, 250], [300, 350], [400, 500], [1000, 1200]]),
        ):
            rejected = eeg_eegrej(eeg, intervals)
            boundaries = [event for event in rejected["event"] if event["type"] == "boundary"]
            latencies = np.asarray([event["latency"] for event in boundaries], dtype=float)
            durations = np.asarray([event["duration"] for event in boundaries], dtype=float)
            cumulative = np.cumsum(durations)
            original_starts = latencies + np.r_[0, cumulative[:-1]]
            reconstructed = np.ceil(np.column_stack([original_starts, original_starts + durations - 1])).astype(int)

            np.testing.assert_array_equal(reconstructed, intervals)


if __name__ == "__main__":
    unittest.main()
