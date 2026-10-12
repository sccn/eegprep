"""Tests for the low-level signal-processing eegrej (sigprocfunc.eegrej)."""

import unittest

import numpy as np

from eegprep.functions.sigprocfunc.eegrej import eegrej
from tests.eeglab_tests import assert_matlab_equal, assert_matlab_near, eeglab_test


UPSTREAM = "unittesting_sigprocfunc/eegrej/sigprocfunc_eegrej_wrapperTest.m"


@eeglab_test(UPSTREAM, "test_pass_general")
@eeglab_test(UPSTREAM, "test_pass_events")
def test_reference_eegrej_explicit_and_omitted_events(eeglab_backend):
    data = np.arange(1.0, 16.0).reshape(3, 5)
    # pass_events constructs an events vector but deliberately does not pass it.
    for arguments in ((np.empty((0, 0)),), ()):
        result, duration, _, boundaries = eeglab_backend(
            "eegrej", data, np.array([[1.0, 1.0], [3.0, 4.0]]), 5.0, *arguments, nargout=4
        )
        assert_matlab_near(result, [[2.0, 5.0], [7.0, 10.0], [12.0, 15.0]])
        assert_matlab_near(duration, [[2.0]])
        assert_matlab_near(boundaries, [[0.5], [1.5]])


@eeglab_test(UPSTREAM, "test_pass_overlap")
def test_reference_eegrej_overlapping_regions(eeglab_backend):
    result, duration, _, boundaries = eeglab_backend(
        "eegrej",
        np.arange(1.0, 19.0).reshape(3, 6),
        np.array([[2.0, 4.0], [3.0, 5.0]]),
        6.0,
        np.empty((0, 0)),
        nargout=4,
    )
    assert_matlab_near(result, [[1.0, 6.0], [7.0, 12.0], [13.0, 18.0]])
    assert_matlab_near(duration, [[2.0]])
    assert_matlab_near(boundaries, [[1.5]])


@eeglab_test(UPSTREAM, "test_passboundary")
def test_reference_eegrej_original_boundary_reconstruction(eeglab_backend):
    eeg = eeglab_backend(
        "pop_importdata",
        "dataformat",
        "array",
        "nbchan",
        1.0,
        "data",
        np.random.default_rng().random((1, 2000)),
        "srate",
        100.0,
        "pnts",
        0.0,
        "xmin",
        0.0,
    )
    for original in (
        np.array([[1.0, 200.0], [1000.0, 1200.0]]),
        np.array([[100.0, 150.0], [200.0, 250.0], [300.0, 350.0], [400.0, 500.0], [1000.0, 1200.0]]),
    ):
        result = eeglab_backend("eeg_eegrej", eeg, np.ceil(original))
        locations = np.concatenate(result["event"]["latency"][0].tolist(), axis=1)
        durations = np.concatenate(result["event"]["duration"][0].tolist(), axis=1)
        cumulative = np.cumsum(durations, axis=1)
        locations = locations + np.concatenate((np.zeros((1, 1)), cumulative[:, :-1]), axis=1)
        reconstructed = np.ceil(np.concatenate((locations, locations + durations - 1.0), axis=0).T)
        assert_matlab_equal(reconstructed, original)


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

    def test_adjacent_regions_merge_to_single_boundary(self):
        # Adjacent regions excise a contiguous block; the two boundaries collapse
        # to one latency after the base-span shift.
        _, _, _, boundevents = eegrej(self.data, [[5, 8], [9, 12]], self.timelength)
        np.testing.assert_array_equal(boundevents, [4.5])


if __name__ == "__main__":
    unittest.main()
