# test_eegrej_unittest.py
import unittest
import numpy as np
from eegprep import eegrej  # replace with the actual module name
from eegprep.eeglabcompat import get_eeglab

class TestEEGRej(unittest.TestCase):
    def setUp(self):
        # 2 channels, 20 samples with simple increasing values
        self.data = np.vstack([np.arange(1, 21), np.arange(101, 121)])
        self.timelength = 10.0  # seconds

    def test_no_regions(self):
        events = [{"latency": 5.0}, {"latency": 10.0}]
        outdata, newt, newevents, boundevents = eegrej(self.data, [], self.timelength, events)
        np.testing.assert_array_equal(outdata, self.data)
        self.assertEqual(newt, self.timelength)
        lats = np.array([ev["latency"] for ev in newevents], dtype=float)
        np.testing.assert_array_equal(lats, [5.0, 10.0])
        self.assertEqual(boundevents.size, 0)

    def test_single_region_no_events(self):
        outdata, newt, newevents, boundevents = eegrej(self.data, [[5, 8]], self.timelength)
        expected_data = np.delete(self.data, np.arange(4, 8), axis=1)
        np.testing.assert_array_equal(outdata, expected_data)
        self.assertAlmostEqual(newt, self.timelength * (expected_data.shape[1] / self.data.shape[1]))
        self.assertIsInstance(newevents, list)
        self.assertEqual(len(newevents), 0)
        np.testing.assert_array_equal(boundevents, [4.5])

    def test_event_shifting_and_removed_inside_region(self):
        events = [{"latency": 3.0}, {"latency": 6.0}, {"latency": 10.0}]  # event at 6 will be removed
        outdata, newt, newevents, boundevents = eegrej(self.data, [[5, 8]], self.timelength, events)
        # After removing samples 5–8, event at 3 stays at 3, event at 10 shifts by -4 to 6,
        # event at 6 (inside region) is removed (MATLAB-style)
        # Ignore boundary events when checking latencies
        lats = np.array([ev["latency"] for ev in newevents if ev.get("type") != "boundary"], dtype=float)
        np.testing.assert_array_equal(lats, [3.0, 6.0])
        np.testing.assert_array_equal(boundevents, [4.5])

    def test_multiple_regions_event_shift(self):
        events = [{"latency": 3.0}, {"latency": 6.0}, {"latency": 15.0}, {"latency": 19.0}]
        outdata, newt, newevents, boundevents = eegrej(self.data, [[5, 8], [12, 14]], self.timelength, events)
        # Removed lengths: first=4 samples, second=3 samples
        # Event at 6 removed; 15 -> 8; 19 -> 12
        lats = np.array([ev["latency"] for ev in newevents if ev.get("type") != "boundary"], dtype=float)
        np.testing.assert_array_equal(lats, [3.0, 8.0, 12.0])
        # Boundary latencies: first at (5-1)=4 -> 4.5, second at (12-1)-4=7 -> 7.5
        np.testing.assert_array_equal(boundevents, [4.5, 7.5])

    def test_overlapping_regions_are_adjusted(self):
        # Overlapping regions should be merged/adjusted
        outdata, newt, newevents, boundevents = eegrej(self.data, [[5, 10], [8, 12]], self.timelength)
        # The overlap means the second region starts after first ends
        reject_mask = np.zeros(self.data.shape[1], dtype=bool)
        reject_mask[4:12] = True  # after adjustment, remove 5–12 (1-based)
        expected_data = self.data[:, ~reject_mask]
        np.testing.assert_array_equal(outdata, expected_data)

    def test_region_at_end_removes_boundary_event(self):
        # Removing last samples should drop boundaries >= newn+1
        outdata, newt, newevents, boundevents = eegrej(self.data, [[18, 20]], self.timelength)
        # 18–20 is 3 samples; boundary latency before removal is (18-1)=17 -> 17.5
        # After removal, newn=17, so boundevents=17.5 remains
        self.assertTrue((boundevents == [17.5]).all())
        
    def test_compare_to_eeglab(self):
        # compare to eeglab
        events = [{"latency": 5.0}, {"latency": 10.0}]
        outdata, newt, newevents, boundevents = eegrej(self.data, [[5, 8]], self.timelength, events)
        # compare to eeglab
        eeglab = get_eeglab()
        eeglab_outdata, eeglab_newevents, eeglab_boundevents = eeglab.eegrej(self.data, np.array([[5, 8]]), self.timelength, events, nargout=3)
        np.testing.assert_array_equal(outdata, eeglab_outdata)

if __name__ == "__main__":
    unittest.main()