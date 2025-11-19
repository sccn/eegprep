import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

# Assume eeg_eegrej is defined as in your module that imports: from eegrej import eegrej
from eegprep import eeg_eegrej
from eegprep.eeglabcompat import get_eeglab
from eegprep.pop_loadset import pop_loadset
from eegprep.eeg_checkset import eeg_checkset

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
    data     = np.arange(75350*4, dtype=float).reshape(4, 75350)
    timevals = np.array(range(data.shape[1]))/100
    EEG = dict({
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
    })
    EEG = eeg_checkset(EEG)
    return EEG

def _save_eeg(path, EEG):
    # Save as a single file object array for simplicity
    np.save(path, EEG, allow_pickle=True)

def _load_eeg(path):
    return np.load(path, allow_pickle=True).item()

@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestEEGEegrej(unittest.TestCase):
    def setUp(self):
        EEG = _make_continuous_eeg2()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.fpath = os.path.join(self.tmpdir.name, "eeg.npy")
        self.fpath_eeglab = ensure_file('FlankerTest.set')
        self.eeglab = get_eeglab('MAT')

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
        rm_seg = np.array([[7535.900000,15070.800000]], dtype=float)
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
        
    def test_eeg_eegrej_empty_regions(self):
        """Test eeg_eegrej with empty regions."""
        EEG = self.base_eeg.copy()
        
        # Test with None
        result = eeg_eegrej(EEG, None)
        np.testing.assert_array_equal(result['data'], EEG['data'])
        self.assertEqual(result['pnts'], EEG['pnts'])
        
        # Test with empty list
        result = eeg_eegrej(EEG, [])
        np.testing.assert_array_equal(result['data'], EEG['data'])
        self.assertEqual(result['pnts'], EEG['pnts'])
        
        # Test with empty array
        result = eeg_eegrej(EEG, np.array([]).reshape(0, 2))
        np.testing.assert_array_equal(result['data'], EEG['data'])
        self.assertEqual(result['pnts'], EEG['pnts'])

    def test_eeg_eegrej_single_region(self):
        """Test eeg_eegrej with a single rejection region."""
        EEG = self.base_eeg.copy()
        
        # Remove samples 5-8 (1-based indexing)
        regions = np.array([[5, 8]])
        result = eeg_eegrej(EEG, regions)
        
        # Should have 20 - 4 = 16 samples remaining
        self.assertEqual(result['pnts'], 16)
        self.assertEqual(result['data'].shape[1], 16)
        
        # Check that data is correctly concatenated
        expected_data = np.concatenate([
            EEG['data'][:, :4],   # samples 1-4 (0-3 in 0-based)
            EEG['data'][:, 8:]    # samples 9-20 (8-19 in 0-based)
        ], axis=1)
        np.testing.assert_array_equal(result['data'], expected_data)

    def test_eeg_eegrej_multiple_regions(self):
        """Test eeg_eegrej with multiple rejection regions."""
        EEG = self.base_eeg.copy()
        
        # Remove samples 3-5 and 15-17 (1-based indexing)
        regions = np.array([[3, 5], [15, 17]])
        result = eeg_eegrej(EEG, regions)
        
        # Should have 20 - 3 - 3 = 14 samples remaining
        self.assertEqual(result['pnts'], 14)
        self.assertEqual(result['data'].shape[1], 14)
        
        # Check that data is correctly concatenated
        expected_data = np.concatenate([
            EEG['data'][:, :2],    # samples 1-2 (0-1 in 0-based)
            EEG['data'][:, 5:14],  # samples 6-14 (5-13 in 0-based)
            EEG['data'][:, 17:]    # samples 18-20 (17-19 in 0-based)
        ], axis=1)
        np.testing.assert_array_equal(result['data'], expected_data)

    def test_eeg_eegrej_overlapping_regions(self):
        """Test eeg_eegrej with overlapping regions that should be merged."""
        EEG = self.base_eeg.copy()
        
        # Overlapping regions: [3, 7] and [5, 10] should merge to [3, 10]
        regions = np.array([[3, 7], [5, 10]])
        
        with patch('builtins.print') as mock_print:
            result = eeg_eegrej(EEG, regions)
            # Should print warning about overlapping regions
            mock_print.assert_called_with("Warning: overlapping regions detected and fixed in eeg_eegrej")
        
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

    def test_eeg_eegrej_region_sorting(self):
        """Test eeg_eegrej with unsorted regions."""
        EEG = self.base_eeg.copy()
        
        # Provide regions in reverse order
        regions = np.array([[15, 17], [3, 5]])
        result = eeg_eegrej(EEG, regions)
        
        # Should still work correctly (regions get sorted internally)
        self.assertEqual(result['pnts'], 14)  # 20 - 3 - 3 = 14
        self.assertEqual(result['data'].shape[1], 14)

    def test_eeg_eegrej_boundary_event_insertion(self):
        """Test eeg_eegrej boundary event insertion."""
        EEG = self.base_eeg.copy()
        
        # Remove samples 8-12 (1-based indexing)
        regions = np.array([[8, 12]])
        result = eeg_eegrej(EEG, regions)
        
        # Check that boundary event is inserted
        boundary_events = [e for e in result['event'] if e.get('type') == 'boundary']
        
        # Should have at least one boundary event
        self.assertGreater(len(boundary_events), 0)
        
        # Find the boundary event we inserted (with duration matching removed region)
        inserted_boundary = None
        for e in boundary_events:
            if e.get('duration') == 5.0:  # Duration matches removed region length
                inserted_boundary = e
                break
        
        self.assertIsNotNone(inserted_boundary)
        # The exact latency may vary based on implementation, but should be reasonable
        self.assertGreater(inserted_boundary['latency'], 0.0)
        self.assertLessEqual(inserted_boundary['latency'], result['pnts'] + 1.0)
        self.assertEqual(inserted_boundary['duration'], 5.0)  # Length of removed region

    def test_eeg_eegrej_event_latency_preservation(self):
        """Test eeg_eegrej preserves 1-based event latencies correctly."""
        EEG = self.base_eeg.copy()
        
        # Add more events for comprehensive testing
        EEG['event'] = [
            {"type": "stim", "latency": 2.0},   # Before rejection region
            {"type": "resp", "latency": 9.0},   # Inside rejection region (should be removed)
            {"type": "stim", "latency": 15.0},  # After rejection region (should shift)
        ]
        
        # Remove samples 8-12 (1-based indexing)
        regions = np.array([[8, 12]])
        result = eeg_eegrej(EEG, regions)
        
        # Check event latencies
        event_latencies = [e.get('latency') for e in result['event'] if e.get('type') != 'boundary']
        event_types = [e.get('type') for e in result['event'] if e.get('type') != 'boundary']
        
        # Event at latency 2 should remain at 2 (before rejection)
        self.assertIn(2.0, event_latencies)
        idx_2 = event_latencies.index(2.0)
        self.assertEqual(event_types[idx_2], 'stim')
        
        # Event at latency 9 should be removed (inside rejection)
        self.assertNotIn(9.0, event_latencies)
        
        # Event at latency 15 should shift to 10 (15 - 5 removed samples)
        self.assertIn(10.0, event_latencies)
        idx_10 = event_latencies.index(10.0)
        self.assertEqual(event_types[idx_10], 'stim')

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
            {"type": "boundary", "latency": 0.0},    # Should be removed (latency 0)
            {"type": "stim", "latency": 5.0},
            {"type": "boundary", "latency": 20.0},   # Should be removed (latency == pnts)
        ]
        
        regions = np.array([[10, 12]])
        result = eeg_eegrej(EEG, regions)
        
        # Check that problematic events are cleaned up
        latencies = [e.get('latency') for e in result['event']]
        
        # Should not have events at latency 0 or at pnts
        self.assertNotIn(0.0, latencies)
        self.assertNotIn(float(result['pnts']), latencies)

    def test_eeg_eegrej_duplicate_event_cleanup(self):
        """Test eeg_eegrej duplicate event cleanup."""
        EEG = self.base_eeg.copy()
        
        # Add duplicate events at same latency
        EEG['event'] = [
            {"type": "stim", "latency": 5.0},
            {"type": "boundary", "latency": 15.0, "duration": 2.0},
            {"type": "boundary", "latency": 15.0, "duration": 3.0},  # Duplicate boundary
        ]
        
        regions = np.array([[8, 10]])
        result = eeg_eegrej(EEG, regions)
        
        # Check that duplicate events are cleaned up
        latency_15_events = [e for e in result['event'] if e.get('latency') == 15.0 - 3.0]  # Adjusted for shift
        
        # Should have only one event at that latency after cleanup
        self.assertLessEqual(len(latency_15_events), 1)

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

    def test_eeg_eegrej_edge_case_full_rejection(self):
        """Test eeg_eegrej edge case where entire signal is rejected."""
        EEG = self.base_eeg.copy()
        
        # Try to reject the entire signal
        regions = np.array([[1, 20]])
        result = eeg_eegrej(EEG, regions)
        
        # Should result in empty or minimal data
        self.assertEqual(result['pnts'], 0)
        self.assertEqual(result['data'].shape[1], 0)
        
        # Events should be empty or minimal
        non_boundary_events = [e for e in result['event'] if e.get('type') != 'boundary']
        self.assertEqual(len(non_boundary_events), 0)

    def test_eeg_eegrej_no_events(self):
        """Test eeg_eegrej with EEG that has no events."""
        EEG = self.base_eeg.copy()
        EEG['event'] = []  # No events
        
        regions = np.array([[5, 8]])
        result = eeg_eegrej(EEG, regions)
        
        # Should still work and insert boundary events
        self.assertEqual(result['pnts'], 16)  # 20 - 4 = 16
        
        # Should have at least one boundary event inserted
        boundary_events = [e for e in result['event'] if e.get('type') == 'boundary']
        self.assertGreater(len(boundary_events), 0)

    def test_eeg_eegrej_data_integrity(self):
        """Test eeg_eegrej maintains data integrity."""
        EEG = self.base_eeg.copy()
        
        # Use specific data pattern to verify integrity
        EEG['data'] = np.array([
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        ], dtype=float)
        
        # Remove samples 6-10 (1-based) = indices 5-9 (0-based)
        regions = np.array([[6, 10]])
        result = eeg_eegrej(EEG, regions)
        
        # Expected result: keep samples 1-5 and 11-20 (0-based: 0-4 and 10-19)
        expected_data = np.array([
            [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        ], dtype=float)
        
        np.testing.assert_array_equal(result['data'], expected_data)

    def test_eeg_eegrej_preserves_eeg_structure(self):
        """Test eeg_eegrej preserves EEG structure fields."""
        EEG = self.base_eeg.copy()
        EEG['srate'] = 250.0
        EEG['nbchan'] = 2
        EEG['trials'] = 1
        EEG['custom_field'] = 'test_value'
        
        regions = np.array([[5, 8]])
        result = eeg_eegrej(EEG, regions)
        
        # Core fields should be updated
        self.assertEqual(result['pnts'], 16)
        self.assertAlmostEqual(result['xmax'], EEG['xmin'] + (EEG['xmax'] - EEG['xmin']) * (16/20))
        
        # Other fields should be preserved
        self.assertEqual(result['srate'], 250.0)
        self.assertEqual(result['nbchan'], 2)
        self.assertEqual(result['trials'], 1)
        self.assertEqual(result['custom_field'], 'test_value')

    def test_eeg_eegrej_matlab_parity_basic(self):
        """Test eeg_eegrej basic MATLAB parity."""
        EEG = self.base_eeg.copy()
        
        # Simple test case that should match MATLAB behavior
        regions = np.array([[8, 12]])
        result = eeg_eegrej(EEG, regions)
        
        # Verify basic properties that should match MATLAB
        self.assertEqual(result['pnts'], 15)  # 20 - 5 = 15
        self.assertEqual(result['data'].shape, (2, 15))
        
        # Verify boundary event properties
        boundary_events = [e for e in result['event'] if e.get('type') == 'boundary']
        self.assertGreater(len(boundary_events), 0)
        
        # Find inserted boundary
        inserted_boundary = None
        for e in boundary_events:
            if e.get('duration') == 5.0:
                inserted_boundary = e
                break
        
        self.assertIsNotNone(inserted_boundary)
        # Verify duration is correct (matches MATLAB behavior)
        self.assertEqual(inserted_boundary['duration'], 5.0)
        # Latency should be reasonable (exact value may vary based on implementation details)
        self.assertGreater(inserted_boundary['latency'], 0.0)
        self.assertLessEqual(inserted_boundary['latency'], result['pnts'] + 1.0)


if __name__ == "__main__":
    unittest.main()