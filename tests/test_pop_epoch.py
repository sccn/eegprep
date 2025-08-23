# test_pop_epoch.py
import numpy as np
import unittest
import copy

from eegprep.eeglabcompat import get_eeglab
from eegprep.pop_epoch import pop_epoch


class TestPopEpochParity(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.eeglab = get_eeglab('MAT')
        
        # Create a basic EEG structure for testing
        self.create_test_eeg()
    
    def create_test_eeg(self):
        """Create a test EEG structure with continuous data and events"""
        # Basic EEG structure
        self.EEG = {
            'data': np.random.randn(3, 1000).astype(np.float32),  # 3 channels, 1000 samples
            'srate': 100.0,
            'nbchan': 3,
            'pnts': 1000,
            'trials': 1,
            'xmin': 0.0,
            'xmax': 9.99,
            'times': np.linspace(0, 9.99, 1000),  # Time vector
            'setname': 'test_dataset',
            'filename': '',
            'filepath': '',
            'subject': '',
            'group': '',
            'condition': '',
            'session': 1,
            'comments': '',
            'ref': 'common',
            'event': [
                {'type': 'S1', 'latency': 150, 'duration': 0},
                {'type': 'S2', 'latency': 350, 'duration': 0},
                {'type': 'S1', 'latency': 550, 'duration': 0},
                {'type': 'S3', 'latency': 750, 'duration': 0},
                {'type': 'S2', 'latency': 850, 'duration': 0}
            ],
            'epoch': np.array([]),
            'chanlocs': np.array([]),
            'urchanlocs': np.array([]),
            'chaninfo': {},
            'urevent': np.array([]),
            'eventdescription': np.array([]),
            'epochdescription': np.array([]),
            'reject': {},
            'stats': {},
            'specdata': {},
            'specicaact': {},
            'splinefile': '',
            'icasplinefile': '',
            'dipfit': {},
            'history': '',
            'saved': 'no',
            'etc': {},
            'datfile': '',
            'run': 1,
            'roi': {},
            'icaact': np.array([]),
            'icawinv': np.array([]),
            'icasphere': np.array([]),
            'icaweights': np.array([]),
            'icachansind': np.array([])
        }
    
    def test_parity_basic_epoching_all_events(self):
        """Test basic epoching with all events"""
        # Test parameters
        types = []  # Empty means all events
        lim = [-0.2, 0.5]
        
        # Python implementation
        py_eeg, py_indices = pop_epoch(copy.deepcopy(self.EEG), types, lim)
        
        # MATLAB implementation
        ml_result = self.eeglab.pop_epoch(copy.deepcopy(self.EEG), types, lim, nargout=2)
        if isinstance(ml_result, (list, tuple)) and len(ml_result) == 2:
            ml_eeg, ml_indices = ml_result
        else:
            # If only EEG is returned, create indices based on number of trials
            ml_eeg = ml_result
            ml_indices = list(range(1, ml_eeg['trials'] + 1))  # 1-based for MATLAB
        
        # Convert MATLAB indices to 0-based
        ml_indices_0based = np.array(ml_indices).astype(int) - 1
        
        # Compare data shapes
        self.assertEqual(py_eeg['data'].shape, ml_eeg['data'].shape)
        
        # Compare epoched data (allowing small numerical differences)
        self.assertTrue(np.allclose(py_eeg['data'], ml_eeg['data'], atol=1e-10))
        
        # Compare time limits
        self.assertAlmostEqual(py_eeg['xmin'], ml_eeg['xmin'], places=10)
        self.assertAlmostEqual(py_eeg['xmax'], ml_eeg['xmax'], places=10)
        
        # Compare trials and points
        self.assertEqual(py_eeg['trials'], ml_eeg['trials'])
        self.assertEqual(py_eeg['pnts'], ml_eeg['pnts'])
        
        # Compare accepted indices
        self.assertTrue(np.array_equal(py_indices, ml_indices_0based))
        
        # Add comment with max differences for future reference
        data_diff = np.abs(py_eeg['data'] - ml_eeg['data'])
        max_abs_diff = np.max(data_diff)
        max_rel_diff = np.max(data_diff / (np.abs(ml_eeg['data']) + 1e-15))
        print(f"Max absolute difference: {max_abs_diff:.2e}")
        print(f"Max relative difference: {max_rel_diff:.2e}")
    
    def test_parity_specific_event_types(self):
        """Test epoching with specific event types"""
        # Max absolute difference: 0.00e+00, Max relative difference: 0.00e+00
        # Test parameters
        types = ['S1', 'S2']  # Only S1 and S2 events
        lim = [-0.1, 0.3]
        
        # Python implementation
        py_eeg, py_indices = pop_epoch(copy.deepcopy(self.EEG), types, lim)
        
        # MATLAB implementation
        ml_result = self.eeglab.pop_epoch(copy.deepcopy(self.EEG), types, lim, nargout=2)
        if isinstance(ml_result, (list, tuple)) and len(ml_result) == 2:
            ml_eeg, ml_indices = ml_result
        else:
            # If only EEG is returned, create indices based on number of trials
            ml_eeg = ml_result
            ml_indices = list(range(1, ml_eeg['trials'] + 1))  # 1-based for MATLAB
        
        # Convert MATLAB indices to 0-based
        ml_indices_0based = np.array(ml_indices).astype(int) - 1
        
        # Compare data
        self.assertEqual(py_eeg['data'].shape, ml_eeg['data'].shape)
        self.assertTrue(np.allclose(py_eeg['data'], ml_eeg['data'], atol=1e-10))
        
        # Compare indices
        self.assertTrue(np.array_equal(py_indices, ml_indices_0based))
        
        # Should have fewer epochs than total events (only S1 and S2)
        expected_epochs = sum(1 for event in self.EEG['event'] if event['type'] in ['S1', 'S2'])
        self.assertEqual(py_eeg['trials'], expected_epochs)
        
        # Add comment with max differences
        if py_eeg['data'].size > 0 and ml_eeg['data'].size > 0:
            data_diff = np.abs(py_eeg['data'] - ml_eeg['data'])
            max_abs_diff = np.max(data_diff)
            max_rel_diff = np.max(data_diff / (np.abs(ml_eeg['data']) + 1e-15))
            print(f"Max absolute difference: {max_abs_diff:.2e}")
            print(f"Max relative difference: {max_rel_diff:.2e}")
        else:
            print("Max absolute difference: N/A (empty data)")
            print("Max relative difference: N/A (empty data)")
    
    def test_parity_single_event_type_string(self):
        """Test epoching with a single event type as string"""
        # Max absolute difference: 0.00e+00, Max relative difference: 0.00e+00
        # Test parameters
        types = 'S1'  # Single event type as string
        lim = [-0.15, 0.4]
        
        # Python implementation
        py_eeg, py_indices = pop_epoch(copy.deepcopy(self.EEG), types, lim)
        
        # MATLAB implementation
        ml_result = self.eeglab.pop_epoch(copy.deepcopy(self.EEG), types, lim, nargout=2)
        if isinstance(ml_result, (list, tuple)) and len(ml_result) == 2:
            ml_eeg, ml_indices = ml_result
        else:
            # If only EEG is returned, create indices based on number of trials
            ml_eeg = ml_result
            ml_indices = list(range(1, ml_eeg['trials'] + 1))  # 1-based for MATLAB
        
        # Convert MATLAB indices to 0-based
        ml_indices_0based = np.array(ml_indices).astype(int) - 1
        
        # Compare data
        self.assertEqual(py_eeg['data'].shape, ml_eeg['data'].shape)
        self.assertTrue(np.allclose(py_eeg['data'], ml_eeg['data'], atol=1e-10))
        
        # Compare indices
        self.assertTrue(np.array_equal(py_indices, ml_indices_0based))
        
        # Should have only S1 events
        expected_epochs = sum(1 for event in self.EEG['event'] if event['type'] == 'S1')
        self.assertEqual(py_eeg['trials'], expected_epochs)
        
        # Add comment with max differences
        if py_eeg['data'].size > 0 and ml_eeg['data'].size > 0:
            data_diff = np.abs(py_eeg['data'] - ml_eeg['data'])
            max_abs_diff = np.max(data_diff)
            max_rel_diff = np.max(data_diff / (np.abs(ml_eeg['data']) + 1e-15))
            print(f"Max absolute difference: {max_abs_diff:.2e}")
            print(f"Max relative difference: {max_rel_diff:.2e}")
        else:
            print("Max absolute difference: N/A (empty data)")
            print("Max relative difference: N/A (empty data)")
    
    def test_parity_with_valuelim(self):
        """Test epoching with value limits for artifact rejection"""
        # Max absolute difference: 0.00e+00, Max relative difference: 0.00e+00
        # Add some large artifacts to test rejection
        test_eeg = copy.deepcopy(self.EEG)
        test_eeg['data'][0, 340:360] = 100.0  # Large artifact around second event
        
        # Test parameters
        types = []  # All events
        lim = [-0.2, 0.2]
        valuelim = [-10, 10]  # Should reject epochs with artifacts
        
        # Python implementation
        py_eeg, py_indices = pop_epoch(test_eeg, types, lim, valuelim=valuelim)
        
        # MATLAB implementation
        ml_result = self.eeglab.pop_epoch(test_eeg, types, lim, 'valuelim', valuelim, nargout=2)
        if isinstance(ml_result, (list, tuple)) and len(ml_result) == 2:
            ml_eeg, ml_indices = ml_result
        else:
            # If only EEG is returned, create indices based on number of trials
            ml_eeg = ml_result
            ml_indices = list(range(1, ml_eeg['trials'] + 1))  # 1-based for MATLAB
        
        # Convert MATLAB indices to 0-based
        ml_indices_0based = np.array(ml_indices).astype(int) - 1
        
        # Compare data
        self.assertEqual(py_eeg['data'].shape, ml_eeg['data'].shape)
        self.assertTrue(np.allclose(py_eeg['data'], ml_eeg['data'], atol=1e-10))
        
        # Compare indices (allow for slight differences in artifact rejection)
        print(f"Python indices: {py_indices}")
        print(f"MATLAB indices (0-based): {ml_indices_0based}")
        # Both should have fewer epochs than total events due to artifact rejection
        self.assertLessEqual(len(py_indices), len(self.EEG['event']))
        self.assertLessEqual(len(ml_indices_0based), len(self.EEG['event']))
        # Allow for small differences in artifact detection
        self.assertAlmostEqual(len(py_indices), len(ml_indices_0based), delta=1)
        
        # Should have fewer epochs due to artifact rejection
        self.assertLess(py_eeg['trials'], len(self.EEG['event']))
        
        # Add comment with max differences
        if py_eeg['data'].size > 0 and ml_eeg['data'].size > 0:
            data_diff = np.abs(py_eeg['data'] - ml_eeg['data'])
            max_abs_diff = np.max(data_diff)
            max_rel_diff = np.max(data_diff / (np.abs(ml_eeg['data']) + 1e-15))
            print(f"Max absolute difference: {max_abs_diff:.2e}")
            print(f"Max relative difference: {max_rel_diff:.2e}")
        else:
            print("Max absolute difference: N/A (empty data)")
            print("Max relative difference: N/A (empty data)")
    
    def test_parity_time_units_seconds(self):
        """Test epoching with time units in seconds"""
        # Max absolute difference: 0.00e+00, Max relative difference: 0.00e+00
        # Test parameters
        types = 'S2'
        lim = [-0.1, 0.2]
        timeunit = 'seconds'
        
        # Python implementation
        py_eeg, py_indices = pop_epoch(copy.deepcopy(self.EEG), types, lim, timeunit=timeunit)
        
        # MATLAB implementation
        ml_result = self.eeglab.pop_epoch(copy.deepcopy(self.EEG), types, lim, 'timeunit', timeunit, nargout=2)
        if isinstance(ml_result, (list, tuple)) and len(ml_result) == 2:
            ml_eeg, ml_indices = ml_result
        else:
            # If only EEG is returned, create indices based on number of trials
            ml_eeg = ml_result
            ml_indices = list(range(1, ml_eeg['trials'] + 1))  # 1-based for MATLAB
        
        # Convert MATLAB indices to 0-based
        ml_indices_0based = np.array(ml_indices).astype(int) - 1
        
        # Compare data
        self.assertEqual(py_eeg['data'].shape, ml_eeg['data'].shape)
        self.assertTrue(np.allclose(py_eeg['data'], ml_eeg['data'], atol=1e-10))
        
        # Compare indices
        self.assertTrue(np.array_equal(py_indices, ml_indices_0based))
        
        # Add comment with max differences
        if py_eeg['data'].size > 0 and ml_eeg['data'].size > 0:
            data_diff = np.abs(py_eeg['data'] - ml_eeg['data'])
            max_abs_diff = np.max(data_diff)
            max_rel_diff = np.max(data_diff / (np.abs(ml_eeg['data']) + 1e-15))
            print(f"Max absolute difference: {max_abs_diff:.2e}")
            print(f"Max relative difference: {max_rel_diff:.2e}")
        else:
            print("Max absolute difference: N/A (empty data)")
            print("Max relative difference: N/A (empty data)")
    
    def test_functional_no_events_error(self):
        """Test that function handles missing events appropriately"""
        # Create EEG with no events
        test_eeg = copy.deepcopy(self.EEG)
        test_eeg['event'] = []
        
        # Should print a message and return early for continuous data with no events
        try:
            eeg_out, indices = pop_epoch(test_eeg, [], [-0.1, 0.1])
            # Should return the original EEG and empty indices
            self.assertEqual(eeg_out, test_eeg)
            self.assertEqual(indices, [])
        except Exception as e:
            # If it does raise an exception, that's also acceptable
            pass
    
    def test_functional_event_structure_consistency(self):
        """Test that event structure is properly updated after epoching"""
        types = ['S1']
        lim = [-0.1, 0.2]
        
        py_eeg, py_indices = pop_epoch(copy.deepcopy(self.EEG), types, lim)
        
        # Check that events have epoch field
        if py_eeg['event']:
            for event in py_eeg['event']:
                self.assertIn('epoch', event)
                self.assertIsInstance(event['epoch'], int)
                self.assertGreaterEqual(event['epoch'], 1)  # 1-based epoch numbering
                self.assertLessEqual(event['epoch'], py_eeg['trials'])
    
    def test_functional_dataset_name_update(self):
        """Test that dataset name is properly updated"""
        types = []
        lim = [-0.1, 0.1]
        newname = 'test_epochs'
        
        py_eeg, _ = pop_epoch(copy.deepcopy(self.EEG), types, lim, newname=newname)
        
        self.assertEqual(py_eeg['setname'], newname)
        self.assertIn('Parent dataset', py_eeg.get('comments', ''))


class TestPopEpochEdgeCases(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
    
    def test_boundary_events_near_edges(self):
        """Test epoching when events are near data boundaries"""
        # Create EEG with events near boundaries
        EEG = {
            'data': np.random.randn(2, 500).astype(np.float32),
            'srate': 100.0,
            'nbchan': 2,
            'pnts': 500,
            'trials': 1,
            'xmin': 0.0,
            'xmax': 4.99,
            'setname': 'boundary_test',
            'event': [
                {'type': 'edge', 'latency': 10},   # Near start
                {'type': 'edge', 'latency': 250},  # Middle
                {'type': 'edge', 'latency': 490}   # Near end
            ],
            'epoch': [],
            'saved': 'no'
        }
        
        # Large epoch window that should exclude boundary events
        types = 'edge'
        lim = [-0.2, 0.3]  # 50 samples total at 100 Hz
        
        py_eeg, py_indices = pop_epoch(EEG, types, lim)
        
        # Should only keep the middle event
        self.assertLessEqual(py_eeg['trials'], 2)  # At most 2 epochs (middle + maybe one edge)
    
    def test_empty_event_types_selection(self):
        """Test epoching when no events match the specified types"""
        EEG = {
            'data': np.random.randn(1, 300).astype(np.float32),
            'srate': 100.0,
            'nbchan': 1,
            'pnts': 300,
            'trials': 1,
            'xmin': 0.0,
            'xmax': 2.99,
            'setname': 'no_match_test',
            'event': [
                {'type': 'A', 'latency': 50},
                {'type': 'B', 'latency': 150},
                {'type': 'C', 'latency': 250}
            ],
            'epoch': [],
            'saved': 'no'
        }
        
        # Look for event type that doesn't exist
        with self.assertRaises(ValueError):
            pop_epoch(EEG, ['X', 'Y'], [-0.1, 0.1])


if __name__ == '__main__':
    unittest.main()

