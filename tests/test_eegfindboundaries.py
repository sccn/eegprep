# test_eeg_findboundaries.py
import unittest
import numpy as np

from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_findboundaries import eeg_findboundaries


class TestEegFindBoundariesParity(unittest.TestCase):

    def setUp(self):
        self.eeglab = get_eeglab('MAT')

    def test_parity_string_types_struct(self):
        EEG = {
            'setname': 'testset',
            'event': [
                {'type': 'boundary', 'latency': 0},
                {'type': 'stim', 'latency': 20},
                {'type': 'boundary123', 'latency': 30},
                {'type': 'resp', 'latency': 40},
            ]
        }
        py_out = eeg_findboundaries(EEG=EEG)
        ml_out = self.eeglab.eeg_findboundaries(EEG)
        # MATLAB returns column vector, Python returns list - compare flattened versions
        self.assertTrue(np.array_equal(np.array(py_out), ml_out.flatten()-1))

    def test_parity_string_types_eventlist(self):
        tmpevent = [
            {'type': 'boundary'},
            {'type': 'foo'},
            {'type': 'bar'},
            {'type': 'boundary_something'},
        ]
        py_out = eeg_findboundaries(EEG=tmpevent)
        ml_out = self.eeglab.eeg_findboundaries(tmpevent)
        # MATLAB returns column vector, Python returns list - compare flattened versions
        self.assertTrue(np.array_equal(np.array(py_out), ml_out.flatten()-1))


class TestEegFindBoundariesFunctional(unittest.TestCase):

    def test_returns_empty_on_empty_input(self):
        self.assertEqual(eeg_findboundaries(EEG={}), [])
        self.assertEqual(eeg_findboundaries(EEG=[]), [])

    def test_missing_type_field(self):
        tmpevent = [{'latency': 1.0}]
        self.assertEqual(eeg_findboundaries(EEG=tmpevent), [])

    def test_string_match_prefix(self):
        tmpevent = [
            {'type': 'boundary'},
            {'type': 'boundary_extra'},
            {'type': 'notboundary'},
            {'type': 'BOUNDARY'},  # case sensitive like MATLAB
        ]
        out = eeg_findboundaries(EEG=tmpevent)
        self.assertEqual(out, [0, 1])

    def test_numeric_option_boundary99_true(self):
        # Toggle the EEG_OPTIONS option_boundary99 for this test
        from eegprep.eeg_options import EEG_OPTIONS
        old = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 1  # Use 1 to match MATLAB convention
        try:
            tmpevent = [{'type': -99}, {'type': 1}, {'type': -99}]
            out = eeg_findboundaries(EEG=tmpevent)
            self.assertEqual(out, [0, 2])
        finally:
            EEG_OPTIONS['option_boundary99'] = old

    def test_numeric_option_boundary99_false(self):
        # Toggle the EEG_OPTIONS option_boundary99 for this test
        from eegprep.eeg_options import EEG_OPTIONS
        old = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 0  # Use 0 to match MATLAB convention
        try:
            tmpevent = [{'type': -99}, {'type': 1}, {'type': -99}]
            out = eeg_findboundaries(EEG=tmpevent)   
            self.assertEqual(out, [])
        finally:
            EEG_OPTIONS['option_boundary99'] = old

    def test_struct_vs_eventlist_path(self):
        EEG = {
            'setname': 'test',
            'event': [{'type': 'stim'}, {'type': 'boundary'}]
        }
        out_struct = eeg_findboundaries(EEG=EEG)
        out_list = eeg_findboundaries(EEG=EEG['event'])
        self.assertEqual(out_struct, out_list)
        # Should find 'boundary' at index 1 (0-based)
        self.assertEqual(out_struct, [1])


class TestEegFindBoundariesEdgeCases(unittest.TestCase):
    """Test edge cases and additional functionality for eeg_findboundaries."""
    
    def test_none_input(self):
        """Test that None input returns empty list."""
        result = eeg_findboundaries(EEG=None)
        self.assertEqual(result, [])
    
    def test_empty_dict_input(self):
        """Test that empty dict input returns empty list."""
        result = eeg_findboundaries(EEG={})
        self.assertEqual(result, [])
    
    def test_empty_list_input(self):
        """Test that empty list input returns empty list."""
        result = eeg_findboundaries(EEG=[])
        self.assertEqual(result, [])
    
    def test_no_events_in_eeg_struct(self):
        """Test EEG struct without events."""
        EEG = {'setname': 'test'}  # Missing 'event' field
        result = eeg_findboundaries(EEG=EEG)
        self.assertEqual(result, [])
    
    def test_empty_events_in_eeg_struct(self):
        """Test EEG struct with empty events list."""
        EEG = {'setname': 'test', 'event': []}
        result = eeg_findboundaries(EEG=EEG)
        self.assertEqual(result, [])
    
    def test_single_event_dict_input(self):
        """Test single event dict (not in list) as input."""
        event = {'type': 'boundary', 'latency': 100}
        result = eeg_findboundaries(EEG=event)
        self.assertEqual(result, [0])  # Should find boundary at index 0
        
        # Non-boundary event
        event = {'type': 'stimulus', 'latency': 100}
        result = eeg_findboundaries(EEG=event)
        self.assertEqual(result, [])
    
    def test_events_without_type_field(self):
        """Test events that lack the 'type' field."""
        events = [
            {'latency': 100},  # No type field
            {'type': 'boundary', 'latency': 200},
            {'latency': 300}   # No type field
        ]
        result = eeg_findboundaries(EEG=events)
        self.assertEqual(result, [])  # Should return empty due to missing type in first event
    
    def test_mixed_type_fields(self):
        """Test events with mixed string and numeric types."""
        events = [
            {'type': 'boundary', 'latency': 100},
            {'type': 1, 'latency': 200},  # Numeric type
            {'type': 'boundary_end', 'latency': 300}
        ]
        result = eeg_findboundaries(EEG=events)
        # Should find string boundaries at indices 0 and 2
        self.assertEqual(result, [0, 2])
    
    def test_boundary_variations(self):
        """Test various boundary string patterns."""
        events = [
            {'type': 'boundary', 'latency': 100},
            {'type': 'boundary123', 'latency': 200},
            {'type': 'boundary_marker', 'latency': 300},
            {'type': 'boundarY', 'latency': 400},  # Different case
            {'type': 'not_boundary', 'latency': 500},  # Doesn't start with boundary
            {'type': 'BOUNDARY', 'latency': 600},  # All caps
        ]
        result = eeg_findboundaries(EEG=events)
        # Should find events that start with 'boundary' (case sensitive)
        # 'boundarY' doesn't start with lowercase 'boundary' - it's case sensitive
        self.assertEqual(result, [0, 1, 2])  # Only lowercase 'boundary' prefixes
    
    def test_no_boundaries_found(self):
        """Test case where no boundaries are found."""
        events = [
            {'type': 'stimulus', 'latency': 100},
            {'type': 'response', 'latency': 200},
            {'type': 'target', 'latency': 300}
        ]
        result = eeg_findboundaries(EEG=events)
        self.assertEqual(result, [])
    
    def test_multiple_adjacent_boundaries(self):
        """Test multiple adjacent boundary events."""
        events = [
            {'type': 'boundary', 'latency': 100},
            {'type': 'boundary', 'latency': 101},
            {'type': 'boundary', 'latency': 102},
            {'type': 'stimulus', 'latency': 200},
            {'type': 'boundary', 'latency': 300}
        ]
        result = eeg_findboundaries(EEG=events)
        self.assertEqual(result, [0, 1, 2, 4])
    
    def test_correct_indexing(self):
        """Test that correct 0-based indices are returned."""
        events = [
            {'type': 'start', 'latency': 0},      # Index 0
            {'type': 'boundary', 'latency': 100}, # Index 1 - should be found
            {'type': 'stimulus', 'latency': 200}, # Index 2
            {'type': 'boundary', 'latency': 300}, # Index 3 - should be found
            {'type': 'end', 'latency': 400}       # Index 4
        ]
        result = eeg_findboundaries(EEG=events)
        self.assertEqual(result, [1, 3])
    
    def test_boundary_labels_preserved(self):
        """Test that function correctly identifies boundaries without modifying input."""
        events = [
            {'type': 'boundary', 'latency': 100, 'duration': 0},
            {'type': 'stimulus', 'latency': 200, 'code': 1}
        ]
        original_events = [ev.copy() for ev in events]  # Deep copy
        
        result = eeg_findboundaries(EEG=events)
        
        # Check result
        self.assertEqual(result, [0])
        # Check that original events weren't modified
        self.assertEqual(events, original_events)
    
    def test_numeric_boundary99_enabled(self):
        """Test numeric boundary detection with option_boundary99 enabled."""
        from eegprep.eeg_options import EEG_OPTIONS
        old_value = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 1
        
        try:
            events = [
                {'type': -99, 'latency': 100},   # Should be found
                {'type': 1, 'latency': 200},     # Should not be found
                {'type': -99, 'latency': 300},   # Should be found
                {'type': 'boundary', 'latency': 400}  # Should not be found (numeric mode)
            ]
            result = eeg_findboundaries(EEG=events)
            self.assertEqual(result, [0, 2])
        finally:
            EEG_OPTIONS['option_boundary99'] = old_value
    
    def test_numeric_boundary99_disabled(self):
        """Test numeric boundary detection with option_boundary99 disabled."""
        from eegprep.eeg_options import EEG_OPTIONS
        old_value = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 0
        
        try:
            events = [
                {'type': -99, 'latency': 100},   # Should not be found
                {'type': 1, 'latency': 200},     # Should not be found
                {'type': -99, 'latency': 300},   # Should not be found
            ]
            result = eeg_findboundaries(EEG=events)
            self.assertEqual(result, [])
        finally:
            EEG_OPTIONS['option_boundary99'] = old_value
    
    def test_mixed_string_numeric_types_with_boundary99(self):
        """Test mixed string and numeric types with boundary99 option."""
        from eegprep.eeg_options import EEG_OPTIONS
        old_value = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 1
        
        try:
            # When first event has string type, function uses string matching mode
            events = [
                {'type': 'boundary', 'latency': 100},  # String type - will be found (string mode)
                {'type': -99, 'latency': 200},          # Numeric type - not found in string mode
                {'type': 'stimulus', 'latency': 300}    # String type - not found
            ]
            result = eeg_findboundaries(EEG=events)
            self.assertEqual(result, [0])  # String 'boundary' found in string mode
            
            # When first event has numeric type, function uses numeric mode
            events_numeric_first = [
                {'type': -99, 'latency': 100},          # Numeric type - will be found (numeric mode)
                {'type': 'boundary', 'latency': 200},   # String type - not found in numeric mode
                {'type': -99, 'latency': 300}           # Numeric type - will be found
            ]
            result = eeg_findboundaries(EEG=events_numeric_first)
            self.assertEqual(result, [0, 2])  # Numeric -99s found in numeric mode
        finally:
            EEG_OPTIONS['option_boundary99'] = old_value
    
    def test_invalid_input_types(self):
        """Test invalid input types return empty list."""
        # String input
        result = eeg_findboundaries(EEG="invalid")
        self.assertEqual(result, [])
        
        # Integer input
        result = eeg_findboundaries(EEG=123)
        self.assertEqual(result, [])
        
        # List of non-dict items
        result = eeg_findboundaries(EEG=[1, 2, 3])
        self.assertEqual(result, [])
    
    def test_eeg_struct_without_setname(self):
        """Test EEG dict that has events but no setname field."""
        EEG = {
            'event': [
                {'type': 'boundary', 'latency': 100},
                {'type': 'stimulus', 'latency': 200}
            ]
            # Missing 'setname' field - should still be treated as event list
        }
        result = eeg_findboundaries(EEG=EEG)
        # Without both 'event' and 'setname', it's treated as event list
        # But since it's a dict with 'event', it should use the event list path
        self.assertEqual(result, [])  # Will be treated as event list, not EEG struct
    
    def test_eeg_struct_identification(self):
        """Test proper identification of EEG struct vs event list."""
        # Valid EEG struct (has both 'event' and 'setname')
        EEG_struct = {
            'setname': 'test',
            'event': [{'type': 'boundary', 'latency': 100}]
        }
        result = eeg_findboundaries(EEG=EEG_struct)
        self.assertEqual(result, [0])
        
        # Dict with only 'event' (treated as event list)
        event_dict = {
            'event': [{'type': 'boundary', 'latency': 100}]
        }
        result = eeg_findboundaries(EEG=event_dict)
        self.assertEqual(result, [])  # No 'type' field at top level
    
    def test_large_event_list(self):
        """Test with a large number of events for performance."""
        events = []
        boundary_indices = []
        
        # Create 1000 events with boundaries at every 100th position
        for i in range(1000):
            if i % 100 == 0:
                events.append({'type': 'boundary', 'latency': i * 10})
                boundary_indices.append(i)
            else:
                events.append({'type': f'event_{i}', 'latency': i * 10})
        
        result = eeg_findboundaries(EEG=events)
        self.assertEqual(result, boundary_indices)
        self.assertEqual(len(result), 10)  # Should find 10 boundaries
    
    def test_boundary_with_additional_fields(self):
        """Test that boundaries with additional fields are correctly identified."""
        events = [
            {
                'type': 'boundary',
                'latency': 100,
                'duration': 0,
                'code': 'boundary_marker',
                'description': 'Data discontinuity'
            },
            {
                'type': 'stimulus',
                'latency': 200,
                'code': 'S1'
            }
        ]
        result = eeg_findboundaries(EEG=events)
        self.assertEqual(result, [0])


if __name__ == '__main__':
    unittest.main()