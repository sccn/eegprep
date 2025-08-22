# test_eeg_findboundaries.py
import unittest
import numpy as np

from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_findboundaries import eeg_findboundaries


# class TestEegFindBoundariesParity(unittest.TestCase):

#     def setUp(self):
#         self.eeglab = get_eeglab('MAT')

#     def test_parity_string_types_struct(self):
#         EEG = {
#             'setname': 'testset',
#             'event': [
#                 {'type': 'boundary', 'latency': 0},
#                 {'type': 'stim', 'latency': 20},
#                 {'type': 'boundary123', 'latency': 30},
#                 {'type': 'resp', 'latency': 40},
#             ]
#         }
#         py_out = eeg_findboundaries(EEG=EEG)
#         ml_out = self.eeglab.eeg_findboundaries(EEG)
#         # MATLAB returns column vector, Python returns list - compare flattened versions
#         self.assertTrue(np.array_equal(np.array(py_out), ml_out.flatten()-1))

    # def test_parity_string_types_eventlist(self):
    #     tmpevent = [
    #         {'type': 'boundary'},
    #         {'type': 'foo'},
    #         {'type': 'bar'},
    #         {'type': 'boundary_something'},
    #     ]
    #     py_out = py_eeg_findboundaries(EEG=tmpevent)
    #     ml_out = self.eeglab.eeg_findboundaries(tmpevent)
    #     # MATLAB returns column vector, Python returns list - compare flattened versions
    #     self.assertTrue(np.array_equal(np.array(py_out), ml_out.flatten()))


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


if __name__ == '__main__':
    unittest.main()