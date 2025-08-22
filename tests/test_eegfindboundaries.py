# test_eeg_findboundaries.py
import unittest
import numpy as np

from eegprep.eeglabcompat import get_eeglab
from eegprep.eeg_findboundaries import eeg_findboundaries as py_eeg_findboundaries


# class TestEegFindBoundariesParity(unittest.TestCase):

#     def setUp(self):
#         self.eeglab = get_eeglab('MAT')

#     def test_parity_string_types_struct(self):
#         EEG = {
#             'setname': 'testset',
#             'event': [
#                 {'type': 'boundary'},
#                 {'type': 'stim'},
#                 {'type': 'boundary123'},
#                 {'type': 'resp'},
#             ]
#         }
#         py_out = py_eeg_findboundaries(EEG=EEG)
#         ml_out = self.eeglab.eeg_findboundaries(EEG)
#         self.assertTrue(np.array_equal(np.array(py_out), np.array(ml_out)))

#     def test_parity_string_types_eventlist(self):
#         tmpevent = [
#             {'type': 'boundary'},
#             {'type': 'foo'},
#             {'type': 'bar'},
#             {'type': 'boundary_something'},
#         ]
#         py_out = py_eeg_findboundaries(EEG=tmpevent)
#         ml_out = self.eeglab.eeg_findboundaries(tmpevent)
#         self.assertTrue(np.array_equal(np.array(py_out), np.array(ml_out)))


class TestEegFindBoundariesFunctional(unittest.TestCase):

    def test_returns_empty_on_empty_input(self):
        self.assertEqual(py_eeg_findboundaries(EEG={}), [])
        self.assertEqual(py_eeg_findboundaries(EEG=[]), [])

    def test_missing_type_field(self):
        tmpevent = [{'latency': 1.0}]
        self.assertEqual(py_eeg_findboundaries(EEG=tmpevent), [])

    def test_string_match_prefix(self):
        tmpevent = [
            {'type': 'boundary'},
            {'type': 'boundary_extra'},
            {'type': 'notboundary'},
            {'type': 'BOUNDARY'},  # case sensitive like MATLAB
        ]
        out = py_eeg_findboundaries(EEG=tmpevent)
        self.assertEqual(out, [0, 1])

    def test_numeric_option_boundary99_true(self):
        # Toggle the EEG_OPTIONS option_boundary99 for this test
        from eegprep.eeg_options import EEG_OPTIONS
        old = EEG_OPTIONS['option_boundary99']
        EEG_OPTIONS['option_boundary99'] = 1  # Use 1 to match MATLAB convention
        try:
            tmpevent = [{'type': -99}, {'type': 1}, {'type': -99}]
            out = py_eeg_findboundaries(EEG=tmpevent)
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
            out = py_eeg_findboundaries(EEG=tmpevent)
            self.assertEqual(out, [])
        finally:
            EEG_OPTIONS['option_boundary99'] = old

    def test_struct_vs_eventlist_path(self):
        EEG = {
            'setname': 'test',
            'event': [{'type': 'stim'}, {'type': 'boundary'}]
        }
        out_struct = py_eeg_findboundaries(EEG=EEG)
        out_list = py_eeg_findboundaries(EEG=EEG['event'])
        self.assertEqual(out_struct, out_list)


if __name__ == '__main__':
    unittest.main()