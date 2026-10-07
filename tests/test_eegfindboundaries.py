# test_eeg_findboundaries.py
import unittest
import os

if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
    raise unittest.SkipTest("MATLAB not available")
import numpy as np

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.eeg_findboundaries import eeg_findboundaries


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
            ],
        }
        py_out = eeg_findboundaries(EEG=EEG)
        ml_out = self.eeglab.eeg_findboundaries(EEG)
        # MATLAB returns column vector, Python returns list - compare flattened versions
        self.assertTrue(np.array_equal(np.array(py_out), ml_out.flatten() - 1))

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
        self.assertTrue(np.array_equal(np.array(py_out), ml_out.flatten() - 1))


if __name__ == '__main__':
    unittest.main()
