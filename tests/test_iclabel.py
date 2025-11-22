import os
import unittest
from copy import deepcopy
import numpy as np
from eegprep import *

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../data/')

@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestICLabelEngines(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set')) 

    def test_basic(self):
        EEG_python = iclabel(self.EEG, algorithm='default', engine=None)
        EEG_matlab = iclabel(self.EEG, algorithm='default', engine='matlab')        
        
        res1 = EEG_python['etc']['ic_classification']['ICLabel']['classifications'].flatten()
        res2 = EEG_matlab['etc']['ic_classification']['ICLabel']['classifications'].flatten()

        self.assertTrue(np.allclose(res1, res2, rtol=1e-5, atol=1e-8), 
                       'ICLabel results differ beyond tolerance')

if __name__ == '__main__':
    unittest.main()
