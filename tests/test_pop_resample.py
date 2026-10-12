import os
import unittest

if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
    raise unittest.SkipTest("MATLAB not available")
import numpy as np
from eegprep import pop_loadset, pop_resample, pop_saveset

# where the test resources
local_url = os.path.join(os.path.dirname(__file__), '../sample_data/')


class TestPopResample(unittest.TestCase):
    def setUp(self):
        self.EEG = pop_loadset(os.path.join(local_url, 'eeglab_data_with_ica_tmp.set'))
        self.new_freq = 96  # New sampling rate in Hz

    def test_basic(self):
        """Test basic resampling functionality with different engines"""
        EEG_python = pop_resample(self.EEG.copy(), self.new_freq, engine='poly')
        pop_saveset(EEG_python, os.path.join(local_url, 'eeglab_data_with_ica_tmp_python.set'))

        EEG_matlab = pop_resample(self.EEG.copy(), self.new_freq, engine='matlab')
        pop_saveset(EEG_matlab, os.path.join(local_url, 'eeglab_data_with_ica_tmp_matlab.set'))

        self.assertEqual(EEG_python['srate'], self.new_freq, 'Python resampling failed')
        print("Sampling rate ok")
        self.assertEqual(EEG_matlab['srate'], self.new_freq, 'MATLAB resampling failed')
        print("Sampling rate ok")

        self.assertEqual(
            EEG_python['data'].shape[1], EEG_matlab['data'].shape[1], 'Data shapes differ between Python and MATLAB'
        )
        print("Data shape ok")
        np.testing.assert_allclose(
            EEG_python['data'],
            EEG_matlab['data'],
            rtol=1e-5,
            atol=1e-8,
            err_msg='Python and MATLAB results differ beyond tolerance',
        )
        print("Data comparison ok")


if __name__ == '__main__':
    unittest.main()
