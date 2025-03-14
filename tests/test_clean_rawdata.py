import os
import unittest
from copy import deepcopy


import numpy as np

from eegprep import *


# where the test resources
web_root = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/'
local_url = os.path.join(os.path.dirname(__file__), '../data/')


def ensure_file(fname: str) -> str:
    """Download a file if it does not exist and return the local path."""
    full_url = f"{web_root}{fname}"
    local_file = f"{local_url}{fname}"
    if not os.path.exists(fname):
        from urllib.request import urlretrieve
        urlretrieve(full_url, local_file)
    return local_file


class TestMATLABAccess(unittest.TestCase):

    def setUp(self):
        self.eeglab = eeglabcompat.get_eeglab('MAT')
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))

    def test_basic(self):
        result = self.eeglab.sqrt(4.0)
        self.assertEqual(result, 2.0, 'MATLAB sqrt() failed')

    def test_eeglab_presence(self):
        result = eeglabcompat.eeg_checkset(self.EEG, eeglab=self.eeglab)
        pass


class TestCleanFlatlines(unittest.TestCase):

    def setUp(self):
        # download file
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))
        self.EEG['data'][5, 1000:2000] = 3.5  # this should trigger
        self.EEG['data'][7, 2000:2100] = 4.5  # this should not (too short)
        self.EEG['data'][9, 3000:4000] = 5.5  # should trigger too
        self.EEG['data'][15, 5000:10000] = np.random.randn(5000)*1e-10  # should not trigger (too large amplitude)
        self.expected = self.EEG['data'][~np.isin(np.arange(self.EEG['nbchan']), [5, 9]), :]

    def test_clean_flatlines(self):
        cleaned_EEG = clean_flatlines(deepcopy(self.EEG), 3.5)
        np.testing.assert_equal(cleaned_EEG['data'], self.expected, err_msg='clean_flatlines() test failed')


class TestCleanDrifts(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('FlankerTest.set'))

    def test_clean_drifts(self):
        eeglab = eeglabcompat.get_eeglab('MAT')

        # compare vs MATLAB
        expected = eeglab.clean_drifts(self.EEG, [3, 4], 75)        
        cleaned1 = clean_drifts(deepcopy(self.EEG), [3, 4], 75, method='fir')
        np.testing.assert_almost_equal(cleaned1['data'], expected['data'],
                                       err_msg='clean_drifts() failed')
        
        # compare FFT vs FIR
        cleaned2 = clean_drifts(deepcopy(self.EEG), [3, 4], 75, method='fft')
        np.testing.assert_almost_equal(cleaned1['data'], cleaned2['data'],
                                       err_msg='clean_drifts() FFT mode test failed')



class TestCleanChannelsNoLocs(unittest.TestCase):

    def setUp(self):
        self.EEG = pop_loadset(ensure_file('EmotionValence.set'))

    def test_clean_channels(self):
        eeglab = eeglabcompat.get_eeglab('MAT')
        expected = eeglab.clean_channels_nolocs(self.EEG, 0.9)
        cleaned, _ = clean_channels_nolocs(deepcopy(self.EEG), 0.9)
        np.testing.assert_almost_equal(cleaned['data'], expected['data'],
                                       err_msg='clean_channels_nolocs() failed')
        pass


if __name__ == "__main__":
    unittest.main()
