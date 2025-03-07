import os
import unittest
from copy import deepcopy


import numpy as np

from eegprep import *


def ensure_file(url: str, fname: str):
    if not os.path.exists(fname):
        from urllib.request import urlretrieve
        urlretrieve(url, fname)


class TestCleanFlatlines(unittest.TestCase):

    web_url = 'https://sccntestdatasets.s3.us-east-2.amazonaws.com/FlankerTest.set'
    local_url = os.path.join(os.path.dirname(__file__), '../data/FlankerTest.set')

    def setUp(self):
        # download file
        ensure_file(self.web_url, self.local_url)

        self.EEG = pop_loadset(self.local_url)
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
        self.myfile = '/home/christian/Intheon/NeuroPype/sample-datasets/neuropype/FlankerTest.set'
        self.EEG = pop_loadset(self.myfile)
        self.expected = pop_loadset('/home/christian/Intheon/Projects/eegprep-refdata/dedrift.set')

    def test_clean_drifts(self):
        # test just the fft_filtfilt function
        cleaned1 = clean_drifts(deepcopy(self.EEG), [3, 4], 75, method='fir')
        np.testing.assert_almost_equal(cleaned1['data'], self.expected['data'],
                                       err_msg='clean_drifts() failed')
        cleaned2 = clean_drifts(deepcopy(self.EEG), [3, 4], 75, method='fft')
        np.testing.assert_almost_equal(cleaned1['data'], cleaned2['data'],
                                       err_msg='clean_drifts() FFT mode test failed')
        pass


class TestCleanChannelsNoLocs(unittest.TestCase):

    def setUp(self):
        self.myfile = '/home/christian/Intheon/NeuroPype/sample-datasets/neuropype/EmotionValence.set'
        self.EEG = pop_loadset(self.myfile)
        self.expected = pop_loadset('/home/christian/Intheon/Projects/eegprep-refdata/cln_chn_nl_0.9.set')

    def test_clean_channels(self):
        cleaned, _ = clean_channels_nolocs(deepcopy(self.EEG), 0.9)
        np.testing.assert_almost_equal(cleaned['data'], self.expected['data'],
                                       err_msg='clean_channels_nolocs() failed')
        pass


if __name__ == "__main__":
    unittest.main()
