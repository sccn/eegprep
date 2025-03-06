import unittest
from copy import deepcopy

import numpy as np

from eegprep import *


class TestCleanFlatlines(unittest.TestCase):

    def setUp(self):
        self.myfile = '/home/christian/Intheon/NeuroPype/sample-datasets/neuropype/FlankerTest.set'
        self.EEG = pop_loadset(self.myfile)
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


if __name__ == "__main__":
    unittest.main()
