import unittest
import numpy as np
from eegprep import pop_loadset, clean_flatlines


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
        cleaned_EEG = clean_flatlines(self.EEG, 3.5)
        np.testing.assert_equal(cleaned_EEG['data'], self.expected, err_msg='clean_flatlines() test failed')

if __name__ == "__main__":
    unittest.main()
