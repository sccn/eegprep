import os
import unittest
import numpy as np

from eegprep import pop_loadset
from eegprep.eegobj import EEGobj
from tests.fixtures import create_test_eeg


class TestEEGobj(unittest.TestCase):

    def test_init_from_dict_and_repr(self):
        eeg = create_test_eeg(n_channels=4, n_samples=100, srate=200.0, n_trials=1)
        obj = EEGobj(eeg)
        s = str(obj)
        self.assertIn('EEG |', s)
        self.assertIn('Channels        : 4', s)
        self.assertIn('Sampling freq.  : 200.0 Hz', s)

    def test_init_from_path(self):
        # Use a known small dataset from data if available
        candidate = os.path.join('data', 'eeglab_data_hdf5.set')
        if not os.path.exists(candidate):
            self.skipTest('sample dataset not available')
        obj = EEGobj(candidate)
        self.assertIsInstance(obj.EEG, dict)
        self.assertIn('data', obj.EEG)

    def test_forward_pop_select_kwargs(self):
        eeg = create_test_eeg(n_channels=4, n_samples=50, srate=100.0, n_trials=5)
        obj = EEGobj(eeg)
        # keep trials 1..3
        out = obj.pop_select(trial=[1, 2, 3])
        self.assertEqual(out['trials'], 3)
        self.assertEqual(out['data'].shape, (4, 50, 3))

    def test_forward_pop_select_keyval(self):
        eeg = create_test_eeg(n_channels=4, n_samples=50, srate=100.0, n_trials=3)
        obj = EEGobj(eeg)
        out = obj.pop_select('channel', [0, 1])
        self.assertEqual(out['nbchan'], 2)
        if out['trials'] > 1 or out['data'].ndim == 3:
            self.assertEqual(out['data'].shape[0], 2)
        else:
            self.assertEqual(out['data'].shape, (2, 50))


if __name__ == '__main__':
    unittest.main()


