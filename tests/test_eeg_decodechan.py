# test_eeg_decodechan_unittest.py
import unittest
import numpy as np

# Bring in the function under test
from eegprep import eeg_decodechan
from tests.eeglab_tests import eeglab_test


@eeglab_test("unittesting_popfunc/eeg_decodechan/popfunc_eeg_decodechan_wrapperTest.m", "test_test_eeg_decodechan")
def test_reference_decodechan_original_channel_names(eeglab_backend, eeglab_suite_root):
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    eeglab_backend("eeg_decodechan", eeg["chanlocs"], np.array([["cz", "pz", "fz"]], dtype=object), nargout=2)
    eeglab_backend("eeg_decodechan", eeg["chanlocs"], "cz pz fz", nargout=2)


class TestEEGDecodeChan(unittest.TestCase):
    def setUp(self):
        self.chanlocs = [
            {"labels": "Fz", "type": "EEG"},
            {"labels": "Cz", "type": "EEG"},
            {"labels": "Pz", "type": "EEG"},
            {"labels": "EOG", "type": "EOG"},
        ]

    def test_mixed_numeric_and_names(self):
        inds, labs = eeg_decodechan(self.chanlocs, ["Pz", 0])
        self.assertEqual(inds, [0, 2])
        self.assertEqual(labs, ["Fz", "Pz"])

    def test_ignoremissing_true_skips_missing(self):
        inds, labs = eeg_decodechan(self.chanlocs, ["Fpz", "cz"], ignoremissing=True)
        self.assertEqual(inds, [1])
        self.assertEqual(labs, ["Cz"])

    def test_match_on_type_field(self):
        inds, types = eeg_decodechan(self.chanlocs, ["eeg"], field="type")
        self.assertEqual(inds, [0, 1, 2])
        self.assertEqual(types, ["EEG", "EEG", "EEG"])

    def test_cell_and_space_separated_channel_labels_match(self):
        cell_indices, cell_labels = eeg_decodechan(self.chanlocs, ["cz", "pz", "fz"])
        text_indices, text_labels = eeg_decodechan(self.chanlocs, "cz pz fz")

        self.assertEqual(cell_indices, [0, 1, 2])
        self.assertEqual(cell_labels, ["Fz", "Cz", "Pz"])
        self.assertEqual(text_indices, cell_indices)
        self.assertEqual(text_labels, cell_labels)


if __name__ == "__main__":
    unittest.main()
