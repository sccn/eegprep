# test_eeg_decodechan_unittest.py
import unittest

# Bring in the function under test
from eegprep import eeg_decodechan

class TestEEGDecodeChan(unittest.TestCase):
    def setUp(self):
        self.chanlocs = [
            {"labels": "Fz", "type": "EEG"},
            {"labels": "Cz", "type": "EEG"},
            {"labels": "Pz", "type": "EEG"},
            {"labels": "EOG", "type": "EOG"},
        ]

    def test_match_by_labels_strings(self):
        inds, labs = eeg_decodechan(self.chanlocs, ["fz", "cz"])
        self.assertEqual(inds, [1, 2])
        self.assertEqual(labs, ["Fz", "Cz"])

    def test_numeric_indices_input(self):
        inds, labs = eeg_decodechan(self.chanlocs, [2, 3])
        self.assertEqual(inds, [2, 3])
        self.assertEqual(labs, ["Cz", "Pz"])

    def test_numeric_passthrough_when_chanlocs_empty(self):
        inds, labs = eeg_decodechan([], [1, 3])
        self.assertEqual(inds, [1, 3])
        self.assertEqual(labs, [1, 3])

    def test_mixed_numeric_and_names(self):
        inds, labs = eeg_decodechan(self.chanlocs, ["Pz", 1])
        self.assertEqual(inds, [1, 3])
        self.assertEqual(labs, ["Fz", "Pz"])

    def test_ignoremissing_false_raises(self):
        with self.assertRaisesRegex(ValueError, "Channel 'fpz' not found"):
            eeg_decodechan(self.chanlocs, ["Fpz"])

    def test_ignoremissing_true_skips_missing(self):
        inds, labs = eeg_decodechan(self.chanlocs, ["Fpz", "cz"], ignoremissing=True)
        self.assertEqual(inds, [2])
        self.assertEqual(labs, ["Cz"])

    def test_match_on_type_field(self):
        inds, types = eeg_decodechan(self.chanlocs, ["eeg"], field="type")
        self.assertEqual(inds, [1, 2, 3])
        self.assertEqual(types, ["EEG", "EEG", "EEG"])

    def test_wrapper_dict_with_key_chanlocs(self):
        wrapper = {"chanlocs": self.chanlocs}
        inds, labs = eeg_decodechan(wrapper, ["cz"])
        self.assertEqual(inds, [2])
        self.assertEqual(labs, ["Cz"])

    def test_duplicate_label_returns_all_matches(self):
        dup = [
            {"labels": "M1", "type": "REF"},
            {"labels": "Cz", "type": "EEG"},
            {"labels": "M1", "type": "REF"},
        ]
        inds, labs = eeg_decodechan(dup, ["m1"])
        self.assertEqual(inds, [1, 3])
        self.assertEqual(labs, ["M1", "M1"])

    def test_out_of_range_indices_raise(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            eeg_decodechan(self.chanlocs, [0])
        with self.assertRaisesRegex(ValueError, "out of range"):
            eeg_decodechan(self.chanlocs, [len(self.chanlocs) + 1])

    def test_invalid_field_raises(self):
        with self.assertRaisesRegex(ValueError, "Field 'bad' not found"):
            eeg_decodechan(self.chanlocs, ["cz"], field="bad")

    def test_non_iterable_chanstr_raises_typeerror(self):
        with self.assertRaises(TypeError):
            eeg_decodechan(self.chanlocs, 123)


if __name__ == "__main__":
    unittest.main()