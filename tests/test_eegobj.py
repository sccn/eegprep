import unittest
import numpy as np
import copy
from eegprep.functions.eegobj.eegobj import EEGobj
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from tests.eeglab_tests import eeglab_test


@eeglab_test("unittesting_adminfunc/eegobj/adminfunc_eegobj_wrapperTest.m", "test_eegobj_simpletests")
def test_reference_eegobj_collection_operations(eeglab_backend, eeglab_suite_root, eeglab_options_directory, request):
    directory = str(eeglab_suite_root / "eeglab/sample_data")
    if request.config.getoption("--eeglab-backend") == "matlab":
        lengths = eeglab_backend("eegprep_source_eegobj", directory)
    else:
        eeglab_backend("pop_editoptions", "option_eegobject", 1.0, nargout=0)
        try:
            eeg = eeglab_backend("pop_loadset", "filename", "eeglab_data_epochs_ica.set", "filepath", directory)
            first = EEGobj(eeg)
            first.pnts = 3.0
            datasets = [first]
            lengths = [len(datasets)]
            datasets.append(copy.deepcopy(datasets[0]))
            lengths.append(len(datasets))
            empty = EEGobj({field: np.empty((0, 0)) for field in first.EEG})
            datasets.extend([copy.deepcopy(empty), copy.deepcopy(datasets[0])])
            lengths.append(len(datasets))
            del datasets[:3]
            lengths.append(len(datasets))
            datasets.extend([copy.deepcopy(empty), copy.deepcopy(datasets[0])])
            datasets.extend(copy.deepcopy(datasets[1:3]))
            lengths.append(len(datasets))
            datasets[1].filename = "test"
            datasets[0].chanlocs[0]["labels"] = "E1"
            datasets[-1] = copy.deepcopy(datasets[0])
            datasets.pop()
            lengths.append(len(datasets))
            lengths = np.array([lengths])
        finally:
            eeglab_backend("pop_editoptions", "option_eegobject", 0.0, nargout=0)
    np.testing.assert_array_equal(lengths, [[1, 2, 4, 1, 5, 4]])


# Helper function to create a dummy EEG dictionary
def create_test_eeg(n_channels=32, n_samples=1000, srate=250.0, n_trials=1):
    eeg = {
        'data': np.random.rand(n_channels, n_samples, n_trials),
        'nbchan': n_channels,
        'pnts': n_samples,
        'trials': n_trials,
        'srate': srate,
        'xmin': 0.0,
        'xmax': (n_samples - 1) / srate,
        'setname': 'test_dataset',
        'filename': 'test.set',
        'filepath': '/tmp',
        'event': [],
        'chanlocs': [{'labels': f'Ch{i + 1}', 'type': 'EEG'} for i in range(n_channels)],
        'icaact': [],  # Add missing field
        'icawinv': [],
        'icasphere': [],
        'icaweights': [],
        'icachansind': [],
        'chaninfo': {},  # Add missing chaninfo field
    }
    eeg = eeg_checkset(eeg)
    return eeg


class TestEEGobj(unittest.TestCase):
    def test_collection_assignment_and_field_mutation_use_python_list_semantics(self):
        first = EEGobj(create_test_eeg(n_channels=2, n_samples=3))
        datasets = [first]

        datasets.append(EEGobj(copy.deepcopy(first.EEG)))
        datasets.extend(EEGobj(copy.deepcopy(first.EEG)) for _ in range(2))
        del datasets[:3]
        datasets.extend(EEGobj(copy.deepcopy(first.EEG)) for _ in range(3))
        datasets[1].filename = "test"
        datasets[0].chanlocs[0]["labels"] = "E1"
        datasets.append(EEGobj(copy.deepcopy(datasets[0].EEG)))
        datasets.pop()

        self.assertEqual(len(datasets), 4)
        self.assertEqual(datasets[1].filename, "test")
        self.assertEqual(datasets[0].chanlocs[0]["labels"], "E1")

    def test_forward_pop_select_kwargs(self):
        eeg = create_test_eeg(n_channels=4, n_samples=50, srate=100.0, n_trials=5)
        obj = EEGobj(eeg)
        # keep trials 1..3
        out = obj.pop_select(trial=[1, 2, 3])
        self.assertEqual(out['trials'], 3)
        self.assertEqual(out['data'].shape[2], 3)
        # Ensure original object is not modified
        self.assertEqual(obj.EEG['trials'], 3)  # obj.EEG should be updated
        self.assertEqual(obj.EEG['data'].shape[2], 3)

    def test_forward_pop_select_keyval(self):
        eeg = create_test_eeg(n_channels=4, n_samples=50, srate=100.0, n_trials=3)
        obj = EEGobj(eeg)
        out = obj.pop_select('channel', [0, 1])
        self.assertEqual(out['nbchan'], 2)
        self.assertEqual(out['data'].shape[0], 2)
        # Ensure original object is not modified
        self.assertEqual(obj.EEG['nbchan'], 2)  # obj.EEG should be updated
        self.assertEqual(obj.EEG['data'].shape[0], 2)

    def test_getattr_unknown_name_raises_on_access(self):
        """Accessing an unknown name (e.g. a field typo) raises AttributeError immediately.

        A typo like obj.icawnv (for icaweights) must fail fast instead of
        silently returning a no-op callable that only errors when called.
        """
        eeg = create_test_eeg()
        obj = EEGobj(eeg)

        with self.assertRaises(AttributeError):
            obj.icawnv  # misspelled field name, not an eegprep function

        # hasattr reflects the same contract.
        self.assertFalse(hasattr(obj, 'not_a_real_field'))


if __name__ == '__main__':
    unittest.main()
