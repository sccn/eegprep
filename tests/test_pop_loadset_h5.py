"""
Test suite for pop_loadset_h5.py - HDF5 EEGLAB file loading utilities.

This module tests the pop_loadset_h5 function which loads EEGLAB datasets
from HDF5 format files.
"""

import os
import unittest

if os.getenv('EEGPREP_SKIP_MATLAB') == '1':
    raise unittest.SkipTest("MATLAB not available")
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5
from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.functions.popfunc.pop_loadset import pop_loadset


@unittest.skipIf(os.getenv('EEGPREP_SKIP_MATLAB') == '1', "MATLAB not available")
class TestPopLoadsetH5Parity(unittest.TestCase):
    """Test parity between Python pop_loadset_h5 and MATLAB pop_loadset for real HDF5 files."""

    def setUp(self):
        """Set up MATLAB connection for parity testing."""
        try:
            from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

            self.eeglab = get_eeglab('MAT')
            self.matlab_available = True
        except Exception as e:
            print(f"MATLAB not available for parity testing: {e}")
            self.matlab_available = False

    def test_parity_continuous_data(self):
        """Test parity with continuous EEG data (single trial)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available for parity testing")

        filepath = 'sample_data/eeglab_data_hdf5.set'

        # Load with Python
        py_eeg = pop_loadset_h5(filepath)

        # Load with MATLAB
        data_dir = os.path.abspath('sample_data')
        ml_eeg = self.eeglab.pop_loadset('filename', 'eeglab_data_hdf5.set', 'filepath', data_dir)

        eeg_compare(py_eeg, ml_eeg)

    def test_parity_epoched_data(self):
        """Test parity with continuous EEG data (single trial)."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available for parity testing")

        filepath = 'sample_data/eeglab_data_epochs_ica_hdf5.set'

        # Load with Python
        py_eeg = pop_loadset_h5(filepath)

        # Load with MATLAB
        data_dir = os.path.abspath('sample_data')
        ml_eeg = self.eeglab.pop_loadset('filename', 'eeglab_data_epochs_ica.set', 'filepath', data_dir)

        eeg_compare(py_eeg, ml_eeg)


class TestPopLoadsetH5RealData(unittest.TestCase):
    """Test pop_loadset_h5 with real HDF5 files without MATLAB dependency."""

    def test_compare_continuous_pop_loadset(self):
        EEG1 = pop_loadset('sample_data/eeglab_data.set')
        EEG2 = pop_loadset_h5('sample_data/eeglab_data_hdf5.set')

        eeg_compare(EEG1, EEG2)

    def test_compare_epochs_pop_loadset(self):
        EEG1 = pop_loadset('sample_data/eeglab_data_epochs_ica.set')
        EEG2 = pop_loadset_h5('sample_data/eeglab_data_epochs_ica_hdf5.set')

        eeg_compare(EEG1, EEG2)

    def test_direct_call_returns_zero_based_urevent_and_urchan(self):
        direct = pop_loadset_h5('sample_data/eeglab_data_hdf5.set')
        via_loadset = pop_loadset('sample_data/eeglab_data_hdf5.set')
        from_set = pop_loadset('sample_data/eeglab_data.set')

        urevents = [int(event['urevent']) for event in direct['event']]
        self.assertEqual(urevents, list(range(len(direct['urevent']))))
        self.assertEqual(urevents, [int(event['urevent']) for event in via_loadset['event']])
        self.assertEqual(urevents, [int(event['urevent']) for event in from_set['event']])
        urchans = [int(chan['urchan']) for chan in direct['chanlocs']]
        self.assertEqual(urchans, list(range(direct['nbchan'])))
        self.assertEqual(urchans, [int(chan['urchan']) for chan in via_loadset['chanlocs']])
        self.assertEqual(urchans, [int(chan['urchan']) for chan in from_set['chanlocs']])


if __name__ == '__main__':
    # test test_load_epoched_data only
    # TestPopLoadsetH5RealData().test_load_epoched_data()

    unittest.main()
