"""Tests for the public eegprep package export surface."""

import unittest

import eegprep
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.sigprocfunc.eegrej import eegrej as sigproc_eegrej


class TestPackageExports(unittest.TestCase):
    def test_eegrej_export_matches_eeglab_low_level_function(self):
        self.assertIs(eegprep.eegrej, sigproc_eegrej)
        self.assertIs(eegprep.eeg_eegrej, eeg_eegrej)
        self.assertIsNot(eegprep.eegrej, eegprep.eeg_eegrej)

    def test_direct_exports_survive_explicit_wrapper_imports(self):
        self.assertIs(eegprep.eeg_checkset, eeg_checkset)
        self.assertIs(eegprep.eeg_checkset_strict_mode, strict_mode)
        self.assertIs(eegprep.pop_epoch, pop_epoch)
        self.assertIs(eegprep.pop_rmbase, pop_rmbase)
        self.assertIs(eegprep.pop_select, pop_select)


if __name__ == "__main__":
    unittest.main()
