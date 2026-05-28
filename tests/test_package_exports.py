"""Tests for the public eegprep package export surface."""

import unittest

import eegprep
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.adminfunc.eeglab import eeglab, gui
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_chanedit import pop_chanedit
from eegprep.functions.popfunc.pop_comments import pop_comments
from eegprep.functions.popfunc.pop_copyset import pop_copyset
from eegprep.functions.popfunc.pop_editset import pop_editset
from eegprep.functions.popfunc.pop_editeventfield import pop_editeventfield
from eegprep.functions.popfunc.pop_editeventvals import pop_editeventvals
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_fileio_brainvision_mat import pop_fileio_brainvision_mat
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset
from eegprep.functions.popfunc.pop_rmdat import pop_rmdat
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.popfunc.pop_selectevent import pop_selectevent
from eegprep.functions.sigprocfunc.eegrej import eegrej as sigproc_eegrej
from eegprep.functions.sigprocfunc.rmbase import rmbase as sigproc_rmbase
from eegprep.plugins.ICLabel.pop_icflag import pop_icflag


class TestPackageExports(unittest.TestCase):
    def test_eegrej_export_matches_eeglab_low_level_function(self):
        self.assertIs(eegprep.eegrej, sigproc_eegrej)
        self.assertIs(eegprep.eeg_eegrej, eeg_eegrej)
        self.assertIsNot(eegprep.eegrej, eegprep.eeg_eegrej)
        self.assertIs(eegprep.rmbase, sigproc_rmbase)

    def test_direct_exports_survive_explicit_wrapper_imports(self):
        self.assertIs(eegprep.eeglab, eeglab)
        self.assertIs(eegprep.gui, gui)
        self.assertIs(eegprep.eeg_checkset, eeg_checkset)
        self.assertIs(eegprep.eeg_checkset_strict_mode, strict_mode)
        self.assertIs(eegprep.pop_chanedit, pop_chanedit)
        self.assertIs(eegprep.pop_comments, pop_comments)
        self.assertIs(eegprep.pop_copyset, pop_copyset)
        self.assertIs(eegprep.pop_editset, pop_editset)
        self.assertIs(eegprep.pop_editeventfield, pop_editeventfield)
        self.assertIs(eegprep.pop_editeventvals, pop_editeventvals)
        self.assertIs(eegprep.pop_epoch, pop_epoch)
        self.assertIs(eegprep.pop_fileio_brainvision_mat, pop_fileio_brainvision_mat)
        self.assertIs(eegprep.pop_mergeset, pop_mergeset)
        self.assertIs(eegprep.pop_rmdat, pop_rmdat)
        self.assertIs(eegprep.pop_rmbase, pop_rmbase)
        self.assertIs(eegprep.pop_select, pop_select)
        self.assertIs(eegprep.pop_selectevent, pop_selectevent)
        self.assertIs(eegprep.pop_icflag, pop_icflag)


if __name__ == "__main__":
    unittest.main()
