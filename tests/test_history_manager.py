import unittest

from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.guifunc.history_manager import HistoryManagerWidget, _extract_function_name

try:
    from PySide6 import QtWidgets
except ImportError:
    QtWidgets = None


@unittest.skipIf(QtWidgets is None, "PySide6 not installed")
class HistoryManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if QtWidgets.QApplication.instance() is None:
            cls.app = QtWidgets.QApplication([])
        else:
            cls.app = QtWidgets.QApplication.instance()

    def setUp(self):
        self.session = EEGPrepSession()
        eeg = eeg_emptyset()
        self.session.store_current(eeg, new=True)
        self.widget = HistoryManagerWidget(self.session)

    def test_extract_function_name(self):
        self.assertEqual(_extract_function_name("EEG = pop_eegfiltnew(EEG, 1, 0);"), "pop_eegfiltnew")
        self.assertEqual(_extract_function_name("pop_saveset(EEG);"), "pop_saveset")

    def test_displays_allcom_commands(self):
        self.session.add_history("EEG = pop_resample(EEG, 250);")
        self.assertEqual(len(self.session.ALLCOM), 1)
        self.assertEqual(self.widget.tree.topLevelItemCount(), 1)
        self.assertEqual(self.widget.tree.topLevelItem(0).text(0), "pop_resample")

    def test_grouping_consecutive_commands(self):
        self.session.add_history("EEG = pop_eegfiltnew(EEG, 1, 0);")
        self.session.add_history("EEG = pop_eegfiltnew(EEG, 2, 0);")
        self.session.add_history("EEG = pop_resample(EEG, 250);")

        self.assertEqual(self.widget.tree.topLevelItemCount(), 2)

        group1 = self.widget.tree.topLevelItem(0)
        self.assertEqual(group1.text(0), "pop_eegfiltnew")
        self.assertEqual(group1.childCount(), 2)

        group2 = self.widget.tree.topLevelItem(1)
        self.assertEqual(group2.text(0), "pop_resample")
        self.assertEqual(group2.childCount(), 1)
