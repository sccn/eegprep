import unittest
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui
from tests.fixtures import create_test_eeg


def _epoched_ica_eeg():
    eeg = create_test_eeg(n_channels=3, n_samples=40, n_trials=3, srate=100)
    eeg["data"] = np.zeros((3, 40, 3))
    eeg["icaweights"] = np.eye(3)
    eeg["icasphere"] = np.eye(3)
    eeg["icawinv"] = np.eye(3)
    eeg["icachansind"] = np.arange(3)
    eeg["reject"] = {
        "gcompreject": np.array([0, 1, 0]),
        "rejthresh": np.array([0, 1, 0]),
        "rejthreshE": np.zeros((3, 3), dtype=bool),
    }
    return eeg


@pytest.mark.gui
@eeglab_test("unittesting_adminfunc/pop_rejmenu/adminfunc_pop_rejmenu_wrapperTest.m", "test_test_pop_rejmenu")
def test_reference_rejection_menu_original_sample(eeglab_backend, request, eeglab_suite_root):
    """Python's modal menu requires real interaction in the deferred GUI phase.

    No acceptance/cancellation is fabricated: the original MATLAB menu is
    nonmodal, whereas the Python call returns only after its dialog is closed.
    """
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data_epochs_ica.set"))
    for mode in (0.0, 1.0):
        eeglab_backend("pop_rejmenu", eeg, mode, nargout=0)
        close_reference_gui(eeglab_backend, request)


class RejectionDialogTests(unittest.TestCase):
    def test_probability_dialog_commands_include_visualization_mode(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "elecrange": "1",
                    "locthresh": "4",
                    "globthresh": "6",
                    "vistype": 2,
                    "superpose": True,
                    "reject": False,
                }

        eeg = _epoched_ica_eeg()
        _joint_out, joint_com = pop_jointprob(eeg, gui=True, renderer=Renderer(), return_com=True, show=False)
        _kurt_out, kurt_com = pop_rejkurt(eeg, gui=True, renderer=Renderer(), return_com=True, show=False)

        self.assertEqual(
            _console_python_command(joint_com),
            "EEG = pop_jointprob(EEG, icacomp=1, elecrange=[1], locthresh=[4], "
            "globthresh=[6], superpose=1, reject=0, vistype=1, topcommand=[], plotflag=0)",
        )
        self.assertEqual(
            _console_python_command(kurt_com),
            "EEG = pop_rejkurt(EEG, icacomp=1, elecrange=[1], locthresh=[4], "
            "globthresh=[6], superpose=1, reject=0, vistype=1, topcommand=[], plotflag=0)",
        )

    def test_viewprops_gui_records_options_and_classifier(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "chanorcomp": "1",
                    "spec_opt": "'freqrange', [2 40]",
                    "erp_opt": "'limits', [-100 500]",
                    "scroll_event": False,
                    "classifier_name": 2,
                }

        eeg = _epoched_ica_eeg()
        eeg["etc"] = {"ic_classification": {"Other": {}, "ICLabel": {}}}
        _figures, com = pop_viewprops(eeg, 0, gui=True, renderer=Renderer(), plot=False, return_com=True)

        self.assertEqual(
            _console_python_command(com),
            "pop_viewprops(EEG, typecomp=0, chanorcomp=[1], spec_opt=\"'freqrange', [2 40]\", "
            "erp_opt=\"'limits', [-100 500]\", scroll_event=0, classifier_name='ICLabel')",
        )

    def test_viewprops_reject_callback_stores_dashboard_component_marks(self):
        session = EEGPrepSession()
        eeg = _epoched_ica_eeg()
        eeg["reject"]["gcompreject"] = np.zeros(3, dtype=int)
        session.store_current(eeg, new=True)
        dispatcher = MenuActionDispatcher(session)
        captured = {}

        def fake_viewprops(selection, **kwargs):
            captured["selection"] = selection
            captured["reject_callback"] = kwargs["reject_callback"]
            return ["figure"], "pop_viewprops(EEG, 0, [1], [], [], 1, '')"

        with mock.patch("eegprep.plugins.ICLabel.pop_viewprops.pop_viewprops", side_effect=fake_viewprops):
            dispatcher.dispatch("pop_viewprops:components")

        captured["selection"]["reject"]["gcompreject"][1] = 1
        captured["reject_callback"](captured["selection"], {2: True})

        np.testing.assert_array_equal(session.EEG["reject"]["gcompreject"], [0, 1, 0])
        np.testing.assert_array_equal(session.ALLEEG[0]["reject"]["gcompreject"], [0, 1, 0])

    def test_reject_marked_epochs_uses_rejglobal_for_ica_menu(self):
        session = EEGPrepSession()
        eeg = _epoched_ica_eeg()
        eeg["reject"]["rejglobal"] = np.array([False, True, False])
        eeg["reject"]["icarejglobal"] = np.array([False, False, False])
        session.store_current(eeg, new=True)
        dispatcher = MenuActionDispatcher(session)

        with mock.patch(
            "eegprep.functions.popfunc.pop_rejepoch.pop_rejepoch",
            return_value=(eeg, "EEG = pop_rejepoch(EEG, [2], 1);"),
        ) as rejepoch:
            dispatcher.dispatch("pop_rejepoch:ica")

        rejepoch.assert_called_once()
        np.testing.assert_array_equal(rejepoch.call_args.args[1], np.array([False, True, False]))

    def test_rejection_dispatch_stores_multi_dataset_results_without_browser_callback(self):
        session = EEGPrepSession()
        first = _epoched_ica_eeg()
        first["setname"] = "first"
        second = _epoched_ica_eeg()
        second["setname"] = "second"
        session.store_current(first, new=True)
        session.store_current(second, new=True)
        session.retrieve([1, 2])
        dispatcher = MenuActionDispatcher(session)
        captured = {}
        command = "EEG = pop_jointprob(EEG, 1, [1], 4, 4, 0, 1, 1);"

        def fake_pop_jointprob(selection, icacomp, **kwargs):
            captured["selection"] = selection
            captured["icacomp"] = icacomp
            captured["kwargs"] = kwargs
            output = [dict(item, setname=f"{item['setname']}-marked") for item in selection]
            return output, command

        with mock.patch("eegprep.functions.popfunc.pop_jointprob.pop_jointprob", side_effect=fake_pop_jointprob):
            dispatcher.dispatch("pop_jointprob:data")

        self.assertEqual([item["setname"] for item in captured["selection"]], ["first", "second"])
        self.assertEqual(captured["icacomp"], 1)
        self.assertTrue(captured["kwargs"]["return_com"])
        self.assertNotIn("command_callback", captured["kwargs"])
        self.assertEqual(session.CURRENTSET, [1, 2])
        self.assertEqual([item["setname"] for item in session.ALLEEG], ["first-marked", "second-marked"])
        self.assertEqual(session.ALLCOM, [command])

    def test_rejection_browser_accept_callback_creates_dataset_without_duplicate_history(self):
        session = EEGPrepSession()
        eeg = _epoched_ica_eeg()
        session.store_current(eeg, new=True)
        dispatcher = MenuActionDispatcher(session)
        captured = {}
        command = "EEG = pop_jointprob(EEG, 1, [1], 4, 4, 0, 1, 1);"

        def fake_pop_jointprob(selection, icacomp, **kwargs):
            captured["selection"] = selection
            captured["icacomp"] = icacomp
            captured["callback"] = kwargs["command_callback"]
            return selection, command

        with mock.patch("eegprep.functions.popfunc.pop_jointprob.pop_jointprob", side_effect=fake_pop_jointprob):
            dispatcher.dispatch("pop_jointprob:data")

        self.assertEqual(session.ALLCOM, [command])
        self.assertEqual(captured["selection"]["setname"], "test_dataset")
        self.assertEqual(captured["icacomp"], 1)

        accepted = dict(eeg)
        accepted["data"] = np.asarray(eeg["data"])[:, :, :2]
        accepted["trials"] = 2
        captured["callback"](accepted, command)

        self.assertEqual(session.CURRENTSET, [2])
        self.assertEqual(session.EEG["trials"], 2)
        self.assertEqual(len(session.ALLEEG), 2)
        self.assertEqual(session.ALLCOM, [command])


if __name__ == "__main__":
    unittest.main()
