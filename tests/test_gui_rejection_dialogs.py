import unittest
from unittest import mock

import numpy as np

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_autorej import pop_autorej_dialog_spec
from eegprep.functions.popfunc.pop_eegthresh import pop_eegthresh, pop_eegthresh_dialog_spec
from eegprep.functions.popfunc.pop_jointprob import pop_jointprob, pop_jointprob_dialog_spec
from eegprep.functions.popfunc.pop_rejchan import pop_rejchan_dialog_spec
from eegprep.functions.popfunc.pop_rejcont import pop_rejcont_dialog_spec
from eegprep.functions.popfunc.pop_rejkurt import pop_rejkurt, pop_rejkurt_dialog_spec
from eegprep.functions.popfunc.pop_rejmenu import pop_rejmenu_dialog_spec
from eegprep.functions.popfunc.pop_rejspec import pop_rejspec_dialog_spec
from eegprep.functions.popfunc.pop_rejtrend import pop_rejtrend_dialog_spec
from eegprep.functions.popfunc.pop_selectcomps import pop_selectcomps_dialog_spec
from eegprep.plugins.ICLabel.pop_viewprops import pop_viewprops, pop_viewprops_dialog_spec
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


class RejectionDialogTests(unittest.TestCase):
    def test_dialog_specs_keep_eeglab_source_and_key_defaults(self):
        eeg = _epoched_ica_eeg()
        specs = [
            pop_eegthresh_dialog_spec(eeg, 1),
            pop_jointprob_dialog_spec(eeg, 1),
            pop_rejkurt_dialog_spec(eeg, 0),
            pop_rejtrend_dialog_spec(eeg, 1),
            pop_rejspec_dialog_spec(eeg, 0),
            pop_rejchan_dialog_spec(eeg),
            pop_rejcont_dialog_spec(eeg),
            pop_autorej_dialog_spec(eeg),
            pop_rejmenu_dialog_spec(eeg, 1),
            pop_selectcomps_dialog_spec(eeg),
            pop_viewprops_dialog_spec(eeg, 0),
        ]

        for spec in specs:
            self.assertIn("pop_", spec.function_name)
            self.assertTrue(spec.eeglab_source.endswith(".m"))
            self.assertIsNotNone(spec.size)

        self.assertEqual(controls_by_tag(specs[0])["elecrange"].value, "1:3")
        self.assertEqual(controls_by_tag(specs[1])["vistype"].value, 2)
        self.assertTrue(controls_by_tag(specs[1])["superpose"].value)
        self.assertEqual(controls_by_tag(specs[2])["vistype"].value, 2)
        self.assertTrue(controls_by_tag(specs[2])["superpose"].value)
        self.assertTrue(controls_by_tag(specs[10])["scroll_event"].enabled)

    def test_viewprops_component_dialog_includes_classifier_dropdown(self):
        eeg = _epoched_ica_eeg()
        eeg["etc"] = {"ic_classification": {"Other": {}, "ICLabel": {}}}
        controls = controls_by_tag(pop_viewprops_dialog_spec(eeg, 0))

        self.assertEqual(controls["classifier_name"].string, "Other|ICLabel")
        self.assertEqual(controls["classifier_name"].value, 2)

    def test_component_probability_dialog_defaults_match_eeglab(self):
        eeg = _epoched_ica_eeg()

        self.assertEqual(controls_by_tag(pop_jointprob_dialog_spec(eeg, 0))["locthresh"].value, "5")
        self.assertEqual(controls_by_tag(pop_jointprob_dialog_spec(eeg, 0))["globthresh"].value, "5")
        self.assertEqual(controls_by_tag(pop_rejkurt_dialog_spec(eeg, 0))["locthresh"].value, "5")
        self.assertEqual(controls_by_tag(pop_rejkurt_dialog_spec(eeg, 0))["globthresh"].value, "5")

    def test_rejection_menu_actions_are_implemented_or_browser_excluded(self):
        implemented = [
            "eeg_rejsuperpose:data_to_ica",
            "pop_autorej",
            "pop_eegthresh:data",
            "pop_jointprob:ica",
            "pop_rejchan",
            "pop_rejcont",
            "pop_rejepoch:data",
            "pop_rejkurt:ica",
            "pop_rejmenu:data",
            "pop_rejspec:ica",
            "pop_rejtrend:data",
            "pop_selectcomps",
            "pop_viewprops:channels",
            "pop_viewprops:components",
        ]
        for action in implemented:
            self.assertEqual(action_kind(action), "implemented")
        self.assertEqual(action_kind("pop_eegplot:reject_data"), "implemented")

    def test_gui_command_is_valid_python_with_keywords(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "elecrange": "1",
                    "negthresh": "-5",
                    "posthresh": "5",
                    "starttime": "0",
                    "endtime": "0.39",
                    "superpose": False,
                    "reject": False,
                }

        eeg = _epoched_ica_eeg()
        eeg["data"][0, 2, 1] = 10
        _out, com = pop_eegthresh(eeg, gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(
            _console_python_command(com),
            "EEG = pop_eegthresh(EEG, icacomp=1, elecrange=[1], negthresh=[-5], "
            "posthresh=[5], starttime=[0], endtime=[0.39], superpose=0, reject=0)",
        )

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
        _joint_out, joint_com = pop_jointprob(eeg, gui=True, renderer=Renderer(), return_com=True)
        _kurt_out, kurt_com = pop_rejkurt(eeg, gui=True, renderer=Renderer(), return_com=True)

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

    def test_viewprops_dispatch_records_history_without_replacing_dataset(self):
        session = EEGPrepSession()
        eeg = _epoched_ica_eeg()
        session.store_current(eeg, new=True)
        dispatcher = MenuActionDispatcher(session)

        with mock.patch(
            "eegprep.plugins.ICLabel.pop_viewprops.pop_viewprops",
            return_value=(["figure"], "pop_viewprops(EEG, 0, [1], [], [], 1, '')"),
        ) as viewprops:
            dispatcher.dispatch("pop_viewprops:components")

        viewprops.assert_called_once()
        self.assertIs(viewprops.call_args.args[0], session.EEG)
        self.assertEqual(viewprops.call_args.kwargs, {"typecomp": 0, "return_com": True})
        self.assertEqual(session.ALLCOM[-1], "pop_viewprops(EEG, 0, [1], [], [], 1, '')")

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


if __name__ == "__main__":
    unittest.main()
