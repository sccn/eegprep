import unittest
from unittest import mock

import numpy as np

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import (
    pop_clean_rawdata,
    pop_clean_rawdata_dialog_spec,
)


def _eeg(*, epoched=False):
    return {
        "data": np.zeros((2, 20, 2), dtype=np.float32) if epoched else np.zeros((2, 40), dtype=np.float32),
        "nbchan": 2,
        "pnts": 20 if epoched else 40,
        "trials": 2 if epoched else 1,
        "srate": 100,
        "xmin": 0,
        "xmax": 0.19 if epoched else 0.39,
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "etc": {},
    }


class PopCleanRawdataGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_matches_clean_rawdata_sections(self):
        spec = pop_clean_rawdata_dialog_spec(_eeg())

        self.assertEqual(spec.title, "pop_clean_rawdata()")
        self.assertEqual(spec.function_name, "pop_clean_rawdata")
        self.assertEqual(spec.eeglab_source, "plugins/clean_rawdata/pop_clean_rawdata.m")
        labels = [(control.style, control.string, control.tag) for control in spec.controls]
        self.assertIn(("checkbox", "Remove channel drift (data not already high-pass filtered)", "filter"), labels)
        self.assertIn(("checkbox", "Process/remove channels", "chanrm"), labels)
        self.assertIn(("checkbox", "Perform Artifact Subspace Reconstruction bad burst correction/rejection", "asr"), labels)
        self.assertIn(("checkbox", "Additional removal of bad data periods", "rejwin"), labels)
        self.assertFalse(controls_by_tag(spec)["vis"].value)

    def test_gui_channel_callbacks_expose_labels(self):
        controls = controls_by_tag(pop_clean_rawdata_dialog_spec(_eeg()))

        self.assertEqual(controls["chanuse_button"].callback.params["channels"], ("Cz", "Pz"))
        self.assertEqual(controls["chanignore_button"].callback.params["channels"], ("Cz", "Pz"))
        self.assertEqual(controls["filter"].callback.name, "toggle_enabled")
        self.assertEqual(controls["filter"].callback.params["targets"], ("filterfreqs",))

    def test_gui_result_runs_clean_artifacts_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "filter": True,
                    "filterfreqs": "0.25 0.75",
                    "chanrm": True,
                    "chanignoreflag": False,
                    "chanignore": "",
                    "chanuseflag": False,
                    "chanuse": "",
                    "rmflat": True,
                    "rmflatsec": "5",
                    "rmcorr": True,
                    "rmcorrval": "0.8",
                    "rmnoise": True,
                    "rmnoiseval": "4",
                    "asr": True,
                    "asrstdval": "20",
                    "distance": False,
                    "rejwin": True,
                    "rejwinval1": "-Inf 7",
                    "rejwinval2": "25",
                    "asrrej": True,
                    "vis": False,
                }

        eeg = _eeg()
        with mock.patch(
            "eegprep.plugins.clean_rawdata.pop_clean_rawdata.clean_artifacts",
            return_value=(dict(eeg, setname="cleaned"), eeg, eeg, np.zeros(2, dtype=bool)),
        ) as clean:
            out, com = pop_clean_rawdata(eeg, gui=True, renderer=Renderer(), return_com=True)

        clean.assert_called_once()
        self.assertEqual(out["setname"], "cleaned")
        self.assertIn("'BurstCriterion', 20", com)
        self.assertIn("'BurstRejection', 'on'", com)

    def test_gui_vis_checkbox_notifies_when_checked(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "filter": False,
                    "filterfreqs": "",
                    "chanrm": False,
                    "chanignoreflag": False,
                    "chanignore": "",
                    "chanuseflag": False,
                    "chanuse": "",
                    "rmflat": False,
                    "rmflatsec": "5",
                    "rmcorr": False,
                    "rmcorrval": "0.8",
                    "rmnoise": False,
                    "rmnoiseval": "4",
                    "asr": False,
                    "asrstdval": "20",
                    "distance": False,
                    "rejwin": False,
                    "rejwinval1": "-Inf 7",
                    "rejwinval2": "25",
                    "asrrej": False,
                    "vis": True,
                }

        eeg = _eeg()
        with (
            mock.patch(
                "eegprep.plugins.clean_rawdata.pop_clean_rawdata.clean_artifacts",
                return_value=(dict(eeg, setname="cleaned"), eeg, eeg, np.zeros(2, dtype=bool)),
            ) as clean,
            mock.patch("eegprep.plugins.clean_rawdata.pop_clean_rawdata._notify_vis_artifacts_unavailable") as notify,
        ):
            out, com = pop_clean_rawdata(eeg, gui=True, renderer=Renderer(), return_com=True)

        clean.assert_called_once()
        notify.assert_called_once()
        self.assertEqual(out["setname"], "cleaned")
        self.assertNotIn("_show_vis_artifacts", com)

    def test_string_channel_lists_use_matlab_cell_history(self):
        eeg = _eeg()
        with mock.patch(
            "eegprep.plugins.clean_rawdata.pop_clean_rawdata.clean_artifacts",
            return_value=(dict(eeg, setname="cleaned"), eeg, eeg, np.zeros(2, dtype=bool)),
        ):
            _out, com = pop_clean_rawdata(
                eeg,
                gui=False,
                Channels=["Cz", "Pz"],
                Channels_ignore=["ECG"],
                return_com=True,
            )

        self.assertIn("'Channels', {'Cz' 'Pz'}", com)
        self.assertIn("'Channels_ignore', {'ECG'}", com)

    def test_epoched_data_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "continuous"):
            pop_clean_rawdata(_eeg(epoched=True), gui=False)


if __name__ == "__main__":
    unittest.main()
