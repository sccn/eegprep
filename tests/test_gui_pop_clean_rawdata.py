import unittest
from unittest import mock

import numpy as np

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import (
    pop_clean_rawdata,
    pop_clean_rawdata_dialog_spec,
)
from eegprep.plugins.clean_rawdata.vis_artifacts import vis_artifacts, vis_artifacts_diagnostics


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
        self.assertEqual(spec.help_text, "pophelp('pop_clean_rawdata')")
        labels = [(control.style, control.string, control.tag) for control in spec.controls]
        self.assertIn(("checkbox", "Remove channel drift (data not already high-pass filtered)", "filter"), labels)
        self.assertIn(("checkbox", "Process/remove channels", "chanrm"), labels)
        self.assertIn(
            ("checkbox", "Perform Artifact Subspace Reconstruction bad burst correction/rejection", "asr"), labels
        )
        self.assertIn(("checkbox", "Additional removal of bad data periods", "rejwin"), labels)
        controls = controls_by_tag(spec)
        self.assertEqual(controls["filter"].font_weight, "bold")
        self.assertEqual(controls["chanrm"].font_weight, "bold")
        self.assertEqual(controls["asr"].font_weight, "bold")
        self.assertEqual(controls["rejwin"].font_weight, "bold")
        self.assertTrue(controls["vis"].value)

    def test_gui_channel_callbacks_expose_labels(self):
        controls = controls_by_tag(pop_clean_rawdata_dialog_spec(_eeg()))

        self.assertEqual(controls["chanuse_button"].callback.params["channels"], ("Cz", "Pz"))
        self.assertEqual(controls["chanignore_button"].callback.params["channels"], ("Cz", "Pz"))
        self.assertEqual(controls["filter"].callback.name, "toggle_enabled")
        self.assertEqual(controls["filter"].callback.params["targets"], ("filterfreqs",))

    def test_gui_channel_callbacks_accept_numpy_chanlocs(self):
        eeg = _eeg()
        eeg["chanlocs"] = np.asarray(eeg["chanlocs"], dtype=object)

        controls = controls_by_tag(pop_clean_rawdata_dialog_spec(eeg))

        self.assertEqual(controls["chanuse_button"].callback.params["channels"], ("Cz", "Pz"))
        self.assertEqual(controls["chanignore_button"].callback.params["channels"], ("Cz", "Pz"))

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

    def test_gui_vis_checkbox_opens_rejected_data_browser_when_checked(self):
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
                return_value=(
                    dict(
                        eeg,
                        setname="cleaned",
                        etc={"clean_sample_mask": np.r_[np.ones(10, dtype=bool), np.zeros(30, dtype=bool)]},
                    ),
                    eeg,
                    eeg,
                    np.zeros(2, dtype=bool),
                ),
            ) as clean,
            mock.patch("eegprep.plugins.clean_rawdata.pop_clean_rawdata.vis_artifacts") as artifacts,
        ):
            out, com = pop_clean_rawdata(eeg, gui=True, renderer=Renderer(), return_com=True)

        clean.assert_called_once()
        artifacts.assert_called_once()
        shown, original = artifacts.call_args.args
        np.testing.assert_array_equal(shown["etc"]["clean_sample_mask"][10:], np.zeros(30, dtype=bool))
        self.assertEqual(original["pnts"], eeg["pnts"])
        np.testing.assert_array_equal(original["data"], eeg["data"])
        self.assertEqual(out["setname"], "cleaned")
        self.assertNotIn("_show_vis_artifacts", com)

    def test_vis_artifacts_diagnostics_summarizes_samples_and_channels(self):
        old = _eeg()
        new = dict(
            old,
            data=old["data"][:, :30],
            pnts=30,
            etc={
                "clean_sample_mask": np.r_[np.ones(10, dtype=bool), np.zeros(5, dtype=bool), np.ones(25, dtype=bool)],
                "clean_channel_mask": np.asarray([True, False]),
            },
        )

        diag = vis_artifacts_diagnostics(new, old)

        self.assertEqual(diag["original_samples"], 40)
        self.assertEqual(diag["clean_samples"], 30)
        self.assertEqual(diag["rejected_sample_count"], 5)
        np.testing.assert_array_equal(diag["rejected_intervals"], [[11, 15]])
        self.assertEqual(diag["removed_channel_indices"], [2])
        self.assertEqual(diag["removed_channel_labels"], ["Pz"])
        self.assertEqual(diag["winrej"].shape, (1, 7))

    def test_vis_artifacts_can_return_diagnostics_without_opening_browser(self):
        old = _eeg()
        new = dict(
            old,
            etc={"clean_sample_mask": np.r_[np.zeros(3, dtype=bool), np.ones(37, dtype=bool)]},
        )

        diag = vis_artifacts(new, old, show=False)

        np.testing.assert_array_equal(diag["rejected_intervals"], [[1, 3]])
        self.assertEqual(diag["rejected_fraction"], 3 / 40)

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
