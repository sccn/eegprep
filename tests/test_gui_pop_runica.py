import unittest
from unittest import mock

import numpy as np

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.popfunc.pop_runica import pop_runica, pop_runica_dialog_spec


def _eeg():
    return {
        "data": np.arange(80, dtype=np.float64).reshape(4, 20),
        "nbchan": 4,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "chanlocs": [
            {"labels": "Fz", "type": "EEG"},
            {"labels": "Cz", "type": "EEG"},
            {"labels": "HEOG", "type": "EOG"},
            {"labels": "VEOG", "type": "EOG"},
        ],
    }


class PopRunicaGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_matches_eeglab_control_order(self):
        spec = pop_runica_dialog_spec(_eeg())

        self.assertEqual(spec.title, "Run ICA decomposition -- pop_runica()")
        self.assertEqual(spec.function_name, "pop_runica")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_runica.m")
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls],
            [
                ("text", "ICA algorithm to use (click to select)", None),
                (
                    "listbox",
                    "Extended Infomax (runica.m; default)|Robust Extended Infomax (runica.m; slow)|"
                    "AMICA (slowest; best)|Infomax picard.m|FastICA picard.m (fastest)",
                    "icatype",
                ),
                ("text", "Commandline options (See help messages)", None),
                ("edit", "", "params"),
                ("checkbox", "Reorder components by variance (if that's not already the case)", "reorder"),
                ("text", "Use only channel type(s) or indices", None),
                ("edit", "", "chantype"),
                ("pushbutton", "... types", "type_button"),
                ("pushbutton", "... channels", "chan_button"),
            ],
        )

    def test_gui_channel_callbacks_expose_types_and_labels(self):
        controls = controls_by_tag(pop_runica_dialog_spec(_eeg()))

        self.assertEqual(controls["type_button"].callback.params["channels"], ("EEG", "EOG"))
        self.assertEqual(controls["chan_button"].callback.params["channels"], ("Fz", "Cz", "HEOG", "VEOG"))

    def test_gui_result_runs_runica_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icatype": 1, "params": "'extended', 1, 'maxsteps', 2", "reorder": True, "chantype": ""}

        eeg = _eeg()
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))
        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated) as runica:
            out, com = pop_runica(eeg, gui=True, renderer=Renderer(), return_com=True)

        runica.assert_called_once()
        self.assertNotIn("lrate", runica.call_args.kwargs)
        self.assertEqual(out["icaweights"].shape, (4, 4))
        self.assertEqual(
            com,
            "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 2, 'interrupt', 'on');",
        )

    def test_gui_numeric_chanind_keeps_one_based_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icatype": 1, "params": "'extended', 1, 'maxsteps', 2", "reorder": True, "chantype": "1 2"}

        eeg = _eeg()
        updated = dict(
            eeg,
            data=eeg["data"][:2],
            nbchan=2,
            chanlocs=eeg["chanlocs"][:2],
            icaweights=np.eye(2),
            icasphere=np.eye(2),
            icawinv=np.eye(2),
            icaact=np.zeros((2, 20, 1)),
        )
        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated):
            out, com = pop_runica(eeg, gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["icaweights"].shape, (2, 2))
        np.testing.assert_array_equal(out["icachansind"], np.array([0, 1]))
        self.assertEqual(
            com,
            "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 2, 'interrupt', 'on', 'chanind', [1 2]);",
        )

    def test_chanind_accepts_one_based_numpy_array(self):
        eeg = _eeg()
        updated = dict(
            eeg,
            data=eeg["data"][:2],
            nbchan=2,
            chanlocs=eeg["chanlocs"][:2],
            icaweights=np.eye(2),
            icasphere=np.eye(2),
            icawinv=np.eye(2),
            icaact=np.zeros((2, 20, 1)),
        )
        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated):
            out, com = pop_runica(eeg, chanind=np.array([1, 2]), return_com=True)

        self.assertEqual(out["icaweights"].shape, (2, 2))
        np.testing.assert_array_equal(out["icachansind"], np.array([0, 1]))
        self.assertEqual(com, "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'chanind', [1 2]);")

    def test_gui_dialog_spec_adds_concatenate_controls_for_multiple_datasets(self):
        spec = pop_runica_dialog_spec([_eeg(), _eeg()])
        tagged = {control.tag: control for control in spec.controls if control.tag}

        self.assertIn("concatenate", tagged)
        self.assertIn("concatcond", tagged)
        self.assertFalse(tagged["concatenate"].value)
        self.assertTrue(tagged["concatcond"].value)

    def test_picard_algorithm_routes_to_eeg_picard(self):
        eeg = _eeg()
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_picard", return_value=updated) as picard:
            out, com = pop_runica(eeg, icatype="picard", options={"maxiter": 7, "mode": "standard"}, return_com=True)

        picard.assert_called_once()
        self.assertEqual(picard.call_args.kwargs["max_iter"], 7)
        self.assertFalse(picard.call_args.kwargs["ortho"])
        self.assertEqual(out["icaweights"].shape, (4, 4))
        self.assertEqual(com, "EEG = pop_runica(EEG, 'icatype', 'picard', 'maxiter', 7, 'mode', 'standard');")

    def test_key_value_options_override_backend_defaults(self):
        eeg = _eeg()
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated) as runica:
            _out, com = pop_runica(eeg, "extended", 0, return_com=True)

        self.assertEqual(runica.call_args.kwargs["extended"], 0)
        self.assertNotIn("lrate", runica.call_args.kwargs)
        self.assertEqual(com, "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 0);")

    def test_amica_algorithm_routes_to_eeg_amica(self):
        eeg = _eeg()
        updated = dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_amica", return_value=updated) as amica:
            out, com = pop_runica(eeg, icatype="runamica15", options={"maxiter": 3}, return_com=True)

        amica.assert_called_once()
        self.assertEqual(amica.call_args.kwargs["max_iter"], 3)
        self.assertEqual(out["icaweights"].shape, (4, 4))
        self.assertEqual(com, "EEG = pop_runica(EEG, 'icatype', 'runamica15', 'maxiter', 3);")

    def test_concatenate_copies_single_decomposition_to_each_dataset(self):
        first = _eeg()
        second = dict(_eeg(), data=np.arange(80, 160, dtype=np.float64).reshape(4, 20))

        def fake_runica(eeg, sortcomps="off", **_kwargs):
            return dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 40, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", side_effect=fake_runica) as runica:
            out, com = pop_runica([first, second], options={"extended": 1}, concatenate="on", return_com=True)

        runica.assert_called_once()
        self.assertEqual(runica.call_args.args[0]["data"].shape, (4, 40))
        self.assertEqual(len(out), 2)
        for eeg in out:
            np.testing.assert_array_equal(eeg["icaweights"], np.eye(4))
            self.assertEqual(eeg["icaact"].size, 0)
        self.assertIn("'concatenate', 'on'", com)

    def test_concatcond_groups_datasets_without_subjects(self):
        first = _eeg()
        second = dict(_eeg(), data=np.arange(80, 160, dtype=np.float64).reshape(4, 20))

        def fake_runica(eeg, sortcomps="off", **_kwargs):
            return dict(eeg, icaweights=np.eye(4), icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 40, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", side_effect=fake_runica) as runica:
            out, com = pop_runica([first, second], options={"extended": 1}, concatcond="on", return_com=True)

        runica.assert_called_once()
        self.assertEqual(runica.call_args.args[0]["data"].shape, (4, 40))
        self.assertEqual(len(out), 2)
        self.assertIn("'concatcond', 'on'", com)

    def test_existing_ica_is_saved_and_iclabel_removed_before_recompute(self):
        eeg = dict(
            _eeg(),
            icaweights=np.eye(4),
            icasphere=np.eye(4),
            icachansind=np.arange(4),
            etc={"ic_classification": {"ICLabel": {"version": "default"}}},
        )
        updated = dict(eeg, icaweights=np.eye(4) * 2, icasphere=np.eye(4), icawinv=np.eye(4), icaact=np.zeros((4, 20, 1)))

        with mock.patch("eegprep.functions.popfunc.pop_runica.eeg_runica", return_value=updated):
            out = pop_runica(eeg, options={"extended": 1})

        self.assertNotIn("ic_classification", out["etc"])
        np.testing.assert_array_equal(out["etc"]["oldicaweights"][0], np.eye(4))

    def test_qt_renderer_reads_listbox_as_one_based_row(self):
        class ListboxWidget:
            def property(self, name):
                return None

            def currentRow(self):
                return 1

            def currentIndex(self):
                raise AssertionError("QListWidget currentIndex returns a QModelIndex")

        self.assertEqual(QtDialogRenderer._read_widget(ListboxWidget()), 2)


if __name__ == "__main__":
    unittest.main()
