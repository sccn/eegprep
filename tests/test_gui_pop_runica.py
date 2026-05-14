import unittest
from unittest import mock

import numpy as np

from eegprep.functions.guifunc.spec import controls_by_tag
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
                ("listbox", "Extended Infomax (runica.m; default)|Robust Extended Infomax (runica.m; slow)", "icatype"),
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
        self.assertEqual(out["icaweights"].shape, (4, 4))
        self.assertEqual(com, "EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'maxsteps', 2);")


if __name__ == "__main__":
    unittest.main()
