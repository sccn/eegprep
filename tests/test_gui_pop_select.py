import unittest
from contextlib import redirect_stdout
import io

import numpy as np

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_select import pop_select, pop_select_dialog_spec


def _eeg():
    return {
        "data": np.arange(40, dtype=np.float32).reshape(4, 10),
        "nbchan": 4,
        "pnts": 10,
        "trials": 1,
        "srate": 100,
        "xmin": 0,
        "xmax": 0.09,
        "times": np.arange(10),
        "chanlocs": [
            {"labels": "Fz", "type": "EEG"},
            {"labels": "Cz", "type": "EEG"},
            {"labels": "HEOG", "type": "EOG"},
            {"labels": "VEOG", "type": "EOG"},
        ],
        "event": [],
        "urevent": [],
        "epoch": [],
        "icaact": np.array([]),
        "icawinv": np.array([]),
        "icasphere": np.array([]),
        "icaweights": np.array([]),
        "icachansind": np.array([]),
        "chaninfo": {},
        "dipfit": [],
        "roi": [],
        "reject": {},
        "stats": {},
        "specdata": [],
        "specicaact": [],
    }


class PopSelectGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_matches_eeglab_control_order(self):
        spec = pop_select_dialog_spec(_eeg())

        self.assertEqual(spec.title, "Select data -- pop_select()")
        self.assertEqual(spec.function_name, "pop_select")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_select.m")
        self.assertEqual(spec.size, (695, 404))
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls[:8]],
            [
                ("text", "Select data in:", None),
                ("text", "Input desired range", None),
                ("text", "on->remove these", None),
                ("text", "Time range [min max] (s)", None),
                ("edit", "", "time"),
                ("spacer", "", None),
                ("checkbox", "    ", "rmtime"),
                ("spacer", "", None),
            ],
        )

    def test_gui_channel_picker_exposes_labels_and_types(self):
        controls = controls_by_tag(pop_select_dialog_spec(_eeg()))

        self.assertEqual(controls["chans_button"].callback.params["channels"], ("Fz", "Cz", "HEOG", "VEOG"))
        self.assertEqual(controls["chantype_button"].callback.params["channels"], ("EEG", "EOG"))

    def test_gui_result_runs_selection_and_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "time": "",
                    "rmtime": False,
                    "point": "",
                    "rmpoint": False,
                    "trial": "",
                    "rmtrial": False,
                    "chans": "Fz Cz",
                    "rmchannel": False,
                    "chantype": "",
                    "rmchantype": False,
                }

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            out, com = pop_select(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["nbchan"], 2)
        self.assertEqual([chan["labels"] for chan in out["chanlocs"]], ["Fz", "Cz"])
        self.assertNotIn("There was an issue storing removed channels", stdout.getvalue())
        self.assertEqual([chan["labels"] for chan in out["chaninfo"]["removedchans"]], ["HEOG", "VEOG"])
        self.assertEqual(com, "EEG = pop_select( EEG, 'channel', {'Fz' 'Cz'});")


if __name__ == "__main__":
    unittest.main()
