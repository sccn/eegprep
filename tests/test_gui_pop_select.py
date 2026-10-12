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


def _epoched_eeg():
    eeg = _eeg()
    eeg["data"] = np.arange(120, dtype=np.float32).reshape(4, 10, 3)
    eeg["pnts"] = 10
    eeg["trials"] = 3
    eeg["xmax"] = 0.09
    eeg["epoch"] = [{}, {}, {}]
    return eeg


class PopSelectGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_accepts_numpy_chanlocs(self):
        # eeg_checkset normalises EEG['chanlocs'] to a numpy array of dicts;
        # the dialog builder must accept that storage form (regression for #229).
        eeg = _eeg()
        eeg["chanlocs"] = np.asarray(eeg["chanlocs"], dtype=object)

        controls = controls_by_tag(pop_select_dialog_spec(eeg))

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

    def test_gui_numeric_channels_keep_one_based_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "time": "",
                    "rmtime": False,
                    "point": "",
                    "rmpoint": False,
                    "trial": "",
                    "rmtrial": False,
                    "chans": "1 2 3",
                    "rmchannel": False,
                    "chantype": "",
                    "rmchantype": False,
                }

        out, com = pop_select(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual([chan["labels"] for chan in out["chanlocs"]], ["Fz", "Cz", "HEOG"])
        self.assertEqual(com, "EEG = pop_select( EEG, 'channel', [1 2 3]);")

    def test_gui_result_accepts_matlab_colon_trial_range(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "time": "",
                    "rmtime": False,
                    "point": "",
                    "rmpoint": False,
                    "trial": "1:2:3",
                    "rmtrial": False,
                    "chans": "",
                    "rmchannel": False,
                    "chantype": "",
                    "rmchantype": False,
                }

        out, com = pop_select(_epoched_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["trials"], 2)
        self.assertEqual(out["data"].shape, (4, 10, 2))
        self.assertEqual(com, "EEG = pop_select( EEG, 'trial', [1 3]);")


if __name__ == "__main__":
    unittest.main()
