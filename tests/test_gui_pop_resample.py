import unittest

import numpy as np

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.popfunc.pop_resample import pop_resample, pop_resample_dialog_spec


def _eeg():
    return {
        "data": np.arange(40, dtype=np.float32).reshape(2, 20),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "xmin": 0,
        "xmax": 0.19,
        "times": np.arange(20),
        "event": [],
        "urevent": [],
    }


class PopResampleGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_matches_eeglab_inputdlg(self):
        spec = pop_resample_dialog_spec(100)

        self.assertEqual(spec.title, "Resample current dataset -- pop_resample()")
        self.assertEqual(spec.function_name, "pop_resample")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_resample.m")
        self.assertEqual([(control.style, control.string, control.tag) for control in spec.controls], [
            ("text", "New sampling rate", None),
            ("edit", "", "freq"),
        ])

    def test_gui_result_resamples_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"freq": "50"}

        out, com = pop_resample(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["srate"], 50)
        self.assertEqual(out["pnts"], 10)
        self.assertEqual(com, "EEG = pop_resample( EEG, 50);")

    def test_gui_validation_rejects_nonpositive_rate(self):
        spec = pop_resample_dialog_spec(100)
        widgets = {"freq": _FakeWidget("0")}

        self.assertEqual(QtDialogRenderer._validation_message(spec, widgets), "New sampling rate must be positive")


class _FakeWidget:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


if __name__ == "__main__":
    unittest.main()
