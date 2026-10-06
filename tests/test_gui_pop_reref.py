import unittest

import numpy as np

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_reref import pop_reref


class PopRerefGuiSpecTests(unittest.TestCase):
    def test_mode_callback_toggles_reference_controls(self):
        class Widget:
            def __init__(self, checked=False):
                self.checked = checked
                self.enabled = True

            def setChecked(self, value):
                self.checked = value

            def setEnabled(self, value):
                self.enabled = value

            def blockSignals(self, value):
                pass

        widgets = {
            "ave": Widget(True),
            "huberef": Widget(False),
            "rerefstr": Widget(False),
            "reref": Widget(False),
            "refbr": Widget(False),
            "keepref": Widget(False),
        }

        QtDialogRenderer._set_reref_mode(widgets, "channels", True)

        self.assertFalse(widgets["ave"].checked)
        self.assertTrue(widgets["rerefstr"].checked)
        self.assertTrue(widgets["reref"].enabled)
        self.assertTrue(widgets["refbr"].enabled)
        self.assertTrue(widgets["keepref"].enabled)

        QtDialogRenderer._set_reref_mode(widgets, "average", True)

        self.assertTrue(widgets["ave"].checked)
        self.assertFalse(widgets["rerefstr"].checked)
        self.assertFalse(widgets["reref"].enabled)
        self.assertFalse(widgets["refbr"].enabled)
        self.assertFalse(widgets["keepref"].enabled)

    def test_gui_refloc_picker_filters_fiducials_like_eeglab(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                self.spec = spec
                return None

        renderer = Renderer()
        eeg = {
            "data": np.zeros((2, 10)),
            "nbchan": 2,
            "pnts": 10,
            "trials": 1,
            "srate": 100,
            "chanlocs": [{"labels": "Fp1"}, {"labels": "Fp2"}],
            "chaninfo": {
                "nodatchans": [
                    {"labels": "Nz", "type": "FID"},
                    {"labels": "M1", "type": "REF"},
                ]
            },
        }

        out = pop_reref(eeg, gui=True, renderer=renderer)
        controls = controls_by_tag(renderer.spec)

        self.assertIs(out, eeg)
        self.assertEqual(controls["refloc_button"].callback.params["channels"], ("M1",))
        self.assertIn("no_channels_message", controls["refloc_button"].callback.params)

    def test_gui_huber_history_uses_numeric_threshold(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "ave": False,
                    "huberef": True,
                    "huberval": "25",
                    "rerefstr": False,
                    "interp": False,
                    "keepref": False,
                    "exclude": "",
                    "refloc": "",
                }

        eeg = {
            "data": np.arange(40, dtype=float).reshape(4, 10),
            "nbchan": 4,
            "pnts": 10,
            "trials": 1,
            "srate": 100,
            "xmin": 0,
            "xmax": 0.09,
            "chanlocs": [{"labels": f"Ch{index + 1}"} for index in range(4)],
            "chaninfo": {},
            "epoch": [],
        }

        _out, com = pop_reref(eeg, gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(com, "EEG = pop_reref( EEG, [], 'huber', 25);")


if __name__ == "__main__":
    unittest.main()
