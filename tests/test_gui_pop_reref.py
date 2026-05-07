import unittest

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_reref import pop_reref_dialog_spec


class PopRerefGuiSpecTests(unittest.TestCase):
    def test_dialog_spec_matches_eeglab_control_order(self):
        spec = pop_reref_dialog_spec("common")

        self.assertEqual(spec.title, "pop_reref - average reference or re-reference data")
        self.assertEqual(spec.function_name, "pop_reref")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_reref.m")
        self.assertEqual(spec.size, (616, 281))
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls],
            [
                ("text", "Current data reference state is: common", None),
                ("checkbox", "Compute average reference", "ave"),
                ("checkbox", "Huber average ref. with threshold", "huberef"),
                ("edit", "", "huberval"),
                ("text", "uV", "scale"),
                ("checkbox", "Re-reference data to channel(s):", "rerefstr"),
                ("edit", "", "reref"),
                ("pushbutton", "...", "refbr"),
                ("checkbox", "Interpolate removed channel(s)", "interp"),
                ("spacer", "", None),
                ("checkbox", "Retain ref. channel(s) in data (will be flat for single-channel ref.)", "keepref"),
                ("text", "Exclude channel indices (EMG, EOG)", None),
                ("edit", "", "exclude"),
                ("pushbutton", "...", "exclude_button"),
                ("text", "Add old ref. channel back to the data", "reflocstr"),
                ("edit", "", "refloc"),
                ("pushbutton", "...", "refloc_button"),
            ],
        )

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

    def test_dialog_callbacks_keep_matlab_metadata(self):
        controls = controls_by_tag(pop_reref_dialog_spec("average", ["Fp1", "Cz"], ["M1"]))

        self.assertEqual(controls["ave"].callback.name, "set_reref_mode")
        self.assertIn("cb_averef", controls["ave"].callback.matlab_callback)
        self.assertEqual(controls["rerefstr"].callback.name, "set_reref_mode")
        self.assertIn("pop_chansel", controls["refbr"].callback.matlab_callback)
        self.assertEqual(controls["refbr"].callback.params["channels"], ("Fp1", "Cz"))
        self.assertEqual(controls["exclude_button"].callback.params["channels"], ("Fp1", "Cz"))
        self.assertEqual(controls["refloc_button"].callback.params["channels"], ("M1",))
        self.assertEqual(pop_reref_dialog_spec().help_text, "pophelp('pop_reref')")


if __name__ == "__main__":
    unittest.main()
