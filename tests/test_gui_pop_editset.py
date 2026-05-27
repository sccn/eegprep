import unittest

from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_editset import pop_editset_dialog_spec


def _eeg():
    return {
        "setname": "demo",
        "subject": "S01",
        "condition": "target",
        "group": "control",
        "run": 1,
        "session": 2,
        "nbchan": 2,
        "pnts": 100,
        "srate": 250.0,
        "xmin": -0.2,
        "ref": "common",
    }


class PopEditsetGuiSpecTests(unittest.TestCase):
    def test_dialog_spec_matches_eeglab_dataset_metadata_order(self):
        spec = pop_editset_dialog_spec(_eeg())

        self.assertEqual(spec.title, "Edit dataset information - pop_editset()")
        self.assertEqual(spec.function_name, "pop_editset")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_editset.m")
        self.assertEqual(spec.size, (688, 389))
        self.assertIsNotNone(spec.extra_stylesheet)
        self.assertIn("QDialog#pop_editset QLabel", spec.extra_stylesheet or "")
        self.assertIn("fields are visible for parity but disabled", spec.known_differences[0])
        self.assertEqual(
            [(control.style, control.string, control.tag) for control in spec.controls[:24]],
            [
                ("text", "Dataset name", None),
                ("edit", "", "setname"),
                ("spacer", "", None),
                ("text", "Data sampling rate (Hz)", None),
                ("edit", "", "srate"),
                ("text", "Subject code", None),
                ("edit", "", "subject"),
                ("text", "Time points per epoch (0->continuous)", None),
                ("edit", "", "pnts"),
                ("text", "Task condition", None),
                ("edit", "", "condition"),
                ("text", "Start time (sec) (only for data epochs)", None),
                ("edit", "", "xmin"),
                ("text", "Subject group", None),
                ("edit", "", "group"),
                ("text", "Number of channels (0->set from data)", None),
                ("edit", "", "nbchan"),
                ("text", "Run number", None),
                ("edit", "", "run"),
                ("text", "Ref. channel indices or mode (see help)", None),
                ("edit", "", "ref"),
                ("text", "Session number", None),
                ("edit", "", "session"),
                ("spacer", "", None),
            ],
        )

    def test_dialog_spec_marks_eeglab_bold_labels_and_read_only_fields(self):
        controls = controls_by_tag(pop_editset_dialog_spec(_eeg()))

        self.assertEqual(controls["setname"].value, "demo")
        self.assertEqual(controls["srate"].value, "250")
        self.assertEqual(controls["pnts"].value, "100")
        self.assertEqual(controls["xmin"].value, "-0.2")
        self.assertFalse(controls["nbchan"].enabled)
        self.assertFalse(controls["ref"].enabled)
        self.assertFalse(controls["chanfile"].enabled)
        self.assertFalse(controls["weightfile"].enabled)
        self.assertFalse(controls["sphfile"].enabled)
        self.assertFalse(controls["icainds"].enabled)


if __name__ == "__main__":
    unittest.main()
