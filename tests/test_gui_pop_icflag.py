import unittest

import numpy as np

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS, pop_icflag, pop_icflag_dialog_spec


def _eeg():
    return {
        "data": np.zeros((3, 20), dtype=np.float32),
        "nbchan": 3,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "icaweights": np.eye(3),
        "icasphere": np.eye(3),
        "icawinv": np.eye(3),
        "icachansind": np.arange(3),
        "reject": {"rejmanual": np.array([1, 0])},
        "etc": {
            "ic_classification": {
                "ICLabel": {
                    "classifications": np.array(
                        [
                            [0.70, 0.10, 0.10, 0.03, 0.02, 0.03, 0.02],
                            [0.02, 0.94, 0.02, 0.01, 0.00, 0.00, 0.01],
                            [0.05, 0.02, 0.91, 0.01, 0.00, 0.00, 0.01],
                        ]
                    )
                }
            }
        },
    }


class PopIcflagGuiTests(unittest.TestCase):
    def test_dialog_spec_matches_eeglab_threshold_prompt(self):
        spec = pop_icflag_dialog_spec()
        controls = controls_by_tag(spec)

        self.assertEqual(spec.title, "Flag components using ICLabel -- pop_icflag()")
        self.assertEqual(spec.function_name, "pop_icflag")
        self.assertEqual(spec.eeglab_source, "plugins/ICLabel/pop_icflag.m")
        self.assertEqual(spec.controls[0].font_weight, "bold")
        self.assertEqual(controls["min_1"].value, "0.9")
        self.assertEqual(controls["max_1"].value, "1")
        self.assertEqual(controls["min_2"].value, "0.9")
        self.assertEqual(controls["max_2"].value, "1")

    def test_gui_result_flags_components_and_returns_replayable_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {
                    "min_0": "",
                    "max_0": "",
                    "min_1": "0.9",
                    "max_1": "1",
                    "min_2": "0.9",
                    "max_2": "1",
                    "min_3": "",
                    "max_3": "",
                    "min_4": "",
                    "max_4": "",
                    "min_5": "",
                    "max_5": "",
                    "min_6": "",
                    "max_6": "",
                }

        out, com = pop_icflag(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        np.testing.assert_array_equal(out["reject"]["gcompreject"], [0, 1, 1])
        np.testing.assert_array_equal(out["reject"]["rejmanual"], [1, 0])
        self.assertEqual(
            _console_python_command(com),
            (
                "EEG = pop_icflag(EEG, thresholds=[[None, None], [0.9, 1], "
                "[0.9, 1], [None, None], [None, None], [None, None], [None, None]])"
            ),
        )

    def test_eeg_icflag_uses_eeglab_open_interval_thresholds(self):
        eeg = _eeg()
        thresholds = np.array(DEFAULT_ICFLAG_THRESHOLDS)
        eeg["etc"]["ic_classification"]["ICLabel"]["classifications"][1, 1] = 0.9

        out = eeg_icflag(eeg, thresholds)

        np.testing.assert_array_equal(out["reject"]["gcompreject"], [0, 0, 1])

    def test_missing_iclabel_raises_clear_error(self):
        eeg = _eeg()
        eeg["etc"] = {}

        with self.assertRaisesRegex(ValueError, "Run pop_iclabel first"):
            pop_icflag(eeg, DEFAULT_ICFLAG_THRESHOLDS)

    def test_missing_iclabel_in_dataset_list_raises_clear_error(self):
        eeg = _eeg()
        missing = _eeg()
        missing["etc"] = {}

        with self.assertRaisesRegex(ValueError, "Run pop_iclabel first"):
            pop_icflag([eeg, missing], DEFAULT_ICFLAG_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
