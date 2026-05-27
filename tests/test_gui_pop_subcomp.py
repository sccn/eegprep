import unittest

import numpy as np

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.popfunc.pop_subcomp import pop_subcomp, pop_subcomp_dialog_spec


def _eeg():
    return {
        "data": np.arange(24, dtype=float).reshape(3, 8),
        "nbchan": 3,
        "pnts": 8,
        "trials": 1,
        "srate": 100,
        "setname": "demo",
        "icaweights": np.eye(3),
        "icasphere": np.eye(3),
        "icawinv": np.eye(3),
        "icachansind": np.arange(3),
        "icaact": np.ones((3, 8)),
        "reject": {"gcompreject": np.array([0, 1, 0]), "rejmanual": np.array([1, 0])},
        "etc": {
            "ic_classification": {
                "ICLabel": {
                    "classifications": np.array(
                        [
                            [0.8, 0.1, 0.1, 0, 0, 0, 0],
                            [0.1, 0.8, 0.1, 0, 0, 0, 0],
                            [0.1, 0.1, 0.8, 0, 0, 0, 0],
                        ]
                    )
                }
            }
        },
    }


class PopSubcompGuiTests(unittest.TestCase):
    def test_dialog_spec_defaults_to_flagged_components(self):
        spec = pop_subcomp_dialog_spec(_eeg())
        controls = controls_by_tag(spec)

        self.assertEqual(spec.title, "Remove components from data -- pop_subcomp()")
        self.assertEqual(spec.function_name, "pop_subcomp")
        self.assertEqual(spec.eeglab_source, "functions/popfunc/pop_subcomp.m")
        self.assertEqual(controls["remove"].value, "2")
        self.assertEqual(controls["retain"].value, "")

    def test_gui_result_removes_components_and_returns_replayable_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"remove": "1 3", "retain": ""}

        out, com = pop_subcomp(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["data"].shape, (3, 8))
        self.assertEqual(out["icaweights"].shape, (1, 3))
        self.assertEqual(out["icawinv"].shape, (3, 1))
        self.assertEqual(out["reject"], {})
        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["classifications"].shape, (1, 7))
        self.assertEqual(_console_python_command(com), "EEG = pop_subcomp(EEG, components=[1, 3], plotag=0)")

    def test_keep_components_removes_the_complement(self):
        out, com = pop_subcomp(_eeg(), [2], keepcomp=1, return_com=True)

        self.assertEqual(out["icaweights"].shape, (1, 3))
        np.testing.assert_array_equal(out["icaweights"], np.eye(3)[1:2])
        self.assertEqual(_console_python_command(com), "EEG = pop_subcomp(EEG, components=[2], plotag=0, keepcomp=1)")

    def test_blank_components_use_reject_flags(self):
        out, com = pop_subcomp(_eeg(), [], return_com=True)

        self.assertEqual(out["icaweights"].shape, (2, 3))
        self.assertEqual(_console_python_command(com), "EEG = pop_subcomp(EEG, components=[], plotag=0)")

    def test_missing_ica_raises_clear_error(self):
        eeg = _eeg()
        eeg["icaweights"] = np.array([])

        with self.assertRaisesRegex(ValueError, "Run pop_runica first"):
            pop_subcomp(eeg, [1])


if __name__ == "__main__":
    unittest.main()
