import unittest
from unittest import mock

import numpy as np

from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel, pop_iclabel_dialog_spec


def _eeg():
    return {
        "data": np.zeros((2, 20), dtype=np.float32),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100,
        "icaweights": np.eye(2),
        "icasphere": np.eye(2),
        "icawinv": np.eye(2),
        "icachansind": np.arange(2),
        "etc": {},
    }


class PopIclabelGuiTests(unittest.TestCase):
    def test_gui_dialog_spec_matches_iclabel_prompt(self):
        spec = pop_iclabel_dialog_spec()

        self.assertEqual(spec.title, "ICLabel")
        self.assertEqual(spec.function_name, "pop_iclabel")
        self.assertEqual(spec.eeglab_source, "plugins/ICLabel/pop_iclabel.m")
        self.assertEqual([(control.style, control.string, control.tag) for control in spec.controls], [
            ("text", "Select which icversion of ICLabel to use:", None),
            ("popupmenu", "Default (recommended)|Lite|Beta", "icversion"),
        ])

    def test_gui_result_runs_iclabel_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icversion": 2}

        eeg = _eeg()
        updated = dict(eeg, etc={"ic_classification": {"ICLabel": {"version": "lite"}}})
        with mock.patch("eegprep.plugins.ICLabel.pop_iclabel.iclabel", return_value=updated) as classify:
            out, com = pop_iclabel(eeg, gui=True, renderer=Renderer(), return_com=True)

        classify.assert_called_once_with(eeg, algorithm="lite", engine=None)
        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["version"], "lite")
        self.assertEqual(com, "EEG = pop_iclabel(EEG, 'lite');")

    def test_missing_ica_raises_clear_error(self):
        eeg = dict(_eeg(), icaweights=np.array([]))

        with self.assertRaisesRegex(ValueError, "requires an ICA decomposition"):
            pop_iclabel(eeg, "default")


if __name__ == "__main__":
    unittest.main()
