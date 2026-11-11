import unittest
from unittest import mock

import numpy as np

from eegprep.plugins.ICLabel.iclabel import iclabel
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel


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
    def test_gui_result_runs_iclabel_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"icversion": 1}

        eeg = _eeg()
        updated = dict(eeg, etc={"ic_classification": {"ICLabel": {"version": "default"}}})
        with mock.patch("eegprep.plugins.ICLabel.pop_iclabel.iclabel", return_value=updated) as classify:
            out, com = pop_iclabel(eeg, gui=True, renderer=Renderer(), return_com=True)

        classify.assert_called_once_with(eeg, algorithm="default", engine=None, feature="pop_iclabel")
        self.assertEqual(out["etc"]["ic_classification"]["ICLabel"]["version"], "default")
        self.assertEqual(com, "EEG = pop_iclabel(EEG, 'default');")

    def test_missing_ica_names_pop_runica_before_the_network_runs(self):
        eeg = _eeg()
        eeg["icaweights"] = np.array([])

        with self.assertRaisesRegex(ValueError, "Run pop_runica"):
            pop_iclabel(eeg, icversion="default", gui=False)

    def test_iclabel_refuses_an_empty_weight_matrix(self):
        with self.assertRaisesRegex(ValueError, "Run pop_runica"):
            iclabel(
                {"data": np.zeros((2, 8)), "icaweights": np.array([]), "etc": {}},
                feature="pop_iclabel",
            )


if __name__ == "__main__":
    unittest.main()
