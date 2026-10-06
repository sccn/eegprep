import unittest

import numpy as np

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.plugins.ICLabel.eeg_icflag import eeg_icflag
from eegprep.plugins.ICLabel.pop_icflag import DEFAULT_ICFLAG_THRESHOLDS, pop_icflag


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


if __name__ == "__main__":
    unittest.main()
