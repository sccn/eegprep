import unittest

import numpy as np

from eegprep.functions.popfunc.pop_resample import pop_resample


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
    def test_gui_result_resamples_and_returns_history(self):
        class Renderer:
            def run(self, spec, initial_values=None):
                return {"freq": "50"}

        out, com = pop_resample(_eeg(), gui=True, renderer=Renderer(), return_com=True)

        self.assertEqual(out["srate"], 50)
        self.assertEqual(out["pnts"], 10)
        self.assertEqual(com, "EEG = pop_resample( EEG, 50);")


if __name__ == "__main__":
    unittest.main()
