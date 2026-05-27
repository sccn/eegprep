from __future__ import annotations

import copy
import os
import unittest

import numpy as np
import numpy.testing as npt
import pytest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_comments import pop_comments
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.fixtures import SAMPLE_DATASET_PATH


def _eeg():
    return {
        "setname": "demo",
        "comments": "old comment",
        "data": np.zeros((1, 4), dtype=np.float32),
        "nbchan": 1,
        "pnts": 4,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.03,
        "times": np.arange(4),
        "event": [],
        "urevent": [],
        "epoch": [],
    }


def test_pop_comments_replaces_eeg_comments_without_mutating_input():
    eeg = _eeg()

    out, com = pop_comments(eeg, "", "new comment", return_com=True)

    assert eeg["comments"] == "old comment"
    assert out["comments"] == "new comment"
    assert com == "EEG = pop_comments(EEG, '', 'new comment');"


def test_pop_comments_concatenates_string_comments():
    out, com = pop_comments("first", "", ["second", "third"], 1, return_com=True)

    assert out == "first\nsecond\nthird"
    assert com == "comments = pop_comments(comments, '', 'second\nthird', 1);"


def test_pop_comments_concat_accepts_only_eeglab_numeric_flag():
    with pytest.raises(TypeError, match="concat"):
        pop_comments("first", "", "second", "on")


def test_pop_comments_gui_uses_renderer_text_and_cancel_returns_original():
    class Renderer:
        def __init__(self, result):
            self.result = result
            self.spec = None

        def run(self, spec, initial_values=None):
            self.spec = spec
            return self.result

    renderer = Renderer({"comments": "from gui"})
    eeg = _eeg()

    out, com = pop_comments(eeg, "About this dataset", gui=True, renderer=renderer, return_com=True)

    assert renderer.spec.title == "Read/Enter text -- pop_comments()"
    assert renderer.spec.function_name == "pop_comments"
    assert out["comments"] == "from gui"
    assert com == "EEG = pop_comments(EEG, '', 'from gui');"

    cancelled = pop_comments(eeg, gui=True, renderer=Renderer(None))
    assert cancelled is not eeg
    npt.assert_array_equal(cancelled["data"], eeg["data"])
    assert cancelled["comments"] == eeg["comments"]
    assert cancelled["setname"] == eeg["setname"]


def test_pop_comments_accepts_sample_data_comments():
    eeg = pop_loadset(str(SAMPLE_DATASET_PATH))

    out = pop_comments(eeg, "", "sample-data note")

    assert out["comments"] == "sample-data note"
    assert str(eeg.get("comments", "")) != "sample-data note"


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestPopCommentsMatlabParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")

    def test_replaces_eeg_comments_like_eeglab(self):
        eeg = _eeg()

        py_out = pop_comments(copy.deepcopy(eeg), "", "matlab parity note")
        ml_out = self.eeglab.pop_comments(copy.deepcopy(eeg), "", "matlab parity note")

        self.assertEqual(py_out["comments"], ml_out["comments"])
        self.assertEqual(py_out["setname"], ml_out["setname"])
