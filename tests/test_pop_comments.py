from __future__ import annotations


import numpy as np
import numpy.testing as npt

from eegprep.functions.popfunc.pop_comments import pop_comments
from tests.eeglab_tests import eeglab_test


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


def _source_strvcat(*rows):
    width = max(map(len, rows))
    return np.array([row.ljust(width) for row in rows])


@eeglab_test("unittesting_popfunc/pop_comments/popfunc_pop_comments_wrapperTest.m", "test_pass_newcomments")
def test_reference_comments_replacement(eeglab_backend):
    comments = np.array([["Line nr. 1", "This is line nr.2", " ", "And this is the final line"]], dtype=object)
    first, second = "New line 1", "And this one is also new                             "
    new_comments = np.array([[first, second]], dtype=object)
    result, _command = eeglab_backend("pop_comments", comments, "", new_comments, nargout=2)
    # near.m compares character codes at 1e-4, hence requires exact characters.
    npt.assert_array_equal(result, _source_strvcat(first, second))


@eeglab_test("unittesting_popfunc/pop_comments/popfunc_pop_comments_wrapperTest.m", "test_pass_newcomments_concat")
def test_reference_comments_concatenation(eeglab_backend):
    old = ("Line nr. 1", "This is line nr.2", " ", "And this is the final line                           ")
    new = ("New line 1", "And this one is also new                             ")
    comments, new_comments = np.array([old], dtype=object), np.array([new], dtype=object)
    result, _command = eeglab_backend("pop_comments", comments, "", new_comments, 1.0, nargout=2)
    npt.assert_array_equal(result, _source_strvcat(*old, *new))


@eeglab_test("unittesting_popfunc/pop_comments/popfunc_pop_comments_wrapperTest.m", "test_test_pop_comments")
def test_reference_comments_original_workflow(eeglab_backend, eeglab_suite_root):
    for old in ("", "another "):
        new = "test pass!"
        new = eeglab_backend("pop_comments", old, "Testing!", new, 0.0)
        new = eeglab_backend("pop_comments", old, "Testing!", new, 1.0)
    new = np.array([["test pass!", "pass again!"]], dtype=object)
    new = eeglab_backend("pop_comments", "another ", "Testing!", new, 0.0)
    new = eeglab_backend("pop_comments", "another ", "Testing!", new, 1.0)
    eeg = eeglab_backend("pop_loadset", str(eeglab_suite_root / "eeglab/sample_data/eeglab_data.set"))
    for concatenate in (1.0, 0.0):
        eeg["comments"] = eeglab_backend(
            "pop_comments",
            eeg["comments"],
            "",
            _source_strvcat("un exemple", " ", "de nouveau dataset"),
            concatenate,
        )


def test_pop_comments_replaces_eeg_comments_without_mutating_input():
    eeg = _eeg()

    out, com = pop_comments(eeg, "", "new comment", return_com=True)

    assert eeg["comments"] == "old comment"
    assert out["comments"] == "new comment"
    assert com == "EEG = pop_comments(EEG, '', 'new comment');"


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


def test_pop_comments_current_suite_replaces_cell_comments():
    old_comments = ["Line nr. 1", "This is line nr.2", " ", "And this is the final line"]

    result = pop_comments(old_comments, "", ["New line 1", "And this one is also new"])

    assert result == "New line 1\nAnd this one is also new"


def test_pop_comments_current_suite_concatenates_cell_comments():
    old_comments = ["Line nr. 1", "This is line nr.2", " ", "And this is the final line"]

    result = pop_comments(old_comments, "", ["New line 1", "And this one is also new"], 1)

    assert result == (
        "Line nr. 1\nThis is line nr.2\n\nAnd this is the final line\nNew line 1\nAnd this one is also new"
    )


def test_pop_comments_current_suite_string_cell_and_dataset_workflow():
    assert pop_comments("", "Testing!", "test pass!", 0) == "test pass!"
    assert pop_comments("", "Testing!", "test pass!", 1) == "test pass!"
    assert pop_comments("another ", "Testing!", "test pass!", 0) == "test pass!"
    assert pop_comments("another ", "Testing!", "test pass!", 1) == "another\ntest pass!"
    assert pop_comments("another ", "Testing!", ["test pass!", "pass again!"], 0) == "test pass!\npass again!"
    assert pop_comments("another ", "Testing!", ["test pass!", "pass again!"], 1) == (
        "another\ntest pass!\npass again!"
    )

    eeg = _eeg()
    eeg["comments"] = pop_comments(eeg["comments"], "", ["un exemple", " ", "de nouveau dataset"], 1)
    assert eeg["comments"].endswith("un exemple\n\nde nouveau dataset")
    eeg["comments"] = pop_comments(eeg["comments"], "", ["un exemple", " ", "de nouveau dataset"], 0)
    assert eeg["comments"] == "un exemple\n\nde nouveau dataset"
