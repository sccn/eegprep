from __future__ import annotations

import copy
import os
import unittest

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.popfunc.pop_editset import pop_editset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from tests.fixtures import SAMPLE_DATASET_PATH


def _eeg():
    return {
        "setname": "demo",
        "filename": "",
        "filepath": "",
        "subject": "",
        "condition": "",
        "group": "",
        "session": "",
        "run": "",
        "comments": "",
        "data": np.zeros((2, 20), dtype=np.float32),
        "nbchan": 2,
        "pnts": 20,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.19,
        "times": np.arange(20),
        "ref": "common",
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "urchanlocs": [],
        "chaninfo": {},
        "event": [{"type": "stim", "latency": 10.0}],
        "urevent": [],
        "epoch": [],
        "icaweights": np.array([]),
        "icasphere": np.array([]),
        "icawinv": np.array([]),
        "icaact": np.array([]),
        "icachansind": np.array([]),
    }


def test_pop_editset_updates_dataset_metadata_without_mutating_input():
    eeg = _eeg()

    out, com = pop_editset(
        eeg,
        "setname",
        "edited",
        "subject",
        "S01",
        "condition",
        "targets",
        "group",
        "control",
        "run",
        2,
        "session",
        3,
        "comments",
        "notes",
        return_com=True,
    )

    assert eeg["setname"] == "demo"
    assert out["setname"] == "edited"
    assert out["subject"] == "S01"
    assert out["condition"] == "targets"
    assert out["group"] == "control"
    assert out["run"] == 2
    assert out["session"] == 3
    assert out["comments"] == "notes"
    assert com == (
        "EEG = pop_editset(EEG, 'setname', 'edited', 'subject', 'S01', "
        "'condition', 'targets', 'group', 'control', 'run', 2, 'session', 3, 'comments', 'notes');"
    )


def test_pop_editset_updates_timing_fields_and_shifts_event_latencies_for_xmin():
    eeg = _eeg()

    out = pop_editset(eeg, srate=200, pnts=10, xmin=-0.1)

    assert out["srate"] == 200
    assert out["pnts"] == 10
    assert out["xmin"] == -0.1
    assert out["xmax"] == pytest.approx(-0.055)
    assert out["event"][0]["latency"] == 20.0
    np.testing.assert_allclose(out["times"], np.linspace(-100.0, -55.0, 10))


def test_pop_editset_gui_renderer_returns_only_changed_fields():
    class Renderer:
        def __init__(self):
            self.spec = None

        def run(self, spec, initial_values=None):
            self.spec = spec
            return {
                "setname": "gui edited",
                "srate": "100",
                "subject": "S02",
                "pnts": "20",
                "condition": "",
                "xmin": "0",
                "group": "",
                "run": "",
                "nbchan": "2",
                "ref": "common",
                "session": "",
            }

    renderer = Renderer()
    out, com = pop_editset(_eeg(), gui=True, renderer=renderer, return_com=True)

    assert renderer.spec.title == "Edit dataset information - pop_editset()"
    assert renderer.spec.function_name == "pop_editset"
    assert out["setname"] == "gui edited"
    assert out["subject"] == "S02"
    assert com == "EEG = pop_editset(EEG, 'setname', 'gui edited', 'subject', 'S02');"


def test_pop_editset_gui_cancel_or_no_changes_returns_original_without_history():
    class Renderer:
        def __init__(self, result):
            self.result = result

        def run(self, spec, initial_values=None):
            return self.result

    eeg = _eeg()
    cancelled, com = pop_editset(eeg, gui=True, renderer=Renderer(None), return_com=True)
    assert cancelled is eeg
    assert com == ""

    unchanged, com = pop_editset(
        eeg,
        gui=True,
        renderer=Renderer(
            {
                "setname": "demo",
                "srate": "100",
                "subject": "",
                "pnts": "20",
                "condition": "",
                "xmin": "0",
                "group": "",
                "run": "",
                "nbchan": "2",
                "ref": "common",
                "session": "",
            }
        ),
        return_com=True,
    )
    assert unchanged is eeg
    assert com == ""


def test_pop_editset_direct_data_and_ica_assignments_update_shape_and_clear_ica_inverse():
    data = np.ones((3, 5, 2), dtype=np.float32)
    weights = np.eye(3)
    out = pop_editset(
        _eeg(),
        "data",
        data,
        "icaweights",
        weights,
        "icachansind",
        [0, 1, 2],
    )

    assert out["data"].shape == (3, 5, 2)
    assert out["nbchan"] == 3
    assert out["pnts"] == 5
    assert out["trials"] == 2
    np.testing.assert_array_equal(out["icaweights"], weights)
    np.testing.assert_array_equal(out["icasphere"], np.eye(3))
    assert out["icawinv"].size == 0
    assert out["icaact"].shape == (3, 5, 2)
    np.testing.assert_array_equal(out["icachansind"], [0, 1, 2])


def test_pop_editset_rejects_unsupported_file_workspace_expressions():
    with pytest.raises(NotImplementedError, match="pop_importdata"):
        pop_editset(_eeg(), "data", "raw.mat")
    with pytest.raises(NotImplementedError, match="pop_chanedit"):
        pop_editset(_eeg(), "chanlocs", "locs.elp")


def test_pop_editset_accepts_sample_data_metadata_edit():
    eeg = pop_loadset(str(SAMPLE_DATASET_PATH))

    out, com = pop_editset(eeg, "setname", "sample edited", "subject", "S99", return_com=True)

    assert out["setname"] == "sample edited"
    assert out["subject"] == "S99"
    assert eeg["setname"] != "sample edited"
    assert com == "EEG = pop_editset(EEG, 'setname', 'sample edited', 'subject', 'S99');"


@unittest.skipIf(os.getenv("EEGPREP_SKIP_MATLAB") == "1", "MATLAB not available")
class TestPopEditsetMatlabParity(unittest.TestCase):
    def setUp(self):
        try:
            self.eeglab = get_eeglab("MAT")
        except Exception as exc:
            self.skipTest(f"MATLAB not available: {exc}")

    def test_metadata_fields_match_eeglab(self):
        eeg = _eeg()

        py_out = pop_editset(
            copy.deepcopy(eeg),
            "setname",
            "matlab edited",
            "subject",
            "S03",
            "condition",
            "oddball",
            "group",
            "patient",
            "run",
            4,
            "session",
            5,
        )
        ml_out = self.eeglab.pop_editset(
            copy.deepcopy(eeg),
            "setname",
            "matlab edited",
            "subject",
            "S03",
            "condition",
            "oddball",
            "group",
            "patient",
            "run",
            4,
            "session",
            5,
        )

        for key in ("setname", "subject", "condition", "group"):
            self.assertEqual(py_out[key], ml_out[key])
        self.assertEqual(int(py_out["run"]), int(ml_out["run"]))
        self.assertEqual(int(py_out["session"]), int(ml_out["session"]))

    def test_xmin_latency_shift_matches_eeglab(self):
        eeg = _eeg()

        py_out = pop_editset(copy.deepcopy(eeg), "xmin", -0.1)
        ml_out = self.eeglab.pop_editset(copy.deepcopy(eeg), "xmin", -0.1)

        self.assertAlmostEqual(float(py_out["xmin"]), float(ml_out["xmin"]))
        self.assertAlmostEqual(float(py_out["event"][0]["latency"]), float(ml_out["event"][0]["latency"]))
