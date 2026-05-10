import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


def _eeg(*, name: str = "demo", saved: str = "no") -> dict:
    pnts = 4
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": name,
            "filename": f"{name}.set",
            "filepath": "/tmp",
            "data": np.zeros((1, pnts), dtype=np.float32),
            "nbchan": 1,
            "pnts": pnts,
            "trials": 1,
            "srate": 100,
            "xmin": 0.0,
            "xmax": (pnts - 1) / 100,
            "times": np.arange(pnts, dtype=float),
            "chanlocs": [{"labels": "Cz", "theta": 0.0, "radius": 0.0, "ref": "common"}],
            "event": np.array([{"type": "stim", "latency": 1}], dtype=object),
            "icaweights": np.eye(1),
            "icasphere": np.eye(1),
            "icawinv": np.eye(1),
            "icachansind": np.arange(1),
            "saved": saved,
        }
    )
    return eeg


def test_eeg_store_appends_modified_dataset_as_unsaved():
    alleeg, checked, index = eeg_store([], _eeg(saved="no"), 0)

    assert index == 1
    assert checked["saved"] == "no"
    assert alleeg[0]["saved"] == "no"


def test_eeg_store_preserves_justloaded_dataset_as_saved():
    alleeg, checked, index = eeg_store([], _eeg(saved="justloaded"), 0)

    assert index == 1
    assert checked["saved"] == "yes"
    assert alleeg[0]["saved"] == "yes"


def test_eeg_store_marks_saved_dataset_unsaved_without_justloaded_marker():
    alleeg, checked, index = eeg_store([], _eeg(saved="yes"), 0)

    assert index == 1
    assert checked["saved"] == "no"
    assert alleeg[0]["saved"] == "no"


def test_eeg_store_handles_multiple_eeg_inputs_with_one_based_indices():
    alleeg, current, indices = eeg_store([], [_eeg(name="first"), _eeg(name="second")], [0, 0])

    assert indices == [1, 2]
    assert [eeg["setname"] for eeg in current] == ["first", "second"]
    assert [eeg["setname"] for eeg in alleeg] == ["first", "second"]


def test_eeg_store_replaces_existing_one_based_slot():
    existing = [_eeg(name="first", saved="yes"), _eeg(name="second", saved="yes")]

    alleeg, checked, index = eeg_store(existing, _eeg(name="replacement", saved="no"), 2)

    assert index == 2
    assert checked["setname"] == "replacement"
    assert checked["saved"] == "no"
    assert [eeg["setname"] for eeg in alleeg] == ["first", "replacement"]


def test_eeg_store_appends_when_index_omitted_or_none():
    alleeg, checked, index = eeg_store(None, _eeg(name="first"), None)

    assert index == 1
    assert checked["setname"] == "first"
    assert alleeg[0]["setname"] == "first"


def test_eeg_store_rejects_mismatched_multiple_indices():
    with pytest.raises(ValueError, match="Length of EEG list"):
        eeg_store([], [_eeg(name="first"), _eeg(name="second")], [1])


def test_eeg_store_rejects_non_positive_explicit_index():
    with pytest.raises(ValueError, match="1-based"):
        eeg_store([], _eeg(), -1)
