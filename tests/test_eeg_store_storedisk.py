from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.adminfunc.storage import OffloadedData
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_newset import pop_newset
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study


@pytest.fixture(autouse=True)
def restore_eeg_options():
    old_options = dict(EEG_OPTIONS)
    try:
        yield
    finally:
        EEG_OPTIONS.clear()
        EEG_OPTIONS.update(old_options)


def _eeg(name: str, offset: float = 0.0) -> dict:
    data = np.arange(6, dtype=np.float32).reshape((2, 3)) + offset
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": name,
            "data": data,
            "nbchan": 2,
            "pnts": 3,
            "trials": 1,
            "srate": 100.0,
            "xmin": 0.0,
            "xmax": 0.02,
            "times": np.arange(3, dtype=float),
            "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
            "event": [],
            "saved": "no",
        }
    )
    return eeg


def _saved_loaded_eeg(tmp_path: Path, name: str, offset: float = 0.0) -> dict:
    set_file = tmp_path / f"{name}.set"
    pop_saveset(_eeg(name, offset), set_file, savemode="twofiles")
    return pop_loadset(set_file)


def test_eeg_store_offloads_saved_non_current_and_retrieve_rehydrates(tmp_path: Path):
    EEG_OPTIONS["option_storedisk"] = 1
    alleeg, current, current_set = eeg_store([], _saved_loaded_eeg(tmp_path, "one"), 0)
    alleeg, current, current_set = eeg_store(alleeg, _saved_loaded_eeg(tmp_path, "two", 10), 0)

    assert current_set == 2
    assert current["setname"] == "two"
    assert isinstance(alleeg[0]["data"], OffloadedData)
    assert not isinstance(alleeg[1]["data"], OffloadedData)

    retrieved, alleeg, retrieved_set = eeg_retrieve(alleeg, 1)

    assert retrieved_set == 1
    assert retrieved["setname"] == "one"
    np.testing.assert_allclose(retrieved["data"], _eeg("one")["data"])
    assert not isinstance(alleeg[0]["data"], OffloadedData)
    assert isinstance(alleeg[1]["data"], OffloadedData)


def test_eeg_store_storedisk_refuses_unsaved_resident_dataset():
    EEG_OPTIONS["option_storedisk"] = 1
    alleeg = [_eeg("unsaved")]

    with pytest.raises(RuntimeError, match="Cannot offload unsaved dataset 1"):
        eeg_store(alleeg, _eeg("new"), 0)


def test_pop_newset_creating_new_dataset_offloads_saved_previous_dataset(tmp_path: Path):
    EEG_OPTIONS["option_storedisk"] = 1
    first = _saved_loaded_eeg(tmp_path, "first")
    second = _saved_loaded_eeg(tmp_path, "second", 20)
    alleeg, current, current_set = eeg_store([], first, 0)

    alleeg, current, current_set, command = pop_newset(alleeg, second, current_set, "setname", "second-copy")

    assert current_set == 2
    assert current["setname"] == "second-copy"
    assert isinstance(alleeg[0]["data"], OffloadedData)
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'second-copy');"


def test_pop_savestudy_resaves_offloaded_study_dataset(tmp_path: Path):
    EEG_OPTIONS["option_storedisk"] = 1
    alleeg, _current, _current_set = eeg_store([], _saved_loaded_eeg(tmp_path, "one"), 0)
    alleeg, _current, _current_set = eeg_store(alleeg, _saved_loaded_eeg(tmp_path, "two", 10), 0)
    study, alleeg = pop_study(None, alleeg, name="Storedisk study")

    saved, command = pop_savestudy(
        study,
        alleeg,
        tmp_path / "storedisk.study",
        resavedatasets="on",
        return_com=True,
    )

    assert saved["saved"] == "yes"
    assert (tmp_path / "storedisk.study").exists()
    assert command.startswith("STUDY = pop_savestudy(")
