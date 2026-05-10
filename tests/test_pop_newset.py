import numpy as np
import pytest

from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_newset import pop_newset


def _eeg(*, name: str = "demo") -> dict:
    pnts = 4
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": name,
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
            "saved": "no",
        }
    )
    return eeg


def test_pop_newset_stores_setname_and_retrieves_dataset():
    alleeg, current, current_set, command = pop_newset([], _eeg(), 0, "setname", "renamed")

    assert current_set == 1
    assert current["setname"] == "renamed"
    assert alleeg[0]["setname"] == "renamed"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"

    _alleeg, retrieved, retrieved_set, retrieve_command = pop_newset(alleeg, current, current_set, "retrieve", 1)

    assert retrieved_set == 1
    assert retrieved["setname"] == "renamed"
    assert retrieve_command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', 1);"


def test_pop_newset_rejects_unknown_keyword_options():
    with pytest.raises(ValueError, match="Unsupported pop_newset"):
        pop_newset([], _eeg(), 0, unknown=True)


def test_pop_newset_empty_retrieve_option_stores_current_dataset():
    alleeg, current, current_set, command = pop_newset([], _eeg(name="stored"), 0, "retrieve", [])

    assert current_set == 1
    assert current["setname"] == "stored"
    assert alleeg[0]["setname"] == "stored"
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"
