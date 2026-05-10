import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


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


def test_eeg_retrieve_returns_deepcopy_and_one_based_index():
    source = _eeg(name="source")
    selected, alleeg, current = eeg_retrieve([source], 1)

    selected["setname"] = "changed"

    assert current == 1
    assert alleeg[0]["setname"] == "source"


def test_eeg_retrieve_handles_multiple_indices_and_empty_slots():
    selected, _alleeg, current = eeg_retrieve([_eeg(name="first"), {}, _eeg(name="third")], [1, 2, 3])

    assert current == [1, 2, 3]
    assert [eeg["setname"] for eeg in selected] == ["first", "", "third"]
    assert selected[1]["ref"] == "common"


def test_eeg_retrieve_accepts_tuple_indices():
    selected, _alleeg, current = eeg_retrieve([_eeg(name="first"), _eeg(name="second")], (2,))

    assert current == [2]
    assert [eeg["setname"] for eeg in selected] == ["second"]


def test_eeg_retrieve_rejects_zero_and_missing_indices():
    with pytest.raises(ValueError, match="1-based"):
        eeg_retrieve([_eeg()], 0)
    with pytest.raises(IndexError, match="No dataset"):
        eeg_retrieve([_eeg()], 2)
