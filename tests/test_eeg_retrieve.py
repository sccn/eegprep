import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from tests.eeglab_tests import eeglab_test


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


@eeglab_test("unittesting_adminfunc/eeg_retrieve/pass_general.m", "test_pass_general")
def test_eeg_retrieve_returns_deepcopy_and_one_based_index():
    first = _eeg(name="first")
    second = _eeg(name="second")
    selected, alleeg, current = eeg_retrieve([first, second], 2)

    selected["setname"] = "changed"

    assert current == 2
    assert alleeg[0]["setname"] == "first"
    assert alleeg[1]["setname"] == "second"


@eeglab_test("unittesting_adminfunc/eeg_retrieve/pass_multiple.m", "test_pass_multiple")
def test_eeg_retrieve_handles_multiple_indices_and_empty_slots():
    selected, _alleeg, current = eeg_retrieve([_eeg(name="first"), {}, _eeg(name="third")], [3, 2, 1])

    assert current == [3, 2, 1]
    assert [eeg["setname"] for eeg in selected] == ["third", "", "first"]
    assert selected[1]["ref"] == "common"


def test_eeg_retrieve_accepts_tuple_indices():
    selected, _alleeg, current = eeg_retrieve([_eeg(name="first"), _eeg(name="second")], (2,))

    assert current == [2]
    assert [eeg["setname"] for eeg in selected] == ["second"]


@eeglab_test("unittesting_adminfunc/eeg_retrieve/fail_outside.m", "test_fail_outside")
def test_eeg_retrieve_rejects_negative_and_missing_indices():
    with pytest.raises(ValueError, match="1-based"):
        eeg_retrieve([_eeg()], -1)
    with pytest.raises(IndexError, match="No dataset"):
        eeg_retrieve([_eeg()], 2)


@eeglab_test("unittesting_adminfunc/eeg_retrieve/pass_zero.m", "test_pass_zero")
def test_eeg_retrieve_zero_returns_empty_dataset_without_changing_alleeg():
    source = [_eeg(name="first"), _eeg(name="second")]

    selected, alleeg, current = eeg_retrieve(source, 0)

    assert current == 0
    assert selected.keys() == eeg_emptyset().keys()
    assert selected["setname"] == ""
    assert selected["nbchan"] == 0
    assert np.asarray(selected["data"]).size == 0
    assert [eeg["setname"] for eeg in alleeg] == ["first", "second"]


@eeglab_test("unittesting_adminfunc/eeg_retrieve/fail_no_arg.m", "test_fail_no_arg")
def test_eeg_retrieve_requires_a_dataset_index():
    with pytest.raises(TypeError):
        eeg_retrieve([_eeg()])


def test_eeg_retrieve_leaves_a_deleted_slot_empty():
    # Retrieving a deleted slot yields an empty EEG, but must not turn the slot into a dataset.
    alleeg = [_eeg(name="first"), {}, _eeg(name="third")]

    selected, alleeg, current = eeg_retrieve(alleeg, 2)

    assert current == 2
    assert selected["setname"] == ""
    assert alleeg[1] == {}


def test_eeg_retrieve_list_leaves_deleted_slots_empty():
    alleeg = [_eeg(name="first"), {}, _eeg(name="third")]

    _selected, alleeg, _current = eeg_retrieve(alleeg, [1, 2, 3])

    assert alleeg[1] == {}
    assert [alleeg[0]["setname"], alleeg[2]["setname"]] == ["first", "third"]
