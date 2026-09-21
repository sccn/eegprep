import numpy as np
import pytest

from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from tests.eeglab_tests import eeglab_test


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


@eeglab_test("unittesting_adminfunc/eeg_store/pass_general.m", "test_pass_general")
def test_eeg_store_appends_modified_dataset_as_unsaved():
    alleeg, checked, index = eeg_store([_eeg(name="first")], _eeg(name="second", saved="no"))

    assert index == 2
    assert checked["saved"] == "no"
    assert [dataset["setname"] for dataset in alleeg] == ["first", "second"]


@eeglab_test("unittesting_adminfunc/eeg_store/pass_multiple.m", "test_pass_multiple")
def test_eeg_store_fills_lowest_empty_slot():
    # EEGLAB eeg_store puts a new dataset into the first slot whose data is empty.
    alleeg = [_eeg(name="first"), {}, _eeg(name="third")]

    alleeg, _checked, index = eeg_store(alleeg, [_eeg(name="second"), _eeg(name="fourth")])

    assert index == [2, 4]
    assert [eeg["setname"] for eeg in alleeg] == ["first", "second", "third", "fourth"]


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


@eeglab_test("unittesting_adminfunc/eeg_store/pass_new.m", "test_pass_new")
def test_eeg_store_handles_multiple_eeg_inputs_with_one_based_indices():
    alleeg, current, indices = eeg_store([], [_eeg(name="first"), _eeg(name="second")], [0, 0])

    assert indices == [1, 2]
    assert [eeg["setname"] for eeg in current] == ["first", "second"]
    assert [eeg["setname"] for eeg in alleeg] == ["first", "second"]


@eeglab_test("unittesting_adminfunc/eeg_store/pass_multiple_new.m", "test_pass_multiple_new")
def test_eeg_store_appends_three_datasets_to_existing_collection():
    alleeg, current, indices = eeg_store(
        [_eeg(name="first")],
        [_eeg(name="second"), _eeg(name="third"), _eeg(name="fourth")],
    )

    assert indices == [2, 3, 4]
    assert [eeg["setname"] for eeg in current] == ["second", "third", "fourth"]
    assert [eeg["setname"] for eeg in alleeg] == ["first", "second", "third", "fourth"]


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


@eeglab_test("unittesting_adminfunc/eeg_store/fail_num_index.m", "test_fail_num_index")
def test_eeg_store_rejects_mismatched_multiple_indices():
    with pytest.raises(ValueError, match="Length of EEG list"):
        eeg_store([], [_eeg(name="first"), _eeg(name="second")], [1])


@eeglab_test("unittesting_adminfunc/eeg_store/fail_negative_index.m", "test_fail_negative_index")
def test_eeg_store_rejects_non_positive_explicit_index():
    with pytest.raises(ValueError, match="1-based"):
        eeg_store([], _eeg(), -1)


@eeglab_test("unittesting_adminfunc/eeg_store/fail_no_arg.m", "test_fail_no_arg")
def test_eeg_store_requires_a_dataset():
    with pytest.raises(TypeError):
        eeg_store([])


@eeglab_test("unittesting_adminfunc/eeg_store/pass_bugzilla_17.m", "test_pass_bugzilla_17")
def test_eeg_store_has_no_legacy_two_hundred_dataset_limit():
    alleeg = []
    eeg = _eeg()

    for _ in range(202):
        alleeg, eeg, current = eeg_store(alleeg, eeg)

    assert len(alleeg) == 202
    assert current == 202
