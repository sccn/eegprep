import numpy as np

from eegprep.functions.adminfunc.eeg_store import eeg_store
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from tests.eeglab_tests import assert_matlab_equal, assert_matlab_near, eeglab_test


EEG_STORE_WRAPPER = "unittesting_adminfunc/eeg_store/adminfunc_eeg_store_wrapperTest.m"


def _source_eeg(eeglab_backend, value):
    eeg = eeglab_backend("eeg_emptyset")
    eeg.update(nbchan=1.0, trials=1.0, pnts=1.0, srate=1.0, xmin=0.0, xmax=0.0, data=np.array([[float(value)]]))
    return eeg


def _source_dataset_row(*datasets):
    return np.array(
        [[tuple(dataset.values()) for dataset in datasets]], dtype=[(field, object) for field in datasets[0]]
    )


def _source_data_cell(*datasets):
    values = np.empty((1, len(datasets)), dtype=object)
    for index, dataset in enumerate(datasets):
        values[0, index] = dataset["data"]
    return values


@eeglab_test(EEG_STORE_WRAPPER, "test_pass_general")
def test_reference_eeg_store_original_general_case(eeglab_backend):
    first, second = (_source_eeg(eeglab_backend, value) for value in (1, 2))
    alleeg, _eeg, _current = eeglab_backend("eeg_store", _source_dataset_row(first), second, nargout=3)
    assert_matlab_near(alleeg[0, 0]["data"], first["data"])
    assert_matlab_near(alleeg[0, 1]["data"], second["data"])


@eeglab_test(EEG_STORE_WRAPPER, "test_pass_multiple")
def test_reference_eeg_store_original_multiple_case(eeglab_backend):
    first, second, third, fourth = (_source_eeg(eeglab_backend, value) for value in (1, 2, 3, 4))
    # Assigning MATLAB struct elements 1 and 3 leaves every field in slot 2 empty.
    empty = {field: np.empty((0, 0)) for field in first}
    alleeg, _eeg, _current = eeglab_backend(
        "eeg_store", _source_dataset_row(first, empty, second), _source_dataset_row(third, fourth), nargout=3
    )
    assert_matlab_equal(alleeg["data"], _source_data_cell(first, third, second, fourth))


@eeglab_test(EEG_STORE_WRAPPER, "test_pass_multiple_new")
def test_reference_eeg_store_original_multiple_new_case(eeglab_backend):
    first, second, third, fourth = (_source_eeg(eeglab_backend, value) for value in (1, 2, 3, 4))
    alleeg, _eeg, _current = eeglab_backend(
        "eeg_store", _source_dataset_row(first), _source_dataset_row(second, third, fourth), nargout=3
    )
    assert_matlab_near(
        np.hstack([dataset["data"] for dataset in alleeg.flat]),
        np.hstack([dataset["data"] for dataset in (first, second, third, fourth)]),
    )


@eeglab_test(EEG_STORE_WRAPPER, "test_pass_new")
def test_reference_eeg_store_original_new_case(eeglab_backend):
    first, second = (_source_eeg(eeglab_backend, value) for value in (1, 2))
    alleeg, _eeg, _current = eeglab_backend(
        "eeg_store", np.empty((0, 0)), _source_dataset_row(second, first), nargout=3
    )
    assert_matlab_near(alleeg[0, 0]["data"], second["data"])
    assert_matlab_near(alleeg[0, 1]["data"], first["data"])


@eeglab_test(EEG_STORE_WRAPPER, "test_pass_bugzilla_17")
def test_reference_eeg_store_original_202_dataset_smoke(eeglab_backend):
    eeg = _source_eeg(eeglab_backend, 1)
    alleeg = np.empty((0, 0))
    eeg = eeglab_backend("eeg_checkset", eeg)
    # The source reduces this only for Octave, which is not a contract backend.
    for _index in range(202):
        alleeg, eeg, _current = eeglab_backend("eeg_store", alleeg, eeg, nargout=3)


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
    alleeg, checked, index = eeg_store([_eeg(name="first")], _eeg(name="second", saved="no"))

    assert index == 2
    assert checked["saved"] == "no"
    assert [dataset["setname"] for dataset in alleeg] == ["first", "second"]


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
