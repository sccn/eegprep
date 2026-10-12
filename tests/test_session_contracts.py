from __future__ import annotations


import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.adminfunc.storage import MemmapData
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.plugins.ICLabel.iclabel import iclabel
from tests.fixtures import (
    create_test_eeg,
)
from tests.eeglab_tests import eeglab_test


@eeglab_test("unittesting_adminfunc/eeg_global/adminfunc_eeg_global_wrapperTest.m", "test_pass_general")
def test_reference_global_workspace_declarations(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        exists = eeglab_backend("eegprep_source_eeg_global")
    else:
        workspace = EEGPrepConsoleWorkspace(EEGPrepSession(), exports={})
        names = ("EEG", "ALLEEG", "CURRENTSET", "LASTCOM", "ALLCOM")
        try:
            for name in names:
                workspace.namespace.pop(name, None)
            workspace.pull_from_session()
            exists = [[name in workspace.namespace for name in names]]
        finally:
            workspace.close()
    assert np.all(exists)


def test_multi_dataset_selection_contract_preserves_order_and_console_shape():
    session = EEGPrepSession()
    for name in ("first", "second", "third"):
        eeg = create_test_eeg(n_channels=2, n_samples=8)
        eeg["setname"] = name
        session.store_current(eeg, new=True)

    selected = session.retrieve([3, 1])
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    try:
        assert isinstance(selected, list)
        assert [dataset["setname"] for dataset in selected] == ["third", "first"]
        assert session.CURRENTSET == [3, 1]
        assert session.selected_dataset_indices() == [3, 1]
        assert session.current_set_value() == [3, 1]
        assert workspace.namespace["EEG"] is session.EEG
        assert workspace.namespace["CURRENTSET"] == [3, 1]
    finally:
        workspace.close()


def test_currentset_empty_single_and_multiple_console_values():
    session = EEGPrepSession()
    assert session.current_set_value() == 0

    session.store_current(create_test_eeg(n_channels=2, n_samples=8), new=True)
    assert session.current_set_value() == 1

    session.store_current(create_test_eeg(n_channels=2, n_samples=8), new=True)
    session.retrieve([1, 2])
    assert session.current_set_value() == [1, 2]


def _session_with(*setnames):
    session = EEGPrepSession()
    for setname in setnames:
        eeg = create_test_eeg(n_channels=2, n_samples=8)
        eeg["setname"] = setname
        eeg["etc"] = {"note": "kept"}
        session.store_current(eeg, new=True)
    return session


def test_dataset_state_token_changes_when_a_dataset_is_stored():
    session = EEGPrepSession()
    token = session.dataset_state_token()

    session.store_current(create_test_eeg(n_channels=2, n_samples=8), new=True)

    assert session.dataset_state_token() != token


@pytest.mark.parametrize(
    "change",
    (
        lambda session: session.add_history("plot(EEG);"),
        lambda session: session.mark_current_saved(),
        lambda session: session.EEG.update(history="edited history"),
        lambda session: session.ALLEEG[0].update(setname="unselected edit"),
    ),
    ids=("session-history", "saved-flag", "dataset-history", "unselected-dataset"),
)
def test_dataset_state_token_ignores_history_saved_flag_and_unselected_datasets(change):
    session = _session_with("unselected", "selected")
    token = session.dataset_state_token()

    change(session)

    assert session.dataset_state_token() == token


@pytest.mark.parametrize(
    "change",
    (
        lambda session: session.EEG["data"].__setitem__((0, 0), 9.0),
        lambda session: np.putmask(session.EEG["data"], session.EEG["data"] > 0, 0.0),
        lambda session: session.EEG["chanlocs"][0].update(labels="renamed"),
        lambda session: session.EEG["etc"].update(history="nested history is dataset content"),
        lambda session: session.ALLEEG[1].update(setname="edited slot"),
    ),
    ids=("data-sample", "putmask", "nested-chanloc", "nested-history-key", "selected-slot-alias"),
)
def test_dataset_state_token_detects_in_place_edits_of_selected_datasets(change):
    session = _session_with("unselected", "selected")
    session.retrieve(2)
    assert session.EEG is not session.ALLEEG[1]
    token = session.dataset_state_token()

    change(session)

    assert session.dataset_state_token() != token


def test_dataset_state_token_detects_edit_in_last_chunk_of_fortran_data():
    session = EEGPrepSession()
    eeg = create_test_eeg(n_channels=2, n_samples=8)
    eeg["data"] = np.asfortranarray(np.zeros((3, 700_000), dtype=np.float32))
    eeg["nbchan"], eeg["pnts"] = eeg["data"].shape
    session.store_current(eeg, new=True)
    assert session.EEG["data"].flags.f_contiguous
    token = session.dataset_state_token()

    session.EEG["data"][-1, -1] = 1.0

    assert session.dataset_state_token() != token


def test_dataset_state_token_matches_again_after_an_edit_is_reverted():
    # Only content is compared, so a result computed from the restored values is still valid.
    session = _session_with("selected")
    token = session.dataset_state_token()
    original = session.EEG["data"][0, 0]

    session.EEG["data"][0, 0] = original + 1.0
    assert session.dataset_state_token() != token
    session.EEG["data"][0, 0] = original

    assert session.dataset_state_token() == token


def test_dataset_state_token_detects_selected_slot_replacement_without_notification():
    session = EEGPrepSession()
    session.store_current(create_test_eeg(n_channels=2, n_samples=8), new=True)
    token = session.dataset_state_token()
    replacement = create_test_eeg(n_channels=2, n_samples=8)
    session.ALLEEG[0] = replacement
    session.EEG = replacement

    assert session.dataset_state_token() != token


def test_dataset_state_token_detects_memmap_data_mutation(tmp_path):
    path = tmp_path / "data.fdt"
    backing = np.memmap(path, dtype="float32", mode="w+", shape=(2, 2), order="F")
    backing[:] = [[1.0, 2.0], [3.0, 4.0]]
    backing.flush()

    eeg = create_test_eeg(n_channels=2, n_samples=2)
    eeg["data"] = MemmapData(path, (2, 2), mode="r+")
    session = EEGPrepSession()
    session.store_current(eeg, new=True)
    token = session.dataset_state_token()

    np.asarray(session.EEG["data"])[0, 0] = 9.0

    assert session.dataset_state_token() != token


# Values other than dicts, lists, tuples, and arrays are hashed through repr(), so an
# in-place edit that leaves repr() unchanged, such as changing a stored value of a
# scipy.sparse matrix, is not detected.
def test_dataset_state_token_detects_edits_in_the_iclabel_classes_object_array(sample_ica_eeg):
    session = EEGPrepSession()
    session.store_current(iclabel(sample_ica_eeg), new=True)
    classes = session.EEG["etc"]["ic_classification"]["ICLabel"]["classes"]
    assert classes.dtype == object
    token = session.dataset_state_token()

    classes[0] = "Renamed"

    assert session.dataset_state_token() != token


def test_dataset_state_token_detects_edits_of_an_array_held_in_an_object_array():
    session = _session_with("selected")
    cells = np.empty(2, dtype=object)
    cells[0], cells[1] = np.zeros(3), "label"
    session.EEG["etc"]["cells"] = cells
    token = session.dataset_state_token()

    cells[0][1] = 5.0

    assert session.dataset_state_token() != token


def test_dataset_state_token_follows_an_object_array_that_contains_itself():
    session = _session_with("selected")
    cells = np.empty(2, dtype=object)
    cells[0], cells[1] = cells, "label"
    session.EEG["etc"]["cells"] = cells
    token = session.dataset_state_token()

    cells[1] = "edited"

    assert session.dataset_state_token() != token


def test_dataset_state_token_follows_reference_cycles_and_still_detects_edits():
    session = _session_with("selected")
    etc = session.EEG["etc"]
    holder = np.empty(1, dtype=object)
    holder[0] = etc
    etc["self"] = etc
    etc["cells"] = holder
    token = session.dataset_state_token()

    etc["note"] = "edited inside the cycle"

    assert session.dataset_state_token() != token


def test_dataset_state_token_hashes_a_strided_view_by_its_visible_content():
    session = _session_with("selected")
    # The view spans several buffered digest chunks.
    base = np.zeros((3, 1_400_000), dtype=np.float32)
    session.EEG["data"] = base[:, ::2]
    token = session.dataset_state_token()

    base[-1, -1] = 1.0
    assert session.dataset_state_token() == token

    base[-1, -2] = 1.0
    assert session.dataset_state_token() != token


def test_require_dataset_state_unchanged_names_the_function_and_the_change():
    session = _session_with("first", "second")
    token = session.dataset_state_token()
    session.require_dataset_state_unchanged(token, "pop_iclabel_async")

    session.EEG["setname"] = "edited"
    with pytest.raises(
        RuntimeError, match=r"pop_iclabel_async result discarded because the selected dataset was modified"
    ):
        session.require_dataset_state_unchanged(token, "pop_iclabel_async")

    session.retrieve(1)
    with pytest.raises(
        RuntimeError, match=r"ICLabel result discarded because the dataset selection changed.*Run ICLabel again"
    ):
        session.require_dataset_state_unchanged(token, "ICLabel")


@pytest.mark.parametrize(
    ("source", "value"),
    (
        ("CURRENTSET = [1, 1]", [1, 1]),
        ("CURRENTSET = -1", -1),
        ("CURRENTSET = [1, -1, 2]", [1, -1, 2]),
        ("CURRENTSET = [0, 1]", [0, 1]),
    ),
)
def test_console_currentset_assignment_rejects_invalid_indices(source, value):
    session = EEGPrepSession()
    for name in ("first", "second"):
        eeg = create_test_eeg(n_channels=2, n_samples=8)
        eeg["setname"] = name
        session.store_current(eeg, new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    try:
        workspace.namespace["CURRENTSET"] = value
        with pytest.raises(ValueError, match="CURRENTSET"):
            workspace.after_execute(source, success=True)
    finally:
        workspace.close()
