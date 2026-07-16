from __future__ import annotations

import ast

import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession, normalize_dataset_indices
from tests.fixtures import (
    assert_eeg_fields_close,
    assert_history_contains_once,
    assert_history_replayable,
    assert_session_synced,
    create_test_eeg,
    fresh_sample_eeg,
    fresh_session_with_sample,
)


def test_sample_data_satisfies_core_eeg_contract():
    eeg = fresh_sample_eeg()
    data = eeg["data"]

    assert isinstance(data, np.ndarray)
    assert data.shape[0] == int(eeg["nbchan"])
    assert data.shape[1] == int(eeg["pnts"])
    assert int(eeg["trials"]) == 1
    assert data.ndim == 2
    assert np.asarray(eeg["times"]).shape == (int(eeg["pnts"]),)

    for field in (
        "event",
        "urevent",
        "epoch",
        "chanlocs",
        "chaninfo",
        "history",
        "icaact",
        "icawinv",
        "icasphere",
        "icaweights",
        "icachansind",
    ):
        assert field in eeg

    events = list(eeg["event"])
    chanlocs = list(eeg["chanlocs"])
    assert events
    assert chanlocs
    assert all(isinstance(event, dict) for event in events)
    assert all(float(event["latency"]) >= 1 for event in events if "latency" in event)
    assert all(isinstance(chanloc, dict) for chanloc in chanlocs)
    assert all("labels" in chanloc for chanloc in chanlocs)
    assert isinstance(eeg["chaninfo"], dict)
    assert isinstance(eeg["history"], str)
    assert np.asarray(eeg["icachansind"]).dtype.kind in {"i", "u"}


def test_sample_session_and_console_namespace_stay_synced():
    command = "EEG = pop_loadset('eeglab_data.set');"
    session = fresh_session_with_sample(command)
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    try:
        assert session.CURRENTSET == [1]
        assert session.selected_dataset_indices() == [1]
        assert session.current_set_value() == 1
        assert_session_synced(session, workspace.namespace)
        assert_history_contains_once(session, command)
    finally:
        workspace.close()


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


def test_dataset_index_normalization_accepts_canonical_empty_values():
    assert normalize_dataset_indices(0) == []
    assert normalize_dataset_indices([0]) == []
    assert normalize_dataset_indices([]) == []


@pytest.mark.parametrize("indices", ([1, 1], [0], [-1], [0, 1], [1.5], True))
def test_dataset_index_normalization_rejects_invalid_selection(indices):
    with pytest.raises(ValueError):
        normalize_dataset_indices(indices, allow_empty=False)


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


def test_history_helpers_accept_replayable_gui_history():
    converted = assert_history_replayable("EEG = pop_reref( EEG, [], 'keepref', 'on');")
    tree = ast.parse(converted)
    statement = tree.body[0]

    assert isinstance(statement, ast.Assign)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    assert statement.value.func.id == "pop_reref"
    assert isinstance(statement.value.args[0], ast.Name)
    assert statement.value.args[0].id == "EEG"
    assert any(
        keyword.arg == "keepref" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "on"
        for keyword in statement.value.keywords
    )


def test_field_comparison_helper_supports_parity_scaffolding():
    left = {"data": np.array([1.0, 2.0]), "setname": "demo"}
    right = {"data": np.array([1.0, 2.0 + 1e-8]), "setname": "demo"}

    assert_eeg_fields_close(left, right, ("data", "setname"))

import threading

def test_session_concurrency_with_store_and_listeners():
    session = EEGPrepSession()
    session.store_current(create_test_eeg(n_channels=2, n_samples=8), new=True)
    
    events = []
    
    def on_change(sess):
        # We append to a local list. In Python, list.append is thread-safe.
        events.append("change")
        
    def on_gui(action, name):
        events.append(f"{action}:{name}")
        
    session.add_change_listener(on_change)
    session.add_gui_action_listener(on_gui)
    
    def worker(i):
        eeg = create_test_eeg(n_channels=2, n_samples=8)
        eeg["setname"] = f"Thread {i}"
        with session.gui_action(f"action_{i}"):
            session.store_current(eeg, new=True, command=f"EEG = pop_test({i});")
            # Also read something to trigger read locks
            _ = session.current_set_value()
            _ = session.menu_statuses()
            _ = session.dataset_summaries()
            _ = session.clone_current()
            _ = session.history_command_at(1)
            
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(session.ALLEEG) == 11  # 1 initial + 10 workers
    assert len(session.ALLCOM) == 10  # 10 commands added
    
    # Check that listener was fired properly
    assert events.count("change") == 10
    
    gui_begins = [e for e in events if e.startswith("begin:")]
    gui_ends = [e for e in events if e.startswith("end:")]
    assert len(gui_begins) == 10
    assert len(gui_ends) == 10
