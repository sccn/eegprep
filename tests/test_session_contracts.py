from __future__ import annotations


import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession
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
