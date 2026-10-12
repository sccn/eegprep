"""Original menu workflow and supplemental Python session contracts."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

import eegprep
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.adminfunc.eeglab_execmenu import eeglab_execmenu
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui
from tests.fixtures import SAMPLE_DATASET_PATH


EEGLAB_EXECMENU_WRAPPER = "unittesting_adminfunc/eeglab_execmenu/adminfunc_eeglab_execmenu_wrapperTest.m"


def test_native_base_workspace_snapshot_restores_values_and_global_bindings(eeglab_matlab_engine):
    """Exercise the state-only harness without launching any GUI workflow."""
    engine = eeglab_matlab_engine
    engine.eval(
        "global eegprep_contract_global; eegprep_contract_global = int16(7); "
        "eegprep_contract_local = struct('field', single([1 2]));",
        nargout=0,
    )
    try:
        engine.eegprep_test_base_workspace("snapshot", nargout=0)
        engine.eval(
            "eegprep_contract_global = 99; clear eegprep_contract_local; "
            "global eegprep_contract_new_global; eegprep_contract_new_global = 42; "
            "eegprep_contract_new_local = 43; clear functions;",
            nargout=0,
        )
        engine.eegprep_test_base_workspace("restore", nargout=0)
        assert engine.eval("isequal(eegprep_contract_global, int16(7))")
        assert engine.eval("isequal(eegprep_contract_local.field, single([1 2]))")
        assert engine.eval("ismember('eegprep_contract_global', who('global'))")
        assert not engine.eval("ismember('eegprep_contract_new_global', who('global'))")
        assert not engine.eval("exist('eegprep_contract_new_local', 'var')")
        assert not engine.mislocked("eegprep_test_base_workspace")
    finally:
        engine.eval(
            "clear global eegprep_contract_global eegprep_contract_new_global; "
            "clear eegprep_contract_local eegprep_contract_new_local;",
            nargout=0,
        )


def _source_cell(*values):
    cell = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cell[0, index] = value
    return cell


@pytest.mark.gui
@eeglab_test(EEGLAB_EXECMENU_WRAPPER, "test_i_pass_general")
def test_reference_execmenu_original_menu_and_history_workflow(
    eeglab_backend,
    request,
    eeglab_suite_root,
    eeglab_working_directory,
):
    """Retain the original menu calls, with file writes confined to scratch."""
    sample_folder = eeglab_working_directory / "sample_data"
    sample_folder.mkdir()
    for filename in ("eeglab_data.set", "eeglab_data.fdt"):
        shutil.copy2(eeglab_suite_root / "eeglab/sample_data" / filename, sample_folder / filename)
    montage = eeglab_suite_root / "eeglab/plugins/dipfit/standard_BEM/elec/standard_1005.elc"
    # Keep the literal DIPFIT lookup source, including any missing dependency;
    # do not replace it with a different packaged Python montage.
    calls = (
        (
            "From ASCII/float file or MATLAB array",
            "pop_importdata",
            ["dataformat", "array", "nbchan", 0.0, "data", "tmpdata", "srate", 100.0],
        ),
        ("Save current dataset as", "pop_saveset", ["test.set"]),
        ("Dataset info", "pop_editset", ["subject", "test2"]),
        ("Resave current dataset(s)", "pop_saveset", ["savemode", "resave"]),
        ("Load existing dataset", "pop_loadset", [str(sample_folder / "eeglab_data.set")]),
        ("Event values", "pop_editeventvals", ["changefield", _source_cell(1.0, "position", 3.0)]),
        ("About this dataset", "pop_comments", [np.array([list("EEGLAB Tutorial Dataset"), list("test".ljust(23))])]),
        ("Channel locations", "pop_chanedit", ["lookup", str(montage)]),
        ("Change sampling rate", "pop_resample", [64.0]),
    )
    tmpdata = np.random.default_rng().random((10, 1000))
    if request.config.getoption("--eeglab-backend") == "matlab":
        eeglab_backend("eegprep_test_base_workspace", "snapshot", nargout=0)
        try:
            eeglab_backend("eeglab", nargout=0)
            eeglab_backend("assignin", "base", "tmpdata", tmpdata, nargout=0)
            for label, function, parameters in calls:
                eeglab_backend("eeglab_execmenu", label, function, _source_cell(*parameters), nargout=0)
            savedcom = eeglab_backend("evalin", "base", "ALLCOM")
            for command in reversed(np.asarray(savedcom, dtype=object).ravel().tolist()):
                # Replay only commands produced by the real menu calls, one at a time.
                eeglab_backend("evalin", "base", str(command), nargout=0)
        finally:
            close_reference_gui(eeglab_backend, request)
            eeglab_backend("eegprep_test_base_workspace", "restore", nargout=0)
    else:
        session = EEGPrepSession()
        window = eeglab_backend("eeglab", session=session)
        workspace = EEGPrepConsoleWorkspace(session, window=window)
        workspace.namespace["tmpdata"] = tmpdata
        try:
            for label, function, parameters in calls:
                parameters = list(parameters)
                if function == "pop_importdata":
                    # Resolve the MATLAB variable-name argument in the actual
                    # Python console namespace; Python imports accept arrays.
                    parameters[5] = workspace.namespace["tmpdata"]
                elif function == "pop_editeventvals":
                    parameters[1] = parameters[1].ravel().tolist()
                elif function == "pop_comments":
                    parameters[0] = ["".join(row).rstrip() for row in parameters[0]]
                eeglab_backend("eeglab_execmenu", label, function, parameters, session=session, nargout=0)
            # eegh prepends newest commands; EEGPrepSession appends them.
            savedcom = list(reversed(session.ALLCOM))
            for command in reversed(savedcom):
                workspace.execute_history_command(command)
        finally:
            workspace.close()
            close_reference_gui(eeglab_backend, request, window=window.window)


def _workflow_calls(save_file: Path) -> list[tuple[str, str, list[object]]]:
    data = np.random.default_rng(37).random((10, 1000))
    montage = Path(eegprep.__file__).resolve().parent / "resources/headplot/Standard-10-5-Cap385.sfp"
    return [
        (
            "From ASCII/float file or MATLAB array",
            "pop_importdata",
            ["dataformat", "array", "nbchan", 0, "data", data, "srate", 100],
        ),
        ("Save current dataset as", "pop_saveset", [save_file]),
        ("Dataset info", "pop_editset", ["subject", "test2"]),
        ("Resave current dataset(s)", "pop_saveset", ["savemode", "resave"]),
        ("Load existing dataset", "pop_loadset", [SAMPLE_DATASET_PATH]),
        ("Event values", "pop_editeventvals", ["changefield", [1, "position", 3]]),
        ("About this dataset", "pop_comments", [["EEGLAB Tutorial Dataset", "test"]]),
        ("Channel locations", "pop_chanedit", ["lookup", montage]),
        ("Change sampling rate", "pop_resample", [64]),
    ]


def _run_workflow(calls: list[tuple[str, str, list[object]]]) -> EEGPrepSession:
    session = EEGPrepSession()
    for label, function, parameters in calls:
        command = eeglab_execmenu(label, function, parameters, session=session)
        assert command == session.LASTCOM
        assert session.ALLCOM[-1] == command
        assert session.CURRENTSET == [1]
        assert session.EEG is session.ALLEEG[0]
    return session


def test_current_eeglab_execmenu_workflow_updates_session_and_replays_deterministically(tmp_path: Path):
    calls = _workflow_calls(tmp_path / "test.set")
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    try:
        import_command = eeglab_execmenu(*calls[0], session=session)
        assert np.asarray(session.EEG["data"]).shape == (10, 1000)
        assert session.EEG["srate"] == 100
        assert import_command.startswith("EEG = pop_importdata(")

        eeglab_execmenu(*calls[1], session=session)
        assert (tmp_path / "test.set").is_file()
        assert session.EEG["filename"] == "test.set"

        eeglab_execmenu(*calls[2], session=session)
        assert session.EEG["subject"] == "test2"

        eeglab_execmenu(*calls[3], session=session)
        assert session.LASTCOM == "EEG = pop_saveset(EEG, 'savemode', 'resave');"

        eeglab_execmenu(*calls[4], session=session)
        assert np.asarray(session.EEG["data"]).shape == (32, 30504)
        assert len(session.ALLEEG) == 1

        eeglab_execmenu(*calls[5], session=session)
        assert session.EEG["event"][0]["position"] == 3

        eeglab_execmenu(*calls[6], session=session)
        assert session.EEG["comments"] == "EEGLAB Tutorial Dataset\ntest"

        eeglab_execmenu(*calls[7], session=session)
        assert session.EEG["chanlocs"][0]["labels"] == "FPz"
        assert session.EEG["chanlocs"][0]["urchan"] == 0
        assert np.isfinite(session.EEG["chanlocs"][0]["X"])
        assert "theta" not in session.EEG["chanlocs"][1]

        eeglab_execmenu(*calls[8], session=session)
        assert session.EEG["srate"] == 64
        assert np.asarray(session.EEG["data"]).shape == (32, 15252)
        assert session.EEG["event"][0]["position"] == 3
        assert session.EEG["comments"] == "EEGLAB Tutorial Dataset\ntest"
        assert len(session.ALLCOM) == len(calls)
        assert workspace.namespace["EEG"] is session.EEG
        assert workspace.namespace["ALLEEG"] is session.ALLEEG
        assert workspace.namespace["CURRENTSET"] == 1
        assert workspace.namespace["LASTCOM"] == session.LASTCOM
        assert workspace.namespace["ALLCOM"] is session.ALLCOM
    finally:
        workspace.close()

    replayed = _run_workflow(calls)
    np.testing.assert_array_equal(replayed.EEG["data"], session.EEG["data"])
    np.testing.assert_array_equal(
        [event["latency"] for event in replayed.EEG["event"]],
        [event["latency"] for event in session.EEG["event"]],
    )
    assert [event["type"] for event in replayed.EEG["event"]] == [event["type"] for event in session.EEG["event"]]
    assert replayed.EEG["event"][0]["position"] == session.EEG["event"][0]["position"] == 3
    assert replayed.EEG["comments"] == session.EEG["comments"]
    assert [chan["labels"] for chan in replayed.EEG["chanlocs"]] == [chan["labels"] for chan in session.EEG["chanlocs"]]
    np.testing.assert_allclose(
        [replayed.EEG["chanlocs"][0][field] for field in ("X", "Y", "Z")],
        [session.EEG["chanlocs"][0][field] for field in ("X", "Y", "Z")],
    )
    assert replayed.ALLCOM == session.ALLCOM
