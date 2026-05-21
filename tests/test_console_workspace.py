from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from eegprep.functions.adminfunc import console as console_module
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession


def _demo_eeg(setname: str = "demo"):
    return {
        "setname": setname,
        "data": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nbchan": 2,
        "pnts": 2,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.01,
        "chanlocs": [{"labels": "Cz"}, {"labels": "Pz"}],
        "event": [],
        "urevent": [],
        "epoch": [],
    }


def _fake_pop_reref(eeg, ref, *, return_com=False):
    output = dict(eeg, setname="reref", ref=list(ref))
    command = "EEG = pop_reref(EEG, []);"
    return (output, command) if return_com else output


def test_workspace_starts_with_eeglab_style_names():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    for name in ("EEG", "ALLEEG", "CURRENTSET", "ALLCOM", "LASTCOM", "STUDY", "CURRENTSTUDY"):
        assert name in workspace.namespace
    assert workspace.namespace["session"] is session


def test_session_changes_update_console_namespace():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    eeg = _demo_eeg()

    session.store_current(eeg, new=True, command="EEG = demo;")

    assert workspace.namespace["EEG"] is session.EEG
    assert workspace.namespace["ALLEEG"] is session.ALLEEG
    assert workspace.namespace["CURRENTSET"] == 1
    assert workspace.namespace["LASTCOM"] == "EEG = demo;"


def test_console_eeg_assignment_stores_current_dataset_and_refreshes():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={})
    edited = dict(session.EEG, setname="edited")

    workspace.namespace["EEG"] = edited
    workspace.after_execute("EEG = edited")

    assert session.EEG["setname"] == "edited"
    assert session.ALLEEG[0]["setname"] == "edited"
    assert session.ALLCOM[-1] == "EEG = edited"
    refresh.assert_called_once()


def test_console_lastcom_assignment_updates_unified_history_once():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    workspace.namespace["LASTCOM"] = "EEG = custom_command(EEG);"
    workspace.after_execute("LASTCOM = 'EEG = custom_command(EEG);'")

    assert session.LASTCOM == "EEG = custom_command(EEG);"
    assert session.ALLCOM == ["EEG = custom_command(EEG);"]


def test_bare_pop_call_updates_session_and_returns_compact_unpackable_result():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    refresh = mock.Mock()
    workspace = EEGPrepConsoleWorkspace(session, refresh=refresh, exports={"pop_reref": _fake_pop_reref})

    result = workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])
    workspace.after_execute("pop_reref(EEG, [])")

    eeg, command = result
    assert eeg is session.EEG
    assert command == "EEG = pop_reref(EEG, []);"
    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]
    assert "data" not in repr(result)
    refresh.assert_called_once()


def test_eegprep_proxy_pop_call_updates_session_like_direct_pop_call():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})

    result = workspace.namespace["eegprep"].pop_reref(workspace.namespace["EEG"], [])
    workspace.after_execute("eegprep.pop_reref(EEG, [])")

    eeg, command = result
    assert eeg is session.EEG
    assert command == "EEG = pop_reref(EEG, []);"
    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]


def test_console_restores_eegprep_proxy_after_user_imports_eegprep():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={})

    workspace.namespace["eegprep"] = console_module.eegprep
    workspace.after_execute("import eegprep")

    assert isinstance(workspace.namespace["eegprep"], console_module.ConsoleEEGPrepModule)


def test_console_restores_pop_wrappers_after_from_import():
    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})
    original_wrapper = workspace.namespace["pop_reref"]

    workspace.namespace["pop_reref"] = _fake_pop_reref
    workspace.after_execute("from eegprep import pop_reref")

    assert workspace.namespace["pop_reref"] is original_wrapper


def test_assignment_style_pop_call_does_not_duplicate_history():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": _fake_pop_reref})

    result = workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])
    workspace.namespace["EEG"], workspace.namespace["LASTCOM"] = result
    workspace.after_execute("EEG, LASTCOM = pop_reref(EEG, [])")

    assert session.EEG["setname"] == "reref"
    assert session.ALLCOM == ["EEG = pop_reref(EEG, []);"]


def test_failed_pop_call_does_not_mutate_session_or_history():
    def failing_pop(eeg, ref, *, return_com=False):
        raise ValueError("bad ref")

    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_reref": failing_pop})

    with pytest.raises(ValueError, match="bad ref"):
        workspace.namespace["pop_reref"](workspace.namespace["EEG"], [])

    assert session.EEG["setname"] == "demo"
    assert session.ALLCOM == []


def test_multiple_selected_datasets_stay_selected_after_pop_call():
    def fake_pop_select(eeg, *, return_com=False):
        output = [dict(item, setname=f"{item['setname']}-selected") for item in eeg]
        command = "EEG = pop_select(EEG);"
        return (output, command) if return_com else output

    session = EEGPrepSession()
    session.store_current(_demo_eeg("one"), new=True)
    session.store_current(_demo_eeg("two"), new=True)
    session.retrieve([1, 2])
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_select": fake_pop_select})

    workspace.namespace["pop_select"](workspace.namespace["EEG"])

    assert session.CURRENTSET == [1, 2]
    assert [item["setname"] for item in session.ALLEEG] == ["one-selected", "two-selected"]


def test_safe_after_execute_sync_error_recovers_namespace_without_crashing():
    session = EEGPrepSession()
    session.store_current(_demo_eeg(), new=True)
    workspace = EEGPrepConsoleWorkspace(session, exports={})
    writes = []

    workspace.namespace["EEG"] = 3
    console_module._safe_after_execute(workspace, "EEG = 3", success=True, write=writes.append)

    assert session.EEG["setname"] == "demo"
    assert workspace.namespace["EEG"] is session.EEG
    assert session.ALLCOM == []
    assert "EEGPrep workspace sync failed" in "".join(writes)


class _FakeEvents:
    def __init__(self):
        self.callbacks = {}

    def register(self, name, callback):
        self.callbacks[name] = callback


class _FakeShell:
    def __init__(self):
        self.events = _FakeEvents()
        self.enabled_gui = None
        self.called = False

    def enable_gui(self, gui):
        self.enabled_gui = gui

    def __call__(self):
        self.called = True
        callback = self.events.callbacks["post_run_cell"]
        callback(SimpleNamespace(info=SimpleNamespace(raw_cell="EEG"), success=True))


def test_run_console_forwards_cli_options_to_gui_launcher():
    shell = _FakeShell()
    captured = {}

    def shell_factory(namespace, banner):
        captured["namespace"] = namespace
        captured["banner"] = banner
        return shell

    def gui_launcher(*args, **kwargs):
        captured["gui_args"] = args
        captured["gui_kwargs"] = kwargs
        return SimpleNamespace(refresh=mock.Mock())

    assert console_module.run_console(
        ["--full", "--no-plugins", "--window-menu-bar"],
        shell_factory=shell_factory,
        gui_launcher=gui_launcher,
    ) == 0

    assert captured["gui_args"] == ("full",)
    assert captured["gui_kwargs"]["include_plugins"] is False
    assert captured["gui_kwargs"]["native_menu_bar"] is False
    assert "EEGPrep interactive console" in captured["banner"]
    assert shell.enabled_gui == "qt"
    assert shell.called is True
    assert "EEG" in captured["namespace"]


def test_ipython_factory_error_is_user_facing_when_dependency_missing():
    with (
        mock.patch.object(console_module.importlib, "import_module", side_effect=ImportError("missing")),
        pytest.raises(RuntimeError, match="IPython is required for eegprep-console"),
    ):
        console_module._ipython_shell_factory({}, "")
