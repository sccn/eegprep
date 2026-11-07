"""Preferences and persistence regressions for F29/F89."""

import importlib
import json
from pathlib import Path

import pytest

try:
    from PySide6.QtWidgets import QCheckBox, QScrollArea
except ImportError:
    pytest.skip("PySide6 is required for GUI workflows", allow_module_level=True)

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.adminfunc.eeglab import eeglab
from eegprep.functions.adminfunc.pop_editoptions import (
    pop_editoptions,
    pop_editoptions_dialog_spec,
)
from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.test_stream11_datasets import accept, dialogs as dialogs
from tests.test_stream11_menus import qapp as qapp

pytestmark = [pytest.mark.gui, pytest.mark.modal_dialogs]


@pytest.fixture
def options(tmp_path, monkeypatch):
    previous = dict(EEG_OPTIONS)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    yield tmp_path / ".eegprep" / "options.json"
    EEG_OPTIONS.clear()
    EEG_OPTIONS.update(previous)


def test_f89_preferences_apply_immediately_and_reload(options, qapp, dialogs):
    EEG_OPTIONS["option_allmenus"] = 0
    window = build_main_window(include_plugins=False)
    pending, _ = dialogs

    def edit(dialog):
        assert dialog.windowTitle() == "Memory options - pop_editoptions()"
        for key in (
            "option_storedisk",
            "option_savetwofiles",
            "option_computeica",
            "option_rememberfolder",
            "option_allmenus",
            "option_boundary99",
            "option_showadvanced",
            "option_checkversion",
        ):
            assert dialog.findChild(QCheckBox, key) is not None
        dialog.findChild(QCheckBox, "option_allmenus").setChecked(True)
        dialog.findChild(QCheckBox, "option_savetwofiles").setChecked(True)
        accept(dialog)

    pending.append(edit)
    try:
        window.dispatcher.dispatch("pop_editoptions", window.window)
        assert window.all_menus
        assert EEG_OPTIONS["option_savetwofiles"] == 1
        assert json.loads(options.read_text())["option_allmenus"] == 1
        EEG_OPTIONS.update(option_allmenus=0, option_savetwofiles=0)
        restarted = build_main_window(include_plugins=False)
        assert restarted.all_menus
        assert EEG_OPTIONS["option_savetwofiles"] == 1
        restarted.window.close()
    finally:
        window.window.close()


def test_importing_pop_editoptions_does_not_load_saved_preferences(options):
    EEG_OPTIONS["option_allmenus"] = 0
    before = dict(EEG_OPTIONS)
    options.parent.mkdir()
    options.write_text(json.dumps({"option_allmenus": 1}))

    importlib.reload(importlib.import_module("eegprep.functions.adminfunc.pop_editoptions"))

    assert EEG_OPTIONS == before


@pytest.mark.parametrize("startup", ["main_window", "eeglab", "console"])
def test_interactive_startup_loads_saved_preferences(options, qapp, startup):
    EEG_OPTIONS["option_allmenus"] = 0
    options.parent.mkdir()
    options.write_text(json.dumps({"option_allmenus": 1}))

    if startup == "console":
        workspace = EEGPrepConsoleWorkspace(EEGPrepSession())
        try:
            assert EEG_OPTIONS["option_allmenus"] == 1
        finally:
            workspace.close()
        return
    window = (
        eeglab(show=False, include_plugins=False) if startup == "eeglab" else build_main_window(include_plugins=False)
    )
    try:
        assert EEG_OPTIONS["option_allmenus"] == 1
        assert window.all_menus
    finally:
        window.window.close()


def test_f29_preferences_cancel_changes_neither_options_nor_file(options, qapp, dialogs):
    previous = dict(EEG_OPTIONS)
    window = build_main_window(include_plugins=False)
    pending, _ = dialogs
    pending.append(lambda dialog: dialog.reject())
    try:
        window.dispatcher.dispatch("pop_editoptions", window.window)
        assert EEG_OPTIONS == previous
        assert not options.exists()
        assert not window.session.ALLCOM
    finally:
        window.window.close()


def test_f89_invalid_preference_is_atomic(options):
    previous = dict(EEG_OPTIONS)
    with pytest.raises(ValueError, match="pop_editoptions.*option_cachesize"):
        pop_editoptions(option_allmenus=1, option_cachesize="abc", persist=True)
    assert EEG_OPTIONS == previous
    assert not options.exists()


def test_f89_preferences_history_records_replayable_values(options):
    command = pop_editoptions(option_allmenus=1, option_cachesize=128, persist=True)
    assert command == "pop_editoptions('option_allmenus', 1, 'option_cachesize', 128);"


def test_f89_default_options_are_visible_without_scrolling(qapp, options):
    values = dict(EEG_OPTIONS, option_showadvanced=0)
    _, dialog, widgets = QtDialogRenderer().build_dialog(pop_editoptions_dialog_spec(values))
    dialog.show()
    qapp.processEvents()
    try:
        viewport = dialog.findChild(QScrollArea).viewport()
        cache = widgets["option_cachesize"]
        assert viewport.rect().contains(cache.mapTo(viewport, cache.rect().center()))
    finally:
        dialog.close()


def test_f27_native_dialog_preference_does_not_enter_matlab_history(options):
    command = pop_editoptions(option_native_dialogs=0, option_allmenus=1, persist=True)
    assert "option_native_dialogs" not in command
    assert json.loads(options.read_text())["option_native_dialogs"] == 0


@pytest.mark.parametrize(
    "key",
    [
        "option_saveversion6",
        "option_saveasstruct",
        "option_parallel",
        "option_single",
        "option_eegobject",
        "option_computeica",
        "option_scaleicarms",
        "option_rememberfolder",
        "option_donotusetoolboxes",
        "option_showpendingplugins",
        "option_htmlingraphics",
        "option_checkversion",
        "option_cachesize",
    ],
)
def test_f89_unavailable_preferences_cannot_be_changed_in_gui(qapp, options, key):
    values = dict(EEG_OPTIONS, option_showadvanced=1)
    _, dialog, widgets = QtDialogRenderer().build_dialog(pop_editoptions_dialog_spec(values))
    try:
        control = widgets[key]
        assert not control.isEnabled()
        assert control.toolTip()
        if isinstance(control, QCheckBox):
            before = control.isChecked()
            control.click()
            assert control.isChecked() == before
    finally:
        dialog.close()
