"""Preferences and persistence regressions for F29/F89."""

import json

import pytest
from PySide6.QtWidgets import QCheckBox, QLineEdit

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.adminfunc.pop_editoptions import pop_editoptions
from eegprep.functions.guifunc.main_window import build_main_window
from tests.test_stream11_datasets import accept, dialogs as dialogs
from tests.test_stream11_menus import qapp as qapp

pytestmark = [pytest.mark.gui, pytest.mark.modal_dialogs]


@pytest.fixture
def options(tmp_path, monkeypatch):
    from pathlib import Path

    previous = dict(EEG_OPTIONS)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    yield tmp_path / ".eegprep" / "options.json"
    EEG_OPTIONS.clear()
    EEG_OPTIONS.update(previous)


def test_f89_preferences_apply_immediately_and_reload(options, qapp, dialogs):
    from eegprep.functions.adminfunc.pop_editoptions import load_eegprep_options

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
        dialog.findChild(QLineEdit, "option_cachesize").setText("256")
        accept(dialog)

    pending.append(edit)
    try:
        window.dispatcher.dispatch("pop_editoptions", window.window)
        assert window.all_menus
        assert EEG_OPTIONS["option_savetwofiles"] == 1
        assert EEG_OPTIONS["option_cachesize"] == 256
        assert json.loads(options.read_text())["option_allmenus"] == 1
        EEG_OPTIONS.update(option_allmenus=0, option_cachesize=500)
        load_eegprep_options(options)
        restarted = build_main_window(include_plugins=False)
        assert restarted.all_menus
        assert EEG_OPTIONS["option_cachesize"] == 256
        restarted.window.close()
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
