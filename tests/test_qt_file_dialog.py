from types import SimpleNamespace
from unittest import mock

import pytest

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.file_dialogs import file_dialog_kwargs
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.qt import _select_file
from eegprep.functions.guifunc.session import EEGPrepSession


def _qt_widgets():
    class DialogOption:
        DontUseNativeDialog = 4
        ShowDirsOnly = 1

    class FileDialog:
        Option = DialogOption

    return SimpleNamespace(QFileDialog=FileDialog)


def test_save_as_can_skip_the_typed_name_overwrite_prompt(monkeypatch):
    class DialogOption:
        DontUseNativeDialog = 4
        DontConfirmOverwrite = 8

    qt_widgets = SimpleNamespace(QFileDialog=SimpleNamespace(Option=DialogOption))
    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 1)

    assert file_dialog_kwargs(qt_widgets, confirm_overwrite=False) == {"options": 8}
    assert file_dialog_kwargs(qt_widgets, native_file_dialogs=False, confirm_overwrite=False) == {"options": 12}


@pytest.mark.parametrize(
    ("configured", "explicit", "expected"),
    [
        (0, None, {"options": 4}),
        (1, None, {}),
        (0, True, {}),
        (1, False, {"options": 4}),
    ],
)
def test_file_dialog_kwargs_precedence(monkeypatch, configured, explicit, expected):
    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", configured)

    assert file_dialog_kwargs(_qt_widgets(), native_file_dialogs=explicit) == expected


@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [(1, False, {"options": 4}), (0, True, {})],
)
def test_dispatch_scopes_constructor_override_for_callback_dialogs(
    monkeypatch,
    configured,
    override,
    expected,
):
    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", configured)
    dispatcher = MenuActionDispatcher(EEGPrepSession(), native_file_dialogs=override)
    observed = {}

    def inspect_callback_policy(_parent):
        observed.update(file_dialog_kwargs(_qt_widgets()))

    with mock.patch.object(dispatcher, "_loadset", side_effect=inspect_callback_policy):
        dispatcher.dispatch("pop_loadset")

    assert observed == expected


def test_callback_driven_select_file_uses_current_global_option(monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QLineEdit

    from eegprep.functions.guifunc.qt import _require_qt

    QApplication.instance() or QApplication([])
    _qt_core, qt_widgets = _require_qt()
    captured = {}

    def get_open_file_name(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "test_file.txt", ""

    def get_save_file_name(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "save_file.txt", ""

    monkeypatch.setattr(qt_widgets.QFileDialog, "getOpenFileName", get_open_file_name)
    monkeypatch.setattr(qt_widgets.QFileDialog, "getSaveFileName", get_save_file_name)

    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 0)
    target = QLineEdit()
    _select_file(
        None,
        target,
        {"caption": "My Open Dialog", "filter": "Text (*.txt)", "mode": "open"},
        {},
    )

    assert target.text() == "test_file.txt"
    assert captured["args"][1] == "My Open Dialog"
    assert captured["args"][3] == "Text (*.txt)"
    assert captured["kwargs"]["options"] == qt_widgets.QFileDialog.Option.DontUseNativeDialog

    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 1)
    _select_file(None, target, {"caption": "My Save Dialog", "mode": "save"}, {})

    assert target.text() == "save_file.txt"
    assert captured["args"][1] == "My Save Dialog"
    assert captured["kwargs"] == {}
