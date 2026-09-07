from types import SimpleNamespace
from unittest import mock

import pytest

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.file_dialogs import file_dialog_kwargs, native_file_dialog_override
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


def test_file_dialog_kwargs_combines_directory_flags(monkeypatch):
    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 0)

    assert file_dialog_kwargs(_qt_widgets(), directories=True) == {"options": 5}


def test_scoped_override_applies_to_callback_dialogs(monkeypatch):
    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 1)

    with native_file_dialog_override(False):
        assert file_dialog_kwargs(_qt_widgets()) == {"options": 4}

    assert file_dialog_kwargs(_qt_widgets()) == {}


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
    captured = {}

    class DialogOption:
        DontUseNativeDialog = 4

    class FileDialog:
        Option = DialogOption

        @staticmethod
        def getOpenFileName(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "test_file.txt", ""

        @staticmethod
        def getSaveFileName(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "save_file.txt", ""

    qt_widgets = SimpleNamespace(QFileDialog=FileDialog)
    monkeypatch.setattr("eegprep.functions.guifunc.qt._require_qt", lambda: (None, qt_widgets))

    class TargetWidget:
        def __init__(self):
            self.value = ""

        def setText(self, val):
            self.value = val

    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 0)
    target = TargetWidget()
    _select_file(
        None,
        target,
        {"caption": "My Open Dialog", "filter": "Text (*.txt)", "mode": "open"},
        {},
    )

    assert target.value == "test_file.txt"
    assert captured["args"][1] == "My Open Dialog"
    assert captured["args"][3] == "Text (*.txt)"
    assert captured["kwargs"] == {"options": 4}

    monkeypatch.setitem(EEG_OPTIONS, "option_native_dialogs", 1)
    _select_file(None, target, {"caption": "My Save Dialog", "mode": "save"}, {})

    assert target.value == "save_file.txt"
    assert captured["args"][1] == "My Save Dialog"
    assert captured["kwargs"] == {}
