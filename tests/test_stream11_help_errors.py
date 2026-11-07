"""Preconditions, actionable errors, rendered help, and update checks."""

from io import BytesIO
import json
from urllib.error import URLError

import pytest

try:
    from PySide6.QtWidgets import QLineEdit, QMessageBox, QTextBrowser
except ImportError:
    pytest.skip("PySide6 is required for GUI workflows", allow_module_level=True)

from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.test_gui_main_window import _demo_eeg
from tests.test_stream11_datasets import accept, dialogs as dialogs
from tests.test_stream11_menus import qapp as qapp

pytestmark = [pytest.mark.gui, pytest.mark.modal_dialogs]


@pytest.mark.parametrize(
    "action",
    [
        "pop_subcomp",
        "pop_selectcomps",
        "pop_eegplot:components",
        "pop_spectopo:components",
        "pop_erpimage:components",
        "pop_newtimef:components",
        "pop_topoplot:components",
        "pop_headplot:components",
        "pop_envtopo",
        "pop_viewprops:components",
        "pop_iclabel",
        "pop_icflag",
    ],
)
def test_f64_f81_no_ica_is_reported_before_opening_settings(qapp, dialogs, action):
    session = EEGPrepSession()
    session.store_current(_demo_eeg(ica=False, epoched=True))
    window = build_main_window(session, include_plugins=False)
    pending, seen = dialogs

    def error(dialog):
        assert isinstance(dialog, QMessageBox)
        assert action.partition(":")[0] in dialog.text()
        assert "icaweights" in dialog.text()
        assert "Decompose data by ICA" in dialog.text()
        dialog.accept()

    pending.append(error)
    try:
        window.dispatcher.dispatch_gui(action, window.window)
        assert len(seen) == 1
        assert not session.ALLCOM
    finally:
        window.window.close()


@pytest.mark.parametrize(
    "action,field,value,required",
    [
        ("pop_editset", "srate", "abc", ("srate", "positive")),
        ("pop_select", "chans", "XYZ", ("channel", "labels")),
        ("pop_select", "trial", "99", ("trial", "1")),
    ],
)
def test_f47_errors_name_function_field_and_recovery(qapp, dialogs, action, field, value, required):
    session = EEGPrepSession()
    session.store_current(_demo_eeg(epoched=True))
    window = build_main_window(session, include_plugins=False)
    pending, _ = dialogs

    def enter(dialog):
        dialog.findChild(QLineEdit, field).setText(value)
        accept(dialog)

    def error(dialog):
        assert isinstance(dialog, QMessageBox)
        assert action in dialog.text()
        for text in required:
            assert text in dialog.text()
        dialog.accept()

    pending.extend([enter, error])
    try:
        window.dispatcher.dispatch_gui(action, window.window)
        assert not session.ALLCOM
    finally:
        window.window.close()


@pytest.mark.parametrize("topic", ["eeg_helpmenu", "pop_editoptions", "pop_resample"])
def test_f96_help_renders_markdown(qapp, topic):
    dialog = pophelp(topic)
    try:
        browser = dialog.findChild(QTextBrowser, "help_browser")
        text = browser.toPlainText()
        assert not text.startswith("# ")
        assert "```" not in text
        assert "`EEG`" not in text
        assert browser.document().begin().blockFormat().headingLevel() == 1
    finally:
        dialog.close()


@pytest.mark.parametrize("release,expected", [("999.0.0", "available"), ("0.0.1", "latest")])
def test_f97_updates_show_version_result_in_app(qapp, dialogs, monkeypatch, release, expected):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: BytesIO(json.dumps({"info": {"version": release}}).encode())
    )
    window = build_main_window(include_plugins=False)
    pending, seen = dialogs

    def message(dialog):
        assert isinstance(dialog, QMessageBox)
        assert expected in dialog.text()
        assert "EEGPrep" in dialog.text()
        dialog.accept()

    pending.append(message)
    try:
        window.dispatcher.dispatch_gui("updates", window.window)
        assert len(seen) == 1
    finally:
        window.window.close()


def test_f97_update_network_failure_is_actionable(qapp, dialogs, monkeypatch):
    def offline(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    window = build_main_window(include_plugins=False)
    pending, _ = dialogs

    def message(dialog):
        assert "connection" in dialog.text()
        assert "try again" in dialog.text()
        dialog.accept()

    pending.append(message)
    try:
        window.dispatcher.dispatch_gui("updates", window.window)
    finally:
        window.window.close()
