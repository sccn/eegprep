from __future__ import annotations

import os

import numpy as np
import pytest

from eegprep.functions.guifunc.inputdlg2 import inputdlg2
from eegprep.functions.guifunc.listdlg2 import build_listdlg2_dialog
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.gui import close_reference_gui


SUPERGUI_WRAPPER = "unittesting_guifunc/supergui/guifunc_supergui_wrapperTest.m"


def _cell_row(*values):
    cells = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cells[0, index] = value
    return cells


def _reference_supergui(eeglab_backend, request, *args, nargout):
    if request.config.getoption("--eeglab-backend") == "matlab":
        # The source requests three outputs, including graphics objects that
        # cannot cross the MAT-file transport. It does not inspect any output.
        eeglab_backend("eegprep_test_gui_call", "supergui", float(nargout), *args, nargout=0)
        window = None
    else:
        # Retain Python's actual dialog only to close the same test-owned
        # window; MATLAB's requested output counts above remain unchanged.
        _, window, _ = eeglab_backend("supergui", *args, nargout=3)
    close_reference_gui(eeglab_backend, request, window=window)


@pytest.mark.gui
@eeglab_test(SUPERGUI_WRAPPER, "test_test_supergui")
def test_reference_supergui(eeglab_backend, request):
    controls = _cell_row(
        _cell_row("style", "radiobutton", "string", "radio"),
        _cell_row("style", "pushbutton", "string", "push"),
    )
    _reference_supergui(eeglab_backend, request, "geomhoriz", _cell_row(1.0, 1.0), "uilist", controls, nargout=0)
    _reference_supergui(
        eeglab_backend,
        request,
        "geom",
        _cell_row(
            _cell_row(1.0, 1.0, np.array([[0.1, 0.1]]), np.array([[0.2, 0.2]])),
            _cell_row(1.0, 1.0, np.array([[0.1, 0.4]]), np.array([[0.4, 0.5]])),
        ),
        "uilist",
        controls,
        "title",
        "MyGUI",
        "userdata",
        np.empty((0, 0)),
        "geomvert",
        np.array([[1.0, 2.0]]),
        "horizontalalignment",
        "left",
        "minwidth",
        10.0,
        "borders",
        np.array([[0.05, 0.04, 0.07, 0.06]]),
        "spacing",
        np.array([[0.02, 0.01]]),
        "inseth",
        0.02,
        "insetv",
        0.02,
        nargout=3,
    )
    _reference_supergui(
        eeglab_backend,
        request,
        "geomhoriz",
        _cell_row(1.0, 1.0),
        "uilist",
        controls,
        "title",
        "MyGUI",
        "userdata",
        np.empty((0, 0)),
        "geomvert",
        np.array([[3.0, 2.0]]),
        "horizontalalignment",
        "center",
        "minwidth",
        10.0,
        "borders",
        np.array([[0.1, 0.08, 0.14, 0.12]]),
        "spacing",
        np.array([[0.02, 0.01]]),
        "inseth",
        0.02,
        "insetv",
        0.02,
        nargout=3,
    )


class RecordingRenderer:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def run(self, spec, initial_values=None):
        self.calls.append((spec, initial_values))
        return self.result


def test_inputdlg2_returns_answers_in_prompt_order():
    renderer = RecordingRenderer({"answer0": "this", "answer1": "that"})

    answer = inputdlg2(
        ["testcase", "another test"],
        "inputdlg2 testcase",
        1,
        ["this", "that"],
        "i_pass_general",
        renderer=renderer,
    )

    spec, initial_values = renderer.calls[0]
    assert answer == ["this", "that"]
    assert initial_values is None
    assert spec.title == "inputdlg2 testcase"
    assert spec.help_text == "i_pass_general"
    assert [control.value for control in spec.controls if control.style == "edit"] == ["this", "that"]


@pytest.fixture
def qt_widgets():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield widgets
    app.processEvents()


def test_listdlg2_defaults_to_multiple_selection(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring=["This", "is", "a", "testcase"])
    list_widget = dialog.findChild(qt_widgets.QListWidget, "listboxvals")

    assert list_widget.count() == 4
    assert list_widget.selectionMode() == qt_widgets.QAbstractItemView.ExtendedSelection
    dialog.close()


def test_listdlg2_selects_one_based_initial_values(qt_widgets):
    _app, dialog = build_listdlg2_dialog(
        liststring=["This", "is", "a", "testcase"],
        initialvalue=[1, 3],
    )
    list_widget = dialog.findChild(qt_widgets.QListWidget, "listboxvals")

    assert [index + 1 for index in range(list_widget.count()) if list_widget.item(index).isSelected()] == [1, 3]
    dialog.close()


def _event_dialog(fields, types, *, selectionmode="multiple"):
    """Field and type edits with EEGLAB's list buttons, and two mirrored frequency edits."""
    controls = (
        ControlSpec(
            "pushbutton",
            "Field",
            tag="field_button",
            callback=CallbackSpec(
                "select_event_field",
                params={
                    "button": "field_button",
                    "target": "field",
                    "fields": fields,
                    "reminder_source": "type",
                    "reminder": "pick a type",
                },
            ),
        ),
        ControlSpec(
            "pushbutton",
            "Types",
            tag="type_button",
            callback=CallbackSpec(
                "select_event_types",
                params={
                    "button": "type_button",
                    "target": "type",
                    "event_types": types,
                    "selectionmode": selectionmode,
                    "no_types_message": "No type field",
                    "reminder_source": "field",
                    "reminder": "pick a field",
                },
            ),
        ),
        ControlSpec("edit", tag="field", value=""),
        ControlSpec("edit", tag="type", value=""),
        ControlSpec("edit", tag="phase", callback=CallbackSpec("mirror_text", {"source": "phase", "target": "coher"})),
        ControlSpec("edit", tag="coher", callback=CallbackSpec("mirror_text", {"source": "coher", "target": "phase"})),
    )
    return DialogSpec(
        title="events", controls=controls, geometry=((1, 1), (1, 1), (1, 1)), function_name="events", eeglab_source=""
    )


def test_event_field_list_fills_the_field_and_reminds_about_an_empty_type(qt_dialogs, monkeypatch):
    shown = {}

    def listdlg2(**kwargs):
        shown.update(kwargs)
        return [2], 1, "rt"

    monkeypatch.setattr(qt_dialogs.qt, "listdlg2", listdlg2)
    widgets = qt_dialogs.build(_event_dialog(["latency", "rt"], ["stim"]))
    widgets["field_button"].click()
    assert widgets["field"].text() == "rt"
    assert shown["promptstring"] == "Select fields" and shown["selectionmode"] == "single"
    assert qt_dialogs.messages == [("Warning", "pick a type")]
    qt_dialogs.messages.clear()
    widgets["type"].setText("stim")
    monkeypatch.setattr(qt_dialogs.qt, "listdlg2", lambda **kwargs: ([], 0, ""))
    widgets["field_button"].click()
    assert widgets["field"].text() == "rt" and qt_dialogs.messages == []
    qt_dialogs.build(_event_dialog([], ["stim"]))["field_button"].click()
    assert qt_dialogs.messages == [("Error", "No events"), ("Warning", "pick a type")]


def test_event_type_list_selects_several_types_and_reminds_about_an_empty_field(qt_dialogs):
    widgets = qt_dialogs.build(_event_dialog(["latency"], ["stim", "rt"]))
    qt_dialogs.answers["chansel"] = ([1, 2], 1, "")
    widgets["type"].setText("stim")
    widgets["type_button"].click()
    assert widgets["type"].text() == "stim rt"
    assert qt_dialogs.messages == [("Warning", "pick a field")]
    qt_dialogs.messages.clear()
    qt_dialogs.build(_event_dialog(["latency"], []))["type_button"].click()
    assert qt_dialogs.messages[0] == ("Error", "No type field")


def test_single_event_type_selection_appends_the_chosen_type(qt_dialogs):
    qt_dialogs.answers["item"] = ("rt", True)
    widgets = qt_dialogs.build(_event_dialog(["latency"], ["stim", "rt"], selectionmode="single"))
    widgets["type"].setText("stim")
    widgets["type_button"].click()
    assert widgets["type"].text() == "stim rt" and qt_dialogs.messages == []


def test_mirrored_edits_copy_only_when_both_are_filled(qt_dialogs):
    widgets = qt_dialogs.build(_event_dialog([], []))
    widgets["phase"].setText("10")
    widgets["phase"].editingFinished.emit()
    assert widgets["coher"].text() == ""
    widgets["coher"].setText("12")
    widgets["phase"].editingFinished.emit()
    assert widgets["coher"].text() == "10"
    widgets["coher"].setText("8 12")
    widgets["coher"].editingFinished.emit()
    assert widgets["phase"].text() == "8 12"
