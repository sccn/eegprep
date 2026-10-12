from __future__ import annotations

import os

import numpy as np
import pytest

from eegprep.functions.guifunc.inputdlg2 import inputdlg2
from eegprep.functions.guifunc.listdlg2 import build_listdlg2_dialog
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
