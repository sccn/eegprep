from __future__ import annotations

import os

import numpy as np
import pytest

from eegprep.functions.guifunc.errordlg2 import build_errordlg2
from eegprep.functions.guifunc.inputdlg2 import inputdlg2, inputdlg2_dialog_spec
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.listdlg2 import build_listdlg2_dialog, listdlg2
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.guifunc.supergui import supergui
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


def _spec(*, geomvert: tuple[float, ...] | None = None, help_text: str | None = None) -> DialogSpec:
    return DialogSpec(
        title="MyGUI",
        controls=(
            ControlSpec("radiobutton", "you have the choice", tag="choice", value=False),
            ControlSpec("pushbutton", "pushme"),
            ControlSpec("edit", tag="entry", value="PLEASE PRESS [Ok]"),
        ),
        geometry=((1, 1), (1,)),
        geomvert=geomvert,
        function_name="test_inputgui",
        eeglab_source="functions/guifunc/inputgui.m",
        help_text=help_text,
    )


class RecordingRenderer:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def run(self, spec, initial_values=None):
        self.calls.append((spec, initial_values))
        return self.result


class RecordingPlotRenderer:
    def __init__(self):
        self.app = RecordingApplication()
        self.dialog = RecordingDialog()
        self.widgets = {"entry": object()}
        self.calls = []

    def build_dialog(self, spec, initial_values=None):
        self.calls.append((spec, initial_values))
        return self.app, self.dialog, self.widgets


class RecordingApplication:
    def __init__(self):
        self.processed = 0

    def processEvents(self):
        self.processed += 1


class RecordingDialog:
    def __init__(self):
        self.shown = 0

    def show(self):
        self.shown += 1


def test_inputgui_requires_a_dialog_spec():
    with pytest.raises(TypeError):
        inputgui()


def test_inputgui_returns_tagged_values_from_renderer():
    renderer = RecordingRenderer({"choice": True, "entry": "accepted"})
    spec = _spec()

    result = inputgui(spec, initial_values={"entry": "initial"}, renderer=renderer)

    assert result == {"choice": True, "entry": "accepted"}
    assert renderer.calls == [(spec, {"entry": "initial"})]


def test_inputgui_preserves_explicit_vertical_geometry():
    renderer = RecordingRenderer({})
    spec = _spec(geomvert=(4, 1))

    inputgui(spec, renderer=renderer)

    assert renderer.calls[0][0].geomvert == (4, 1)


def test_inputgui_preserves_help_target_for_the_renderer():
    renderer = RecordingRenderer({})
    spec = _spec(help_text="pophelp('pop_editoptions')")

    inputgui(spec, renderer=renderer)

    assert renderer.calls[0][0].help_text == "pophelp('pop_editoptions')"
    assert renderer.calls[0][0].show_help_button


def test_inputgui_represents_multiple_help_actions_as_explicit_controls(qt_widgets):
    spec = DialogSpec(
        title="MyGUI",
        controls=(
            ControlSpec(
                "pushbutton",
                "Help gui",
                tag="help_gui",
                callback=CallbackSpec(
                    "show_message",
                    {"button": "help_gui", "title": "Help gui", "message": "First help action"},
                ),
            ),
            ControlSpec(
                "pushbutton",
                "More help",
                tag="more_help",
                callback=CallbackSpec(
                    "show_message",
                    {"button": "more_help", "title": "More help", "message": "Second help action"},
                ),
            ),
        ),
        geometry=((1, 1),),
        function_name="test_inputgui_help",
        eeglab_source="functions/guifunc/inputgui.m",
        show_help_button=False,
    )

    _app, dialog, widgets = inputgui(spec, mode="plot")

    assert isinstance(widgets["help_gui"], qt_widgets.QPushButton)
    assert isinstance(widgets["more_help"], qt_widgets.QPushButton)
    assert widgets["help_gui"].text() == "Help gui"
    assert widgets["more_help"].text() == "More help"
    assert dialog.isVisible()
    dialog.close()


def test_inputgui_rejects_a_numeric_help_target():
    spec = _spec(help_text=100)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="help_text must be a string"):
        inputgui(spec, renderer=RecordingRenderer({}))


def test_inputgui_accepts_its_three_python_arguments_positionally():
    renderer = RecordingRenderer({"entry": "accepted"})
    spec = _spec()

    result = inputgui(spec, {"entry": "initial"}, renderer)

    assert result == {"entry": "accepted"}
    assert renderer.calls == [(spec, {"entry": "initial"})]


def test_inputgui_plot_mode_returns_a_visible_nonmodal_dialog():
    renderer = RecordingPlotRenderer()
    spec = _spec()

    result = inputgui(spec, {"entry": "preview"}, renderer, mode="plot")

    assert result == (renderer.app, renderer.dialog, renderer.widgets)
    assert renderer.calls == [(spec, {"entry": "preview"})]
    assert renderer.dialog.shown == 1
    assert renderer.app.processed == 1


def test_inputgui_tagged_mapping_is_the_python_rich_output():
    renderer = RecordingRenderer({"choice": True, "entry": "accepted"})

    result = inputgui(_spec(), initial_values={"entry": "initial"}, renderer=renderer)

    assert result == {"choice": True, "entry": "accepted"}
    assert "pushme" not in result


def test_inputgui_renderer_can_be_reused_without_retaining_initial_values():
    renderer = RecordingRenderer({"entry": "accepted"})
    spec = _spec()

    inputgui(spec, initial_values={"entry": "first"}, renderer=renderer)
    inputgui(spec, initial_values={"entry": "second"}, renderer=renderer)

    assert renderer.calls == [(spec, {"entry": "first"}), (spec, {"entry": "second"})]


def test_supergui_builds_geometry_controls_and_layout_options(qt_widgets):
    spec = DialogSpec(
        title="MyGUI",
        controls=(
            ControlSpec("radiobutton", "radio", tag="radio"),
            ControlSpec("pushbutton", "push", tag="push"),
        ),
        geometry=((1, 1),),
        geomvert=(3, 2),
        function_name="test_supergui",
        eeglab_source="functions/guifunc/supergui.m",
        content_margins=(10, 8, 14, 12),
        row_spacing=2,
    )

    _app, dialog, widgets = supergui(spec)

    assert spec.title == "MyGUI"
    assert spec.geometry == ((1, 1),)
    assert spec.geomvert == (3, 2)
    assert [(control.style, control.string) for control in spec.controls] == [
        ("radiobutton", "radio"),
        ("pushbutton", "push"),
    ]
    assert spec.content_margins == (10, 8, 14, 12)
    assert spec.row_spacing == 2
    assert dialog.windowTitle() == "MyGUI"
    assert isinstance(widgets["radio"], qt_widgets.QRadioButton)
    assert isinstance(widgets["push"], qt_widgets.QPushButton)
    assert widgets["radio"].text() == "radio"
    assert widgets["push"].text() == "push"
    dialog.close()


def test_inputdlg2_requires_a_prompt_and_title():
    with pytest.raises(TypeError):
        inputdlg2()


def test_inputdlg2_rejects_mismatched_prompts_and_defaults():
    with pytest.raises(ValueError, match="same length"):
        inputdlg2_dialog_spec(["testcase", "another test"], "inputdlg2 testcase", 1, ["this"])


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


def test_inputdlg2_uses_vertical_rows_for_a_multiline_prompt():
    spec = inputdlg2_dialog_spec([["test", "case"], "another test"], "inputdlg2 testcase", 1, ["this", "that"])

    assert spec.controls[0].string == "test\ncase"
    assert spec.geometry == ((1,), (1,), (1, 0.6))
    assert spec.geomvert == (2, 1)


def test_inputdlg2_omits_help_when_no_function_name_is_given():
    spec = inputdlg2_dialog_spec(["testcase", "another test"], "inputdlg2 testcase", 1, ["this", "that"])

    assert spec.function_name == "inputdlg2"
    assert spec.help_text is None
    assert not spec.show_help_button


@pytest.fixture
def qt_widgets():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield widgets
    app.processEvents()


def test_errordlg2_builds_a_critical_message_with_requested_text(qt_widgets):
    _app, dialog = build_errordlg2("Explanation of error", "testcase for errordlg2")

    assert dialog.text() == "Explanation of error"
    assert dialog.icon() == qt_widgets.QMessageBox.Icon.Critical
    assert dialog.standardButtons() & qt_widgets.QMessageBox.StandardButton.Ok
    dialog.close()


def test_listdlg2_requires_list_items():
    with pytest.raises(TypeError):
        listdlg2()


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


def test_listdlg2_displays_prompt_text(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring=["one", "two"], promptstring=["Choose", "values"])
    prompt = dialog.findChild(qt_widgets.QLabel, "prompt")

    assert prompt.text() == "Choose\nvalues"
    dialog.close()


def test_listdlg2_uses_requested_window_size(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring=["one", "two"], listsize=(420, 260))

    assert (dialog.width(), dialog.height()) == (420, 260)
    dialog.close()


def test_listdlg2_uses_requested_window_title(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring=["one", "two"], name="My list")

    assert dialog.windowTitle() == "My list"
    dialog.close()


def test_listdlg2_single_mode_selects_only_one_item(qt_widgets):
    _app, dialog = build_listdlg2_dialog(
        liststring=["This", "is", "a", "testcase"],
        selectionmode="single",
    )
    list_widget = dialog.findChild(qt_widgets.QListWidget, "listboxvals")

    assert list_widget.selectionMode() == qt_widgets.QAbstractItemView.SingleSelection
    assert list_widget.item(0).isSelected()
    dialog.close()


def test_listdlg2_treats_a_string_as_one_list_item(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring="This is a testcase")
    list_widget = dialog.findChild(qt_widgets.QListWidget, "listboxvals")

    assert list_widget.count() == 1
    assert list_widget.item(0).text() == "This is a testcase"
    assert list_widget.selectionMode() == qt_widgets.QAbstractItemView.SingleSelection
    dialog.close()


def test_listdlg2_uses_custom_ok_label_and_accepts(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring=["one", "two"], okstring="TEST_OK")
    ok = dialog.findChild(qt_widgets.QPushButton, "ok")

    assert ok.text() == "TEST_OK"
    assert ok.width() >= ok.sizeHint().width()
    ok.click()
    assert dialog.result() == qt_widgets.QDialog.Accepted
    dialog.close()


def test_listdlg2_uses_custom_cancel_label_and_rejects(qt_widgets):
    _app, dialog = build_listdlg2_dialog(liststring=["one", "two"], cancelstring="TEST_CANCEL")
    cancel = dialog.findChild(qt_widgets.QPushButton, "cancel")

    assert cancel.text() == "TEST_CANCEL"
    assert cancel.width() >= cancel.sizeHint().width()
    cancel.click()
    assert dialog.result() == qt_widgets.QDialog.Rejected
    dialog.close()
