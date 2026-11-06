"""Release-readiness regressions for the scroll window (issue 547, stream 3)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej
from eegprep.functions.popfunc.pop_eegplot import apply_eegplot_rejections, pop_eegplot
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.eegplot import eegplot2event, normalize_events, trial2eegplot, winrej_to_array
from tests.fixtures import (
    SAMPLE_DATASET_PATH,
    create_test_eeg,
    create_test_eeg_with_ica,
    matlab_engine_available,
)


_INSTRUCTIONS = (
    "Mark stretches of continuous data for rejection\n"
    "by dragging the left mouse button. Click on marked\n"
    'stretches to unmark. When done,press "REJECT" to\n'
    "excise marked stretches (Note: Leaves rejection\n"
    "boundary markers in the event table)."
)
_EPOCH_QUESTION = "Are you sure, you want to reject the labeled trials ?"
_MANUAL_COLOR = [1.0, 0.9, 0.9]


def test_string_event_types_use_eegplot_unique_palette() -> None:
    # eegplot.m:1164-1171 sorts string types and paints them from
    # {r, [0 0.8 0], m, c, k, b, [0 0.8 0]}.
    from eegprep.functions.guifunc.eegbrowser import _EVENT_COLORS

    events = normalize_events(
        [
            {"type": "square", "latency": 10},
            {"type": "rt", "latency": 20},
            {"type": "square", "latency": 30},
        ]
    )
    by_type = {event.type: event.color_index for event in events}

    assert _EVENT_COLORS[0] == (255, 0, 0)
    assert _EVENT_COLORS[1] == (0, 204, 0)
    assert _EVENT_COLORS[6] == (0, 204, 0)
    assert by_type == {"rt": 0, "square": 1}


def test_show_false_omitted_reject_does_not_open_a_dialog() -> None:
    eeg = create_test_eeg(n_channels=1, n_samples=8, srate=10)
    out, command = pop_eegplot(eeg, return_com=True, show=False)

    assert out is eeg
    assert command == "pop_eegplot(EEG, 1, 0, 1)"


def test_apply_eegplot_rejections_empty_continuous_reject_records_nothing() -> None:
    eeg = _continuous_eeg()
    original = np.array(eeg["data"], copy=True)

    out, command = apply_eegplot_rejections(eeg, np.zeros((0, 6)), reject=1, return_com=True)

    assert command == ""
    assert out["pnts"] == eeg["pnts"]
    np.testing.assert_array_equal(out["data"], original)


def test_apply_eegplot_rejections_epoch_command_matches_pop_rejepoch_history() -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=4, n_trials=4, srate=10)
    eeg["data"] = np.arange(32, dtype=float).reshape(2, 4, 4)
    rows = trial2eegplot([False, True, False, True], np.zeros((2, 4), dtype=bool), 4, _MANUAL_COLOR)

    out, command = apply_eegplot_rejections(eeg, rows, return_com=True)

    assert command == "EEG = pop_rejepoch( EEG, [2 4] ,0);"
    assert out["trials"] == 2
    np.testing.assert_array_equal(out["data"], eeg["data"][:, :, [0, 2]])


def test_scroll_window_region_removed_by_eeg_eegrej_is_exact() -> None:
    eeg = _continuous_eeg(n_samples=40)
    rows = np.array([[13, 24, 1.0, 0.9, 0.9, 1.0]])
    original = np.array(eeg["data"], copy=True)

    out, command = apply_eegplot_rejections(eeg, rows, return_com=True)

    assert command == "EEG = eeg_eegrej( EEG, [13 24] );"
    assert out["pnts"] == 28
    np.testing.assert_array_equal(out["data"], np.concatenate([original[:, :12], original[:, 24:]], axis=1))
    assert out["event"][0]["type"] == "boundary"


@pytest.mark.matlab
@pytest.mark.skipif(not matlab_engine_available(), reason="MATLAB engine not available or skipped")
def test_scroll_window_eeg_eegrej_region_matches_matlab() -> None:
    from eegprep.functions.adminfunc.eeglabcompat import get_eeglab

    eeglab = get_eeglab("MAT")
    eeg = pop_loadset(str(SAMPLE_DATASET_PATH))
    rows = np.array([[13, 24, 1.0, 0.9, 0.9, 1.0]])
    regions = eegplot2event(rows, -1)
    py = eeg_eegrej(eeg, regions)
    mat = eeglab.eeg_eegrej(eeg, regions)
    mat_regions = np.asarray(eeglab.eegplot2event(rows, -1), dtype=float)

    np.testing.assert_array_equal(regions, mat_regions)
    assert int(py["pnts"]) == int(np.asarray(mat["pnts"]).reshape(-1)[0])
    np.testing.assert_array_equal(np.asarray(py["data"]), np.asarray(mat["data"]))


def _close_browser_windows(app) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    for widget in list(app.topLevelWidgets()):
        if isinstance(widget, EEGBrowserWindow):
            widget.close()
            widget.deleteLater()
    app.processEvents()


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _close_browser_windows(app)
    yield app
    _close_browser_windows(app)


def test_gui_explicit_reject_skips_instruction_dialog(qapp) -> None:
    eeg = _continuous_eeg()
    window = _open_browser(eeg, reject=1)
    assert window.model.state.winrej == []
    window.close()
    qapp.processEvents()


def test_gui_continuous_drag_marks_one_based_samples_and_click_clears(qapp) -> None:
    eeg = _continuous_eeg(n_samples=80)
    window = _open_browser(eeg, reject=1)
    canvas = window.canvas
    start = _view_sample(canvas, 1.2)
    end = _view_sample(canvas, 2.3)
    assert start == int(round(1.2 * 10)) + 1
    assert end == int(round(2.3 * 10)) + 1

    _mouse(canvas, "press", 1.2)
    _mouse(canvas, "release", 2.3)
    rows = winrej_to_array(canvas.model.state.winrej, canvas.model.data.n_channels)
    np.testing.assert_array_equal(rows[:, :2], [[start, end]])

    _mouse(canvas, "press", 1.6)
    _mouse(canvas, "release", 1.6)
    assert canvas.model.state.winrej == []
    window.close()
    qapp.processEvents()


def test_gui_epoch_click_marks_one_epoch_and_drag_does_not_spread(qapp) -> None:
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=6, srate=100)
    eeg["data"] = np.zeros((2, 10, 6))
    window = _open_browser(eeg, reject=1)
    canvas = window.canvas

    _mouse(canvas, "press", 15)
    _mouse(canvas, "release", 25)
    rows = winrej_to_array(canvas.model.state.winrej, 2)
    np.testing.assert_array_equal(rows[:, :2], [[10, 19]])

    _mouse(canvas, "press", 35)
    _mouse(canvas, "release", 35)
    rows = winrej_to_array(canvas.model.state.winrej, 2)
    np.testing.assert_array_equal(rows[:, :2], [[10, 19], [30, 39]])

    _mouse(canvas, "press", 15)
    _mouse(canvas, "release", 15)
    rows = winrej_to_array(canvas.model.state.winrej, 2)
    np.testing.assert_array_equal(rows[:, :2], [[30, 39]])
    window.close()
    qapp.processEvents()


def test_gui_component_epoch_click_marks_that_epoch(qapp) -> None:
    eeg = create_test_eeg_with_ica(n_channels=3, n_samples=20, n_trials=4, n_components=2, srate=100)
    window = _open_browser(eeg, icacomp=0, reject=1)
    canvas = window.canvas
    assert canvas.model.data.epoched

    _mouse(canvas, "press", 30)
    _mouse(canvas, "release", 30)
    rows = winrej_to_array(canvas.model.state.winrej, canvas.model.data.n_channels)
    np.testing.assert_array_equal(rows[:, :2], [[20, 39]])
    window.close()
    qapp.processEvents()


def test_gui_event_lines_keep_distinct_eegplot_colors(qapp) -> None:
    eeg = _continuous_eeg(n_samples=40)
    eeg["event"] = [
        {"type": "square", "latency": 5},
        {"type": "rt", "latency": 15},
        {"type": "square", "latency": 25},
    ]
    window = _open_browser(eeg, reject=1)
    colors = set()
    for line in window.canvas._event_line_items:
        color = line.pen().color()
        colors.add((color.red(), color.green(), color.blue()))

    assert (255, 0, 0) in colors
    assert (0, 204, 0) in colors
    window.close()
    qapp.processEvents()


@pytest.mark.modal_dialogs
def test_gui_inspect_by_eye_cancel_and_close_do_not_open_browser(qapp) -> None:
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    session = EEGPrepSession()
    session.store_current(_continuous_eeg(), new=True)
    parent = qt_widgets.QWidget()
    parent.show()
    dispatcher = MenuActionDispatcher(session)

    state = _drive(qapp, qt_widgets, [_click_message("Cancel")])
    dispatcher.dispatch("pop_eegplot:data", parent)

    assert state["seen"] == [("Warning", _INSTRUCTIONS)]
    assert session.ALLCOM == []
    assert _browser_windows() == []

    state = _drive(qapp, qt_widgets, [_dismiss_message])
    dispatcher.dispatch("pop_eegplot:data", parent)

    assert state["seen"][0][0] == "Warning"
    assert session.ALLCOM == []
    assert _browser_windows() == []
    parent.close()


@pytest.mark.modal_dialogs
def test_gui_channel_scroll_skips_instruction_dialog(qapp) -> None:
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    session = EEGPrepSession()
    session.store_current(_continuous_eeg(), new=True)
    parent = qt_widgets.QWidget()
    parent.show()
    state = {"box": False, "watch": True}

    def watch() -> None:
        if not state["watch"]:
            return
        child = qapp.activeModalWidget()
        if isinstance(child, qt_widgets.QMessageBox):
            state["box"] = True
            child.reject()
            return
        from PySide6 import QtCore

        QtCore.QTimer.singleShot(10, watch)

    from PySide6 import QtCore

    QtCore.QTimer.singleShot(0, watch)
    MenuActionDispatcher(session).dispatch("pop_eegplot:channels", parent)
    state["watch"] = False

    assert state["box"] is False
    assert len(_browser_windows()) == 1
    assert session.ALLCOM == ["pop_eegplot(EEG, 1, 1, 1)"]
    _browser_windows()[0].close()
    parent.close()
    qapp.processEvents()


@pytest.mark.modal_dialogs
def test_gui_reject_cancel_keeps_dataset_and_ok_records_eeg_eegrej_and_pop_newset(qapp) -> None:
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    eeg = _continuous_eeg(n_samples=80)
    original = np.array(eeg["data"], copy=True)
    session = EEGPrepSession()
    session.store_current(eeg, new=True)
    parent = qt_widgets.QWidget()
    parent.show()
    dispatcher = MenuActionDispatcher(session)

    _drive(qapp, qt_widgets, [_click_message("Continue")])
    dispatcher.dispatch("pop_eegplot:data", parent)
    window = _browser_windows()[0]
    _mouse(window.canvas, "press", 1.2)
    _mouse(window.canvas, "release", 2.3)
    rows = winrej_to_array(window.canvas.model.state.winrej, 1)
    assert rows.shape[0] == 1
    _drive(qapp, qt_widgets, [_click_named("cancel")])
    _reject_button(window).click()
    qapp.processEvents()

    assert len(session.ALLEEG) == 1
    np.testing.assert_array_equal(session.EEG["data"], original)
    assert session.ALLCOM == ["pop_eegplot(EEG, 1, 0, 1)"]
    assert _browser_windows() == []

    _drive(qapp, qt_widgets, [_click_message("Continue")])
    dispatcher.dispatch("pop_eegplot:data", parent)
    window = _browser_windows()[0]
    _mouse(window.canvas, "press", 1.2)
    _mouse(window.canvas, "release", 2.3)
    rows = winrej_to_array(window.canvas.model.state.winrej, 1)
    start, end = int(rows[0, 0]), int(rows[0, 1])
    assert [start, end] == [int(round(1.2 * 10)) + 1, int(round(2.3 * 10)) + 1]
    _drive(qapp, qt_widgets, [_click_named("ok")])
    _reject_button(window).click()
    qapp.processEvents()

    expected = np.concatenate([original[:, : start - 1], original[:, end:]], axis=1)
    assert len(session.ALLEEG) == 2
    assert session.ALLEEG[0]["pnts"] == original.shape[1]
    np.testing.assert_array_equal(session.EEG["data"], expected)
    assert "EEG = eeg_eegrej( EEG, [13 24] );" in session.ALLCOM
    assert any(command.startswith("[ALLEEG EEG CURRENTSET] = pop_newset(") for command in session.ALLCOM)
    parent.close()


@pytest.mark.modal_dialogs
def test_gui_epoch_reject_no_and_close_keep_trials_yes_opens_pop_newset(qapp) -> None:
    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    eeg = create_test_eeg(n_channels=2, n_samples=10, n_trials=6, srate=100)
    eeg["data"] = np.arange(120, dtype=float).reshape(2, 10, 6)
    session = EEGPrepSession()
    session.store_current(eeg, new=True)
    parent = qt_widgets.QWidget()
    parent.show()
    dispatcher = MenuActionDispatcher(session)

    dispatcher.dispatch("pop_eegplot:reject_data", parent)
    window = _browser_windows()[0]
    _mouse(window.canvas, "press", 15)
    _mouse(window.canvas, "release", 15)
    state = _drive(qapp, qt_widgets, [_click_message("NO")])
    _reject_button(window).click()
    qapp.processEvents()

    assert state["seen"] == [(_epoch_title(), _EPOCH_QUESTION)]
    assert session.EEG["trials"] == 6
    assert len(session.ALLEEG) == 1
    assert not any("pop_rejepoch" in command for command in session.ALLCOM)

    dispatcher.dispatch("pop_eegplot:reject_data", parent)
    window = _browser_windows()[0]
    _mouse(window.canvas, "press", 15)
    _mouse(window.canvas, "release", 15)
    _drive(qapp, qt_widgets, [_dismiss_message])
    _reject_button(window).click()
    qapp.processEvents()

    assert session.EEG["trials"] == 6
    assert not any("pop_rejepoch" in command for command in session.ALLCOM)

    dispatcher.dispatch("pop_eegplot:reject_data", parent)
    window = _browser_windows()[0]
    _mouse(window.canvas, "press", 15)
    _mouse(window.canvas, "release", 15)
    _mouse(window.canvas, "press", 35)
    _mouse(window.canvas, "release", 35)
    state = _drive(qapp, qt_widgets, [_click_message("YES"), _click_named("ok")])
    _reject_button(window).click()
    qapp.processEvents()

    assert state["seen"][0] == (_epoch_title(), _EPOCH_QUESTION)
    assert state["seen"][1][0] == "Dataset info -- pop_newset()"
    assert session.EEG["trials"] == 4
    np.testing.assert_array_equal(session.EEG["data"], eeg["data"][:, :, [0, 2, 4, 5]])
    assert "EEG = pop_rejepoch( EEG, [2 4] ,0);" in session.ALLCOM
    assert any(command.startswith("[ALLEEG EEG CURRENTSET] = pop_newset(") for command in session.ALLCOM)
    parent.close()


def _continuous_eeg(n_samples: int = 40) -> dict:
    eeg = create_test_eeg(n_channels=1, n_samples=n_samples, srate=10)
    eeg["data"] = np.arange(n_samples, dtype=float).reshape(1, n_samples)
    eeg["event"] = []
    return eeg


def _epoch_title() -> str:
    return "Reject pre-labelled epochs -- pop_rejepoch()"


def _qt_app():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("Qt application is not running")
    return app


def _browser_windows(*, visible_only: bool = True) -> list:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    return [
        widget
        for widget in _qt_app().topLevelWidgets()
        if isinstance(widget, EEGBrowserWindow) and (widget.isVisible() or not visible_only)
    ]


def _open_browser(eeg: dict, **kwargs):
    before = {id(window) for window in _browser_windows(visible_only=False)}
    pop_eegplot(eeg, return_com=True, command_callback=lambda *_args: None, **kwargs)
    opened = [window for window in _browser_windows() if id(window) not in before]
    assert len(opened) == 1
    window = opened[0]
    app = _qt_app()
    window.show()
    window.resize(960, 560)
    app.processEvents()
    window.canvas.redraw()
    app.processEvents()
    return window


def _view_sample(canvas, x_value: float) -> int:
    point = _viewport_point(canvas, x_value)
    from PySide6 import QtCore, QtGui

    local = QtCore.QPointF(point)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        local,
        local,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    scene = canvas._event_scene_point(event)
    return canvas._event_sample(scene)


def _mouse(canvas, kind: str, x_value: float) -> None:
    from PySide6 import QtCore, QtGui, QtWidgets

    point = QtCore.QPointF(_viewport_point(canvas, x_value))
    event_type = QtCore.QEvent.Type.MouseButtonPress if kind == "press" else QtCore.QEvent.Type.MouseButtonRelease
    event = QtGui.QMouseEvent(
        event_type,
        point,
        point,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(canvas.plot.viewport(), event)
    _qt_app().processEvents()


def _viewport_point(canvas, x_value: float):
    from PySide6 import QtCore

    view_box = canvas.plot.getViewBox()
    y_value = float(view_box.viewRect().center().y())
    scene_point = view_box.mapViewToScene(QtCore.QPointF(x_value, y_value))
    return canvas.plot.mapFromScene(scene_point)


def _reject_button(window):
    from PySide6 import QtWidgets

    return next(button for button in window.findChildren(QtWidgets.QPushButton) if button.text() == "Reject")


def _drive(app, qt_widgets, actions: list) -> dict:
    from PySide6 import QtCore

    state: dict = {"step": 0, "seen": [], "tries": 0, "timeout": False}

    def poll() -> None:
        if state["step"] >= len(actions):
            return
        child = app.activeModalWidget()
        if child is None or not child.isVisible():
            state["tries"] += 1
            if state["tries"] > 200:
                state["timeout"] = True
                return
            QtCore.QTimer.singleShot(10, poll)
            return
        state["tries"] = 0
        if isinstance(child, qt_widgets.QMessageBox):
            text = child.text()
        else:
            label = child.findChild(qt_widgets.QLabel, "questdlg2_prompt")
            text = label.text() if label is not None else ""
        state["seen"].append((child.windowTitle(), text))
        actions[state["step"]](child)
        state["step"] += 1
        if state["step"] < len(actions):
            QtCore.QTimer.singleShot(10, poll)

    QtCore.QTimer.singleShot(0, poll)
    return state


def _click_message(label: str):
    def click(box) -> None:
        from PySide6 import QtWidgets

        for button in box.findChildren(QtWidgets.QPushButton):
            if button.text() == label:
                button.click()
                return
        raise AssertionError(f"missing dialog button {label}")

    return click


def _dismiss_message(box) -> None:
    box.reject()


def _click_named(object_name: str):
    def click(dialog) -> None:
        from PySide6 import QtWidgets

        button = dialog.findChild(QtWidgets.QPushButton, object_name)
        if button is None:
            raise AssertionError(f"missing dialog button {object_name}")
        button.click()

    return click
