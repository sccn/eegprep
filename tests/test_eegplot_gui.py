from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.eegplot import build_eegplot_model


SAMPLE_DATASET = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    yield app


def test_gui_eegbrowser_renders_nonblank_sample_data(qapp) -> None:
    qt_gui = pytest.importorskip("PySide6.QtGui")
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    eeg = pop_loadset(str(SAMPLE_DATASET))
    data_before = np.array(eeg["data"], copy=True)
    model = build_eegplot_model(eeg, winlength=1, dispchans=8)
    window = EEGBrowserWindow(model)
    window.show()
    qapp.processEvents()

    image = window.canvas.grab().toImage().convertToFormat(qt_gui.QImage.Format_RGBA8888)
    pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)

    assert pixels[..., :3].std() > 0
    assert np.unique(pixels[..., :3].reshape(-1, 3), axis=0).shape[0] >= 5
    np.testing.assert_array_equal(eeg["data"], data_before)
    window.close()


def test_gui_navigation_buttons_and_position_field_update_visible_window(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.arange(100, dtype=float).reshape(1, 100), srate=10, winlength=2, spacing=1)
    window = EEGBrowserWindow(model)

    window.controls.forward_step_button.click()
    qapp.processEvents()

    assert model.state.time == pytest.approx(2.0)
    assert window.controls.position.value() == pytest.approx(2.0)

    window.controls.position.setValue(3.0)
    qapp.processEvents()

    assert model.state.time == pytest.approx(3.0)
    curve = next(item for item in window.canvas._items if hasattr(item, "getData"))
    x_values, _y_values = curve.getData()
    assert x_values[0] == pytest.approx(3.0)
    window.close()


def test_gui_large_continuous_data_decimates_visible_traces(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    samples = 120_000
    srate = 512
    times = np.arange(samples, dtype=np.float32) / float(srate)
    data = np.vstack([np.sin(2 * np.pi * (channel + 1) * times) + channel * 0.01 for channel in range(16)]).astype(
        np.float32
    )
    model = build_eegplot_model(data, srate=srate, winlength=30, dispchans=12, spacing=1, show=False)
    window = EEGBrowserWindow(model)
    window.resize(960, 560)
    window.show()
    qapp.processEvents()
    window.canvas.redraw()

    curves = [item for item in window.canvas._items if hasattr(item, "getData")]
    max_points = max(len(curve.getData()[0]) for curve in curves)

    assert model.data.total_samples == samples
    assert len(curves) >= 12
    assert max_points <= window.canvas.width() * 2 + 2

    window.scroll_time(1.5)
    qapp.processEvents()

    assert model.state.time == pytest.approx(45.0)
    window.close()


def test_gui_linked_child_windows_synchronize_time_and_channels(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    parent_model = build_eegplot_model(np.zeros((4, 100)), srate=10, winlength=2, dispchans=2, spacing=1)
    child_model = build_eegplot_model(np.zeros((4, 100)), srate=10, winlength=2, dispchans=2, spacing=1)
    sibling_model = build_eegplot_model(np.zeros((4, 100)), srate=10, winlength=2, dispchans=2, spacing=1)
    parent = EEGBrowserWindow(parent_model)
    child = EEGBrowserWindow(child_model)
    sibling = EEGBrowserWindow(sibling_model)
    parent.link_child(child)
    parent.link_child(sibling)

    parent.scroll_time(1.0)
    qapp.processEvents()

    assert child_model.state.time == pytest.approx(parent_model.state.time)
    assert sibling_model.state.time == pytest.approx(parent_model.state.time)

    child.scroll_channels(1)
    qapp.processEvents()

    assert parent_model.state.channel_offset == child_model.state.channel_offset
    assert sibling_model.state.channel_offset == child_model.state.channel_offset
    parent.close()


def test_gui_accept_callback_receives_winrej_and_cancel_does_not(qapp) -> None:
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    accepted = []
    model = build_eegplot_model(np.zeros((1, 20)), srate=10, spacing=1)
    window = EEGBrowserWindow(model, accept_callback=accepted.append)
    window.canvas.mark_samples(2, 6)

    window.close()
    assert accepted == []

    window = EEGBrowserWindow(model, accept_callback=accepted.append)
    accept_button = next(button for button in window.findChildren(qt_widgets.QPushButton) if button.text() == "Reject")
    accept_button.click()

    assert accepted
    np.testing.assert_array_equal(accepted[-1][:, :2], [[2, 6]])
    window.close()


@pytest.mark.modal_dialogs
def test_gui_cancel_browser_menu_dialog_keeps_menu_bar(qapp) -> None:
    qt_core = pytest.importorskip("PySide6.QtCore")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    is_valid = pytest.importorskip("shiboken6").isValid
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(
        np.zeros((2, 40)),
        srate=10,
        winlength=2,
        spacing=1,
        plottitle="kept",
    )
    window = EEGBrowserWindow(model)
    window.show()
    qapp.processEvents()
    before_bar = _browser_menu_bar(window)
    assert [item[0] for item in before_bar] == ["Figure", "Display", "Settings", "Help"]
    watched = _watched_browser_actions(window)
    before_actions = [_action_state(action) for action in watched]
    before_tree = _browser_menu_tree(window)

    _popup_browser_menu(qapp, window, "Display")
    gc.collect()
    assert all(is_valid(action) for action in watched), "opening Display collected browser menu actions"

    def cancel() -> None:
        dialog = qapp.activeModalWidget()
        button = _cancel_button(dialog, qt_widgets) if dialog is not None else None
        if button is None:
            qt_core.QTimer.singleShot(0, cancel)
            return
        gc.collect()
        button.click()

    qt_core.QTimer.singleShot(0, cancel)
    window.title_action.trigger()
    qapp.processEvents()
    gc.collect()

    assert _browser_menu_bar(window) == before_bar
    assert [_action_state(action) for action in watched] == before_actions
    assert all(is_valid(action) for action in watched)
    assert _browser_menu_tree(window) == before_tree
    assert model.state.plottitle == "kept"
    window.close()
    qapp.processEvents()


def _watched_browser_actions(window):
    return (
        window.accept_action,
        window.title_action,
        window.marks_action,
        window.xgrid_action,
        window.time_range_action,
        window.zoom_on_action,
        window.events_on_action,
        window.help_action,
    )


def _action_state(action):
    return (action.text(), action.isEnabled())


def _browser_menu_bar(window):
    menubar = window.menuBar()
    return [
        (
            action.text(),
            action.isEnabled(),
            menubar.actionGeometry(action).x(),
            menubar.actionGeometry(action).width(),
        )
        for action in menubar.actions()
    ]


def _browser_menu_tree(window):
    return [
        (action.text(), _menu_tree(action.menu()) if action.menu() is not None else None)
        for action in window.menuBar().actions()
    ]


def _menu_tree(menu):
    return [
        (action.text(), action.isEnabled(), _menu_tree(action.menu()) if action.menu() is not None else None)
        for action in menu.actions()
    ]


def _popup_browser_menu(qapp, window, label: str) -> None:
    menubar = window.menuBar()
    for action in menubar.actions():
        if action.text() == label and action.menu() is not None:
            menu = action.menu()
            menu.popup(menubar.mapToGlobal(menubar.actionGeometry(action).bottomLeft()))
            qapp.processEvents()
            return
    raise AssertionError(f"no browser menu named {label}")


def _cancel_button(dialog, qt_widgets):
    if dialog is None:
        return None
    for button in dialog.findChildren(qt_widgets.QPushButton):
        if button.text().replace("&", "") == "Cancel":
            return button
    return None
