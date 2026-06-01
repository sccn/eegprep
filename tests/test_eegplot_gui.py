from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.eegplot import build_eegplot_model, eegplot


SAMPLE_DATASET = Path(__file__).resolve().parents[1] / "sample_data" / "eeglab_data.set"


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
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
    assert np.unique(pixels[..., :3].reshape(-1, 3), axis=0).shape[0] > 10
    np.testing.assert_array_equal(eeg["data"], data_before)
    window.close()


def test_gui_display_menu_labels_refresh_after_toggle(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((1, 20)), srate=10, spacing=1, xgrid="off", ygrid="on", scale="on")
    window = EEGBrowserWindow(model)
    window.show()
    qapp.processEvents()

    assert window.xgrid_action.text() == "X grid on"
    assert window.ygrid_action.text() == "Y grid off"
    assert window.scale_action.text() == "Show scale"
    assert window.scale_action.isChecked() is True
    assert window.controls.event_button.isHidden() is True

    window.xgrid_action.trigger()
    window.ygrid_action.trigger()
    window.scale_action.trigger()

    assert window.xgrid_action.text() == "X grid off"
    assert window.ygrid_action.text() == "Y grid on"
    assert window.scale_action.text() == "Show scale"
    assert window.scale_action.isChecked() is False
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


def test_gui_shell_controls_use_eeglab_browser_geometry(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((8, 100)), srate=10, winlength=2, dispchans=6, spacing=1)
    window = EEGBrowserWindow(model)
    window.resize(960, 560)
    window.show()
    qapp.processEvents()

    assert window.channel_slider.geometry().x() < window.canvas.geometry().x()
    assert window.scale_indicator.geometry().x() >= window.canvas.geometry().right() - 2
    assert window.controls.stack_button.geometry().y() < window.canvas.geometry().y()
    assert window.controls.norm_button.geometry().y() > window.controls.stack_button.geometry().y()
    assert window.controls.cancel_button.geometry().x() < window.controls.back_page_button.geometry().x()
    assert window.controls.back_step_button.geometry().right() <= window.controls.position.geometry().x()
    assert window.controls.position.geometry().right() <= window.controls.forward_step_button.geometry().x()
    assert window.controls.plus_button.geometry().y() < window.controls.minus_button.geometry().y()
    window.close()


def test_gui_scale_buttons_stack_and_norm_controls_update_state(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    data = np.vstack([np.linspace(0, 9, 100), np.linspace(10, 19, 100)])
    model = build_eegplot_model(data, srate=10, winlength=2, dispchans=2, spacing=10)
    window = EEGBrowserWindow(model)

    window.controls.plus_button.click()
    qapp.processEvents()

    assert model.state.spacing == pytest.approx(11.0)
    assert window.controls.spacing.value() == pytest.approx(11.0)

    window.controls.minus_button.click()
    qapp.processEvents()

    assert model.state.spacing == pytest.approx(9.9)

    window.controls.stack_button.click()
    window.controls.norm_button.click()
    qapp.processEvents()

    assert model.state.stacked is True
    assert model.state.normalized is True
    assert window.controls.stack_button.text() == "Spread"
    assert window.controls.norm_button.text() == "Denorm"
    assert model.state.spacing == pytest.approx(5.0)

    window.controls.norm_button.click()
    qapp.processEvents()

    assert model.state.normalized is False
    assert model.state.spacing == pytest.approx(9.9)
    window.close()


def test_gui_spacing_floor_matches_spinbox_floor(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((1, 20)), srate=10, spacing=1)
    window = EEGBrowserWindow(model)

    window.set_spacing(0)
    qapp.processEvents()

    assert model.state.spacing == pytest.approx(0.001)
    assert window.controls.spacing.value() == pytest.approx(0.001)
    window.close()


def test_gui_menus_toggle_events_marks_scale_and_channel_labels(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(
        np.zeros((2, 30)),
        srate=10,
        spacing=1,
        events=[{"type": "stim", "latency": 5, "duration": 4}],
        winrej=[[1, 10, 0.7, 1.0, 0.9, 1, 1]],
    )
    window = EEGBrowserWindow(model)

    assert model.state.show_events is True
    window.events_off_action.trigger()
    assert model.state.show_events is False
    window.events_on_action.trigger()
    assert model.state.show_events is True

    assert model.state.show_marks is True
    window.marks_action.trigger()
    assert model.state.show_marks is False
    assert window.marks_action.text() == "Show marks"

    assert model.state.scale is True
    window.scale_action.trigger()
    assert model.state.scale is False
    assert window.scale_action.text() == "Show scale"

    assert window.event_duration_action.isEnabled() is True
    assert model.state.show_event_durations is False
    window.event_duration_action.trigger()
    assert model.state.show_event_durations is True
    assert window.event_duration_action.text() == "Hide event duration"

    window.show_numbers_action.trigger()
    assert model.state.channel_label_mode == "numbers"
    window.show_labels_action.trigger()
    assert model.state.channel_label_mode == "labels"
    assert [action.text() for action in window.show_labels_action.parent().actions()] == [
        "Show number",
        "Show label",
        "Load .loc(s) file",
    ]
    window.close()


def test_gui_winrej_regions_are_visibly_saturated(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(
        np.zeros((2, 30)),
        srate=10,
        spacing=1,
        winrej=[[1, 10, 0.7, 1.0, 0.9, 1, 1]],
    )
    window = EEGBrowserWindow(model)

    regions = [item for item in window.canvas._items if item.__class__.__name__ == "LinearRegionItem"]

    assert len(regions) == 1
    assert regions[0].brush.color().alpha() >= 200
    window.close()


def test_gui_event_lines_stay_visible_and_labels_sit_above_plot_box(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(
        np.zeros((2, 100)),
        srate=10,
        spacing=1,
        winlength=5,
        dispchans=2,
        events=[{"type": "stim", "latency": 10}, {"type": "resp", "latency": 30}],
    )
    window = EEGBrowserWindow(model)
    window.resize(960, 560)
    window.show()
    qapp.processEvents()

    view_rect = window.canvas.plot.getPlotItem().vb.sceneBoundingRect()
    assert [label.text() for label in window.canvas._event_label_items] == ["stim", "resp"]
    assert all(label.sceneBoundingRect().bottom() <= view_rect.top() for label in window.canvas._event_label_items)

    assert len(window.canvas._event_line_items) == 2
    assert all(line.zValue() == 34 for line in window.canvas._event_line_items)
    assert all(line.line().y1() == view_rect.top() for line in window.canvas._event_line_items)
    assert all(line.line().y2() == view_rect.bottom() for line in window.canvas._event_line_items)
    plot_boxes = [item for item in window.canvas._scene_items if item.__class__.__name__ == "QGraphicsRectItem"]
    assert len(plot_boxes) == 1
    assert plot_boxes[0].rect() == view_rect
    window.close()


def test_gui_displayed_channels_and_vertical_slider_update_visible_state(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((5, 100)), srate=10, spacing=1, winlength=2, dispchans=2)
    window = EEGBrowserWindow(model)

    assert window.channel_slider.isHidden() is False
    window.channel_slider.setValue(2)
    qapp.processEvents()

    assert model.state.channel_offset == 2

    window.set_displayed_channels(5)
    qapp.processEvents()

    assert model.state.dispchans == 5
    assert window.channel_slider.isVisible() is False
    window.close()


def test_gui_cancel_and_accept_buttons_set_close_state(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    close_model = build_eegplot_model(np.zeros((1, 20)), srate=10, spacing=1)
    close_window = EEGBrowserWindow(close_model)
    close_window.controls.cancel_button.click()
    qapp.processEvents()

    assert close_model.state.cancelled is True

    accept_model = build_eegplot_model(
        np.zeros((1, 20)), srate=10, spacing=1, command="TMPREJ = []", butlabel="Update Marks"
    )
    accept_window = EEGBrowserWindow(accept_model)
    assert accept_window.controls.accept_button.isHidden() is False
    assert accept_window.controls.accept_button.text() == "Update Marks"

    accept_window.controls.accept_button.click()
    qapp.processEvents()

    assert accept_model.state.accepted is True


def test_gui_keyboard_shortcuts_match_navigation_and_scale(qapp) -> None:
    qt_core = pytest.importorskip("PySide6.QtCore")
    qt_test = pytest.importorskip("PySide6.QtTest")
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((3, 100)), srate=10, winlength=2, dispchans=2, spacing=10)
    window = EEGBrowserWindow(model)
    window.show()
    window.activateWindow()
    qapp.processEvents()

    qt_test.QTest.keyClick(window, qt_core.Qt.Key.Key_Right)
    qt_test.QTest.keyClick(window, qt_core.Qt.Key.Key_Down)
    qt_test.QTest.keyClick(window, qt_core.Qt.Key.Key_Minus)
    qapp.processEvents()

    assert model.state.time == pytest.approx(2.0)
    assert model.state.channel_offset == 1
    assert model.state.spacing == pytest.approx(9.0)
    window.close()


def test_gui_continuous_trace_axis_uses_seconds(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.arange(10, dtype=float).reshape(1, 10), srate=10, winlength=0.5, spacing=1)
    window = EEGBrowserWindow(model)
    qapp.processEvents()

    curve = next(item for item in window.canvas._items if hasattr(item, "getData"))
    x_values, _y_values = curve.getData()

    np.testing.assert_allclose(x_values, [0.0, 0.1, 0.2, 0.3, 0.4])
    window.close()


def test_gui_canvas_mark_and_unmark_continuous_region(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((2, 20)), srate=10, spacing=1)
    window = EEGBrowserWindow(model)

    window.canvas.mark_samples(3, 8)
    assert len(model.state.winrej) == 1
    assert (model.state.winrej[0].start, model.state.winrej[0].end) == (3, 8)

    window.canvas.toggle_sample(5)
    assert model.state.winrej == []
    window.close()


def test_gui_canvas_channel_specific_toggle(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((3, 20)), srate=10, spacing=1, setelectrode=True)
    window = EEGBrowserWindow(model)

    window.canvas.mark_samples(3, 8, channel_index=1)
    assert model.state.winrej[0].channel_mask == (False, True, False)

    window.canvas.toggle_sample(5, channel_index=1)
    assert model.state.winrej == []
    window.close()


def test_gui_channel_specific_marks_draw_red_trace_overlay(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    data = np.vstack([np.arange(20, dtype=float), np.arange(20, dtype=float), np.arange(20, dtype=float)])
    model = build_eegplot_model(
        data,
        srate=10,
        spacing=10,
        winrej=[[3, 8, 0.7, 1.0, 0.9, 0, 1, 0]],
        show=False,
    )
    window = EEGBrowserWindow(model)

    curves = [item for item in window.canvas._items if hasattr(item, "getData")]

    assert len(curves) == 4
    x_values, _y_values = curves[-1].getData()
    np.testing.assert_allclose(x_values, np.arange(2, 8) / 10)
    window.close()


def test_gui_data2_overlay_draws_second_trace_per_channel(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    data = np.vstack([np.arange(20, dtype=float), np.arange(20, dtype=float)])
    model = build_eegplot_model(data, data2=data + 10.0, srate=10, spacing=10, dispchans=2, show=False)
    window = EEGBrowserWindow(model)

    curves = [item for item in window.canvas._items if hasattr(item, "getData")]

    assert len(curves) == 4
    np.testing.assert_allclose(curves[0].getData()[0], curves[1].getData()[0])
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


def test_gui_eegplot_children_option_links_opened_browser(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    child_model = build_eegplot_model(np.zeros((4, 100)), srate=10, winlength=2, dispchans=2, spacing=1)
    child = EEGBrowserWindow(child_model)

    parent = eegplot(np.zeros((4, 100)), srate=10, winlength=2, dispchans=2, spacing=1, children=child)

    assert child in parent._linked_windows
    assert parent in child._linked_windows
    parent.close()


def test_gui_noui_publication_view_hides_and_restores_browser_controls(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    model = build_eegplot_model(np.zeros((3, 100)), srate=10, winlength=2, dispchans=2, spacing=1, noui="on")
    window = EEGBrowserWindow(model)
    window.show()
    qapp.processEvents()

    assert window.menuBar().isHidden() is True
    assert window.channel_slider.isHidden() is True
    assert window.controls.cancel_button.isHidden() is True
    assert window.scale_indicator.isHidden() is True
    assert window.canvas.isVisible() is True

    window.set_publication_view(False)
    qapp.processEvents()

    assert window.menuBar().isHidden() is False
    assert window.channel_slider.isHidden() is False
    assert window.controls.cancel_button.isHidden() is False
    assert window.scale_indicator.isHidden() is False
    window.close()


def test_gui_spectral_mode_uses_frequency_axis_and_labels(qapp) -> None:
    from eegprep.functions.guifunc.eegbrowser import EEGBrowserWindow

    freqs = np.arange(1, 11, dtype=float)
    model = build_eegplot_model(
        np.arange(10, dtype=float).reshape(1, 10),
        freqs=freqs,
        freqlimits=[4, 6],
        spacing=1,
        winlength=2,
        show=False,
    )
    window = EEGBrowserWindow(model)

    curve = next(item for item in window.canvas._items if hasattr(item, "getData"))
    x_values, _y_values = curve.getData()

    np.testing.assert_allclose(x_values, [4.0, 5.0, 6.0])
    assert window.controls.time_label.text() == "Freq"
    assert window.controls.value_label.text() == "Power"
    window.close()


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
