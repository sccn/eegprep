from __future__ import annotations

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

    assert window.xgrid_action.text() == "X grid on"
    assert window.ygrid_action.text() == "Y grid off"
    assert window.scale_action.text() == "Hide scale"

    window.xgrid_action.trigger()
    window.ygrid_action.trigger()
    window.scale_action.trigger()

    assert window.xgrid_action.text() == "X grid off"
    assert window.ygrid_action.text() == "Y grid on"
    assert window.scale_action.text() == "Show scale"
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
