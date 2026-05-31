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
