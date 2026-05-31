"""Qt/pyqtgraph rendering foundation for the EEGPrep scrolling EEG browser."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.sigprocfunc.eegplot import (
    BrowserModel,
    decimate_minmax,
    event_latency_to_sample,
    visible_sample_bounds,
)

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtCore, QtWidgets
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtCore = None
    QtWidgets = None
    pg = None

_QMainWindow: Any = QtWidgets.QMainWindow if QtWidgets is not None else object
_QWidget: Any = QtWidgets.QWidget if QtWidgets is not None else object


_EVENT_COLORS = (
    (220, 40, 40),
    (0, 150, 60),
    (190, 0, 190),
    (0, 160, 170),
    (40, 40, 40),
    (40, 80, 210),
)


def open_eegbrowser(model: BrowserModel) -> Any:
    """Create, show, and return an EEG browser window."""
    qt_widgets, _pg = _require_gui()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    window = EEGBrowserWindow(model)
    window.show()
    app.processEvents()
    return window


class EEGBrowserWindow(_QMainWindow):
    """Basic EEGLAB-like scrolling browser window."""

    def __init__(self, model: BrowserModel):
        super().__init__()
        qt_widgets, _pg = _require_gui()
        self.model = model
        self.setObjectName("eegbrowser")
        self.setWindowTitle(model.state.title)
        self.resize(960, 560)
        central = qt_widgets.QWidget()
        layout = qt_widgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.canvas = EEGBrowserCanvas(model)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(_BrowserControls(model, self.canvas), 0)
        self.setCentralWidget(central)
        self._build_menus()

    def _build_menus(self) -> None:
        menu = self.menuBar().addMenu("Display")
        self.xgrid_action = menu.addAction("")
        self.xgrid_action.triggered.connect(self._toggle_xgrid)
        self.ygrid_action = menu.addAction("")
        self.ygrid_action.triggered.connect(self._toggle_ygrid)
        self.scale_action = menu.addAction("")
        self.scale_action.triggered.connect(self._toggle_scale)
        self._refresh_display_menu_labels()

    def _refresh_display_menu_labels(self) -> None:
        self.xgrid_action.setText("X grid off" if self.model.state.xgrid else "X grid on")
        self.ygrid_action.setText("Y grid off" if self.model.state.ygrid else "Y grid on")
        self.scale_action.setText("Hide scale" if self.model.state.scale else "Show scale")

    def _toggle_xgrid(self) -> None:
        self.model.state.xgrid = not self.model.state.xgrid
        self._refresh_display_menu_labels()
        self.canvas.redraw()

    def _toggle_ygrid(self) -> None:
        self.model.state.ygrid = not self.model.state.ygrid
        self._refresh_display_menu_labels()
        self.canvas.redraw()

    def _toggle_scale(self) -> None:
        self.model.state.scale = not self.model.state.scale
        self._refresh_display_menu_labels()
        self.canvas.redraw()


class EEGBrowserCanvas(_QWidget):
    """pyqtgraph canvas for traces, events, and marked windows."""

    def __init__(self, model: BrowserModel):
        super().__init__()
        qt_widgets, plot_graphics = _require_gui()
        self.model = model
        self.setObjectName("eegbrowser_canvas")
        self.setMinimumSize(680, 380)
        self._items: list[Any] = []
        layout = qt_widgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = plot_graphics.PlotWidget(background="w")
        self.plot.setObjectName("eegbrowser_plot")
        self.plot.showGrid(x=model.state.xgrid, y=model.state.ygrid, alpha=0.25)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.getPlotItem().setClipToView(True)
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        layout.addWidget(self.plot)
        self.redraw()

    def redraw(self) -> None:
        """Redraw all browser layers for the current state."""
        self.model.state.clamp_to_data(self.model.data)
        self.plot.clear()
        self._items.clear()
        self.plot.showGrid(x=self.model.state.xgrid, y=self.model.state.ygrid, alpha=0.25)
        self._configure_axes()
        self._draw_winrej()
        self._draw_events()
        self._draw_traces()
        self._draw_scale()

    def _configure_axes(self) -> None:
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        self.plot.setXRange(self._sample_edge_to_x_value(start), self._sample_edge_to_x_value(stop), padding=0)
        self.plot.setYRange(-0.5, float(self.model.state.dispchans) - 0.5, padding=0.04)
        labels = []
        for screen_index, channel_index in enumerate(self._visible_channels()):
            labels.append((float(screen_index), self.model.data.channel_labels[channel_index]))
        self.plot.getAxis("left").setTicks([labels])
        axis_label, axis_units = self._bottom_axis_label()
        self.plot.getAxis("bottom").setLabel(axis_label, units=axis_units)
        self.plot.getAxis("bottom").setTicks([self._x_ticks(start, stop)])
        if self.model.state.plottitle:
            self.plot.setTitle(self.model.state.plottitle)

    def _draw_winrej(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        for region in self.model.state.winrej:
            region_start = _winrej_frame_to_index(region.start)
            region_stop = _winrej_frame_to_index(region.end) + 1
            if region_stop < start or region_start > stop:
                continue
            color = tuple(int(np.clip(component, 0, 1) * 255) for component in region.color)
            item = plot_graphics.LinearRegionItem(
                values=(
                    self._sample_edge_to_x_value(max(region_start, start)),
                    self._sample_edge_to_x_value(min(region_stop, stop)),
                ),
                orientation=plot_graphics.LinearRegionItem.Vertical,
                brush=(*color, 70),
                movable=False,
            )
            item.lines[0].setPen(plot_graphics.mkPen((*color, 120), width=1))
            item.lines[1].setPen(plot_graphics.mkPen((*color, 120), width=1))
            self.plot.addItem(item)
            self._items.append(item)

    def _draw_events(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        if not self.model.state.show_events:
            return
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        for event in self.model.state.events:
            sample = event_latency_to_sample(event.latency, self.model.data)
            if sample < start or sample > stop:
                continue
            color = _EVENT_COLORS[event.color_index % len(_EVENT_COLORS)]
            x_value = self._sample_to_x_value(sample)
            line = plot_graphics.InfiniteLine(
                pos=x_value,
                angle=90,
                pen=plot_graphics.mkPen(color, width=2 if event.type.lower() == "boundary" else 1),
                movable=False,
            )
            self.plot.addItem(line)
            label = plot_graphics.TextItem(event.type[:20], color=color, anchor=(0, 1), angle=90)
            label.setPos(x_value, float(self.model.state.dispchans) - 0.2)
            self.plot.addItem(label)
            self._items.extend([line, label])

    def _draw_traces(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        data = self.model.data
        state = self.model.state
        start, stop = visible_sample_bounds(data, state)
        x_values = self._sample_range_to_x_values(start, stop)
        for screen_index, channel_index in enumerate(self._visible_channels()):
            y_values = np.asarray(data.flat_data[channel_index, start:stop], dtype=float)
            if state.submean and y_values.size:
                y_values = y_values - np.nanmean(y_values)
            x_dec, y_dec = decimate_minmax(x_values, y_values, max(1, self.width()))
            shifted = self._trace_to_axis_y(y_dec, screen_index)
            curve = self.plot.plot(
                x_dec,
                shifted,
                pen=plot_graphics.mkPen(_trace_color(state.colors, screen_index), width=1),
                skipFiniteCheck=True,
            )
            curve.setClipToView(True)
            self._items.append(curve)
            if data.flat_data2 is not None:
                self._draw_overlay_trace(channel_index, screen_index, start, stop)

    def _draw_overlay_trace(self, channel_index: int, screen_index: int, start: int, stop: int) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        overlay = self.model.data.flat_data2
        if overlay is None:
            return
        x_values = self._sample_range_to_x_values(start, stop)
        y_values = np.asarray(overlay[channel_index, start:stop], dtype=float)
        if self.model.state.submean and y_values.size:
            y_values = y_values - np.nanmean(y_values)
        x_dec, y_dec = decimate_minmax(x_values, y_values, max(1, self.width()))
        curve = self.plot.plot(
            x_dec,
            self._trace_to_axis_y(y_dec, screen_index),
            pen=plot_graphics.mkPen((210, 35, 35), width=1),
            skipFiniteCheck=True,
        )
        curve.setClipToView(True)
        self._items.append(curve)

    def _draw_scale(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        if not self.model.state.scale:
            return
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        x_start = self._sample_edge_to_x_value(start)
        x_stop = self._sample_edge_to_x_value(stop)
        x_span = max(abs(x_stop - x_start), np.finfo(float).eps)
        x = x_stop - x_span / 20.0
        y0 = -0.25
        y1 = y0 + 0.42
        pen = plot_graphics.mkPen((0, 0, 0), width=1)
        for xs, ys in (
            ([x, x], [y0, y1]),
            ([x - 0.01 * x_span, x + 0.01 * x_span], [y0, y0]),
            ([x - 0.01 * x_span, x + 0.01 * x_span], [y1, y1]),
        ):
            self._items.append(self.plot.plot(xs, ys, pen=pen))
        label = plot_graphics.TextItem(f"{self.model.state.spacing:g}", color=(0, 0, 0), anchor=(0, 0.5))
        label.setPos(x, (y0 + y1) / 2.0)
        self.plot.addItem(label)
        self._items.append(label)

    def _trace_to_axis_y(self, values: np.ndarray, screen_index: int) -> np.ndarray:
        spacing = max(float(self.model.state.spacing), np.finfo(float).eps)
        return float(screen_index) + np.asarray(values, dtype=float) / spacing * 0.42

    def _visible_channels(self) -> range:
        state = self.model.state
        return range(state.channel_offset, state.channel_offset + state.dispchans)

    def _bottom_axis_label(self) -> tuple[str, str]:
        if self.model.data.mode == "spectral":
            return "Frequency", "Hz"
        if self.model.data.epoched:
            return "Epoch", ""
        return "Time", "s"

    def _sample_range_to_x_values(self, start: int, stop: int) -> np.ndarray:
        if self.model.data.x_values is not None:
            return self.model.data.x_values[start:stop]
        if self.model.data.epoched:
            return np.arange(start, stop, dtype=float)
        return np.arange(start, stop, dtype=float) / float(self.model.state.srate)

    def _sample_to_x_value(self, sample: int) -> float:
        if self.model.data.x_values is not None:
            index = max(0, min(int(sample), self.model.data.total_samples - 1))
            return float(self.model.data.x_values[index])
        if self.model.data.epoched:
            return float(sample)
        return float(sample) / float(self.model.state.srate)

    def _sample_edge_to_x_value(self, sample: int) -> float:
        if self.model.data.x_values is None:
            return self._sample_to_x_value(sample)
        index = int(sample)
        values = self.model.data.x_values
        if index <= 0:
            return float(values[0])
        if index >= values.size:
            return float(values[-1])
        return float(values[index])

    def _x_ticks(self, start: int, stop: int) -> list[tuple[float, str]]:
        if self.model.data.epoched:
            first_epoch = int(start // max(1, self.model.data.pnts)) + 1
            last_epoch = int(np.ceil(stop / max(1, self.model.data.pnts)))
            return [
                (float((epoch - 1) * self.model.data.pnts), str(epoch))
                for epoch in range(first_epoch, last_epoch + 1)
                if start <= (epoch - 1) * self.model.data.pnts <= stop
            ]
        start_x = self._sample_edge_to_x_value(start)
        stop_x = self._sample_edge_to_x_value(stop)
        duration = max(1.0, abs(stop_x - start_x))
        step = _nice_time_step(duration)
        first = np.ceil(start_x / step) * step
        ticks = []
        current = first
        while current <= stop_x:
            ticks.append((float(current), f"{current:g}"))
            current += step
        return ticks


class _BrowserControls(_QWidget):
    def __init__(self, model: BrowserModel, canvas: EEGBrowserCanvas):
        super().__init__()
        qt_widgets, _pg = _require_gui()
        self.model = model
        self.canvas = canvas
        layout = qt_widgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for label, fraction in (("<<", -0.9), ("<", -0.2), (">", 0.2), (">>", 0.9)):
            button = qt_widgets.QPushButton(label)
            button.clicked.connect(lambda _checked=False, step=fraction: self._scroll_time(step))
            layout.addWidget(button)
        layout.addSpacing(12)
        layout.addWidget(qt_widgets.QLabel("Position"))
        self.position = qt_widgets.QDoubleSpinBox()
        self.position.setDecimals(3)
        self.position.setSingleStep(0.1)
        self.position.setValue(model.state.time + (1.0 if model.data.epoched else 0.0))
        self.position.valueChanged.connect(self._set_position)
        layout.addWidget(self.position)
        layout.addWidget(qt_widgets.QLabel("Spacing"))
        self.spacing = qt_widgets.QDoubleSpinBox()
        self.spacing.setDecimals(3)
        self.spacing.setRange(0.001, 1_000_000.0)
        self.spacing.setValue(model.state.spacing)
        self.spacing.valueChanged.connect(self._set_spacing)
        layout.addWidget(self.spacing)
        layout.addStretch(1)
        close = qt_widgets.QPushButton("Close")
        close.clicked.connect(self._close_window)
        layout.addWidget(close)

    def _scroll_time(self, fraction: float) -> None:
        self.model.state.time += self.model.state.winlength * fraction
        self.model.state.clamp_to_data(self.model.data)
        self.position.blockSignals(True)
        self.position.setValue(self.model.state.time + (1.0 if self.model.data.epoched else 0.0))
        self.position.blockSignals(False)
        self.canvas.redraw()

    def _set_position(self, value: float) -> None:
        self.model.state.time = float(value) - (1.0 if self.model.data.epoched else 0.0)
        self.model.state.clamp_to_data(self.model.data)
        self.canvas.redraw()

    def _set_spacing(self, value: float) -> None:
        self.model.state.spacing = float(value)
        self.canvas.redraw()

    def _close_window(self) -> None:
        window = self.window()
        if window is not None:
            window.close()


def _trace_color(colors: tuple[Any, ...], index: int) -> Any:
    color = colors[index % len(colors)]
    if isinstance(color, str):
        return color
    values = tuple(color)
    if values and max(values) <= 1:
        return tuple(int(float(value) * 255) for value in values[:3])
    return tuple(int(float(value)) for value in values[:3])


def _winrej_frame_to_index(value: float) -> int:
    frame = int(round(float(value)))
    if frame <= 0:
        return 0
    return frame - 1


def _nice_time_step(duration: float) -> float:
    target = max(duration / 6.0, 1e-6)
    magnitude = 10 ** np.floor(np.log10(target))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        step = float(multiplier * magnitude)
        if step >= target:
            return step
    return float(10.0 * magnitude)


def _require_gui() -> tuple[Any, Any]:
    if QtWidgets is None or QtCore is None or pg is None:
        raise RuntimeError(
            "PySide6 and pyqtgraph are required for EEGPrep eegplot. Install them with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtWidgets, pg


__all__ = ["EEGBrowserCanvas", "EEGBrowserWindow", "open_eegbrowser"]
