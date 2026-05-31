"""Qt/pyqtgraph rendering foundation for the EEGPrep scrolling EEG browser."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.sigprocfunc.eegplot import (
    BrowserModel,
    browser_window_duration,
    decimate_minmax,
    event_latency_to_sample,
    visible_sample_bounds,
)

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtCore = None
    QtGui = None
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
_SCALE_STEP = 0.1
_MIN_SPACING = 0.001
_BUTTON_STYLE = """
QWidget { background: #eaf2ff; color: #000000; }
QPushButton {
    background: #d7e8ff;
    border: 1px solid #7f8fa6;
    border-radius: 3px;
    color: #000000;
    min-width: 34px;
    padding: 2px 6px;
}
QPushButton:disabled { color: #777777; background: #d8d8d8; }
QPushButton#eegbrowser_event_types { min-width: 86px; }
QPushButton#eegbrowser_cancel { min-width: 56px; }
QDoubleSpinBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #7f8fa6;
    color: #000000;
    min-width: 74px;
}
QLabel#eegbrowser_status_value {
    background: transparent;
    color: #000000;
    min-width: 48px;
    padding: 1px 3px;
}
"""


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
        self._pre_normalize_spacing: float | None = None
        self._legend_window: Any = None
        self._message_window: Any = None
        self.setObjectName("eegbrowser")
        self.setWindowTitle(model.state.title)
        self.setStyleSheet(_BUTTON_STYLE)
        self.resize(960, 560)
        central = qt_widgets.QWidget()
        layout = qt_widgets.QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        plot_layout = qt_widgets.QHBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        self.channel_slider = qt_widgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.channel_slider.setObjectName("eegbrowser_channel_slider")
        self.channel_slider.setInvertedAppearance(True)
        self.channel_slider.valueChanged.connect(self._set_channel_offset)
        self.canvas = EEGBrowserCanvas(model)
        plot_layout.addWidget(self.channel_slider, 0)
        plot_layout.addWidget(self.canvas, 1)
        layout.addLayout(plot_layout, 1)
        self.controls = _BrowserControls(self)
        layout.addWidget(self.controls, 0)
        self.setCentralWidget(central)
        self._build_menus()
        self._build_shortcuts()
        self._sync_controls()

    def _build_menus(self) -> None:
        figure_menu = self.menuBar().addMenu("Figure")
        print_menu = figure_menu.addMenu("Print")
        self.print_portrait_action = print_menu.addAction("Portrait")
        self.print_portrait_action.triggered.connect(
            lambda: self._show_status_message("Print portrait is not available.")
        )
        self.print_landscape_action = print_menu.addAction("Landscape")
        self.print_landscape_action.triggered.connect(
            lambda: self._show_status_message("Print landscape is not available.")
        )
        self.print_action = print_menu.addAction("Print")
        self.print_action.triggered.connect(lambda: self._show_status_message("Use your desktop print command."))
        self.edit_figure_action = figure_menu.addAction("Edit figure")
        self.edit_figure_action.triggered.connect(self._edit_figure)
        self.accept_action = figure_menu.addAction("Accept and close")
        self.accept_action.triggered.connect(self.accept_and_close)
        self.cancel_action = figure_menu.addAction("Cancel and close")
        self.cancel_action.triggered.connect(self.cancel_and_close)

        display_menu = self.menuBar().addMenu("Display")
        mark_menu = display_menu.addMenu("Data select/mark")
        self.marks_action = mark_menu.addAction("")
        self.marks_action.triggered.connect(self._toggle_marks)
        self.mark_color_action = mark_menu.addAction("Choose color")
        self.mark_color_action.triggered.connect(self._choose_mark_color)
        self.event_duration_action = display_menu.addAction("Plot event duration")
        self.event_duration_action.setEnabled(False)
        grid_menu = display_menu.addMenu("Grid")
        self.xgrid_action = grid_menu.addAction("")
        self.xgrid_action.triggered.connect(self._toggle_xgrid)
        self.ygrid_action = grid_menu.addAction("")
        self.ygrid_action.triggered.connect(self._toggle_ygrid)
        grid_style_menu = grid_menu.addMenu("Grid Style")
        for label in ("- -", "_ .", ". .", "__"):
            action = grid_style_menu.addAction(label)
            action.setEnabled(False)
        self.submean_action = display_menu.addAction("")
        self.submean_action.triggered.connect(self._toggle_submean)
        self.scale_action = display_menu.addAction("")
        self.scale_action.triggered.connect(self._toggle_scale)
        self.title_action = display_menu.addAction("Title")
        self.title_action.triggered.connect(self._prompt_title)
        self.stack_action = display_menu.addAction("")
        self.stack_action.triggered.connect(self.toggle_stack)
        self.norm_action = display_menu.addAction("")
        self.norm_action.triggered.connect(self.toggle_normalize)

        settings_menu = self.menuBar().addMenu("Settings")
        self.time_range_action = settings_menu.addAction("Time range to display")
        self.time_range_action.triggered.connect(self._prompt_time_range)
        self.dispchans_action = settings_menu.addAction("Number of channels to display")
        self.dispchans_action.triggered.connect(self._prompt_displayed_channels)
        channel_label_menu = settings_menu.addMenu("Channel labels")
        self.show_numbers_action = channel_label_menu.addAction("Show number")
        self.show_numbers_action.setCheckable(True)
        self.show_numbers_action.triggered.connect(lambda: self.set_channel_label_mode("numbers"))
        self.show_labels_action = channel_label_menu.addAction("Show label")
        self.show_labels_action.setCheckable(True)
        self.show_labels_action.triggered.connect(lambda: self.set_channel_label_mode("labels"))
        self.hide_labels_action = channel_label_menu.addAction("Hide labels")
        self.hide_labels_action.setCheckable(True)
        self.hide_labels_action.triggered.connect(lambda: self.set_channel_label_mode("none"))
        self.load_locs_action = channel_label_menu.addAction("Load .loc(s) file")
        self.load_locs_action.setEnabled(False)
        zoom_menu = settings_menu.addMenu("Zoom off/on")
        self.zoom_on_action = zoom_menu.addAction("Zoom on")
        self.zoom_on_action.triggered.connect(lambda: self.set_zoom_enabled(True))
        self.zoom_off_action = zoom_menu.addAction("Zoom off")
        self.zoom_off_action.triggered.connect(lambda: self.set_zoom_enabled(False))
        events_menu = settings_menu.addMenu("Events")
        self.events_on_action = events_menu.addAction("Events on")
        self.events_on_action.triggered.connect(lambda: self.set_events_visible(True))
        self.events_off_action = events_menu.addAction("Events off")
        self.events_off_action.triggered.connect(lambda: self.set_events_visible(False))
        self.event_string_length_action = events_menu.addAction("Events' string length")
        self.event_string_length_action.setEnabled(False)
        self.event_legend_action = events_menu.addAction("Events' legend")
        self.event_legend_action.triggered.connect(self.show_event_legend)

        help_menu = self.menuBar().addMenu("Help")
        self.help_action = help_menu.addAction("eegplot help")
        self.help_action.triggered.connect(lambda: self._show_status_message("EEGPrep eegplot browser"))
        self._refresh_menu_labels()

    def _build_shortcuts(self) -> None:
        shortcuts = {
            QtCore.Qt.Key.Key_Left: lambda: self.scroll_time(-1.0),
            QtCore.Qt.Key.Key_Right: lambda: self.scroll_time(1.0),
            QtCore.Qt.Key.Key_PageUp: lambda: self.scroll_time(-5.0),
            QtCore.Qt.Key.Key_PageDown: lambda: self.scroll_time(5.0),
            QtCore.Qt.Key.Key_Home: self.scroll_to_start,
            QtCore.Qt.Key.Key_End: self.scroll_to_end,
            QtCore.Qt.Key.Key_Up: lambda: self.scroll_channels(-1),
            QtCore.Qt.Key.Key_Down: lambda: self.scroll_channels(1),
            QtCore.Qt.Key.Key_Plus: lambda: self.adjust_spacing(1),
            QtCore.Qt.Key.Key_Equal: lambda: self.adjust_spacing(1),
            QtCore.Qt.Key.Key_Minus: lambda: self.adjust_spacing(-1),
        }
        for key, callback in shortcuts.items():
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)

    def _refresh_menu_labels(self) -> None:
        self.accept_action.setEnabled(self.model.state.accept_label is not None)
        self.marks_action.setText("Hide marks" if self.model.state.show_marks else "Show marks")
        self.xgrid_action.setText("X grid off" if self.model.state.xgrid else "X grid on")
        self.ygrid_action.setText("Y grid off" if self.model.state.ygrid else "Y grid on")
        self.submean_action.setText("Remove DC offset off" if self.model.state.submean else "Remove DC offset on")
        self.scale_action.setText("Hide scale" if self.model.state.scale else "Show scale")
        self.stack_action.setText("Spread channels" if self.model.state.stacked else "Stack channels")
        self.norm_action.setText("Denormalize channels" if self.model.state.normalized else "Normalize channels")
        has_events = bool(self.model.state.events)
        self.events_on_action.setEnabled(has_events)
        self.events_off_action.setEnabled(has_events)
        self.event_legend_action.setEnabled(has_events)
        self.show_numbers_action.setChecked(self.model.state.channel_label_mode == "numbers")
        self.show_labels_action.setChecked(self.model.state.channel_label_mode == "labels")
        self.hide_labels_action.setChecked(self.model.state.channel_label_mode == "none")

    def _toggle_xgrid(self) -> None:
        self.model.state.xgrid = not self.model.state.xgrid
        self._redraw()

    def _toggle_ygrid(self) -> None:
        self.model.state.ygrid = not self.model.state.ygrid
        self._redraw()

    def _toggle_scale(self) -> None:
        self.model.state.scale = not self.model.state.scale
        self._redraw()

    def _toggle_marks(self) -> None:
        self.model.state.show_marks = not self.model.state.show_marks
        self._redraw()

    def _toggle_submean(self) -> None:
        self.model.state.submean = not self.model.state.submean
        self._redraw()

    def scroll_time(self, fraction: float) -> None:
        self.model.state.time += self.model.state.winlength * fraction
        self._redraw()

    def scroll_to_start(self) -> None:
        self.model.state.time = 0.0
        self._redraw()

    def scroll_to_end(self) -> None:
        self.model.state.time = browser_window_duration(self.model.data, self.model.state)
        self._redraw()

    def scroll_channels(self, amount: int) -> None:
        self.model.state.channel_offset += amount
        self._redraw()

    def set_position(self, value: float) -> None:
        self.model.state.time = float(value) - (1.0 if self.model.data.epoched else 0.0)
        self._redraw()

    def set_spacing(self, value: float) -> None:
        self.model.state.spacing = max(float(value), _MIN_SPACING)
        self._redraw()

    def adjust_spacing(self, direction: int) -> None:
        if direction > 0:
            self.model.state.spacing += self.model.state.spacing * _SCALE_STEP
        else:
            self.model.state.spacing = max(
                _MIN_SPACING,
                self.model.state.spacing - self.model.state.spacing * _SCALE_STEP,
            )
        self._redraw()

    def set_displayed_channels(self, value: int) -> None:
        self.model.state.dispchans = int(value)
        self._redraw()

    def set_channel_label_mode(self, mode: str) -> None:
        if mode not in {"labels", "numbers", "none"}:
            raise ValueError(f"unknown channel label mode: {mode}")
        self.model.state.channel_label_mode = mode
        self._redraw()

    def set_events_visible(self, visible: bool) -> None:
        self.model.state.show_events = bool(visible)
        self._redraw()

    def set_zoom_enabled(self, enabled: bool) -> None:
        self.model.state.zoom_enabled = bool(enabled)
        self.canvas.plot.setMouseEnabled(x=enabled, y=enabled)

    def toggle_stack(self) -> None:
        self.model.state.stacked = not self.model.state.stacked
        self._redraw()

    def toggle_normalize(self) -> None:
        state = self.model.state
        if state.normalized:
            state.normalized = False
            if self._pre_normalize_spacing is not None:
                state.spacing = self._pre_normalize_spacing
            self._pre_normalize_spacing = None
        else:
            self._pre_normalize_spacing = state.spacing
            state.normalized = True
            state.spacing = 5.0
        self._redraw()

    def show_event_legend(self) -> None:
        qt_widgets, _pg = _require_gui()
        if not self.model.state.events:
            return
        dialog = qt_widgets.QDialog(self)
        dialog.setObjectName("eegbrowser_event_legend")
        dialog.setWindowTitle("Event types")
        layout = qt_widgets.QVBoxLayout(dialog)
        seen: dict[str, tuple[int, int, int]] = {}
        for event in self.model.state.events:
            seen.setdefault(event.type, _EVENT_COLORS[event.color_index % len(_EVENT_COLORS)])
        for label, color in seen.items():
            item = qt_widgets.QLabel(label)
            item.setStyleSheet(f"color: rgb({color[0]}, {color[1]}, {color[2]});")
            layout.addWidget(item)
        dialog.show()
        self._legend_window = dialog

    def accept_and_close(self) -> None:
        self.model.state.accepted = True
        self.close()

    def cancel_and_close(self) -> None:
        self.model.state.cancelled = True
        self.close()

    def _set_channel_offset(self, value: int) -> None:
        self.model.state.channel_offset = int(value)
        self._redraw()

    def _redraw(self) -> None:
        self.model.state.clamp_to_data(self.model.data)
        self.canvas.redraw()
        self._sync_controls()
        self._refresh_menu_labels()

    def _sync_controls(self) -> None:
        self.model.state.clamp_to_data(self.model.data)
        maximum_offset = max(0, self.model.data.n_channels - self.model.state.dispchans)
        self.channel_slider.blockSignals(True)
        self.channel_slider.setRange(0, maximum_offset)
        self.channel_slider.setValue(self.model.state.channel_offset)
        self.channel_slider.setVisible(maximum_offset > 0)
        self.channel_slider.blockSignals(False)
        self.controls.sync_from_state()

    def _prompt_title(self) -> None:
        qt_widgets, _pg = _require_gui()
        title, ok = qt_widgets.QInputDialog.getText(self, "Title", "Plot title", text=self.model.state.plottitle)
        if ok:
            self.model.state.plottitle = str(title)
            self._redraw()

    def _prompt_time_range(self) -> None:
        qt_widgets, _pg = _require_gui()
        value, ok = qt_widgets.QInputDialog.getDouble(
            self,
            "Time range to display",
            "Seconds" if not self.model.data.epoched else "Epochs",
            self.model.state.winlength,
            0.001,
            1_000_000.0,
            3,
        )
        if ok:
            self.model.state.winlength = float(value)
            self._redraw()

    def _prompt_displayed_channels(self) -> None:
        qt_widgets, _pg = _require_gui()
        value, ok = qt_widgets.QInputDialog.getInt(
            self,
            "Number of channels to display",
            "Channels",
            self.model.state.dispchans,
            1,
            self.model.data.n_channels,
        )
        if ok:
            self.set_displayed_channels(value)

    def _choose_mark_color(self) -> None:
        if QtGui is None or QtWidgets is None:
            return
        current = tuple(int(np.clip(component, 0, 1) * 255) for component in self.model.state.mark_color)
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*current), self, "Choose marking color")
        if color.isValid():
            self.model.state.mark_color = (color.redF(), color.greenF(), color.blueF())

    def _edit_figure(self) -> None:
        self.controls.hide()
        self.menuBar().hide()

    def _show_status_message(self, text: str) -> None:
        qt_widgets, _pg = _require_gui()
        if self._message_window is not None:
            self._message_window.close()
        message = qt_widgets.QMessageBox(self)
        message.setObjectName("eegbrowser_message")
        message.setWindowTitle("EEGPrep")
        message.setText(text)
        message.show()
        self._message_window = message


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
        self.plot.setMouseEnabled(x=model.state.zoom_enabled, y=model.state.zoom_enabled)
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
        if self.model.state.channel_label_mode != "none":
            for screen_index, channel_index in enumerate(self._visible_channels()):
                if self.model.state.channel_label_mode == "numbers":
                    label = str(channel_index + 1)
                else:
                    label = self.model.data.channel_labels[channel_index]
                labels.append((float(screen_index), label))
        self.plot.getAxis("left").setTicks([labels])
        axis_label, axis_units = self._bottom_axis_label()
        self.plot.getAxis("bottom").setLabel(axis_label, units=axis_units)
        self.plot.getAxis("bottom").setTicks([self._x_ticks(start, stop)])
        self.plot.setTitle(self.model.state.plottitle)

    def _draw_winrej(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        if not self.model.state.show_marks:
            return
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
            label.setPos(x_value, float(self.model.state.dispchans) - 0.6)
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
            if state.normalized and y_values.size:
                y_values = y_values / _channel_std(data.flat_data[channel_index])
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
        if self.model.state.normalized and y_values.size:
            y_values = y_values / _channel_std(self.model.data.flat_data[channel_index])
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
        baseline = (float(self.model.state.dispchans) - 1.0) / 2.0 if self.model.state.stacked else float(screen_index)
        return baseline + np.asarray(values, dtype=float) / spacing * 0.42

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
    def __init__(self, browser: EEGBrowserWindow):
        super().__init__()
        qt_widgets, _pg = _require_gui()
        self.browser = browser
        self.model = browser.model
        self.setStyleSheet(_BUTTON_STYLE)
        layout = qt_widgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.cancel_button = qt_widgets.QPushButton("Cancel" if self.model.state.accept_label is not None else "Close")
        self.cancel_button.setObjectName("eegbrowser_cancel")
        self.cancel_button.clicked.connect(browser.cancel_and_close)
        layout.addWidget(self.cancel_button)
        self.event_button = qt_widgets.QPushButton("Event types")
        self.event_button.setObjectName("eegbrowser_event_types")
        self.event_button.setMinimumWidth(90)
        self.event_button.clicked.connect(browser.show_event_legend)
        layout.addWidget(self.event_button)
        self.back_page_button = self._button("<<", lambda: browser.scroll_time(-5.0))
        self.back_step_button = self._button("<", lambda: browser.scroll_time(-1.0))
        self.forward_step_button = self._button(">", lambda: browser.scroll_time(1.0))
        self.forward_page_button = self._button(">>", lambda: browser.scroll_time(5.0))
        for button in (
            self.back_page_button,
            self.back_step_button,
            self.forward_step_button,
            self.forward_page_button,
        ):
            layout.addWidget(button)
        layout.addSpacing(12)

        self.channel_label = self._status_label("Chan.")
        self.channel_value = self._status_value("")
        self.time_label = self._status_label("Time")
        self.time_value = self._status_value("0.00")
        self.value_label = self._status_label("Value")
        self.value_value = self._status_value("0.00")
        for widget in (
            self.channel_label,
            self.channel_value,
            self.time_label,
            self.time_value,
            self.value_label,
            self.value_value,
        ):
            layout.addWidget(widget)

        self.position = qt_widgets.QDoubleSpinBox()
        self.position.setObjectName("eegbrowser_position")
        self.position.setDecimals(3)
        self.position.setSingleStep(0.1)
        self.position.valueChanged.connect(browser.set_position)
        layout.addWidget(self.position)

        self.spacing = qt_widgets.QDoubleSpinBox()
        self.spacing.setObjectName("eegbrowser_spacing")
        self.spacing.setDecimals(3)
        self.spacing.setRange(_MIN_SPACING, 1_000_000.0)
        self.spacing.valueChanged.connect(browser.set_spacing)
        layout.addWidget(self.spacing)
        self.plus_button = self._button("+", lambda: browser.adjust_spacing(1))
        self.minus_button = self._button("-", lambda: browser.adjust_spacing(-1))
        layout.addWidget(self.plus_button)
        layout.addWidget(self.minus_button)
        self.norm_button = self._button("Norm", browser.toggle_normalize)
        self.stack_button = self._button("Stack", browser.toggle_stack)
        layout.addWidget(self.norm_button)
        layout.addWidget(self.stack_button)
        layout.addStretch(1)
        self.accept_button = qt_widgets.QPushButton(self.model.state.accept_label or "REJECT")
        self.accept_button.setObjectName("eegbrowser_accept")
        self.accept_button.clicked.connect(browser.accept_and_close)
        layout.addWidget(self.accept_button)

    def sync_from_state(self) -> None:
        state = self.model.state
        max_time = browser_window_duration(self.model.data, state) - state.winlength
        state_at_start = state.time <= 0
        state_at_end = state.time >= max(0.0, max_time)
        self.back_page_button.setEnabled(not state_at_start)
        self.back_step_button.setEnabled(not state_at_start)
        self.forward_step_button.setEnabled(not state_at_end)
        self.forward_page_button.setEnabled(not state_at_end)
        self.position.blockSignals(True)
        self.position.setRange(
            1.0 if self.model.data.epoched else 0.0, max(0.0, browser_window_duration(self.model.data, state))
        )
        self.position.setValue(state.time + (1.0 if self.model.data.epoched else 0.0))
        self.position.blockSignals(False)
        self.spacing.blockSignals(True)
        self.spacing.setValue(state.spacing)
        self.spacing.blockSignals(False)
        self.event_button.setVisible(bool(state.events))
        self.accept_button.setVisible(state.accept_label is not None)
        if state.accept_label is not None:
            self.accept_button.setText(state.accept_label)
        self.cancel_button.setText("Cancel" if state.accept_label is not None else "Close")
        self.stack_button.setText("Spread" if state.stacked else "Stack")
        self.norm_button.setText("Denorm" if state.normalized else "Norm")
        self.time_label.setText("Freq" if self.model.data.mode == "spectral" else "Time")
        self.value_label.setText("Power" if self.model.data.mode == "spectral" else "Value")

    def _button(self, label: str, callback: Any) -> Any:
        qt_widgets, _pg = _require_gui()
        button = qt_widgets.QPushButton(label)
        button.clicked.connect(callback)
        return button

    def _status_label(self, text: str) -> Any:
        qt_widgets, _pg = _require_gui()
        label = qt_widgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return label

    def _status_value(self, text: str) -> Any:
        label = self._status_label(text)
        label.setObjectName("eegbrowser_status_value")
        return label


def _trace_color(colors: tuple[Any, ...], index: int) -> Any:
    color = colors[index % len(colors)]
    if isinstance(color, str):
        return color
    values = tuple(color)
    if values and max(values) <= 1:
        return tuple(int(float(value) * 255) for value in values[:3])
    return tuple(int(float(value)) for value in values[:3])


def _channel_std(values: np.ndarray) -> float:
    sample = np.asarray(values[: min(1000, values.size)], dtype=float)
    std = float(np.nanstd(sample))
    if not np.isfinite(std) or std <= 0:
        return 1.0
    return std


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
    if QtWidgets is None or QtCore is None or QtGui is None or pg is None:
        raise RuntimeError(
            "PySide6 and pyqtgraph are required for EEGPrep eegplot. Install them with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtWidgets, pg


__all__ = ["EEGBrowserCanvas", "EEGBrowserWindow", "open_eegbrowser"]
