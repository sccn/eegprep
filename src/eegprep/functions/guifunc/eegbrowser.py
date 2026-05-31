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
    (255, 0, 0),
    (0, 204, 0),
    (255, 0, 255),
    (0, 255, 255),
    (0, 0, 0),
    (0, 0, 255),
    (0, 204, 0),
)
_SCALE_STEP = 0.1
_MIN_SPACING = 0.001
_TRACE_SCALE = 0.72
_AXES_POSITION = (0.0964286, 0.15, 0.842, 0.75)
_CONTROL_POSITIONS = {
    "back_page": (0.2464, 0.0254, 0.0385, 0.0339),
    "back_step": (0.2924, 0.0254, 0.0288, 0.0339),
    "position": (0.3287, 0.0203, 0.0561, 0.0390),
    "forward_step": (0.3924, 0.0254, 0.0299, 0.0339),
    "forward_page": (0.4297, 0.0254, 0.0385, 0.0339),
    "spacing": (0.6744, 0.0236, 0.0582, 0.0390),
    "channel_value": (0.4762, 0.0100, 0.0582, 0.0390),
    "time_value": (0.5256, 0.0100, 0.0707, 0.0390),
    "data_value": (0.6006, 0.0100, 0.0582, 0.0390),
    "channel_label": (0.4762, 0.0500, 0.0582, 0.0390),
    "time_label": (0.5256, 0.0500, 0.0707, 0.0390),
    "data_label": (0.6006, 0.0500, 0.0582, 0.0390),
    "plus": (0.7437, 0.0458, 0.0275, 0.0270),
    "minus": (0.7437, 0.0134, 0.0275, 0.0270),
    "accept": (0.8000, 0.0200, 0.1400, 0.0500),
    "cancel": (0.0500, 0.0200, 0.0700, 0.0500),
    "events": (0.1400, 0.0200, 0.0900, 0.0500),
    "channel_slider": (0.0300, 0.1500, 0.0150, 0.8000),
    "norm": (0.9380, 0.8700, 0.0600, 0.0480),
    "stack": (0.9380, 0.9300, 0.0600, 0.0480),
}
_BUTTON_STYLE = """
QWidget { background: #eaf2ff; color: #000000; }
QPushButton {
    background: #d7e8ff;
    border: 1px solid #7f8fa6;
    border-radius: 3px;
    color: #000000;
    min-width: 0px;
    padding: 1px 2px;
}
QPushButton:disabled { color: #777777; background: #d8d8d8; }
QPushButton#eegbrowser_event_types { min-width: 86px; }
QPushButton#eegbrowser_cancel { min-width: 56px; }
QLineEdit {
    background: #ffffff;
    border: 1px solid #7f8fa6;
    color: #000000;
    padding: 1px 3px;
}
QLabel#eegbrowser_status_value {
    background: transparent;
    color: #000000;
    padding: 1px 3px;
}
QLabel#eegbrowser_status_label {
    background: transparent;
    color: #000000;
    font-weight: 600;
}
QScrollBar:vertical {
    background: #d7e8ff;
    border: 1px solid #7f8fa6;
    width: 16px;
}
QScrollBar::handle:vertical {
    background: #bfbfbf;
    border: 1px solid #7f8fa6;
    min-height: 24px;
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
        self.setMinimumSize(720, 420)
        central = qt_widgets.QWidget()
        central.setObjectName("eegbrowser_content")
        self._content = central
        self.channel_slider = qt_widgets.QScrollBar(QtCore.Qt.Orientation.Vertical, central)
        self.channel_slider.setObjectName("eegbrowser_channel_slider")
        self.channel_slider.valueChanged.connect(self._set_channel_offset)
        self.canvas = EEGBrowserCanvas(model, parent=central)
        self.scale_indicator = _ScaleIndicator(model, parent=central)
        self.controls = _BrowserControls(self)
        self.setCentralWidget(central)
        self._build_menus()
        self._build_shortcuts()
        self._layout_children()
        self._sync_controls()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._layout_children()

    def _layout_children(self) -> None:
        width = max(1, self._content.width())
        height = max(1, self._content.height())
        axes_geometry = _rect_from_normalized(_AXES_POSITION, width, height)
        self.canvas.setGeometry(
            axes_geometry.adjusted(
                -42,
                -5,
                1,
                28,
            )
        )
        self.channel_slider.setGeometry(_rect_from_normalized(_CONTROL_POSITIONS["channel_slider"], width, height))
        scale_x = _AXES_POSITION[0] + _AXES_POSITION[2]
        scale_width = max(0.001, 1.0 - scale_x)
        self.scale_indicator.setGeometry(
            _rect_from_normalized((scale_x, _AXES_POSITION[1], scale_width, _AXES_POSITION[3]), width, height)
        )
        self.controls.set_geometry(width, height)

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
        self.event_duration_action.triggered.connect(self._toggle_event_durations)
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
        self.scale_action.setCheckable(True)
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

        self.help_action = self.menuBar().addAction("Help")
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
        self.submean_action.setText("Do not remove DC offset" if self.model.state.submean else "Remove DC offset")
        self.scale_action.setText("Show scale")
        self.scale_action.setChecked(self.model.state.scale)
        self.stack_action.setText("Spread channels" if self.model.state.stacked else "Stack channels")
        self.norm_action.setText("Denormalize channels" if self.model.state.normalized else "Normalize channels")
        has_events = bool(self.model.state.events)
        has_event_durations = any(_event_duration_samples(event) is not None for event in self.model.state.events)
        self.event_duration_action.setEnabled(has_event_durations)
        self.event_duration_action.setText(
            "Hide event duration" if self.model.state.show_event_durations else "Plot event duration"
        )
        self.events_on_action.setEnabled(has_events)
        self.events_off_action.setEnabled(has_events)
        self.event_legend_action.setEnabled(has_events)
        self.show_numbers_action.setChecked(self.model.state.channel_label_mode == "numbers")
        self.show_labels_action.setChecked(self.model.state.channel_label_mode == "labels")

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

    def _toggle_event_durations(self) -> None:
        self.model.state.show_event_durations = not self.model.state.show_event_durations
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
        self.scale_indicator.update()
        self._sync_controls()
        self._refresh_menu_labels()

    def _sync_controls(self) -> None:
        self.model.state.clamp_to_data(self.model.data)
        maximum_offset = max(0, self.model.data.n_channels - self.model.state.dispchans)
        self.channel_slider.blockSignals(True)
        self.channel_slider.setRange(0, maximum_offset)
        self.channel_slider.setPageStep(max(1, self.model.state.dispchans))
        self.channel_slider.setSingleStep(1)
        self.channel_slider.setValue(self.model.state.channel_offset)
        self.channel_slider.setVisible(maximum_offset > 0)
        self.channel_slider.blockSignals(False)
        self.scale_indicator.setVisible(self.model.state.scale)
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

    def __init__(self, model: BrowserModel, parent: Any | None = None):
        super().__init__(parent)
        qt_widgets, plot_graphics = _require_gui()
        self.model = model
        self.setObjectName("eegbrowser_canvas")
        self._items: list[Any] = []
        self._scene_items: list[Any] = []
        self._event_label_items: list[Any] = []
        layout = qt_widgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = plot_graphics.PlotWidget(background=None)
        self.plot.setObjectName("eegbrowser_plot")
        self.plot.getViewBox().setBackgroundColor("w")
        self.plot.showGrid(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=model.state.zoom_enabled, y=model.state.zoom_enabled)
        self.plot.hideButtons()
        self.plot.getPlotItem().setClipToView(True)
        self.plot.getPlotItem().setDownsampling(auto=True, mode="peak")
        self.plot.getViewBox().invertY(True)
        self._style_axes()
        layout.addWidget(self.plot)
        self.redraw()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self.redraw()
        QtCore.QTimer.singleShot(0, self.redraw)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.redraw)

    def redraw(self) -> None:
        """Redraw all browser layers for the current state."""
        self.model.state.clamp_to_data(self.model.data)
        self._clear_scene_items()
        self.plot.clear()
        self._items.clear()
        self.plot.showGrid(x=False, y=False)
        self._configure_axes()
        self._draw_grid()
        self._draw_winrej()
        self._draw_event_durations()
        self._draw_traces()
        self._draw_event_lines_and_labels()
        self._draw_plot_box()

    def _clear_scene_items(self) -> None:
        scene = self.plot.scene()
        for item in self._scene_items:
            scene.removeItem(item)
        self._scene_items.clear()
        self._event_label_items.clear()

    def _configure_axes(self) -> None:
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        self.plot.setXRange(self._sample_edge_to_x_value(start), self._sample_edge_to_x_value(stop), padding=0)
        self.plot.setYRange(-0.5, float(self._plot_channel_count()) - 0.5, padding=0)
        labels = []
        if self.model.state.channel_label_mode != "none":
            for screen_index, channel_index in enumerate(self._visible_channels()):
                if self.model.state.channel_label_mode == "numbers":
                    label = str(channel_index + 1)
                else:
                    label = self.model.data.channel_labels[channel_index]
                labels.append((float(screen_index), label))
        self.plot.getAxis("left").setTicks([labels])
        self.plot.getAxis("bottom").setLabel("")
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

    def _draw_event_durations(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        if not self.model.state.show_events or not self.model.state.show_event_durations:
            return
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        for event in self.model.state.events:
            sample = event_latency_to_sample(event.latency, self.model.data)
            if sample < start or sample > stop:
                continue
            color = _EVENT_COLORS[event.color_index % len(_EVENT_COLORS)]
            x_value = self._sample_to_x_value(sample)
            duration = _event_duration_samples(event)
            if duration is None:
                continue
            duration_stop = min(self.model.data.total_samples - 1, sample + duration)
            region = plot_graphics.LinearRegionItem(
                values=(x_value, self._sample_to_x_value(duration_stop)),
                orientation=plot_graphics.LinearRegionItem.Vertical,
                brush=(*color, 45),
                movable=False,
            )
            region.setZValue(-5)
            region.lines[0].setPen(plot_graphics.mkPen((*color, 90), width=1))
            region.lines[1].setPen(plot_graphics.mkPen((*color, 90), width=1))
            self.plot.addItem(region)
            self._items.append(region)

    def _draw_event_lines_and_labels(self) -> None:
        if QtGui is None:
            return
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
                pen=plot_graphics.mkPen(color, width=2.5 if event.type.lower() == "boundary" else 1),
                movable=False,
            )
            line.setZValue(25)
            self.plot.addItem(line)
            self._items.append(line)
            self._add_event_label(event.type[:20], color, x_value)

    def _add_event_label(self, text: str, color: tuple[int, int, int], x_value: float) -> None:
        if QtGui is None or QtWidgets is None:
            return
        label = QtWidgets.QGraphicsSimpleTextItem(text.replace("_", "-"))
        font = label.font()
        font.setPointSize(10)
        label.setFont(font)
        label.setBrush(QtGui.QBrush(QtGui.QColor(*color)))
        label.setRotation(90)
        label.setZValue(40)
        view_box = self.plot.getPlotItem().vb
        scene_pos = view_box.mapViewToScene(QtCore.QPointF(x_value, 0.0))
        view_rect = view_box.sceneBoundingRect()
        label.setPos(scene_pos.x(), max(0.0, view_rect.top() - label.boundingRect().width() - 2.0))
        self.plot.scene().addItem(label)
        self._scene_items.append(label)
        self._event_label_items.append(label)

    def _draw_plot_box(self) -> None:
        if QtGui is None or QtWidgets is None:
            return
        view_rect = self.plot.getPlotItem().vb.sceneBoundingRect()
        frame = QtWidgets.QGraphicsRectItem(view_rect)
        frame.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        frame.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
        frame.setZValue(35)
        self.plot.scene().addItem(frame)
        self._scene_items.append(frame)

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

    def _trace_to_axis_y(self, values: np.ndarray, screen_index: int) -> np.ndarray:
        spacing = max(float(self.model.state.spacing), np.finfo(float).eps)
        baseline = (float(self._plot_channel_count()) - 1.0) / 2.0 if self.model.state.stacked else float(screen_index)
        return baseline - np.asarray(values, dtype=float) / spacing * _TRACE_SCALE

    def _visible_channels(self) -> range:
        state = self.model.state
        return range(state.channel_offset, state.channel_offset + self._plot_channel_count())

    def _plot_channel_count(self) -> int:
        state = self.model.state
        remaining = self.model.data.n_channels - state.channel_offset
        return max(1, min(remaining, state.dispchans + 2))

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
        step = 1.0
        first = np.ceil(start_x / step) * step
        ticks = []
        current = first
        while current <= stop_x:
            ticks.append((float(current), f"{current:g}"))
            current += step
        return ticks

    def _style_axes(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        plot_item = self.plot.getPlotItem()
        plot_item.showAxis("top", True)
        plot_item.showAxis("right", True)
        plot_item.getAxis("top").setStyle(showValues=False, tickLength=0)
        plot_item.getAxis("right").setStyle(showValues=False, tickLength=0)
        axis_pen = plot_graphics.mkPen((0, 0, 0), width=1)
        for name in ("left", "bottom", "top", "right"):
            axis = plot_item.getAxis(name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
            axis.setTickPen(axis_pen)
        plot_item.getAxis("left").setWidth(42)
        plot_item.getAxis("bottom").setHeight(28)
        plot_item.getAxis("top").setHeight(1)
        plot_item.getAxis("right").setWidth(1)
        plot_item.getAxis("bottom").setGrid(False)
        plot_item.getAxis("left").setGrid(False)

    def _draw_grid(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        pen = plot_graphics.mkPen((225, 225, 225), width=1)
        if self.model.state.xgrid:
            for x_value, _label in self._x_ticks(start, stop):
                line = plot_graphics.InfiniteLine(pos=x_value, angle=90, pen=pen, movable=False)
                self.plot.addItem(line)
                self._items.append(line)
        if self.model.state.ygrid:
            for y_value in range(self._plot_channel_count()):
                line = plot_graphics.InfiniteLine(pos=float(y_value), angle=0, pen=pen, movable=False)
                self.plot.addItem(line)
                self._items.append(line)


class _ScaleIndicator(_QWidget):
    def __init__(self, model: BrowserModel, parent: Any | None = None):
        super().__init__(parent)
        self.model = model
        self.setObjectName("eegbrowser_scale_indicator")

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        super().paintEvent(event)
        if not self.model.state.scale or QtGui is None or QtCore is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        width = self.width()
        height = self.height()
        x = max(10, width // 2)
        top = int(height * 0.82)
        bottom = int(height * 0.98)
        cap = max(7, int(width * 0.22))
        painter.drawText(QtCore.QRect(0, max(0, top - 70), width, 24), QtCore.Qt.AlignmentFlag.AlignCenter, "Scale")
        painter.drawText(
            QtCore.QRect(0, max(0, top - 42), width, 24),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            _format_control_number(self.model.state.spacing),
        )
        painter.drawLine(x - cap, top, x + cap, top)
        painter.drawLine(x, top, x, bottom)
        painter.drawLine(x - cap, bottom, x + cap, bottom)
        painter.end()


_NumericLineEditBase = QtWidgets.QLineEdit if QtWidgets is not None else object


class _NumericLineEdit(_NumericLineEditBase):
    if QtCore is not None:  # pragma: no branch - class definition guard
        valueChanged = QtCore.Signal(float)

    def __init__(self, value: float = 0.0, *, minimum: float = 0.0, maximum: float = 1_000_000.0):
        super().__init__()
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.editingFinished.connect(self._emit_value)
        self.setValue(value)

    def setRange(self, minimum: float, maximum: float) -> None:  # noqa: N802 - Qt-style compatibility
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        if not self.signalsBlocked():
            self.setValue(self.value())

    def setValue(self, value: float) -> None:  # noqa: N802 - Qt-style compatibility
        clamped = min(max(float(value), self._minimum), self._maximum)
        self.setText(_format_control_number(clamped))
        if not self.signalsBlocked():
            self.valueChanged.emit(clamped)

    def value(self) -> float:
        try:
            value = float(self.text())
        except ValueError:
            value = self._minimum
        return min(max(value, self._minimum), self._maximum)

    def _emit_value(self) -> None:
        self.setValue(self.value())


class _BrowserControls:
    def __init__(self, browser: EEGBrowserWindow):
        qt_widgets, _pg = _require_gui()
        self.browser = browser
        self.model = browser.model
        parent = browser._content
        self._widgets: list[Any] = []

        self.cancel_button = qt_widgets.QPushButton(
            "CANCEL" if self.model.state.accept_label is not None else "CLOSE", parent
        )
        self.cancel_button.setObjectName("eegbrowser_cancel")
        self.cancel_button.clicked.connect(browser.cancel_and_close)
        self._widgets.append(self.cancel_button)
        self.event_button = qt_widgets.QPushButton("Event types", parent)
        self.event_button.setObjectName("eegbrowser_event_types")
        self.event_button.clicked.connect(browser.show_event_legend)
        self._widgets.append(self.event_button)
        self.back_page_button = self._button("<<", lambda: browser.scroll_time(-5.0))
        self.back_step_button = self._button("<", lambda: browser.scroll_time(-1.0))
        self.forward_step_button = self._button(">", lambda: browser.scroll_time(1.0))
        self.forward_page_button = self._button(">>", lambda: browser.scroll_time(5.0))

        self.channel_label = self._status_label("Chan.")
        self.channel_value = self._status_value("")
        self.time_label = self._status_label("Time")
        self.time_value = self._status_value("0.00")
        self.value_label = self._status_label("Value")
        self.value_value = self._status_value("0.00")

        self.position = _NumericLineEdit()
        self.position.setObjectName("eegbrowser_position")
        self.position.valueChanged.connect(browser.set_position)
        self.position.setParent(parent)
        self._widgets.append(self.position)

        self.spacing = _NumericLineEdit(minimum=_MIN_SPACING)
        self.spacing.setObjectName("eegbrowser_spacing")
        self.spacing.setRange(_MIN_SPACING, 1_000_000.0)
        self.spacing.valueChanged.connect(browser.set_spacing)
        self.spacing.setParent(parent)
        self._widgets.append(self.spacing)
        self.plus_button = self._button("+", lambda: browser.adjust_spacing(1))
        self.minus_button = self._button("-", lambda: browser.adjust_spacing(-1))
        self.norm_button = self._button("Norm", browser.toggle_normalize)
        self.stack_button = self._button("Stack", browser.toggle_stack)
        self.accept_button = qt_widgets.QPushButton(self.model.state.accept_label or "REJECT", parent)
        self.accept_button.setObjectName("eegbrowser_accept")
        self.accept_button.clicked.connect(browser.accept_and_close)
        self._widgets.append(self.accept_button)

    def set_geometry(self, width: int, height: int) -> None:
        for name, widget in (
            ("cancel", self.cancel_button),
            ("events", self.event_button),
            ("back_page", self.back_page_button),
            ("back_step", self.back_step_button),
            ("position", self.position),
            ("forward_step", self.forward_step_button),
            ("forward_page", self.forward_page_button),
            ("channel_label", self.channel_label),
            ("channel_value", self.channel_value),
            ("time_label", self.time_label),
            ("time_value", self.time_value),
            ("data_label", self.value_label),
            ("data_value", self.value_value),
            ("spacing", self.spacing),
            ("plus", self.plus_button),
            ("minus", self.minus_button),
            ("norm", self.norm_button),
            ("stack", self.stack_button),
            ("accept", self.accept_button),
        ):
            widget.setGeometry(_rect_from_normalized(_CONTROL_POSITIONS[name], width, height))

    def hide(self) -> None:
        for widget in self._widgets:
            widget.hide()

    def show(self) -> None:
        for widget in self._widgets:
            widget.show()

    def sync_from_state(self) -> None:
        state = self.model.state
        self.back_page_button.setEnabled(True)
        self.back_step_button.setEnabled(True)
        self.forward_step_button.setEnabled(True)
        self.forward_page_button.setEnabled(True)
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
        self.cancel_button.setText("CANCEL" if state.accept_label is not None else "CLOSE")
        self.stack_button.setText("Spread" if state.stacked else "Stack")
        self.norm_button.setText("Denorm" if state.normalized else "Norm")
        self.time_label.setText("Freq" if self.model.data.mode == "spectral" else "Time")
        self.value_label.setText("Power" if self.model.data.mode == "spectral" else "Value")

    def _button(self, label: str, callback: Any) -> Any:
        qt_widgets, _pg = _require_gui()
        button = qt_widgets.QPushButton(label, self.browser._content)
        button.clicked.connect(callback)
        self._widgets.append(button)
        return button

    def _status_label(self, text: str) -> Any:
        qt_widgets, _pg = _require_gui()
        label = qt_widgets.QLabel(text, self.browser._content)
        label.setObjectName("eegbrowser_status_label")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._widgets.append(label)
        return label

    def _status_value(self, text: str) -> Any:
        label = self._status_label(text)
        label.setObjectName("eegbrowser_status_value")
        return label


def _rect_from_normalized(rect: tuple[float, float, float, float], width: int, height: int) -> Any:
    if QtCore is None:
        raise RuntimeError("QtCore is required for EEG browser layout")
    x, y, rect_width, rect_height = rect
    return QtCore.QRect(
        int(round(x * width)),
        int(round(height - (y + rect_height) * height)),
        max(1, int(round(rect_width * width))),
        max(1, int(round(rect_height * height))),
    )


def _format_control_number(value: float) -> str:
    return f"{float(value):.4g}"


def _event_duration_samples(event: Any) -> int | None:
    if event.duration is None:
        return None
    duration = float(event.duration)
    if not np.isfinite(duration) or duration <= 0:
        return None
    return int(round(duration))


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


def _require_gui() -> tuple[Any, Any]:
    if QtWidgets is None or QtCore is None or QtGui is None or pg is None:
        raise RuntimeError(
            "PySide6 and pyqtgraph are required for EEGPrep eegplot. Install them with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtWidgets, pg


__all__ = ["EEGBrowserCanvas", "EEGBrowserWindow", "open_eegbrowser"]
