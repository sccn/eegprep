"""Qt/pyqtgraph rendering foundation for the EEGPrep scrolling EEG browser."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat
from eegprep.functions.sigprocfunc.eegplot import (
    BrowserModel,
    add_winrej_region,
    browser_max_channel_offset,
    browser_visible_channel_count,
    browser_window_duration,
    decimate_minmax,
    event_latency_to_sample,
    toggle_winrej_at_sample,
    visible_sample_bounds,
    winrej_to_array,
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
_WINREJ_BRUSH_ALPHA = 210
_WINREJ_EDGE_ALPHA = 230
_EPOCHED_XTICK_LABEL_ROTATION = -45.0
_AXES_POSITION = (0.0964286, 0.15, 0.842, 0.75)
_NOUI_AXES_POSITION = (0.055, 0.075, 0.90, 0.87)
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


def open_eegbrowser(model: BrowserModel, accept_callback: Callable[[np.ndarray], None] | None = None) -> Any:
    """Create, show, and return an EEG browser window."""
    qt_widgets, _pg = _require_gui()
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    window = EEGBrowserWindow(model, accept_callback=accept_callback)
    window.show()
    app.processEvents()
    return window


def link_eegbrowser_windows(parent: Any, children: Any) -> tuple[Any, ...]:
    """Link child browser windows so scrolling changes stay synchronized."""
    child_windows = _normalize_child_windows(children)
    for child in child_windows:
        parent.link_child(child)
    return child_windows


class EEGBrowserWindow(_QMainWindow):
    """Basic EEGLAB-like scrolling browser window."""

    def __init__(self, model: BrowserModel, accept_callback: Callable[[np.ndarray], None] | None = None):
        super().__init__()
        qt_widgets, _pg = _require_gui()
        self.model = model
        self.accept_callback = accept_callback
        self._pre_normalize_spacing: float | None = None
        self._legend_window: Any = None
        self._message_window: Any = None
        self._linked_windows: list[Any] = []
        self._child_windows: list[Any] = []
        self._syncing_link = False
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
        if model.state.noui:
            self.set_publication_view(True)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._layout_children()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        for child in list(self._child_windows):
            child.close()
        for linked in list(self._linked_windows):
            linked._unlink_window(self)
        self._linked_windows.clear()
        self._child_windows.clear()
        super().closeEvent(event)

    def _layout_children(self) -> None:
        width = max(1, self._content.width())
        height = max(1, self._content.height())
        if self.model.state.noui:
            self.canvas.setGeometry(_rect_from_normalized(_NOUI_AXES_POSITION, width, height))
            self.channel_slider.hide()
            self.scale_indicator.hide()
            self.controls.hide()
            self.menuBar().hide()
            return
        self.channel_slider.show()
        self.scale_indicator.show()
        self.controls.show()
        self.menuBar().show()
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
        self.controls.sync_from_state()

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
        self.accept_action.setEnabled(self._can_accept())
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

    def set_publication_view(self, enabled: bool = True) -> None:
        """Toggle EEGLAB ``noui``-style publication display."""
        self.model.state.noui = bool(enabled)
        self._layout_children()
        self._sync_controls()

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
        if self.accept_callback is not None:
            winrej = winrej_to_array(self.model.state.winrej, self.model.data.n_channels)
            self.accept_callback(winrej)
        self.close()

    def cancel_and_close(self) -> None:
        self.model.state.cancelled = True
        self.close()

    def _set_channel_offset(self, value: int) -> None:
        self.model.state.channel_offset = int(value)
        self._redraw()

    def _redraw(self) -> None:
        self._redraw_local()
        if not self._syncing_link:
            self._sync_linked_windows({id(self)})

    def _redraw_local(self) -> None:
        self.model.state.clamp_to_data(self.model.data)
        self.canvas.redraw()
        self.scale_indicator.update()
        self._sync_controls()
        self._refresh_menu_labels()

    def _sync_controls(self) -> None:
        self.model.state.clamp_to_data(self.model.data)
        if self.model.state.noui:
            self.channel_slider.setVisible(False)
            self.scale_indicator.setVisible(False)
            self.controls.hide()
            self.menuBar().hide()
            return
        self.controls.show()
        self.menuBar().show()
        maximum_offset = browser_max_channel_offset(self.model.data, self.model.state)
        self.channel_slider.blockSignals(True)
        self.channel_slider.setRange(0, maximum_offset)
        self.channel_slider.setPageStep(browser_visible_channel_count(self.model.data, self.model.state))
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
        self.set_publication_view(True)

    def link_window(self, other: Any) -> None:
        """Link this browser with another browser for synchronized scrolling."""
        if other is self:
            return
        self._add_linked_window(other)
        other._add_linked_window(self)
        other._apply_linked_scroll(self)

    def link_child(self, child: Any) -> None:
        """Link and register a child browser window."""
        self.link_window(child)
        if child not in self._child_windows:
            self._child_windows.append(child)

    def _add_linked_window(self, other: Any) -> None:
        if other not in self._linked_windows:
            self._linked_windows.append(other)

    def _unlink_window(self, other: Any) -> None:
        if other in self._linked_windows:
            self._linked_windows.remove(other)
        if other in self._child_windows:
            self._child_windows.remove(other)

    def _sync_linked_windows(self, visited: set[int]) -> None:
        for linked in list(self._linked_windows):
            if id(linked) in visited:
                continue
            linked._apply_linked_scroll(self, visited)

    def _apply_linked_scroll(self, source: Any, visited: set[int] | None = None) -> None:
        if visited is None:
            visited = {id(source)}
        visited.add(id(self))
        self._syncing_link = True
        try:
            self.model.state.time = source.model.state.time
            self.model.state.winlength = source.model.state.winlength
            self.model.state.channel_offset = source.model.state.channel_offset
            self._redraw_local()
        finally:
            self._syncing_link = False
        self._sync_linked_windows(visited)

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

    def _can_accept(self) -> bool:
        return self.model.state.accept_label is not None or self.accept_callback is not None


class EEGBrowserCanvas(_QWidget):
    """pyqtgraph canvas for traces, events, and marked windows."""

    def __init__(self, model: BrowserModel, parent: Any | None = None):
        super().__init__(parent)
        qt_widgets, plot_graphics = _require_gui()
        self.model = model
        self.setObjectName("eegbrowser_canvas")
        self._items: list[Any] = []
        self._scene_items: list[Any] = []
        self._event_line_items: list[Any] = []
        self._event_label_items: list[Any] = []
        self._trial_boundary_items: list[Any] = []
        self._bottom_tick_label_items: list[Any] = []
        self._drag_start_sample: int | None = None
        self._drag_start_pos: Any = None
        self._drag_channel_index: int | None = None
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
        self.plot.viewport().installEventFilter(self)
        self.redraw()

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API name
        """Handle EEGLAB-style mark/unmark mouse gestures on the plot viewport."""
        if QtCore is None or watched is not self.plot.viewport():
            return super().eventFilter(watched, event)
        event_type = event.type()
        press = QtCore.QEvent.Type.MouseButtonPress
        release = QtCore.QEvent.Type.MouseButtonRelease
        if event_type in {press, release} and event.button() != QtCore.Qt.MouseButton.LeftButton:
            self._clear_drag()
            return super().eventFilter(watched, event)
        if event_type == press:
            scene_point = self._event_scene_point(event)
            if not self.plot.getPlotItem().vb.sceneBoundingRect().contains(scene_point):
                return super().eventFilter(watched, event)
            sample = self._event_sample(scene_point)
            self._drag_start_sample = sample
            self._drag_start_pos = event.position() if hasattr(event, "position") else event.pos()
            self._drag_channel_index = self._event_channel(scene_point) if self.model.state.setelectrode else None
            return True
        if event_type == release:
            if self._drag_start_sample is None:
                return super().eventFilter(watched, event)
            scene_point = self._event_scene_point(event)
            sample = self._event_sample(scene_point)
            if self._mouse_moved(event):
                self.mark_samples(self._drag_start_sample, sample, channel_index=self._drag_channel_index)
            else:
                self.toggle_sample(sample, channel_index=self._drag_channel_index)
            self._clear_drag()
            return True
        return super().eventFilter(watched, event)

    def mark_samples(self, start: int, end: int, *, channel_index: int | None = None) -> None:
        """Add a mark from browser sample coordinates and redraw."""
        self.model.state.winrej = add_winrej_region(
            self.model.state.winrej,
            start,
            end,
            n_channels=self.model.data.n_channels,
            total_samples=self.model.data.total_samples,
            color=self.model.state.mark_color,
            channel_index=channel_index,
            pnts=self.model.data.pnts if self.model.data.epoched else None,
        )
        self.redraw()

    def toggle_sample(self, sample: int, *, channel_index: int | None = None) -> None:
        """Remove a mark at a sample or toggle a marked channel."""
        self.model.state.winrej = toggle_winrej_at_sample(
            self.model.state.winrej,
            sample,
            n_channels=self.model.data.n_channels,
            channel_index=channel_index,
        )
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
        self._trial_boundary_items.clear()
        self._bottom_tick_label_items.clear()
        self.plot.showGrid(x=False, y=False)
        self._configure_axes()
        self._draw_grid()
        self._draw_winrej()
        self._draw_event_durations()
        self._draw_traces()
        self._draw_marked_channel_segments()
        self._draw_event_lines_and_labels()
        self._draw_plot_box()
        self._draw_epoched_latency_tick_labels()

    def _clear_scene_items(self) -> None:
        scene = self.plot.scene()
        for item in self._scene_items:
            scene.removeItem(item)
        self._scene_items.clear()
        self._event_line_items.clear()
        self._event_label_items.clear()
        self._bottom_tick_label_items.clear()

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
        bottom_axis = self.plot.getAxis("bottom")
        bottom_axis.setLabel("")
        if self.model.data.epoched:
            bottom_axis.setHeight(44)
            bottom_axis.setTicks([[(x_value, "") for x_value, _label in self._bottom_x_ticks(start, stop)]])
        else:
            bottom_axis.setHeight(28)
            bottom_axis.setTicks([self._bottom_x_ticks(start, stop)])
        top_axis = self.plot.getAxis("top")
        if self.model.data.epoched:
            top_axis.setHeight(22)
            top_axis.setStyle(showValues=True, tickLength=0)
            top_axis.setTicks([self._epoch_ticks(start, stop)])
        else:
            top_axis.setHeight(1)
            top_axis.setStyle(showValues=False, tickLength=0)
            top_axis.setTicks([[]])
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
                brush=(*color, _WINREJ_BRUSH_ALPHA),
                movable=False,
            )
            item.lines[0].setPen(plot_graphics.mkPen((*color, _WINREJ_EDGE_ALPHA), width=1))
            item.lines[1].setPen(plot_graphics.mkPen((*color, _WINREJ_EDGE_ALPHA), width=1))
            item.setZValue(-20)
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
        if QtGui is None or QtWidgets is None:
            return
        if not self.model.state.show_events:
            return
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        for event in self.model.state.events:
            sample = event_latency_to_sample(event.latency, self.model.data)
            if sample < start or sample > stop:
                continue
            color = _EVENT_COLORS[event.color_index % len(_EVENT_COLORS)]
            x_value = self._sample_to_x_value(sample)
            self._add_event_line(event.type, color, x_value)
            self._add_event_label(event.type[:20], color, x_value)

    def _add_event_line(self, event_type: str, color: tuple[int, int, int], x_value: float) -> None:
        if QtGui is None or QtWidgets is None:
            return
        view_box = self.plot.getPlotItem().vb
        view_rect = view_box.sceneBoundingRect()
        scene_pos = view_box.mapViewToScene(QtCore.QPointF(x_value, 0.0))
        line = QtWidgets.QGraphicsLineItem(scene_pos.x(), view_rect.top(), scene_pos.x(), view_rect.bottom())
        pen = QtGui.QPen(QtGui.QColor(*color))
        pen.setWidthF(2.5 if event_type.lower() == "boundary" else 1.5)
        line.setPen(pen)
        line.setZValue(34)
        self.plot.scene().addItem(line)
        self._scene_items.append(line)
        self._event_line_items.append(line)

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
        label.setPos(scene_pos.x(), view_rect.top() - label.boundingRect().width() - 2.0)
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

    def _draw_epoched_latency_tick_labels(self) -> None:
        if QtCore is None or QtGui is None or QtWidgets is None:
            return
        if not self.model.data.epoched:
            return
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        view_box = self.plot.getPlotItem().vb
        view_rect = view_box.sceneBoundingRect()
        font = QtGui.QFont()
        font.setPointSize(9)
        for x_value, label_text in self._bottom_x_ticks(start, stop):
            if not label_text:
                continue
            scene_pos = view_box.mapViewToScene(QtCore.QPointF(float(x_value), 0.0))
            label = QtWidgets.QGraphicsSimpleTextItem(label_text)
            label.setFont(font)
            label.setBrush(QtGui.QBrush(QtGui.QColor("#000000")))
            label.setRotation(_EPOCHED_XTICK_LABEL_ROTATION)
            label.setZValue(45)
            label.setPos(scene_pos.x() - 2.0, view_rect.bottom() + 18.0)
            self.plot.scene().addItem(label)
            self._scene_items.append(label)
            self._bottom_tick_label_items.append(label)

    def _draw_traces(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        data = self.model.data
        state = self.model.state
        start, stop = visible_sample_bounds(data, state)
        x_values = self._sample_range_to_x_values(start, stop)
        for screen_index, channel_index in enumerate(self._visible_channels()):
            y_values = self._trace_values(channel_index, start, stop)
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

    def _draw_marked_channel_segments(self) -> None:
        _qt_widgets, plot_graphics = _require_gui()
        start, stop = visible_sample_bounds(self.model.data, self.model.state)
        visible_channels = tuple(self._visible_channels())
        visible_x = self._sample_range_to_x_values(start, stop)
        for region in self.model.state.winrej:
            mask = np.asarray(region.channel_mask, dtype=bool)
            if mask.size == 0 or mask.all():
                continue
            segment_start = max(_winrej_frame_to_index(region.start), start)
            segment_stop = min(_winrej_frame_to_index(region.end) + 1, stop)
            if segment_stop <= segment_start:
                continue
            rel_start = segment_start - start
            rel_stop = segment_stop - start
            for screen_index, channel_index in enumerate(visible_channels):
                if channel_index >= mask.size or not bool(mask[channel_index]):
                    continue
                y_values = self._trace_values(channel_index, start, stop)[rel_start:rel_stop]
                x_values = visible_x[rel_start:rel_stop]
                x_dec, y_dec = decimate_minmax(x_values, y_values, max(1, self.width()))
                curve = self.plot.plot(
                    x_dec,
                    self._trace_to_axis_y(y_dec, screen_index),
                    pen=plot_graphics.mkPen((200, 0, 0), width=1.4),
                    skipFiniteCheck=True,
                )
                curve.setClipToView(True)
                curve.setZValue(20)
                self._items.append(curve)

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

    def _trace_values(self, channel_index: int, start: int, stop: int) -> np.ndarray:
        data = self.model.data
        state = self.model.state
        y_values = np.asarray(data.flat_data[channel_index, start:stop], dtype=float)
        if state.submean and y_values.size:
            y_values = y_values - np.nanmean(y_values)
        if state.normalized and y_values.size:
            y_values = y_values / _channel_std(data.flat_data[channel_index])
        return y_values

    def _trace_to_axis_y(self, values: np.ndarray, screen_index: int) -> np.ndarray:
        spacing = max(float(self.model.state.spacing), np.finfo(float).eps)
        baseline = (float(self._plot_channel_count()) - 1.0) / 2.0 if self.model.state.stacked else float(screen_index)
        return baseline - np.asarray(values, dtype=float) / spacing * _TRACE_SCALE

    def _visible_channels(self) -> range:
        state = self.model.state
        return range(state.channel_offset, state.channel_offset + self._plot_channel_count())

    def _plot_channel_count(self) -> int:
        return browser_visible_channel_count(self.model.data, self.model.state)

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

    def _bottom_x_ticks(self, start: int, stop: int) -> list[tuple[float, str]]:
        if self.model.data.epoched:
            return self._latency_ticks(start, stop)
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

    def _epoch_ticks(self, start: int, stop: int) -> list[tuple[float, str]]:
        pnts = max(1, int(self.model.data.pnts))
        first_epoch = max(0, int(start // pnts))
        last_epoch = min(int(self.model.data.trials) - 1, int(np.ceil(stop / pnts)) - 1)
        ticks = []
        for epoch in range(first_epoch, last_epoch + 1):
            epoch_start = epoch * pnts
            epoch_stop = min((epoch + 1) * pnts, self.model.data.total_samples)
            center = (max(start, epoch_start) + min(stop, epoch_stop)) / 2.0
            ticks.append((float(center), str(epoch + 1)))
        return ticks

    def _trial_boundary_ticks(self, start: int, stop: int) -> list[float]:
        pnts = max(1, int(self.model.data.pnts))
        first_epoch = max(0, int(start // pnts))
        last_epoch = min(int(self.model.data.trials) - 1, int(np.ceil(stop / pnts)) - 1)
        return [float(epoch * pnts) for epoch in range(first_epoch, last_epoch + 1) if start <= epoch * pnts <= stop]

    def _latency_ticks(self, start: int, stop: int) -> list[tuple[float, str]]:
        pnts = max(1, int(self.model.data.pnts))
        srate = float(self.model.state.srate)
        limits = np.asarray(self.model.state.limits, dtype=float)
        increment = _epoch_latency_increment_ms(float(self.model.state.winlength), float(limits[1] - limits[0]))
        first_epoch = max(0, int(start // pnts))
        last_epoch = min(int(self.model.data.trials) - 1, int(np.ceil(stop / pnts)) - 1)
        ticks: list[tuple[float, str]] = []
        for epoch in range(first_epoch, last_epoch + 1):
            first_ms = np.ceil(limits[0] / increment) * increment
            for latency_ms in np.arange(first_ms, limits[1] + increment * 0.001, increment):
                point = (latency_ms * 0.001 - limits[0] * 0.001) * srate + 1.0 + epoch * pnts
                x_value = float(point - 1.0)
                if start <= x_value <= stop:
                    label = eeg_point2lat(point, epoch + 1, srate, limits, 1e-3)[0]
                    ticks.append((x_value, _format_tick_label(float(label))))
        return ticks

    def _event_scene_point(self, event: Any) -> Any:
        point = event.position() if hasattr(event, "position") else event.pos()
        return self.plot.viewport().mapToScene(point.toPoint() if hasattr(point, "toPoint") else point)

    def _event_sample(self, scene_point: Any) -> int:
        view_point = self.plot.getPlotItem().vb.mapSceneToView(scene_point)
        x_value = float(view_point.x())
        if self.model.data.x_values is not None:
            sample = int(np.argmin(np.abs(self.model.data.x_values - x_value)))
            max_sample = self.model.data.total_samples - 1
        elif self.model.data.epoched:
            # 0-based: epoched winrej rows follow trial2eegplot sample offsets.
            sample = int(round(x_value))
            max_sample = self.model.data.total_samples - 1
        else:
            # 1-based: continuous winrej rows feed eeg_eegrej latency ranges.
            sample = int(round(x_value * float(self.model.state.srate))) + 1
            max_sample = self.model.data.total_samples
        return max(0, min(sample, max_sample))

    def _event_channel(self, scene_point: Any) -> int | None:
        view_point = self.plot.getPlotItem().vb.mapSceneToView(scene_point)
        screen_index = int(round(float(view_point.y())))
        channel_index = self.model.state.channel_offset + screen_index
        if 0 <= channel_index < self.model.data.n_channels:
            return channel_index
        return None

    def _mouse_moved(self, event: Any) -> bool:
        if self._drag_start_pos is None:
            return False
        point = event.position() if hasattr(event, "position") else event.pos()
        dx = float(point.x() - self._drag_start_pos.x())
        dy = float(point.y() - self._drag_start_pos.y())
        return dx * dx + dy * dy > 9.0

    def _clear_drag(self) -> None:
        self._drag_start_sample = None
        self._drag_start_pos = None
        self._drag_channel_index = None

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
            for x_value, _label in self._bottom_x_ticks(start, stop):
                line = plot_graphics.InfiniteLine(pos=x_value, angle=90, pen=pen, movable=False)
                self.plot.addItem(line)
                self._items.append(line)
        if self.model.data.epoched:
            boundary_pen = plot_graphics.mkPen((0, 0, 180), width=1, style=QtCore.Qt.PenStyle.DashLine)
            for x_value in self._trial_boundary_ticks(start, stop):
                line = plot_graphics.InfiniteLine(pos=x_value, angle=90, pen=boundary_pen, movable=False)
                line.setZValue(-1)
                self.plot.addItem(line)
                self._items.append(line)
                self._trial_boundary_items.append(line)
        if self.model.state.ygrid:
            for y_value in range(self._plot_channel_count()):
                line = plot_graphics.InfiniteLine(pos=float(y_value), angle=0, pen=pen, movable=False)
                self.plot.addItem(line)
                self._items.append(line)


def _epoch_latency_increment_ms(winlength: float, epoch_width_ms: float) -> float:
    divisions = 20.0 / max(float(winlength), np.finfo(float).eps)
    candidates = np.asarray(
        [
            100000 / 1,
            100000 / 2,
            100000 / 4,
            100000 / 5,
            10000 / 1,
            10000 / 2,
            10000 / 4,
            10000 / 5,
            1000 / 1,
            1000 / 2,
            1000 / 4,
            1000 / 5,
            100 / 1,
            100 / 2,
            100 / 4,
            100 / 5,
            100 / 10,
            100 / 20,
        ],
        dtype=float,
    )
    index = int(np.argmin(np.abs(divisions * candidates - float(epoch_width_ms))))
    return float(candidates[index])


def _format_tick_label(value: float) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


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

        self.cancel_button = qt_widgets.QPushButton("CANCEL" if self.browser._can_accept() else "CLOSE", parent)
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
        self.accept_button = qt_widgets.QPushButton(self.model.state.accept_label or "Reject", parent)
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
        self.accept_button.setVisible(self.browser._can_accept())
        self.accept_button.setText(state.accept_label or "Reject")
        self.cancel_button.setText("CANCEL" if self.browser._can_accept() else "CLOSE")
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


def _normalize_child_windows(children: Any) -> tuple[Any, ...]:
    if isinstance(children, EEGBrowserWindow):
        return (children,)
    if isinstance(children, (list, tuple, set)):
        out = tuple(children)
        if all(isinstance(item, EEGBrowserWindow) for item in out):
            return out
    raise TypeError("children must be an EEGBrowserWindow or a sequence of EEGBrowserWindow objects")


def _require_gui() -> tuple[Any, Any]:
    if QtWidgets is None or QtCore is None or QtGui is None or pg is None:
        raise RuntimeError(
            "PySide6 and pyqtgraph are required for EEGPrep eegplot. Install them with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtWidgets, pg


__all__ = ["EEGBrowserCanvas", "EEGBrowserWindow", "link_eegbrowser_windows", "open_eegbrowser"]
