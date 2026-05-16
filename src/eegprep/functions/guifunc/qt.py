"""Optional PySide6 renderer for EEGLAB-like dialog specs."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.popfunc.pop_chansel import pop_chansel

from .spec import CallbackSpec, ControlSpec, DialogSpec

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtWidgets import QDialog
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtCore = None
    QtWidgets = None
    QDialog = None

_VALUE_PROPERTY = "eegprep_value"
_MULTI_SELECT_PROPERTY = "eegprep_multiselect"


def _require_qt() -> tuple[Any, Any]:
    if QtCore is None or QtWidgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI dialogs. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtCore, QtWidgets


class QtDialogRenderer:
    """Render :class:`DialogSpec` using PySide6 widgets."""

    def run(
        self,
        spec: DialogSpec,
        initial_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        app, dialog, widgets = self.build_dialog(spec, initial_values=initial_values)
        app.processEvents()
        if dialog.exec() != self._QDialog().Accepted:
            return None
        return {tag: self._read_widget(widget) for tag, widget in widgets.items()}

    @staticmethod
    def _QDialog() -> Any:
        if QDialog is None:
            raise RuntimeError(
                "PySide6 is required for EEGPrep GUI dialogs. Install it with "
                "`pip install -e .[gui]` or `pip install eegprep[gui]`."
            )
        return QDialog

    def build_dialog(
        self,
        spec: DialogSpec,
        initial_values: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Build a dialog without executing it, for screenshot capture."""
        qt_core, qt_widgets = _require_qt()

        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
        dialog = qt_widgets.QDialog()
        dialog.setObjectName(spec.function_name)
        dialog.setWindowTitle(spec.title)
        self._apply_eeglab_style(dialog)
        layout = qt_widgets.QVBoxLayout(dialog)
        layout.setContentsMargins(*spec.content_margins)
        layout.setSpacing(4)

        initial_values = initial_values or {}
        widgets: dict[str, Any] = {}
        index = 0
        for row_index, row_geometry in enumerate(spec.geometry):
            weights = self._row_weights(row_geometry)
            row_container = qt_widgets.QWidget()
            row_layout = qt_widgets.QHBoxLayout(row_container)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            added_visible_widget = False
            for weight in weights:
                if index >= len(spec.controls):
                    break
                control = spec.controls[index]
                widget = self._build_widget(qt_widgets, control, initial_values)
                if control.tag:
                    widgets[control.tag] = widget
                stretch = max(1, round(float(weight) * 100))
                if control.style.lower() == "spacer":
                    row_layout.addStretch(stretch)
                else:
                    row_layout.addWidget(widget, stretch, qt_core.Qt.AlignVCenter)
                    added_visible_widget = True
                index += 1
            if added_visible_widget:
                layout.addWidget(row_container, self._row_stretch(spec, row_index))
            else:
                layout.addSpacing(8)

        for control in spec.controls:
            self._connect_callback(control.callback, widgets)

        self._add_buttons(qt_widgets, layout, spec, dialog, widgets)
        dialog.adjustSize()
        self._apply_spec_size(dialog, spec)
        return app, dialog, widgets

    @staticmethod
    def _apply_eeglab_style(dialog: Any) -> None:
        dialog.setStyleSheet(
            """
            QDialog {
                background: #a8c2ff;
                color: #000066;
                font-size: 16px;
            }
            QLabel, QCheckBox, QPushButton, QLineEdit, QComboBox, QListWidget {
                font-size: 16px;
            }
            QLabel, QCheckBox {
                color: #000066;
                background: transparent;
            }
            QLineEdit {
                background: white;
                border: 1px solid #7f7f7f;
                min-height: 18px;
                max-height: 18px;
                margin-left: 1px;
                padding: 0 3px;
                color: #000066;
            }
            QLineEdit:disabled {
                background: #dce6ff;
                color: #7c86a8;
            }
            QComboBox {
                background: white;
                border: 1px solid #7f7f7f;
                min-height: 20px;
                max-height: 20px;
                color: #000066;
            }
            QListWidget {
                background: white;
                border: 1px solid #7f7f7f;
                min-height: 74px;
                max-height: 74px;
                color: #000066;
            }
            QPushButton {
                background: #eeeeee;
                border: 1px solid #7f7f7f;
                min-height: 18px;
                max-height: 18px;
                padding: 0 10px;
                color: #000066;
            }
            QPushButton:disabled {
                color: #b0b0b0;
            }
            QPushButton#events_button {
                min-width: 130px;
                max-width: 130px;
            }
            QPushButton#scroll {
                min-width: 159px;
                max-width: 159px;
            }
            QPushButton#refbr, QPushButton#exclude_button, QPushButton#refloc_button {
                min-width: 33px;
                max-width: 33px;
                padding: 0;
            }
            QPushButton#interp_nondatchan,
            QPushButton#interp_removedchans,
            QPushButton#interp_datchan,
            QPushButton#interp_selectchan,
            QPushButton#interp_uselist {
                min-width: 434px;
                max-width: 434px;
                padding: 0;
            }
            QDialog#pop_interp QPushButton {
                padding: 0;
            }
            QDialog#pop_interp QLineEdit,
            QDialog#pop_interp QComboBox {
                min-width: 198px;
                max-width: 198px;
            }
            QDialog#pop_reref QLineEdit {
                min-width: 163px;
                max-width: 163px;
            }
            QDialog#pop_runica QListWidget {
                min-height: 102px;
                max-height: 102px;
            }
            QCheckBox {
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                background: white;
                border: 1px solid #7f7f7f;
            }
            """
        )

    @staticmethod
    def _row_weights(row_geometry: Any) -> list[float]:
        if isinstance(row_geometry, (list, tuple)):
            return [max(0.01, float(value)) for value in row_geometry]
        return [1.0] * max(1, int(row_geometry))

    @staticmethod
    def _row_stretch(spec: DialogSpec, row_index: int) -> int:
        if spec.geomvert is None:
            return 0
        value = spec.geomvert[min(row_index, len(spec.geomvert) - 1)]
        return max(1, round(float(value) * 100))

    @staticmethod
    def _add_buttons(
        QtWidgets: Any,
        layout: Any,
        spec: DialogSpec,
        dialog: Any,
        widgets: dict[str, Any],
    ) -> None:
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 18, 0, 0)
        button_layout.setSpacing(16)
        if spec.help_text:
            help_button = QtWidgets.QPushButton("Help")
            help_button.setObjectName("help")
            help_button.setFixedWidth(80)
            help_button.clicked.connect(lambda: QtDialogRenderer._show_help(QtWidgets, dialog, spec))
            button_layout.addWidget(help_button)
        button_layout.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel")
        ok_button = QtWidgets.QPushButton("OK")
        cancel_button.setObjectName("cancel")
        ok_button.setObjectName("ok")
        cancel_button.setFixedWidth(80)
        ok_button.setFixedWidth(80)
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(lambda: QtDialogRenderer._accept_if_valid(dialog, spec, widgets))
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addWidget(button_container)

    @staticmethod
    def _apply_spec_size(dialog: Any, spec: DialogSpec) -> None:
        if spec.size is None:
            return
        dialog.resize(*spec.size)

    def _build_widget(self, QtWidgets: Any, control: ControlSpec, initial_values: Mapping[str, Any]) -> Any:
        style = control.style.lower()
        value = initial_values.get(control.tag, control.value)
        if style == "text":
            widget = QtWidgets.QLabel(control.string)
        elif style == "edit":
            widget = QtWidgets.QLineEdit("" if value is None else str(value))
        elif style == "pushbutton":
            widget = QtWidgets.QPushButton(control.string)
        elif style == "checkbox":
            widget = QtWidgets.QCheckBox(control.string)
            widget.setChecked(bool(value))
        elif style == "popupmenu":
            widget = QtWidgets.QComboBox()
            widget.addItems([item.strip() for item in control.string.split("|")])
            try:
                index = int(value) - 1
            except (TypeError, ValueError):
                index = 0
            if 0 <= index < widget.count():
                widget.setCurrentIndex(index)
        elif style == "listbox":
            widget = QtWidgets.QListWidget()
            widget.addItems([item.strip() for item in control.string.split("|")])
            if _is_sequence_value(value):
                widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
                widget.setProperty(_MULTI_SELECT_PROPERTY, True)
                selected_rows = [int(item) - 1 for item in value]
                if selected_rows and 0 <= selected_rows[0] < widget.count():
                    widget.setCurrentRow(selected_rows[0])
                for row in selected_rows:
                    if 0 <= row < widget.count():
                        widget.item(row).setSelected(True)
            else:
                try:
                    index = int(value) - 1
                except (TypeError, ValueError):
                    index = 0
                if 0 <= index < widget.count():
                    widget.setCurrentRow(index)
        elif style == "spacer":
            widget = QtWidgets.QWidget()
        else:
            raise ValueError(f"Unsupported GUI control style: {control.style}")

        if control.tag:
            widget.setObjectName(control.tag)
        if control.tooltip:
            widget.setToolTip(control.tooltip)
        self._apply_widget_size_policy(QtWidgets, widget, style)
        widget.setEnabled(control.enabled)
        return widget

    @staticmethod
    def _apply_widget_size_policy(QtWidgets: Any, widget: Any, style: str) -> None:
        policy = QtWidgets.QSizePolicy
        if style in {"edit", "popupmenu", "listbox", "pushbutton"}:
            widget.setSizePolicy(policy.Expanding, policy.Fixed)
            return
        if style in {"text", "checkbox"}:
            widget.setMinimumWidth(0)
            widget.setSizePolicy(policy.Expanding, policy.Fixed)

    def _connect_callback(self, callback: CallbackSpec | None, widgets: dict[str, Any]) -> None:
        if callback is None:
            return
        params = callback.params
        if callback.name == "sync_time_to_samples":
            source = widgets[params["source"]]
            target = widgets[params["target"]]
            multiplier = float(params["srate"])
            source.editingFinished.connect(lambda: self._sync_numeric(source, target, multiplier))
        elif callback.name == "sync_samples_to_time":
            source = widgets[params["source"]]
            target = widgets[params["target"]]
            multiplier = 1.0 / float(params["srate"])
            source.editingFinished.connect(lambda: self._sync_numeric(source, target, multiplier))
        elif callback.name == "select_event_types":
            button = widgets[params["button"]]
            target = widgets[params["target"]]
            button.clicked.connect(lambda: self._select_event_types(button, target, params))
        elif callback.name == "select_channels":
            button = widgets[params["button"]]
            target = widgets[params["target"]]
            button.clicked.connect(lambda: self._select_channels(button, target, params))
        elif callback.name == "set_reref_mode":
            source = widgets[params["source"]]
            source.toggled.connect(lambda checked: self._set_reref_mode(widgets, params["mode"], checked))
        elif callback.name == "toggle_enabled":
            source = widgets[params["source"]]
            targets = [widgets[tag] for tag in params["targets"] if tag in widgets]
            source.toggled.connect(lambda checked: self._set_enabled(targets, checked))
            self._set_enabled(targets, source.isChecked())
        elif callback.name == "select_interp_channels":
            button = widgets[params["button"]]
            target = widgets[params["target"]]
            button.clicked.connect(lambda: self._select_interp_channels(button, target, params))

    @staticmethod
    def _accept_if_valid(dialog: Any, spec: DialogSpec, widgets: dict[str, Any]) -> None:
        message = QtDialogRenderer._validation_message(spec, widgets)
        if message:
            _qt_core, qt_widgets = _require_qt()
            qt_widgets.QMessageBox.warning(dialog, "Warning", message)
            return
        dialog.accept()

    @staticmethod
    def _validation_message(spec: DialogSpec, widgets: dict[str, Any]) -> str | None:
        if spec.function_name == "pop_reref":
            return QtDialogRenderer._validate_pop_reref_dialog(spec, widgets)
        if spec.function_name == "pop_interp":
            return QtDialogRenderer._validate_pop_interp_dialog(spec, widgets)
        if spec.function_name == "pop_resample":
            text = QtDialogRenderer._widget_text(widgets.get("freq")).strip()
            if not text:
                return "New sampling rate is required"
            try:
                value = float(text)
            except ValueError:
                return "New sampling rate must be numeric"
            if value <= 0:
                return "New sampling rate must be positive"
        if spec.function_name == "pop_runica" and "dataset" in widgets:
            if not QtDialogRenderer._read_widget(widgets["dataset"]):
                return "Select at least one dataset"
        return None

    @staticmethod
    def _validate_pop_reref_dialog(spec: DialogSpec, widgets: dict[str, Any]) -> str | None:
        if QtDialogRenderer._widget_checked(widgets.get("huberef")):
            huber_text = QtDialogRenderer._widget_text(widgets.get("huberval")).strip()
            if huber_text:
                try:
                    float(huber_text)
                except ValueError:
                    return f"could not convert string to float: '{huber_text}'"

        channel_labels = QtDialogRenderer._callback_channels(spec, "refbr")
        if QtDialogRenderer._widget_checked(widgets.get("rerefstr")):
            ref_text = QtDialogRenderer._widget_text(widgets.get("reref")).strip()
            if not ref_text:
                return "Aborting: you must enter one or more reference channels"
            message = QtDialogRenderer._validate_channel_text(ref_text, channel_labels, "Channel")
            if message:
                return message

        exclude_text = QtDialogRenderer._widget_text(widgets.get("exclude")).strip()
        if exclude_text:
            message = QtDialogRenderer._validate_channel_text(exclude_text, channel_labels, "Channel")
            if message:
                return message

        refloc_text = QtDialogRenderer._widget_text(widgets.get("refloc")).strip()
        if refloc_text:
            refloc_labels = QtDialogRenderer._callback_channels(spec, "refloc_button")
            return QtDialogRenderer._validate_channel_text(refloc_text, refloc_labels, "Reference location")
        return None

    @staticmethod
    def _validate_pop_interp_dialog(spec: DialogSpec, widgets: dict[str, Any]) -> str | None:
        for control in spec.controls:
            if control.callback is None or control.callback.name != "validate_numeric_range":
                continue
            widget = widgets.get(control.tag or "")
            text = QtDialogRenderer._widget_text(widget).strip()
            if not text:
                continue
            params = control.callback.params
            try:
                values = QtDialogRenderer._parse_numeric_text(text)
            except ValueError:
                return "Time/point range must contain numeric values"
            if len(values) != int(params.get("columns", 2)):
                return "Time/point range must contain 2 columns exactly"
            if min(values) < float(params["lower"]):
                return "Time/point range exceed lower data limits"
            if math.floor(max(values)) > float(params["upper"]):
                return "Time/point range exceed upper data limits"
        return None

    @staticmethod
    def _callback_channels(spec: DialogSpec, tag: str) -> tuple[str, ...]:
        for control in spec.controls:
            if control.tag == tag and control.callback is not None:
                return tuple(str(value) for value in control.callback.params.get("channels", ()))
        return ()

    @staticmethod
    def _validate_channel_text(text: str, labels: tuple[str, ...], label: str) -> str | None:
        values = QtDialogRenderer._parse_channel_text(text)
        lower_labels = [value.lower() for value in labels]
        for value in values:
            if QtDialogRenderer._is_int_text(value):
                index = int(value)
                if index < 0 or index >= len(labels):
                    return f"{label} index out of range"
                continue
            if value.lower() not in lower_labels:
                return f"{label} '{value}' not found"
        return None

    @staticmethod
    def _parse_channel_text(text: str) -> list[str]:
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        tokens = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", text)
        return [next(part for part in token if part) for token in tokens]

    @staticmethod
    def _parse_numeric_text(text: str) -> list[float]:
        cleaned = text.strip().strip("[]")
        if not cleaned:
            return []
        return [float(value) for value in re.split(r"[\s,]+", cleaned) if value]

    @staticmethod
    def _is_int_text(value: str) -> bool:
        return bool(re.fullmatch(r"[+-]?\d+", value.strip()))

    @staticmethod
    def _widget_checked(widget: Any) -> bool:
        return bool(widget is not None and hasattr(widget, "isChecked") and widget.isChecked())

    @staticmethod
    def _widget_text(widget: Any) -> str:
        if widget is None or not hasattr(widget, "text"):
            return ""
        return str(widget.text())

    @staticmethod
    def _sync_numeric(source: Any, target: Any, multiplier: float) -> None:
        text = source.text().strip()
        if not text:
            target.setText("")
            return
        try:
            value = float(text) * multiplier
        except ValueError:
            return
        target.setText(f"{value:g}")

    @staticmethod
    def _select_event_types(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        event_types = [str(value) for value in params.get("event_types", ())]
        if not event_types:
            return
        current = target.text().strip()
        _qt_core, qt_widgets = _require_qt()
        value, accepted = qt_widgets.QInputDialog.getItem(
            button,
            "Select event type",
            "Event type",
            event_types,
            0,
            editable=False,
        )
        if accepted and value:
            target.setText((current + " " + value).strip())

    @staticmethod
    def _select_channels(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        channels = [str(value) for value in params.get("channels", ())]
        if channels:
            _chanlist, value, _allchanstr = pop_chansel(
                channels,
                withindex="on",
                select=target.text().strip(),
                selectionmode=str(params.get("selectionmode", "multiple")),
                parent=button,
            )
            accepted = bool(value)
        else:
            _qt_core, qt_widgets = _require_qt()
            no_channels_message = str(params.get("no_channels_message", "")).strip()
            if no_channels_message:
                qt_widgets.QMessageBox.warning(button, "Warning", no_channels_message)
                return
            value, accepted = qt_widgets.QInputDialog.getText(
                button,
                "Select channel",
                "Channel index or label",
            )
        if not accepted or not value:
            return
        target.setText(value.strip())

    @staticmethod
    def _select_interp_channels(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        source = str(params.get("source", "")).lower()
        chanlocs = [dict(chan) for chan in params.get("chanlocs", ())]
        removedchans = [dict(chan) for chan in params.get("removedchans", ())]
        alleeg = [dict(eeg) for eeg in params.get("alleeg", ())]

        if source in {"removedchans", "nondatchan"}:
            labels = [str(chan.get("labels", "")) for chan in removedchans]
            chanlist, chanliststr, _allchanstr = pop_chansel(labels, parent=button)
            if not chanlist:
                return
            selected = [removedchans[index - 1] for index in chanlist]
            chanstr = "EEG.chaninfo.removedchans([" + " ".join(str(index) for index in chanlist) + "])"
            QtDialogRenderer._store_interp_selection(target, selected, chanstr, chanliststr)
            return

        if source == "datchan":
            labels = [str(chan.get("labels", "")) for chan in chanlocs]
            chanlist, chanliststr, _allchanstr = pop_chansel(labels, parent=button)
            if not chanlist:
                return
            selected = [index - 1 for index in chanlist]
            chanstr = "[" + " ".join(str(index) for index in chanlist) + "]"
            QtDialogRenderer._store_interp_selection(target, selected, chanstr, chanliststr)
            return

        _qt_core, qt_widgets = _require_qt()
        dataset_index, accepted = qt_widgets.QInputDialog.getInt(
            button,
            "Choose dataset",
            "Dataset index",
            1,
            1,
            max(1, len(alleeg)),
        )
        if not accepted:
            return
        if dataset_index < 1 or dataset_index > len(alleeg):
            qt_widgets.QMessageBox.warning(button, "Warning", "Wrong index")
            return

        other = alleeg[dataset_index - 1]
        other_chanlocs = [dict(chan) for chan in other.get("chanlocs", ())]
        if source == "selectchan":
            labels = [str(chan.get("labels", "")) for chan in other_chanlocs]
            chanlist, _chanliststr, _allchanstr = pop_chansel(labels, parent=button)
        else:
            chanlist = list(range(1, len(other_chanlocs) + 1))
        if not chanlist:
            return

        current_labels = {str(chan.get("labels", "")).lower() for chan in chanlocs}
        selected_indices = [
            index for index in chanlist
            if str(other_chanlocs[index - 1].get("labels", "")).lower() not in current_labels
        ]
        if not selected_indices:
            qt_widgets.QMessageBox.warning(button, "Warning", "No new channels selected")
            return

        if len(chanlist) == len(other_chanlocs):
            selected = other_chanlocs
            chanstr = f"ALLEEG({dataset_index}).chanlocs"
        else:
            selected_indices = sorted(selected_indices)
            selected = [other_chanlocs[index - 1] for index in selected_indices]
            chanstr = f"ALLEEG({dataset_index}).chanlocs([" + " ".join(str(index) for index in selected_indices) + "])"
        display = " ".join(str(other_chanlocs[index - 1].get("labels", "")) for index in selected_indices)
        QtDialogRenderer._store_interp_selection(target, selected, chanstr, display)

    @staticmethod
    def _store_interp_selection(target: Any, chans: Any, chanstr: str, display: str) -> None:
        target.setProperty(_VALUE_PROPERTY, {"chans": chans, "chanstr": chanstr})
        target.setText(display.strip())

    @staticmethod
    def _set_reref_mode(widgets: dict[str, Any], mode: str, checked: bool) -> None:
        if not checked:
            if mode == "channels" and "ave" in widgets:
                widgets["ave"].setChecked(True)
            return

        average_mode = mode in {"average", "huber"}
        for tag in ("ave", "huberef", "rerefstr"):
            if tag in widgets:
                widgets[tag].blockSignals(True)
                widgets[tag].setChecked(
                    (tag == "ave" and mode == "average")
                    or (tag == "huberef" and mode == "huber")
                    or (tag == "rerefstr" and mode == "channels")
                )
                widgets[tag].blockSignals(False)

        for tag in ("reref", "refbr", "keepref"):
            if tag in widgets:
                widgets[tag].setEnabled(not average_mode)
        if average_mode and "keepref" in widgets:
            widgets["keepref"].setChecked(False)

    @staticmethod
    def _set_enabled(widgets: list[Any], enabled: bool) -> None:
        for widget in widgets:
            widget.setEnabled(enabled)

    @staticmethod
    def _show_help(_qt_widgets: Any, dialog: Any, spec: DialogSpec) -> None:
        dialog._eegprep_help_dialog = pophelp(spec.help_text or spec.function_name, parent=dialog)

    @staticmethod
    def _read_widget(widget: Any) -> Any:
        stored_value = widget.property(_VALUE_PROPERTY)
        if stored_value is not None:
            return stored_value
        if hasattr(widget, "isChecked"):
            return widget.isChecked()
        if widget.property(_MULTI_SELECT_PROPERTY) and hasattr(widget, "selectedIndexes"):
            return sorted({index.row() + 1 for index in widget.selectedIndexes()})
        if hasattr(widget, "currentRow"):
            return widget.currentRow() + 1
        if hasattr(widget, "currentIndex"):
            return widget.currentIndex() + 1
        if hasattr(widget, "text"):
            return widget.text()
        return None


def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, set))
