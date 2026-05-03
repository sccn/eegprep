"""Optional PySide6 renderer for EEGLAB-like dialog specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .spec import CallbackSpec, ControlSpec, DialogSpec


class QtDialogRenderer:
    """Render :class:`DialogSpec` using PySide6 widgets."""

    def run(
        self,
        spec: DialogSpec,
        initial_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        try:
            from PySide6 import QtCore, QtWidgets
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "PySide6 is required for EEGPrep GUI dialogs. Install it with "
                "`pip install -e .[gui]` or `pip install eegprep[gui]`."
            ) from exc

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(spec.title)
        self._apply_eeglab_style(dialog)
        layout = QtWidgets.QGridLayout(dialog)
        layout.setContentsMargins(42, 17, 42, 13)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        initial_values = initial_values or {}
        widgets: dict[str, Any] = {}
        index = 0
        for row, row_geometry in enumerate(spec.geometry):
            row_width = self._row_width(spec.geometry, row)
            for column in range(row_width):
                if index >= len(spec.controls):
                    break
                control = spec.controls[index]
                widget = self._build_widget(QtWidgets, control, initial_values)
                if control.tag:
                    widgets[control.tag] = widget
                column_span = 1
                if isinstance(row_geometry, (int, float)) and row_geometry == 1 and row_width == 1:
                    column_span = max(1, len(spec.geometry[0]) if spec.geometry else 1)
                self._add_widget(QtCore, layout, widget, control, row, column, column_span)
                index += 1

        self._apply_column_stretches(layout, spec)
        for control in spec.controls:
            self._connect_callback(control.callback, widgets)

        self._add_buttons(QtWidgets, layout, spec, dialog)

        dialog.adjustSize()
        self._apply_spec_size(dialog, spec)
        app.processEvents()
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None

        return {tag: self._read_widget(widget) for tag, widget in widgets.items()}

    @staticmethod
    def _row_width(geometry: tuple[Any, ...], row: int) -> int:
        value = geometry[min(row, len(geometry) - 1)]
        if isinstance(value, (list, tuple)):
            return max(1, len(value))
        return max(1, int(value))

    def build_dialog(
        self,
        spec: DialogSpec,
        initial_values: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Build a dialog without executing it, for screenshot capture."""
        try:
            from PySide6 import QtCore, QtWidgets
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "PySide6 is required for EEGPrep GUI dialogs. Install it with "
                "`pip install -e .[gui]` or `pip install eegprep[gui]`."
            ) from exc

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(spec.title)
        self._apply_eeglab_style(dialog)
        layout = QtWidgets.QGridLayout(dialog)
        layout.setContentsMargins(42, 17, 42, 13)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        initial_values = initial_values or {}
        widgets: dict[str, Any] = {}
        index = 0
        for row, row_geometry in enumerate(spec.geometry):
            row_width = self._row_width(spec.geometry, row)
            for column in range(row_width):
                if index >= len(spec.controls):
                    break
                control = spec.controls[index]
                widget = self._build_widget(QtWidgets, control, initial_values)
                if control.tag:
                    widgets[control.tag] = widget
                column_span = 1
                if isinstance(row_geometry, (int, float)) and row_geometry == 1 and row_width == 1:
                    column_span = max(1, self._row_width(spec.geometry, 0))
                self._add_widget(QtCore, layout, widget, control, row, column, column_span)
                index += 1

        self._apply_column_stretches(layout, spec)
        for control in spec.controls:
            self._connect_callback(control.callback, widgets)

        self._add_buttons(QtWidgets, layout, spec, dialog)
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
            QLabel, QCheckBox, QPushButton, QLineEdit {
                font-size: 16px;
            }
            QLabel, QCheckBox {
                color: #000066;
                background: transparent;
            }
            QLineEdit {
                background: white;
                border: 1px solid #7f7f7f;
                min-width: 215px;
                max-width: 215px;
                min-height: 16px;
                max-height: 16px;
                padding: 0 3px;
            }
            QPushButton {
                background: #eeeeee;
                border: 1px solid #7f7f7f;
                min-width: 81px;
                max-width: 81px;
                min-height: 20px;
                max-height: 20px;
                padding: 0 10px;
                color: #000066;
            }
            QPushButton#events_button {
                min-width: 132px;
                max-width: 132px;
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
    def _apply_column_stretches(layout: Any, spec: DialogSpec) -> None:
        first_row = spec.geometry[0] if spec.geometry else ()
        if not isinstance(first_row, (list, tuple)):
            return
        for column, stretch in enumerate(first_row):
            layout.setColumnStretch(column, max(1, int(float(stretch) * 100)))

    @staticmethod
    def _add_buttons(QtWidgets: Any, layout: Any, spec: DialogSpec, dialog: Any) -> None:
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 17, 0, 0)
        button_layout.setSpacing(16)
        button_layout.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel")
        ok_button = QtWidgets.QPushButton("OK")
        cancel_button.setObjectName("cancel")
        ok_button.setObjectName("ok")
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addWidget(
            button_container,
            len(spec.geometry),
            0,
            1,
            max(1, QtDialogRenderer._row_width(spec.geometry, 0)),
        )

    @staticmethod
    def _add_widget(
        QtCore: Any,
        layout: Any,
        widget: Any,
        control: ControlSpec,
        row: int,
        column: int,
        column_span: int,
    ) -> None:
        if control.style.lower() == "edit":
            layout.addWidget(widget, row, column, 1, column_span, QtCore.Qt.AlignLeft)
            return
        layout.addWidget(widget, row, column, 1, column_span)

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
        elif style == "spacer":
            widget = QtWidgets.QWidget()
        else:
            raise ValueError(f"Unsupported GUI control style: {control.style}")

        if control.tag:
            widget.setObjectName(control.tag)
        if control.tooltip:
            widget.setToolTip(control.tooltip)
        widget.setEnabled(control.enabled)
        return widget

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
        from PySide6 import QtWidgets

        event_types = [str(value) for value in params.get("event_types", ())]
        if not event_types:
            return
        current = target.text().strip()
        value, accepted = QtWidgets.QInputDialog.getItem(
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
    def _read_widget(widget: Any) -> Any:
        if hasattr(widget, "isChecked"):
            return widget.isChecked()
        if hasattr(widget, "text"):
            return widget.text()
        return None
