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
        *,
        initial_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        try:
            from PySide6 import QtWidgets
        except ImportError as exc:
            raise RuntimeError(
                "PySide6 is required for EEGPREP GUI dialogs. "
                "Install it with `pip install -e .[gui]` or `pip install eegprep[gui]`."
            ) from exc

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(spec.title)
        layout = QtWidgets.QGridLayout(dialog)
        widgets: dict[str, Any] = {}
        initial_values = dict(initial_values or {})

        row = 0
        col = 0
        for control in spec.controls:
            widget = self._build_widget(QtWidgets, control, initial_values)
            if control.tag:
                widgets[control.tag] = widget
            layout.addWidget(widget, row, col)
            col += 1
            if self._row_width(spec.geometry, row) <= col:
                row += 1
                col = 0
        for control in spec.controls:
            self._connect_callback(control.callback, widgets)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons, row + 1, 0, 1, max(1, col or self._row_width(spec.geometry, 0)))

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        return {tag: self._read_widget(widget) for tag, widget in widgets.items()}

    @staticmethod
    def _row_width(geometry: tuple[Any, ...], row: int) -> int:
        if not geometry:
            return 1
        row_geom = geometry[min(row, len(geometry) - 1)]
        if isinstance(row_geom, (list, tuple)):
            return max(1, len(row_geom))
        return 1

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
        if callback.name == "sync_time_to_samples":
            source = widgets.get(callback.params["source"])
            target = widgets.get(callback.params["target"])
            srate = float(callback.params["srate"])
            if source is not None and target is not None:
                source.editingFinished.connect(lambda: self._sync_numeric(source, target, srate))
        elif callback.name == "sync_samples_to_time":
            source = widgets.get(callback.params["source"])
            target = widgets.get(callback.params["target"])
            srate = float(callback.params["srate"])
            if source is not None and target is not None:
                source.editingFinished.connect(lambda: self._sync_numeric(source, target, 1.0 / srate))
        elif callback.name == "select_event_types":
            button = widgets.get(callback.params["button"])
            target = widgets.get(callback.params["target"])
            if button is not None and target is not None:
                button.clicked.connect(lambda: self._select_event_types(button, target, callback.params))

    @staticmethod
    def _sync_numeric(source: Any, target: Any, multiplier: float) -> None:
        text = source.text().strip()
        if not text:
            target.setText("")
            return
        try:
            target.setText(f"{float(text) * multiplier:g}")
        except ValueError:
            return

    @staticmethod
    def _select_event_types(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        from PySide6 import QtWidgets

        event_types = params.get("event_types", ())
        text, accepted = QtWidgets.QInputDialog.getItem(
            button,
            "Select event type",
            "Event type",
            [str(event_type) for event_type in event_types],
            editable=False,
        )
        if accepted and text:
            current = target.text().strip()
            target.setText(text if not current else f"{current} {text}")

    @staticmethod
    def _read_widget(widget: Any) -> Any:
        if hasattr(widget, "isChecked"):
            return widget.isChecked()
        if hasattr(widget, "text"):
            return widget.text()
        return None
