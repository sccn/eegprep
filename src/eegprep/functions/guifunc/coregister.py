"""Qt implementation of EEGLAB's manual ``coregister`` editor."""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.popfunc.pop_chansel import pop_chansel
from eegprep.functions.sigprocfunc.coregister import (
    ElectrodeSet,
    apply_coregistration_transform,
    electrode_subset_indices,
    estimate_coregistration_transform,
    is_fiducial_label,
    load_coregistration_electrodes,
    normalise_coregistration_transform,
)
from eegprep.functions.sigprocfunc.headplot import HeadplotMesh, load_headplot_mesh

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtWidgets import QDialog
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:  # pragma: no cover - optional GUI dependency
    QtCore = None
    QtWidgets = None
    QDialog = None
    FigureCanvas = None

_BaseDialog = QDialog if QDialog is not None else object
_EEGLAB_BLUE = "#000066"
_EEGLAB_BG = "#a8c2ff"
_PLOT_BG = "#edf5ff"
_SOURCE_COLOR = "#00a000"
_REFERENCE_COLOR = "#cc9985"
_REFERENCE_SUBSETS = ("21 elec (10/20 system)", "86 elec (10/10 system)", "all elec (10/5 system)")


def run_coregister_dialog(
    chanlocs: Any,
    reference: Any,
    *,
    chaninfo: dict[str, Any] | None = None,
    meshfile: str | None = None,
    transform: Any = None,
    parent: Any = None,
    title: str = "Co-registration plot for headplot mesh",
) -> list[float] | None:
    """Open the manual coregistration editor and return the chosen transform."""
    if QtWidgets is None or QDialog is None or FigureCanvas is None:
        raise RuntimeError(
            "PySide6 and the Qt Matplotlib backend are required for manual coregistration. "
            "Install EEGPrep with the GUI extra."
        )
    app, dialog = build_coregister_dialog(
        chanlocs,
        reference,
        chaninfo=chaninfo,
        meshfile=meshfile,
        transform=transform,
        parent=parent,
        title=title,
    )
    app.processEvents()
    if int(dialog.exec()) != 1:
        return None
    return dialog.transform.tolist()


def build_coregister_dialog(
    chanlocs: Any,
    reference: Any,
    *,
    chaninfo: dict[str, Any] | None = None,
    meshfile: str | None = None,
    transform: Any = None,
    parent: Any = None,
    title: str = "Co-registration plot for headplot mesh",
) -> tuple[Any, "CoregisterDialog"]:
    """Build the manual coregistration dialog without executing it."""
    if QtWidgets is None or QDialog is None or FigureCanvas is None:
        raise RuntimeError(
            "PySide6 and the Qt Matplotlib backend are required for manual coregistration. "
            "Install EEGPrep with the GUI extra."
        )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = CoregisterDialog(
        chanlocs,
        reference,
        chaninfo=chaninfo,
        meshfile=meshfile,
        transform=transform,
        parent=parent,
        title=title,
    )
    return app, dialog


class CoregisterDialog(_BaseDialog):
    """Interactive editor for EEGLAB-style 9-parameter coregistration."""

    def __init__(
        self,
        chanlocs: Any,
        reference: Any,
        *,
        chaninfo: dict[str, Any] | None,
        meshfile: str | None,
        transform: Any,
        parent: Any,
        title: str,
    ) -> None:
        if QtWidgets is None or QtCore is None or FigureCanvas is None:
            raise RuntimeError("PySide6 is required for manual coregistration")
        super().__init__(parent)
        self.setObjectName("coregister")
        self.setWindowTitle("coregister()")
        self.source = load_coregistration_electrodes(chanlocs, chaninfo=chaninfo)
        self.reference = load_coregistration_electrodes(reference) if reference else None
        self.mesh = load_headplot_mesh(meshfile) if meshfile else None
        self.transform = normalise_coregistration_transform(transform, default=[0, 0, 0, 0, 0, 0, 1, 1, 1])
        self._title = title
        self._show_source_labels = False
        self._show_reference_labels = False
        self._mesh_visible = True
        self._source_indices = np.arange(len(self.source.labels), dtype=int)
        self._reference_indices = (
            electrode_subset_indices(self.reference, _REFERENCE_SUBSETS[-1])
            if self.reference is not None
            else np.asarray([], dtype=int)
        )
        self._field_tags = ("right", "forward", "up", "pitch", "roll", "yaw", "resizex", "resizey", "resizez")
        self._fields: dict[str, Any] = {}
        self.figure = Figure(figsize=(8.6, 6.0), facecolor=_PLOT_BG)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self._build_ui()
        self._sync_fields()
        self._redraw()

    def _build_ui(self) -> None:
        if QtWidgets is None:
            return
        self.resize(980, 720)
        self.setStyleSheet(
            f"""
            QDialog {{
                background: {_EEGLAB_BG};
                color: {_EEGLAB_BLUE};
                font-size: 13px;
            }}
            QLabel, QPushButton, QLineEdit {{
                color: {_EEGLAB_BLUE};
                font-size: 13px;
            }}
            QPushButton {{
                background: #eeeeee;
                border: 1px solid #777777;
                min-height: 22px;
                padding: 1px 8px;
            }}
            QLineEdit {{
                background: white;
                border: 1px solid #777777;
                min-height: 21px;
                padding: 0 3px;
            }}
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(self._plot_row(), 1)
        layout.addLayout(self._transform_grid())

    def _plot_row(self) -> Any:
        if QtWidgets is None:
            return None
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addLayout(self._left_button_column())
        row.addWidget(self.canvas, 1)
        row.addLayout(self._right_button_column())
        return row

    def _left_button_column(self) -> Any:
        if QtWidgets is None:
            return None
        column = QtWidgets.QVBoxLayout()
        column.setSpacing(2)
        source_labels = QtWidgets.QPushButton("Labels on")
        source_labels.setStyleSheet(f"background: {_SOURCE_COLOR}; color: white;")
        source_labels.clicked.connect(lambda: self._toggle_labels(source_labels, source=True))
        source_labels.setFixedWidth(130)
        column.addWidget(source_labels)
        source_electrodes = QtWidgets.QPushButton("Electrodes")
        source_electrodes.setStyleSheet(f"background: {_SOURCE_COLOR}; color: white;")
        source_electrodes.clicked.connect(self._select_source_electrodes)
        source_electrodes.setFixedWidth(130)
        column.addWidget(source_electrodes)

        reference_labels = QtWidgets.QPushButton("Labels on")
        reference_labels.setStyleSheet(f"background: {_REFERENCE_COLOR}; color: black;")
        reference_labels.setEnabled(self.reference is not None)
        reference_labels.clicked.connect(lambda: self._toggle_labels(reference_labels, source=False))
        reference_labels.setFixedWidth(130)
        column.addWidget(reference_labels)
        reference_electrodes = QtWidgets.QPushButton("Electrodes")
        reference_electrodes.setStyleSheet(f"background: {_REFERENCE_COLOR}; color: black;")
        reference_electrodes.setEnabled(self.reference is not None)
        reference_electrodes.clicked.connect(self._select_reference_electrodes)
        reference_electrodes.setFixedWidth(130)
        column.addWidget(reference_electrodes)

        mesh_button = QtWidgets.QPushButton("Mesh off")
        mesh_button.clicked.connect(lambda: self._toggle_mesh(mesh_button))
        mesh_button.setFixedWidth(130)
        column.addWidget(mesh_button)
        column.addStretch(1)
        return column

    def _right_button_column(self) -> Any:
        if QtWidgets is None:
            return None
        column = QtWidgets.QVBoxLayout()
        column.setSpacing(2)
        help_button = QtWidgets.QPushButton("Help me")
        help_button.clicked.connect(self._show_help_message)
        help_button.setFixedWidth(130)
        column.addWidget(help_button)
        function_help = QtWidgets.QPushButton("Funct. help")
        function_help.clicked.connect(lambda: pophelp("coregister", parent=self))
        function_help.setFixedWidth(130)
        column.addWidget(function_help)
        column.addStretch(1)
        return column

    def _transform_grid(self) -> Any:
        if QtWidgets is None:
            return None
        grid = QtWidgets.QGridLayout()
        labels = (
            ("Move right {mm}", "right", 0, 0),
            ("Move front {mm}", "forward", 1, 0),
            ("Move up {mm}", "up", 2, 0),
            ("Pitch (rad)", "pitch", 0, 2),
            ("Roll (rad)", "roll", 1, 2),
            ("Yaw (rad)", "yaw", 2, 2),
            ("Resize {x}", "resizex", 0, 4),
            ("Resize {y}", "resizey", 1, 4),
            ("Resize {z}", "resizez", 2, 4),
        )
        for label, tag, row, column in labels:
            grid.addWidget(QtWidgets.QLabel(label), row, column)
            field = QtWidgets.QLineEdit()
            field.setObjectName(tag)
            field.editingFinished.connect(self._fields_changed)
            self._fields[tag] = field
            grid.addWidget(field, row, column + 1)

        align_button = QtWidgets.QPushButton("Align montages")
        align_button.clicked.connect(lambda: self._fit_transform("globalrescale"))
        grid.addWidget(align_button, 0, 6)
        warp_button = QtWidgets.QPushButton("Warp montage")
        warp_button.clicked.connect(lambda: self._fit_transform("traditional"))
        grid.addWidget(warp_button, 1, 6)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        grid.addWidget(cancel_button, 2, 6)
        ok_button = QtWidgets.QPushButton("Ok")
        ok_button.setObjectName("ok")
        ok_button.clicked.connect(self.accept)
        grid.addWidget(ok_button, 2, 7)
        return grid

    def _sync_fields(self) -> None:
        for tag, value in zip(self._field_tags, self.transform, strict=False):
            self._fields[tag].blockSignals(True)
            self._fields[tag].setText(f"{value:.6g}")
            self._fields[tag].blockSignals(False)

    def _fields_changed(self) -> None:
        try:
            self.transform = normalise_coregistration_transform([self._fields[tag].text() for tag in self._field_tags])
        except ValueError as exc:
            if QtWidgets is not None:
                QtWidgets.QMessageBox.warning(self, "Warning", str(exc))
            self._sync_fields()
            return
        self._redraw()

    def _toggle_mesh(self, button: Any) -> None:
        self._mesh_visible = not self._mesh_visible
        button.setText("Mesh off" if self._mesh_visible else "Mesh on")
        self._redraw()

    def _toggle_labels(self, button: Any, *, source: bool) -> None:
        if source:
            self._show_source_labels = not self._show_source_labels
            button.setText("Labels off" if self._show_source_labels else "Labels on")
        else:
            self._show_reference_labels = not self._show_reference_labels
            button.setText("Labels off" if self._show_reference_labels else "Labels on")
        self._redraw()

    def _select_source_electrodes(self) -> None:
        indices, _value, _all = pop_chansel(
            self.source.labels,
            withindex="off",
            select=[int(index) + 1 for index in self._source_indices],
            parent=self,
        )
        if indices:
            self._source_indices = np.asarray(indices, dtype=int) - 1
            self._redraw()

    def _select_reference_electrodes(self) -> None:
        if self.reference is None or QtWidgets is None:
            return
        value, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Reference electrodes",
            "show only",
            list(_REFERENCE_SUBSETS),
            2,
            editable=False,
        )
        if accepted and value:
            self._reference_indices = electrode_subset_indices(self.reference, value)
            self._redraw()

    def _fit_transform(self, method: str) -> None:
        if self.reference is None or QtWidgets is None:
            return
        try:
            self.transform = estimate_coregistration_transform(
                self.source,
                self.reference,
                initial=self.transform,
                method=method,
            )
        except (NotImplementedError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Warning", str(exc))
            return
        self._sync_fields()
        self._redraw()

    def _show_help_message(self) -> None:
        if QtWidgets is None:
            return
        QtWidgets.QMessageBox.information(
            self,
            "Warning",
            "User channels are shown in green and reference channels in brown.\n\n"
            "Use Align montages to fit a shared-scale transform to common labels, "
            "or Warp montage to fit the EEGLAB traditional 9-parameter transform. "
            "You can then fine-tune the values in the transform boxes and press Ok.",
        )

    def _redraw(self) -> None:
        self.axes.clear()
        self.axes.set_facecolor(_PLOT_BG)
        source_points = apply_coregistration_transform(self.source.points, self.transform)
        plotted: list[np.ndarray] = []
        if self.mesh is not None and self._mesh_visible:
            self._plot_mesh(self.mesh)
            plotted.append(self.mesh.vertices)
        self._plot_electrodes(
            ElectrodeSet(self.source.labels, source_points, self.source.source),
            self._source_indices,
            _SOURCE_COLOR,
            labels=self._show_source_labels,
        )
        plotted.append(source_points[self._source_indices])
        if self.reference is not None:
            self._plot_electrodes(
                self.reference,
                self._reference_indices,
                _REFERENCE_COLOR,
                labels=self._show_reference_labels,
            )
            plotted.append(self.reference.points[self._reference_indices])
        self._plot_axes(plotted)
        self.axes.set_axis_off()
        self.axes.set_box_aspect((1, 1, 1))
        self.canvas.draw_idle()

    def _plot_mesh(self, mesh: HeadplotMesh) -> None:
        collection = Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolor=(1.0, 0.75, 0.65, 0.22),
            edgecolor=(0.55, 0.55, 0.55, 0.18),
            linewidths=0.2,
        )
        self.axes.add_collection3d(collection)

    def _plot_electrodes(self, electrodes: ElectrodeSet, indices: np.ndarray, color: str, *, labels: bool) -> None:
        if indices.size == 0:
            return
        points = electrodes.points[indices]
        sizes = [70 if is_fiducial_label(electrodes.labels[int(index)]) else 32 for index in indices]
        self.axes.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=sizes, depthshade=False)
        if not labels:
            return
        for point, index in zip(points, indices, strict=False):
            self.axes.text(
                point[0] * 1.03,
                point[1] * 1.03,
                point[2] * 1.03,
                electrodes.labels[int(index)],
                color=color,
                fontsize=8,
            )

    def _plot_axes(self, point_groups: list[np.ndarray]) -> None:
        all_points = [points for points in point_groups if points.size]
        if not all_points:
            all_points = [np.zeros((1, 3))]
        points = np.vstack(all_points)
        center = np.nanmean(points, axis=0)
        span = float(np.nanmax(np.ptp(points, axis=0))) / 2
        span = max(span, 1.0)
        lim = max(float(np.nanmax(np.abs(points))), span)
        self.axes.plot([0, lim], [0, 0], [0, 0], "b--", linewidth=0.8)
        self.axes.plot([0, 0], [0, lim], [0, 0], "g--", linewidth=0.8)
        self.axes.plot([0, 0], [0, 0], [0, lim], "r--", linewidth=0.8)
        self.axes.plot([0.08 * lim, 0.12 * lim], [0, 0], [0, 0], "r", linewidth=3)
        self.axes.text(lim * 1.05, 0, 0, "X", ha="center", va="center")
        self.axes.text(0, lim * 1.05, 0, "Y", ha="center", va="center")
        self.axes.text(0, 0, lim * 1.05, "Z", ha="center", va="center")
        self.axes.set_xlim(center[0] - span, center[0] + span)
        self.axes.set_ylim(center[1] - span, center[1] + span)
        self.axes.set_zlim(center[2] - span, center[2] + span)


def prepare_coregister_display(*args: Any, **kwargs: Any) -> tuple[Any, CoregisterDialog]:
    """Build the dialog for visual capture without executing it."""
    if matplotlib.get_backend().lower() == "agg":
        matplotlib.use("qtagg", force=True)
    return build_coregister_dialog(*args, **kwargs)


__all__ = ["CoregisterDialog", "build_coregister_dialog", "prepare_coregister_display", "run_coregister_dialog"]
