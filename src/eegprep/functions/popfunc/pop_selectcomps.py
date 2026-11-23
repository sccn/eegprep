"""Inspect and mark ICA components for rejection."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import component_rejection_flags, copy_eeg, one_based_indices
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot


PLOTS_PER_FIGURE = 35


def pop_selectcomps(
    EEG: dict[str, Any],
    compnum: Any = None,
    fig: Any = None,
    *,
    reject: Any = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: bool = True,
    return_com: bool = False,
):
    """Inspect components and mark them in ``EEG.reject.gcompreject``."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = compnum is None and reject is None
    if gui:
        result = inputgui(pop_selectcomps_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        compnum = result.get("compnum", "")
        if plot and renderer is None:
            out = copy_eeg(EEG)
            plotted = one_based_indices(compnum, limit=_component_count(out), default_all=True)
            if len(plotted) > PLOTS_PER_FIGURE:
                from eegprep.functions.guifunc.questdlg2 import questdlg2

                answer = questdlg2(
                    f"More than {PLOTS_PER_FIGURE} components so this function will pop-up several windows",
                    "Confirmation",
                    "Cancel",
                    "OK",
                    "OK",
                )
                if answer in {None, "", "Cancel"}:
                    return (EEG, "") if return_com else EEG
            flags = _run_component_browser(out, plotted)
            if flags is None:
                return (EEG, "") if return_com else EEG
            if not isinstance(out.get("reject"), dict):
                out["reject"] = {}
            out["reject"]["gcompreject"] = flags
            command = _history_command(plotted, [])
            command += " [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);"
            return (out, command) if return_com else out
    out = copy_eeg(EEG)
    n_components = _component_count(out)
    if n_components == 0:
        raise ValueError("ICA decomposition is required")
    compnum = one_based_indices(compnum, limit=n_components, default_all=True)
    reject_values = one_based_indices(reject, limit=n_components, default_all=False)
    if reject_values:
        flags = component_rejection_flags(out, n_components, create=False).astype(int)
        flags[np.asarray(reject_values, dtype=int) - 1] = 1
        if not isinstance(out.get("reject"), dict):
            out["reject"] = {}
        out["reject"]["gcompreject"] = flags
    if plot and compnum:
        pop_topoplot(out, 0, compnum, "Reject components by map - pop_selectcomps()", [], 0, gui=False)
    command = _history_command(compnum, reject_values)
    return (out, command) if return_com else out


def pop_selectcomps_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like component selection prompt."""
    n_components = _component_count(EEG)
    return DialogSpec(
        title="Reject comp. by map -- pop_selectcomps",
        function_name="pop_selectcomps",
        eeglab_source="functions/popfunc/pop_selectcomps.m",
        help_text="pophelp('pop_selectcomps')",
        size=(530, 266),
        geometry=((1, 1), 1, 1),
        controls=(
            ControlSpec("text", "Components to plot:"),
            ControlSpec("edit", tag="compnum", value=f"1:{n_components}"),
            ControlSpec("spacer"),
            ControlSpec(
                "text",
                "Note: in the next interface, click on buttons to see component properties and label them "
                'for rejection. To actually reject labelled components use "Tools > Remove components" '
                "or use STUDY menus.",
            ),
        ),
    )


def _history_command(compnum: list[int], reject: list[int]) -> str:
    args = [format_history_value(compnum, cell_for_sequence=None)]
    if reject:
        args.append(f"reject={format_history_value(reject, cell_for_sequence=None)}")
    return "EEG = pop_selectcomps(EEG, " + ", ".join(args) + ");"


def _component_count(EEG: dict[str, Any]) -> int:
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0


def _run_component_browser(EEG: dict[str, Any], compnum: list[int]) -> np.ndarray | None:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            "pop_selectcomps: the component map window needs PySide6. Install it with pip install eegprep[gui]."
        ) from exc

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    count = _component_count(EEG)
    flags = component_rejection_flags(EEG, count, create=True).astype(int).copy()
    accepted_any = False
    for start in range(0, len(compnum), PLOTS_PER_FIGURE):
        page = compnum[start : start + PLOTS_PER_FIGURE]
        if not _run_component_page(EEG, page, flags, app, QtCore, QtGui, QtWidgets):
            return flags if accepted_any else None
        accepted_any = True
    return flags


def _run_component_page(EEG, page, flags, app, qt_core, qt_gui, qt_widgets) -> bool:
    dialog = qt_widgets.QDialog()
    dialog.setObjectName("pop_selectcomps")
    dialog.setWindowTitle("Reject components by map - pop_selectcomps()")
    dialog.setWindowModality(qt_core.Qt.WindowModality.ApplicationModal)
    layout = qt_widgets.QVBoxLayout(dialog)
    grid = qt_widgets.QGridLayout()
    buttons: dict[int, Any] = {}

    def paint(button: Any, rejected: bool) -> None:
        button.setStyleSheet("background-color: #f6c1c1;" if rejected else "background-color: #d9ead3;")

    def open_component(component: int) -> None:
        _show_component_property(EEG, component, flags, buttons, dialog, qt_core, qt_widgets, paint)

    columns = 7
    for index, component in enumerate(page):
        cell = qt_widgets.QWidget()
        cell_layout = qt_widgets.QVBoxLayout(cell)
        cell_layout.setContentsMargins(2, 2, 2, 2)
        scalp = qt_widgets.QLabel()
        scalp.setObjectName(f"map{component}")
        pixmap = _component_map_pixmap(EEG, component, qt_gui)
        if pixmap is not None:
            scalp.setPixmap(pixmap)
        cell_layout.addWidget(scalp)
        button = qt_widgets.QPushButton(str(component))
        button.setObjectName(f"comp{component}")
        button.setAutoDefault(False)
        button.setFixedSize(72, 28)
        paint(button, bool(flags[component - 1]))
        button.clicked.connect(lambda _checked=False, component=component: open_component(component))
        cell_layout.addWidget(button)
        grid.addWidget(cell, index // columns, index % columns)
        buttons[component] = button
    layout.addLayout(grid)
    actions = qt_widgets.QHBoxLayout()
    for label, name, enabled, tip in (
        ("Cancel", "cancel", True, ""),
        (
            "Set thresholds",
            "set_thresholds",
            False,
            "pop_selectcomps: Set thresholds needs pop_icathresh, which is not yet available, "
            "and EEG.stats.compenta. Mark components here, then use Tools > Remove components.",
        ),
        (
            "See comp. stats",
            "comp_stats",
            False,
            "pop_selectcomps: component statistics need EEG.stats.compenta, which this dataset does not have.",
        ),
        ("OK", "ok", True, ""),
    ):
        action = qt_widgets.QPushButton(label)
        action.setObjectName(name)
        action.setEnabled(enabled)
        action.setAutoDefault(name == "ok")
        if tip:
            action.setToolTip(tip)
        if name == "cancel":
            action.clicked.connect(dialog.reject)
        elif name == "ok":
            action.clicked.connect(dialog.accept)
        actions.addWidget(action)
    layout.addLayout(actions)
    app.processEvents()
    return dialog.exec() == qt_widgets.QDialog.DialogCode.Accepted


def _show_component_property(EEG, component, flags, buttons, parent, qt_core, qt_widgets, paint) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from eegprep.functions.popfunc.pop_prop import _plot_one_property

    figure = _plot_one_property(EEG, 0, component, {"freqrange": [1, 50]})
    window = qt_widgets.QDialog(parent)
    window.setObjectName("pop_prop")
    window.setWindowTitle(f"pop_prop() - component {component}")
    window.setWindowModality(qt_core.Qt.WindowModality.ApplicationModal)
    layout = qt_widgets.QVBoxLayout(window)
    layout.addWidget(FigureCanvasQTAgg(figure))
    toggle = qt_widgets.QPushButton("REJECT" if flags[component - 1] else "ACCEPT")
    toggle.setObjectName("accept_reject")
    toggle.setAutoDefault(False)

    def flip() -> None:
        flags[component - 1] = 0 if flags[component - 1] else 1
        toggle.setText("REJECT" if flags[component - 1] else "ACCEPT")
        paint(buttons[component], bool(flags[component - 1]))

    toggle.clicked.connect(flip)
    layout.addWidget(toggle)
    window.finished.connect(lambda _code: plt.close(figure))
    window.exec()


def _component_map_pixmap(EEG: dict[str, Any], component: int, qt_gui):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from eegprep.functions.sigprocfunc.topoplot import topoplot

    weights = np.asarray(EEG.get("icawinv", []), dtype=float)
    if weights.ndim != 2 or component > weights.shape[1]:
        return None
    figure = plt.figure(figsize=(1.15, 1.15), dpi=50)
    try:
        axis = figure.add_axes((0.02, 0.02, 0.96, 0.96))
        electrodes = "off" if int(EEG.get("nbchan", 0) or 0) > 64 else "on"
        topoplot(
            weights[:, component - 1],
            EEG.get("chanlocs") or [],
            axes=axis,
            electrodes=electrodes,
            numcontour=8,
        )
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        width, height = canvas.get_width_height()
        buffer = np.ascontiguousarray(canvas.buffer_rgba())
        image = qt_gui.QImage(buffer.data, width, height, buffer.strides[0], qt_gui.QImage.Format.Format_RGBA8888)
        return qt_gui.QPixmap.fromImage(image.copy())
    except (TypeError, ValueError, RuntimeError):
        return None
    finally:
        plt.close(figure)
