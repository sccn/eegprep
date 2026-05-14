"""Screenshot capture entrypoint for EEGPrep visual parity cases."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.listdlg2 import build_listdlg2_dialog
from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents_dialog_spec
from eegprep.functions.popfunc.pop_chansel import pop_chansel_display_values
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec
from eegprep.functions.popfunc.pop_reref import pop_reref_dialog_spec
from eegprep.functions.popfunc.pop_resample import pop_resample_dialog_spec
from eegprep.functions.popfunc.pop_runica import pop_runica_dialog_spec
from eegprep.functions.popfunc.pop_select import pop_select_dialog_spec
from eegprep.plugins.ICLabel.pop_iclabel import pop_iclabel_dialog_spec
from eegprep.plugins.clean_rawdata.pop_clean_rawdata import pop_clean_rawdata_dialog_spec


def _demo_eeg() -> dict:
    return {
        "data": np.zeros((1, 1000), dtype=np.float32),
        "nbchan": 1,
        "pnts": 1000,
        "trials": 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 3.996,
        "event": [
            {"type": "stim", "latency": 100.0},
            {"type": "resp", "latency": 350.0},
            {"type": "boundary", "latency": 500.5, "duration": 20.0},
        ],
    }


def _demo_reref_eeg() -> dict:
    chanlocs = [
        {
            "labels": "Fp1",
            "ref": "common",
            "theta": -18.0,
            "radius": 0.42,
            "X": -0.25,
            "Y": 0.75,
            "Z": 0.55,
            "type": "EEG",
        },
        {
            "labels": "Fp2",
            "ref": "common",
            "theta": 18.0,
            "radius": 0.42,
            "X": 0.25,
            "Y": 0.75,
            "Z": 0.55,
            "type": "EEG",
        },
        {
            "labels": "Cz",
            "ref": "common",
            "theta": 0.0,
            "radius": 0.0,
            "X": 0.0,
            "Y": 0.0,
            "Z": 1.0,
            "type": "EEG",
        },
        {
            "labels": "Oz",
            "ref": "common",
            "theta": 180.0,
            "radius": 0.42,
            "X": 0.0,
            "Y": -0.8,
            "Z": 0.55,
            "type": "EEG",
        },
    ]
    return {
        "data": np.zeros((4, 1000), dtype=np.float32),
        "nbchan": 4,
        "pnts": 1000,
        "trials": 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 3.996,
        "chanlocs": chanlocs,
        "chaninfo": {
            "nodatchans": [{"labels": "M1", "theta": -90.0, "radius": 0.5, "type": "REF"}],
            "removedchans": [
                {
                    "labels": "Pz",
                    "theta": 180.0,
                    "radius": 0.25,
                    "X": 0.0,
                    "Y": -0.4,
                    "Z": 0.8,
                    "type": "EEG",
                }
            ],
        },
        "ref": "common",
    }


def _demo_interp_eeg(*, epoched: bool = False, removed: bool = False) -> dict:
    chanlocs = [
        {"labels": "Fp1", "theta": -18.0, "radius": 0.42, "X": -0.25, "Y": 0.75, "Z": 0.55},
        {"labels": "Fp2", "theta": 18.0, "radius": 0.42, "X": 0.25, "Y": 0.75, "Z": 0.55},
        {"labels": "Cz", "theta": 0.0, "radius": 0.0, "X": 0.0, "Y": 0.0, "Z": 1.0},
        {"labels": "Oz", "theta": 180.0, "radius": 0.42, "X": 0.0, "Y": -0.8, "Z": 0.55},
    ]
    eeg = {
        "data": np.zeros((4, 500, 2), dtype=np.float32) if epoched else np.zeros((4, 1000), dtype=np.float32),
        "nbchan": 4,
        "pnts": 500 if epoched else 1000,
        "trials": 2 if epoched else 1,
        "srate": 250.0,
        "xmin": 0.0,
        "xmax": 1.996 if epoched else 3.996,
        "chanlocs": chanlocs,
        "epoch": [{"event": [1]}, {"event": [2]}] if epoched else [],
        "chaninfo": {},
    }
    if removed:
        eeg["chaninfo"]["removedchans"] = [
            {"labels": "M1", "theta": -90.0, "radius": 0.5, "X": -0.8, "Y": 0.0, "Z": 0.4},
            {"labels": "M2", "theta": 90.0, "radius": 0.5, "X": 0.8, "Y": 0.0, "Z": 0.4},
        ]
    return eeg


def _demo_main_eeg(*, epoched: bool = False, setname: str = "menu demo") -> dict:
    eeg = _demo_reref_eeg()
    if epoched:
        eeg.update(
            {
                "data": np.zeros((4, 250, 2), dtype=np.float32),
                "pnts": 250,
                "trials": 2,
                "xmin": -0.2,
                "xmax": 0.796,
                "epoch": [{"event": [1]}, {"event": [2]}],
            }
        )
    eeg.update(
        {
            "setname": setname,
            "filename": f"{setname.replace(' ', '_')}.set",
            "filepath": "/tmp",
            "event": [
                {"type": "stim", "latency": 100.0, "duration": 0.0},
                {"type": "resp", "latency": 350.0, "duration": 0.0},
            ],
            "urevent": [],
            "history": "",
            "icaweights": np.eye(4, dtype=np.float32),
            "icasphere": np.eye(4, dtype=np.float32),
            "icawinv": np.eye(4, dtype=np.float32),
            "icachansind": np.arange(4),
            "icaact": np.zeros((4, 250, 2), dtype=np.float32) if epoched else np.zeros((4, 1000), dtype=np.float32),
        }
    )
    return eeg


def _configure_main_window_session(session: EEGPrepSession, state: str) -> None:
    if state == "startup":
        return
    if state == "continuous":
        session.store_current(_demo_main_eeg(), new=True)
        return
    if state == "epoched":
        session.store_current(_demo_main_eeg(epoched=True, setname="menu epoched"), new=True)
        return
    if state == "multiple":
        session.store_current(_demo_main_eeg(setname="menu one"), new=True)
        session.store_current(_demo_main_eeg(setname="menu two"), new=True)
        session.retrieve([1, 2])
        return
    if state == "study":
        session.store_current(_demo_main_eeg(setname="study demo"), new=True)
        session.STUDY = {
            "name": "menu study",
            "filename": "menu_study.study",
            "filepath": "/tmp",
            "task": "demo task",
            "subject": ["S01"],
            "condition": ["C1"],
            "session": [],
            "group": [],
            "cluster": [{"name": "ParentCluster"}],
        }
        session.CURRENTSTUDY = 1
        return
    raise ValueError(f"unsupported main-window state: {state}")


def _main_window_menu_state(menu_label: str | None, state: str) -> str:
    if menu_label is None or state != "startup":
        return state
    return "study" if menu_label == "Study" else "continuous"


def _grab_dialog(dialog, output: pathlib.Path, app) -> None:
    dialog.show()
    dialog.raise_()
    app.processEvents()
    pixmap = _matlab_scaled_pixmap(dialog.grab(), app)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    dialog.close()
    app.processEvents()


def capture_main_window(output: pathlib.Path, *, state: str = "startup", menu_label: str | None = None) -> None:
    """Render and capture the EEGPrep main window, optionally with a menu open."""
    session = EEGPrepSession()
    state = _main_window_menu_state(menu_label, state)
    _configure_main_window_session(session, state)
    window = build_main_window(session, native_menu_bar=False if menu_label else None)
    if menu_label:
        window.window.setWindowFlag(window._qt_core.Qt.WindowStaysOnTopHint, True)
    window.show()
    window.window.activateWindow()
    window.window.raise_()
    window.app.processEvents()
    menu = None
    if menu_label:
        menu = window.open_menu(menu_label)
    if menu is not None:
        pixmap = _grab_main_window_with_menu(window, menu_label, menu)
    else:
        pixmap = window.window.grab()
    pixmap = _matlab_scaled_pixmap(pixmap, window.app)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    window.window.close()
    window.app.processEvents()


def _grab_main_window_with_menu(window, menu_label: str, menu):
    window_pixmap = window.window.grab()
    menu_pixmap = menu.grab()
    ratio = float(window_pixmap.devicePixelRatio() or 1)
    window_pixmap.setDevicePixelRatio(1)
    menu_pixmap.setDevicePixelRatio(1)
    pos = _menu_popup_position(window, menu_label, ratio)
    width = max(window_pixmap.width(), pos[0] + menu_pixmap.width())
    height = max(window_pixmap.height(), pos[1] + menu_pixmap.height())
    canvas = window._qt_gui.QPixmap(width, height)
    canvas.fill(window._qt_gui.QColor("#a8c2ff"))
    painter = window._qt_gui.QPainter(canvas)
    painter.drawPixmap(0, 0, window_pixmap)
    painter.drawPixmap(pos[0], pos[1], menu_pixmap)
    painter.end()
    return canvas


def _menu_popup_position(window, menu_label: str, ratio: float) -> tuple[int, int]:
    menubar = window.window.menuBar()
    for action in menubar.actions():
        if action.text() != menu_label:
            continue
        geometry = menubar.actionGeometry(action)
        x = round((menubar.x() + geometry.x()) * ratio)
        y = round((menubar.y() + geometry.bottom()) * ratio)
        return max(0, x), max(0, y)
    return 0, round(menubar.height() * ratio)


def _matlab_scaled_pixmap(pixmap, app):
    if sys.platform != "darwin":
        return pixmap
    screen = app.primaryScreen()
    ratio = float(pixmap.devicePixelRatio() or 1)
    if screen is not None:
        ratio = max(ratio, float(screen.devicePixelRatio() or 1))
    if ratio <= 1:
        return pixmap
    matlab_ratio = 1.5
    scale = matlab_ratio / ratio
    return pixmap.scaled(max(1, round(pixmap.width() * scale)), max(1, round(pixmap.height() * scale)))


def capture_adjust_events_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_adjustevents dialog."""
    eeg = _demo_eeg()
    event_types = [event["type"] for event in eeg["event"]]
    spec = pop_adjustevents_dialog_spec(float(eeg["srate"]), event_types)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_reref_dialog(output: pathlib.Path, *, variant: str = "average") -> None:
    """Render and capture the pop_reref dialog."""
    eeg = _demo_reref_eeg()
    labels = [chan["labels"] for chan in eeg["chanlocs"]]
    refloc_labels = [chan["labels"] for chan in eeg["chaninfo"]["nodatchans"]]
    spec = pop_reref_dialog_spec("common", labels, refloc_labels)
    renderer = QtDialogRenderer()
    app, dialog, widgets = renderer.build_dialog(spec)
    if variant == "channels":
        widgets["rerefstr"].setChecked(True)
        renderer._set_reref_mode(widgets, "channels", True)
        widgets["reref"].setText("Fp1")
        widgets["keepref"].setChecked(True)
    elif variant == "huber":
        widgets["huberef"].setChecked(True)
        renderer._set_reref_mode(widgets, "huber", True)
    elif variant == "interp_removed":
        widgets["interp"].setChecked(True)
    _grab_dialog(dialog, output, app)


def capture_pop_interp_dialog(output: pathlib.Path, *, variant: str = "continuous") -> None:
    """Render and capture the pop_interp dialog."""
    eeg = _demo_interp_eeg(
        epoched=variant in {"epoched", "epoched_removed"},
        removed=variant in {"removed", "epoched_removed"},
    )
    spec = pop_interp_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_select_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_select dialog."""
    eeg = _demo_main_eeg()
    spec = pop_select_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_resample_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_resample dialog."""
    eeg = _demo_main_eeg()
    spec = pop_resample_dialog_spec(eeg["srate"])
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_runica_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_runica dialog."""
    eeg = _demo_main_eeg()
    spec = pop_runica_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_iclabel_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_iclabel dialog."""
    spec = pop_iclabel_dialog_spec()
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_clean_rawdata_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_clean_rawdata dialog."""
    eeg = _demo_main_eeg()
    spec = pop_clean_rawdata_dialog_spec(eeg)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    _grab_dialog(dialog, output, app)


def capture_pop_chansel_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_chansel/listdlg2 channel picker."""
    labels = ["Fp1", "Fp2", "Cz", "Oz"]
    display_values = pop_chansel_display_values(labels, withindex="on")
    app, dialog = build_listdlg2_dialog(
        promptstring="(use shift|Ctrl to\nselect several)",
        liststring=display_values,
        selectionmode="multiple",
    )
    _grab_dialog(dialog, output, app)


def capture_dataset_index_dialog(output: pathlib.Path) -> None:
    """Render and capture the dataset index prompt used by pop_interp."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    dialog.setObjectName("inputdlg2")
    dialog.setWindowTitle("Choose dataset")
    dialog.resize(200, 127)
    dialog.setStyleSheet(
        """
        QDialog {
            background: #a8c2ff;
            color: #000066;
            font-size: 16px;
        }
        QLabel {
            color: #000066;
            background: transparent;
            font-size: 16px;
        }
        QLineEdit {
            background: white;
            border: 1px solid #7f7f7f;
            min-height: 18px;
            max-height: 18px;
            font-size: 16px;
        }
        QPushButton {
            background: #eeeeee;
            border: 1px solid #7f7f7f;
            min-width: 47px;
            max-width: 47px;
            min-height: 18px;
            max-height: 18px;
            padding: 0;
            color: #000066;
            font-size: 16px;
        }
        """
    )
    widgets = [
        (QtWidgets.QLabel("Dataset index", dialog), "", (9, 16, 180, 20)),
        (QtWidgets.QLineEdit(dialog), "input", (9, 44, 179, 19)),
        (QtWidgets.QPushButton("Help", dialog), "help", (9, 90, 49, 19)),
        (QtWidgets.QPushButton("Cancel", dialog), "cancel", (87, 90, 49, 19)),
        (QtWidgets.QPushButton("OK", dialog), "ok", (140, 90, 49, 19)),
    ]
    for widget, name, geometry in widgets:
        if name:
            widget.setObjectName(name)
        widget.setGeometry(*geometry)
    _grab_dialog(dialog, output, app)


def capture_pophelp_dialog(output: pathlib.Path, function_name: str) -> None:
    """Render and capture the pophelp browser for a pop function."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = pophelp(function_name)
    _grab_dialog(dialog, output, app)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    if args.case == "adjust_events_dialog":
        capture_adjust_events_dialog(args.output)
    elif args.case == "main_window":
        capture_main_window(args.output)
    elif args.case == "main_window_continuous":
        capture_main_window(args.output, state="continuous")
    elif args.case == "main_window_epoched":
        capture_main_window(args.output, state="epoched")
    elif args.case == "main_window_multiple":
        capture_main_window(args.output, state="multiple")
    elif args.case == "main_window_study":
        capture_main_window(args.output, state="study")
    elif args.case == "file_menu":
        capture_main_window(args.output, menu_label="File")
    elif args.case == "edit_menu":
        capture_main_window(args.output, menu_label="Edit")
    elif args.case == "tools_menu":
        capture_main_window(args.output, menu_label="Tools")
    elif args.case == "plot_menu":
        capture_main_window(args.output, menu_label="Plot")
    elif args.case == "study_menu":
        capture_main_window(args.output, menu_label="Study")
    elif args.case == "datasets_menu":
        capture_main_window(args.output, menu_label="Datasets")
    elif args.case == "help_menu":
        capture_main_window(args.output, menu_label="Help")
    elif args.case == "reref_dialog":
        capture_reref_dialog(args.output)
    elif args.case == "reref_dialog_channel_ref":
        capture_reref_dialog(args.output, variant="channels")
    elif args.case == "reref_dialog_huber_ref":
        capture_reref_dialog(args.output, variant="huber")
    elif args.case == "reref_dialog_interp_removed":
        capture_reref_dialog(args.output, variant="interp_removed")
    elif args.case == "pop_interp_dialog":
        capture_pop_interp_dialog(args.output)
    elif args.case == "pop_interp_removed_dialog":
        capture_pop_interp_dialog(args.output, variant="removed")
    elif args.case == "pop_interp_epoched_dialog":
        capture_pop_interp_dialog(args.output, variant="epoched")
    elif args.case == "pop_select_dialog":
        capture_pop_select_dialog(args.output)
    elif args.case == "pop_resample_dialog":
        capture_pop_resample_dialog(args.output)
    elif args.case == "pop_runica_dialog":
        capture_pop_runica_dialog(args.output)
    elif args.case == "pop_iclabel_dialog":
        capture_pop_iclabel_dialog(args.output)
    elif args.case == "pop_clean_rawdata_dialog":
        capture_pop_clean_rawdata_dialog(args.output)
    elif args.case == "pop_chansel_dialog":
        capture_pop_chansel_dialog(args.output)
    elif args.case == "pop_interp_dataset_index_dialog":
        capture_dataset_index_dialog(args.output)
    elif args.case == "pop_reref_help_dialog":
        capture_pophelp_dialog(args.output, "pop_reref")
    elif args.case == "pop_interp_help_dialog":
        capture_pophelp_dialog(args.output, "pop_interp")
    else:
        parser.error(f"unsupported EEGPrep visual capture case: {args.case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
