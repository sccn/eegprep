"""Screenshot capture entrypoint for EEGPrep visual parity cases."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.guifunc.listdlg2 import build_listdlg2_dialog
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents_dialog_spec
from eegprep.functions.popfunc.pop_chansel import pop_chansel_display_values
from eegprep.functions.popfunc.pop_interp import pop_interp_dialog_spec
from eegprep.functions.popfunc.pop_reref import pop_reref_dialog_spec


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


def _grab_dialog(dialog, output: pathlib.Path, app) -> None:
    dialog.show()
    dialog.raise_()
    app.processEvents()
    pixmap = dialog.grab()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    dialog.close()
    app.processEvents()


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
