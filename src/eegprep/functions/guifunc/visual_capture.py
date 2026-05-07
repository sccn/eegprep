"""Screenshot capture entrypoint for EEGPrep visual parity cases."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from eegprep.functions.guifunc.qt import QtDialogRenderer
from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents_dialog_spec


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


def capture_adjust_events_dialog(output: pathlib.Path) -> None:
    """Render and capture the pop_adjustevents dialog."""
    eeg = _demo_eeg()
    event_types = [event["type"] for event in eeg["event"]]
    spec = pop_adjustevents_dialog_spec(float(eeg["srate"]), event_types)
    renderer = QtDialogRenderer()
    app, dialog, _widgets = renderer.build_dialog(spec)
    dialog.show()
    dialog.raise_()
    app.processEvents()
    pixmap = dialog.grab()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(output), "PNG"):
        raise RuntimeError(f"failed to save screenshot: {output}")
    dialog.close()
    app.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    if args.case != "adjust_events_dialog":
        parser.error(f"unsupported EEGPrep visual capture case: {args.case}")

    capture_adjust_events_dialog(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
