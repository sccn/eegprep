"""Export EEG data or ICA activity to a text file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._file_io import channel_labels
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args


def pop_export(EEG: dict[str, Any], filename: str | Path, *args: Any, **kwargs: Any) -> str:
    """Export EEG data or ICA activity to a delimited text file."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    data = _selected_data(EEG, ica=_is_on(options.get("ica", "off")))
    if _is_on(options.get("erp", "off")) and data.ndim == 3:
        data = data.mean(axis=2)
    elif data.ndim == 3:
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    if options.get("expr"):
        raise NotImplementedError("pop_export expr is not supported in EEGPrep yet")
    if _is_on(options.get("time", "on")):
        time = np.tile(
            np.linspace(float(EEG.get("xmin", 0)), float(EEG.get("xmax", 0)), int(EEG["pnts"]))
            / float(options.get("timeunit", 1e-3)),
            int(EEG.get("trials", 1)) if not _is_on(options.get("erp", "off")) else 1,
        )
        data = np.vstack([time, data])
    separator = str(options.get("separator", "\t"))
    precision = int(options.get("precision", 7))
    labels = ["Time", *channel_labels(EEG)] if _is_on(options.get("time", "on")) else channel_labels(EEG)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    transpose = _is_on(options.get("transpose", "off"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=separator)
        if transpose:
            if _is_on(options.get("elec", "on")):
                writer.writerow(labels)
            writer.writerows(_format_row(row, precision) for row in data.T)
        else:
            for index, row in enumerate(data):
                values = _format_row(row, precision)
                if _is_on(options.get("elec", "on")):
                    values = [labels[index], *values]
                writer.writerow(values)
    return _history_command(filename, options)


def _selected_data(EEG: dict[str, Any], *, ica: bool) -> np.ndarray:
    if not ica:
        return np.asarray(EEG["data"])
    icaact = np.asarray(EEG.get("icaact", []))
    if icaact.size:
        return icaact
    weights = np.asarray(EEG.get("icaweights", []))
    sphere = np.asarray(EEG.get("icasphere", []))
    if not weights.size or not sphere.size:
        raise ValueError("No ICA activity or ICA weights are available")
    channels = np.asarray(EEG.get("icachansind", np.arange(EEG["nbchan"])), dtype=int)
    data = np.asarray(EEG["data"])[channels, ...]
    activations = weights @ sphere @ data.reshape((len(channels), -1))
    return activations.reshape((activations.shape[0], int(EEG["pnts"]), int(EEG.get("trials", 1))))


def _format_row(row: np.ndarray, precision: int) -> list[str]:
    return [f"{float(value):.{precision}g}" for value in row]


def _is_on(value: Any) -> bool:
    return str(value).lower() in {"on", "yes", "true", "1"}


def _history_command(filename: str | Path, options: dict[str, Any]) -> str:
    pieces = [format_history_value(str(filename))]
    for key in ["ica", "time", "timeunit", "elec", "transpose", "erp", "precision", "separator"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"LASTCOM = pop_export(EEG, {', '.join(pieces)});"
