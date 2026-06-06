"""Reject channels using average spectral power."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_numeric_sequence
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_select import pop_select


def pop_rejchanspec(
    EEG: dict[str, Any] | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
):
    """Reject bad channels from spectral outliers.

    When precomputed spectra are not supplied, EEGPrep uses a raw FFT
    periodogram. This keeps the wrapper standalone, but absolute dB thresholds
    do not exactly match EEGLAB's ``spectopo``/Welch scaling.
    """
    if EEG is None:
        return (None, "") if return_com else (None, [], [], [])
    options = _normalise_options(parse_key_value_args(args, kwargs, lowercase_kwargs=True), EEG)
    working = copy.deepcopy(EEG)
    if str(options.get("averef", "off")).lower() == "on":
        include = set(int(index) for index in options["elec"])
        exclude_zero_based = [
            index - 1 for index in range(1, int(working.get("nbchan", 0) or 0) + 1) if index not in include
        ]
        working = pop_reref(working, [], exclude=exclude_zero_based, gui=False)
    specdata, specfreqs = _spectrum(working, options)
    rejected = _rejected_channels(specdata, specfreqs, options)
    out = copy.deepcopy(EEG)
    if rejected and str(options.get("indexonly", "off")).lower() != "on":
        out = pop_select(out, "nochannel", [index - 1 for index in rejected], gui=False)
    command = _history_command(options)
    return (out, command) if return_com else (out, rejected, specdata, specfreqs)


def _normalise_options(options: dict[str, Any], EEG: dict[str, Any]) -> dict[str, Any]:
    out = dict(options)
    nbchan = int(EEG.get("nbchan", np.asarray(EEG.get("data", [])).shape[0]) or 0)
    out["elec"] = parse_numeric_sequence(out.get("elec", list(range(1, nbchan + 1))), dtype=int)
    if not out["elec"]:
        out["elec"] = list(range(1, nbchan + 1))
    if any(index < 1 or index > nbchan for index in out["elec"]):
        raise ValueError("elec values must be 1-based channel indices")
    out["freqlims"] = _matrix_option(out.get("freqlims", [35, float(EEG.get("srate", 1)) / 2]), columns=2)
    out["stdthresh"] = _matrix_option(out.get("stdthresh", [5]), columns=2, allow_single=True)
    out["absthresh"] = parse_numeric_sequence(out.get("absthresh", []), dtype=float)
    out.setdefault("averef", "off")
    out.setdefault("indexonly", "off")
    return out


def _spectrum(EEG: dict[str, Any], options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if np.asarray(options.get("specdata", [])).size:
        specdata = np.asarray(options["specdata"], dtype=float)
        specfreqs = np.asarray(options.get("specfreqs", []), dtype=float).ravel()
        if specdata.ndim != 2 or specfreqs.size != specdata.shape[1]:
            raise ValueError("specdata must be channels x frequencies and match specfreqs")
        return specdata, specfreqs
    data = np.asarray(EEG.get("data"), dtype=float)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1, order="F")
    if data.ndim != 2:
        raise ValueError("EEG.data must be 2-D or 3-D")
    demeaned = data - data.mean(axis=1, keepdims=True)
    spectra = np.fft.rfft(demeaned, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1 / float(EEG.get("srate", 1) or 1))
    power = 10 * np.log10(np.maximum(np.abs(spectra) ** 2, np.finfo(float).tiny))
    return power, freqs


def _rejected_channels(specdata: np.ndarray, specfreqs: np.ndarray, options: dict[str, Any]) -> list[int]:
    rejected: set[int] = set()
    selected_rows = np.asarray(options["elec"], dtype=int) - 1
    for row, limits in enumerate(options["freqlims"]):
        low, high = limits
        freq_mask = (specfreqs >= low) & (specfreqs <= high)
        if not freq_mask.any():
            continue
        selected = np.mean(specdata[np.ix_(selected_rows, freq_mask)], axis=1)
        absolute = options.get("absthresh", [])
        if absolute:
            if len(absolute) == 1:
                mask = selected >= absolute[0]
            else:
                mask = (selected <= absolute[0]) | (selected >= absolute[-1])
        else:
            median = float(np.median(selected))
            std = float(np.std(selected))
            thresholds = options["stdthresh"][min(row, len(options["stdthresh"]) - 1)]
            if len(thresholds) == 1:
                mask = selected > median + std * thresholds[0]
            else:
                mask = (selected <= median + std * thresholds[0]) | (selected >= median + std * thresholds[-1])
        rejected.update(int(options["elec"][index]) for index in np.flatnonzero(mask))
    return sorted(rejected)


def _matrix_option(value: Any, *, columns: int, allow_single: bool = False) -> list[list[float]]:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2:
            return arr.tolist()
        values = arr.ravel().tolist()
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple, np.ndarray)):
        return [parse_numeric_sequence(row, dtype=float) for row in value]
    else:
        values = parse_numeric_sequence(value, dtype=float)
    if allow_single and len(values) == 1:
        return [values]
    return [
        values[index : index + columns] for index in range(0, len(values), columns) if values[index : index + columns]
    ]


def _history_command(options: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("elec", "freqlims", "stdthresh", "absthresh", "averef", "indexonly"):
        value = options.get(key)
        if value is None or (isinstance(value, list) and not value):
            continue
        pieces.extend([format_history_value(key), format_history_value(value, cell_for_sequence=None)])
    return "EEG = pop_rejchanspec(EEG, " + ", ".join(pieces) + ");"


__all__ = ["pop_rejchanspec"]
