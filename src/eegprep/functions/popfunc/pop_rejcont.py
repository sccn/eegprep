"""Reject continuous portions of data by spectral thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc._rejection import copy_eeg, one_based_indices, parse_numeric_sequence
from eegprep.functions.popfunc.pop_select import pop_select


def pop_rejcont(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Reject continuous data portions based on high-frequency power."""
    if EEG is None:
        return (None, "") if return_com else (None, [])
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options
    if gui:
        gui_options = _run_gui(EEG, renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else (EEG, [])
        options.update(gui_options)
    out, selected, command = _apply_one(EEG, options)
    return (out, command) if return_com else (out, selected)


def pop_rejcont_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rejcont``."""
    return DialogSpec(
        title="Reject continuous portions of data - pop_rejcont()",
        function_name="pop_rejcont",
        eeglab_source="functions/popfunc/pop_rejcont.m",
        help_text="pophelp('pop_rejcont')",
        size=(560, 332),
        geometry=((2, 1),) * 7,
        controls=(
            ControlSpec("text", "Channel range"),
            ControlSpec("edit", tag="elecrange", value=f"1:{int(EEG.get('nbchan', 0) or 0)}"),
            ControlSpec("text", "Frequency range (Hz)"),
            ControlSpec("edit", tag="freqlimit", value="20 40"),
            ControlSpec("text", "Frequency threshold in dB"),
            ControlSpec("edit", tag="threshold", value="10"),
            ControlSpec("text", "Epoch segment length (s)"),
            ControlSpec("edit", tag="epochlength", value="0.5"),
            ControlSpec("text", "Minimum number of contiguous epochs"),
            ControlSpec("edit", tag="contiguous", value="4"),
            ControlSpec("text", "Add trails before and after regions (s)"),
            ControlSpec("edit", tag="addlength", value="0.25"),
            ControlSpec("text", "Use hanning window before computing FFT"),
            ControlSpec("checkbox", tag="taper", value=True),
        ),
    )


def _run_gui(EEG: dict[str, Any], renderer: Any | None) -> dict[str, Any] | None:
    result = inputgui(pop_rejcont_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    return {
        "elecrange": result.get("elecrange", ""),
        "freqlimit": result.get("freqlimit", "20 40"),
        "threshold": result.get("threshold", "10"),
        "epochlength": result.get("epochlength", "0.5"),
        "contiguous": result.get("contiguous", "4"),
        "addlength": result.get("addlength", "0.25"),
        "taper": "hamming" if result.get("taper", False) else "none",
    }


def _apply_one(EEG: dict[str, Any], options: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, str]:
    out = copy_eeg(EEG)
    if int(out.get("trials", 1) or 1) != 1:
        raise ValueError("pop_rejcont requires continuous data")
    data = np.asarray(out.get("data"), dtype=float)
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        raise ValueError("EEG data must be 2-D continuous data")
    srate = float(out.get("srate", 1.0))
    elecrange = one_based_indices(
        options.get("elecrange"), limit=int(out.get("nbchan", data.shape[0])), default_all=True
    )
    epochlength = float(parse_numeric_sequence(options.get("epochlength", 0.5), dtype=float)[0])
    overlap = float(parse_numeric_sequence(options.get("overlap", 0.25), dtype=float)[0])
    threshold = float(parse_numeric_sequence(options.get("threshold", 10), dtype=float)[0])
    contiguous = int(parse_numeric_sequence(options.get("contiguous", 4), dtype=float)[0])
    addlength = float(parse_numeric_sequence(options.get("addlength", 0.25), dtype=float)[0])
    freqlimit = parse_numeric_sequence(options.get("freqlimit", [35, min(128, srate / 2)]), dtype=float)
    selected = _selected_regions(
        data,
        [item - 1 for item in elecrange],
        srate,
        epochlength,
        overlap,
        freqlimit,
        threshold,
        contiguous,
        addlength,
        options,
    )
    if selected.size and str(options.get("onlyreturnselection", "off")).lower() != "on":
        if str(options.get("correct", "remove")).lower() == "blank":
            for start, stop in selected.astype(int):
                data[np.asarray(elecrange, dtype=int) - 1, start - 1 : stop] = 0
            out["data"] = data
        else:
            out = pop_select(out, "nopoint", selected.tolist(), gui=False)
    command = _history_command(options)
    return out, selected, command


def _selected_regions(
    data: np.ndarray,
    rows: list[int],
    srate: float,
    epochlength: float,
    overlap: float,
    freqlimit: list[float],
    threshold: float,
    contiguous: int,
    addlength: float,
    options: dict[str, Any],
) -> np.ndarray:
    window = max(1, int(round(epochlength * srate)))
    step = max(1, int(round((epochlength - overlap) * srate)))
    if window > data.shape[1]:
        return np.empty((0, 2), dtype=int)
    starts = list(range(0, data.shape[1] - window + 1, step))
    flagged = np.zeros(len(starts), dtype=bool)
    taper = np.hamming(window) if str(options.get("taper", "none")).lower() == "hamming" else np.ones(window)
    freqs = np.fft.rfftfreq(window, d=1 / srate)
    if len(freqlimit) < 2:
        freqlimit = [35, min(128, srate / 2)]
    freq_mask = (freqs >= freqlimit[0]) & (freqs <= freqlimit[-1])
    if not freq_mask.any():
        return np.empty((0, 2), dtype=int)
    for index, start in enumerate(starts):
        segment = data[rows, start : start + window] * taper
        power = 10 * np.log10(np.maximum(np.abs(np.fft.rfft(segment, axis=1)) ** 2, np.finfo(float).tiny))
        score = (
            power[:, freq_mask].max(axis=1)
            if str(options.get("mode", "max")).lower() == "max"
            else power[:, freq_mask].mean(axis=1)
        )
        flagged[index] = bool(np.any(score > threshold))
    regions = _runs_to_regions(flagged, starts, window, contiguous)
    if not regions:
        return np.empty((0, 2), dtype=int)
    pad = int(round(addlength * srate))
    padded = [[max(1, start + 1 - pad), min(data.shape[1], stop + pad)] for start, stop in regions]
    return _merge_regions(np.asarray(padded, dtype=int))


def _runs_to_regions(flagged: np.ndarray, starts: list[int], window: int, contiguous: int) -> list[list[int]]:
    regions = []
    index = 0
    while index < flagged.size:
        if not flagged[index]:
            index += 1
            continue
        run_start = index
        while index < flagged.size and flagged[index]:
            index += 1
        if index - run_start >= contiguous:
            regions.append([starts[run_start], starts[index - 1] + window])
    return regions


def _merge_regions(regions: np.ndarray) -> np.ndarray:
    regions = regions[np.argsort(regions[:, 0])]
    merged = [regions[0].tolist()]
    for start, stop in regions[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], int(stop))
        else:
            merged.append([int(start), int(stop)])
    return np.asarray(merged, dtype=int)


def _history_command(options: dict[str, Any]) -> str:
    values: list[Any] = []
    for key in (
        "elecrange",
        "freqlimit",
        "threshold",
        "epochlength",
        "contiguous",
        "addlength",
        "taper",
        "correct",
        "onlyreturnselection",
    ):
        if key in options:
            values.extend([key, options[key]])
    return (
        "EEG = pop_rejcont(EEG, "
        + ", ".join(format_history_value(item, cell_for_sequence=None) for item in values)
        + ");"
    )
