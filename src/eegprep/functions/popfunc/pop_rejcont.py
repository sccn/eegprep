"""Reject continuous portions of data by spectral thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc._rejection import copy_eeg, one_based_indices, parse_numeric_sequence
from eegprep.functions.popfunc.eeg_findboundaries import eeg_findboundaries
from eegprep.functions.popfunc.eeg_regepochs import eeg_regepochs
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.sigprocfunc.eegplot import eegplot
from eegprep.functions.sigprocfunc.nan_mean import nan_mean


_REJCONT_WINREJ_COLOR = (0.0, 0.9, 0.0)
_REJCONT_TAPER = "hamming"


def pop_rejcont(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
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
    out, selected, command = _apply_one(
        EEG,
        options,
        display=bool(gui or str(options.get("eegplot", "off")).lower() == "on"),
        command_callback=command_callback,
        show=show,
    )
    return (out, command) if return_com else (out, selected)


def pop_rejcont_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rejcont``."""
    return DialogSpec(
        title="Reject continuous portions of data - pop_rejcont()",
        function_name="pop_rejcont",
        eeglab_source="functions/popfunc/pop_rejcont.m",
        help_text=None,
        show_help_button=False,
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
            # EEGLAB labels this "hanning" but maps the checked option to a Hamming taper.
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
        "taper": _REJCONT_TAPER if result.get("taper", False) else "none",
        "eegplot": "on",
    }


def _apply_one(
    EEG: dict[str, Any],
    options: dict[str, Any],
    *,
    display: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], np.ndarray, str]:
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
        out,
        elecrange,
        srate,
        epochlength,
        overlap,
        freqlimit,
        threshold,
        contiguous,
        addlength,
        options,
    )
    command = _history_command(options)
    browser_display = display and str(options.get("onlyreturnselection", "off")).lower() != "on"
    if browser_display:
        _open_rejcont_browser(
            out,
            data,
            selected,
            elecrange,
            options,
            command=command,
            command_callback=command_callback,
            show=show,
        )
    elif selected.size and str(options.get("onlyreturnselection", "off")).lower() != "on":
        if str(options.get("correct", "remove")).lower() == "blank":
            for start, stop in selected.astype(int):
                data[np.asarray(elecrange, dtype=int) - 1, start - 1 : stop] = 0
            out["data"] = data
        else:
            out = pop_select(out, "nopoint", selected.tolist(), gui=False)
    return (EEG if browser_display else out), selected, "" if browser_display else command


def _open_rejcont_browser(
    EEG: dict[str, Any],
    data: np.ndarray,
    selected: np.ndarray,
    elecrange: list[int],
    options: dict[str, Any],
    *,
    command: str,
    command_callback: Any | None,
    show: bool,
) -> Any:
    selected_rows = np.asarray(elecrange, dtype=int) - 1
    selected_data = np.asarray(data, dtype=float)[selected_rows, :]
    winrej = _rejcont_winrej(selected, len(elecrange))

    def _accept(winrej_rows: np.ndarray) -> None:
        eeg_out = _apply_rejcont_winrej(EEG, winrej_rows, elecrange, options)
        if command_callback is not None:
            command_callback(eeg_out, command)

    kwargs: dict[str, Any] = {
        "srate": EEG.get("srate", 256),
        "limits": [
            float(EEG.get("xmin", 0.0) or 0.0) * 1000.0,
            float(EEG.get("xmax", 0.0) or 0.0) * 1000.0,
        ],
        "events": EEG.get("event", []),
        "winlength": 50,
        "xgrid": "off",
        "spacing": _eegplot_spacing(selected_data),
        "winrej": winrej,
        "wincolor": _REJCONT_WINREJ_COLOR,
        "butlabel": "Reject",
        "eloc_file": _selected_chanlocs(EEG, selected_rows),
        "title": f"Scroll channel activities -- eegplot() -- {EEG.get('setname', '')}".rstrip(),
        "show": show,
    }
    if show:
        kwargs["command_callback"] = _accept
    return eegplot(selected_data, **kwargs)


def _apply_rejcont_winrej(
    EEG: dict[str, Any],
    winrej_rows: np.ndarray,
    elecrange: list[int],
    options: dict[str, Any],
) -> dict[str, Any]:
    out = copy_eeg(EEG)
    selected = _continuous_rows_to_regions(winrej_rows, int(out.get("pnts", np.asarray(out.get("data")).shape[1])))
    if selected.size == 0:
        return out
    if str(options.get("correct", "remove")).lower() == "blank":
        data = np.asarray(out.get("data"), dtype=float)
        if data.ndim == 3:
            data = data[:, :, 0]
        for start, stop in selected.astype(int):
            data[np.asarray(elecrange, dtype=int) - 1, start - 1 : stop] = 0
        out["data"] = data
        return out
    return pop_select(out, "nopoint", selected.tolist(), gui=False)


def _continuous_rows_to_regions(winrej_rows: np.ndarray, pnts: int) -> np.ndarray:
    rows = np.asarray(winrej_rows if winrej_rows is not None else [], dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.size == 0:
        return np.empty((0, 2), dtype=int)
    regions = []
    for row in rows:
        start = max(1, min(int(round(float(row[0]))), pnts))
        stop = max(1, min(int(round(float(row[1]))), pnts))
        if stop < start:
            start, stop = stop, start
        if stop > start:
            regions.append([start, stop])
    return _merge_regions(np.asarray(regions, dtype=int)) if regions else np.empty((0, 2), dtype=int)


def _rejcont_winrej(selected: np.ndarray, channel_count: int) -> np.ndarray:
    if selected.size == 0:
        return np.zeros((0, 5 + channel_count), dtype=float)
    rows = np.zeros((selected.shape[0], 5 + channel_count), dtype=float)
    rows[:, 0:2] = selected[:, 0:2]
    rows[:, 2:5] = np.asarray(_REJCONT_WINREJ_COLOR, dtype=float)
    return rows


def _selected_chanlocs(EEG: dict[str, Any], selected_rows: np.ndarray) -> list[dict[str, Any]]:
    chanlocs = chanlocs_as_list(EEG.get("chanlocs"))
    return [chanlocs[index] for index in selected_rows if index < len(chanlocs)]


def _eegplot_spacing(data: np.ndarray) -> float:
    # eegplot.m:457-473. std is the sample standard deviation.
    sample_count = min(1000, int(data.shape[1]))
    stds = np.std(np.asarray(data, dtype=float)[:, :sample_count], axis=1, ddof=1)
    stds = np.sort(stds[np.isfinite(stds)])
    if stds.size > 2:
        spacing = float(np.mean(stds[1:-1]) * 3.0)
    elif stds.size:
        spacing = float(np.mean(stds) * 3.0)
    else:
        spacing = 1.0
    if spacing > 10:
        spacing = float(np.floor(spacing + 0.5))
    if spacing == 0 or np.isnan(spacing):
        spacing = 1.0
    return spacing


def _selected_regions(
    EEG: dict[str, Any],
    elecrange: list[int],
    srate: float,
    epochlength: float,
    overlap: float,
    freqlimit: list[float],
    threshold: float,
    contiguous: int,
    addlength: float,
    options: dict[str, Any],
) -> np.ndarray:
    # pop_rejcont.m:202 passes overlap as eeg_regepochs recurrence, with limits [0 epochlength].
    work = copy_eeg(EEG)
    work["event"] = _boundary_events(work.get("event"))
    work["urevent"] = []
    try:
        epoched = eeg_regepochs(work, overlap, [0.0, epochlength], np.nan)
    except ValueError as exc:
        raise ValueError(f"pop_rejcont: {exc}. Use a shorter epoch length or a positive overlap.") from exc
    spectra, frequencies = _epoch_log_spectrum(epoched, srate, str(options.get("taper", "none")))
    if spectra.size == 0 or not elecrange:
        return np.empty((0, 2), dtype=int)
    selected_rows = [index - 1 for index in elecrange]
    averaged = np.asarray(nan_mean(spectra[selected_rows], axis=0), dtype=float)
    if averaged.ndim == 1:
        averaged = averaged[:, np.newaxis]
    rejected = _threshold_epochs(
        averaged,
        frequencies,
        freqlimit if len(freqlimit) >= 2 else [35.0, min(128.0, srate / 2.0)],
        threshold,
        str(options.get("mode", "max")),
    )
    rows = _windows_for_epochs(epoched, rejected, epochlength, srate)
    grouplen = contiguous / 2.0 * epochlength * srate + 1.0
    rows = _merge_rejcont_windows(rows, grouplen, contiguous)
    point_count = int(EEG.get("pnts", np.asarray(EEG.get("data")).shape[1]))
    padded = [
        [max(1.0, row[0] - addlength * srate), min(float(point_count), row[1] + addlength * srate)] for row in rows
    ]
    if not padded:
        return np.empty((0, 2), dtype=int)
    return np.rint(np.asarray(padded, dtype=float)).astype(int)


def _boundary_events(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, dict):
        records = [value]
    elif isinstance(value, list):
        records = [item for item in value if isinstance(item, dict)]
    else:
        records = []
    if not records:
        return []
    bounds = eeg_findboundaries(EEG=records)
    return [records[index] for index in bounds] if bounds else []


def _epoch_log_spectrum(epoched: dict[str, Any], srate: float, taper: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(epoched.get("data"), dtype=float)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    if data.ndim != 3 or data.shape[1] < 2:
        return np.empty((0, 0, 0), dtype=float), np.empty(0, dtype=float)
    if taper.lower() == _REJCONT_TAPER:
        data = data * np.hamming(data.shape[1])[np.newaxis, :, np.newaxis]
    spectrum = np.fft.fft(data, axis=1)
    half = spectrum.shape[1] // 2
    # pop_rejcont.m:211-213 stretches linspace(0, Nyquist, N/2) and drops DC, rather than using bin frequencies.
    frequencies = np.linspace(0.0, srate / 2.0, half)[1:]
    spectrum = spectrum[:, 1:half, :]
    power = 10.0 * np.log10(np.maximum(np.abs(spectrum) ** 2, np.finfo(float).tiny))
    power = power - np.mean(power, axis=2, keepdims=True)
    return power, frequencies


def _threshold_epochs(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    freqlimit: list[float],
    threshold: float,
    mode: str,
) -> np.ndarray:
    # pop_rejcont.m:239 expands a single threshold to [-100, threshold] before either mode.
    low = -100.0
    high = float(threshold)
    if frequencies.size == 0:
        return np.empty(0, dtype=int)
    if mode.lower() == "max":
        timerange = (float(frequencies[0]), float(frequencies[-1]))
        span = timerange[1] - timerange[0]
        if span == 0:
            return np.empty(0, dtype=int)
        frame_low = max(1, int(np.floor((freqlimit[0] - timerange[0]) / span * (frequencies.size - 1))) + 1)
        frame_high = int(np.floor((freqlimit[-1] - timerange[0]) / span * (frequencies.size - 1))) + 1
        frame_high = min(frequencies.size, max(frame_low, frame_high))
        segment = spectrum[frame_low - 1 : frame_high, :]
    else:
        nearest_start = int(np.argmin(np.abs(frequencies - freqlimit[0])))
        nearest_stop = int(np.argmin(np.abs(frequencies - freqlimit[-1])))
        if nearest_stop < nearest_start:
            nearest_start, nearest_stop = nearest_stop, nearest_start
        segment = np.mean(spectrum[nearest_start : nearest_stop + 1, :], axis=0, keepdims=True)
    if segment.size == 0:
        return np.empty(0, dtype=int)
    rejected = (np.min(segment, axis=0) < low) | (np.max(segment, axis=0) > high)
    return np.flatnonzero(rejected) + 1


def _windows_for_epochs(
    epoched: dict[str, Any], rejected: np.ndarray, epochlength: float, srate: float
) -> list[list[float]]:
    events = epoched.get("event")
    if events is None:
        events = []
    if isinstance(events, np.ndarray):
        events = events.reshape(-1).tolist()
    if any("epoch" not in event for event in events):
        raise ValueError(
            "pop_rejcont: some events are not associated with any epoch. "
            "Reload the continuous dataset and run pop_rejcont again."
        )
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_epoch.setdefault(int(event["epoch"]), []).append(event)
    urevents = epoched.get("urevent")
    if urevents is None:
        urevents = []
    if isinstance(urevents, np.ndarray):
        urevents = urevents.reshape(-1).tolist()
    rows: list[list[float]] = []
    width = epochlength * srate - 1.0
    for epoch_number in rejected.tolist():
        matched = by_epoch.get(int(epoch_number))
        if not matched:
            continue
        event = matched[0]
        if str(event.get("type")) != "X":
            raise ValueError(
                "pop_rejcont: an epoch event was not type 'X'. "
                "Run pop_rejcont on continuous data; boundary events are kept and regular epochs are rebuilt."
            )
        latency = float(urevents[int(event["urevent"])]["latency"])
        rows.append([latency, latency + width])
    return rows


def _merge_rejcont_windows(rows: list[list[float]], grouplen: float, contiguous: int) -> list[list[float]]:
    # pop_rejcont.m:293-316, including the backwards deletion order.
    index = len(rows) - 1
    while index >= 0:
        count = len(rows)
        start, stop = rows[index][0], rows[index][1]
        if contiguous >= 2 and count >= index + 1 and stop - start > grouplen:
            del rows[index]
        elif contiguous >= 2 and index == 0 and count > 1 and rows[1][0] - stop > grouplen:
            del rows[index]
        elif contiguous >= 2 and index == count - 1 and count > 1 and start - rows[index - 1][1] > grouplen:
            del rows[index]
        elif (
            contiguous >= 2
            and 0 < index < count - 1
            and count > 1
            and rows[index + 1][0] - stop > grouplen
            and start - rows[index - 1][1] > grouplen
        ):
            del rows[index]
        elif index < count - 1 and count > 1 and rows[index + 1][0] - stop <= grouplen:
            rows[index][1] = rows[index + 1][1]
            del rows[index + 1]
        index -= 1
    return rows


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
        "overlap",
        "mode",
        "contiguous",
        "addlength",
        "taper",
        "correct",
        "onlyreturnselection",
        "eegplot",
    ):
        if key in options:
            values.extend([key, options[key]])
    return (
        "EEG = pop_rejcont(EEG, "
        + ", ".join(format_history_value(item, cell_for_sequence=None) for item in values)
        + ");"
    )
