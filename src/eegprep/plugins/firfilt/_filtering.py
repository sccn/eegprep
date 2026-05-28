"""Shared FIR filtering helpers for the bundled firfilt plugin."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.signal import fftconvolve, lfilter, minimum_phase, remez
from scipy.signal import windows as signal_windows

from eegprep.functions.miscfunc.misc import round_mat
from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.plugins.firfilt.firws import firws


WINDOW_TYPES = ("rectangular", "hann", "hamming", "blackman", "kaiser")
FILTER_TYPES = ("bandpass", "highpass", "lowpass", "bandstop")


def design_eegfiltnew(
    srate: float,
    *,
    locutoff: float | None = None,
    hicutoff: float | None = None,
    filtorder: int | None = None,
    revfilt: bool = False,
    minphase: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Design the Hamming-windowed FIR used by EEGLAB ``pop_eegfiltnew``."""
    locutoff = _none_if_zero(locutoff)
    hicutoff = _none_if_zero(hicutoff)
    revfilt = bool(revfilt)
    if hicutoff is None:
        if locutoff is None:
            raise ValueError("Not enough input arguments.")
        hicutoff = locutoff
        locutoff = None
        revfilt = not revfilt

    nyquist = float(srate) / 2.0
    edge_array = np.sort(np.asarray([value for value in (locutoff, hicutoff) if value is not None], dtype=float))
    if edge_array.size == 0:
        raise ValueError("Not enough input arguments.")
    if np.any(edge_array < 0) or np.any(edge_array >= nyquist):
        raise ValueError("Cutoff frequency out of range")

    max_tbw = edge_array.copy()
    if not revfilt:
        max_tbw[-1] = nyquist - edge_array[-1]
    elif edge_array.size == 2:
        max_tbw = np.asarray([np.diff(edge_array)[0] / 2.0])
    max_df = float(np.min(max_tbw))
    if max_df <= 0:
        raise ValueError("Cutoff frequency out of range")

    if filtorder is None:
        if revfilt:
            df = min(max(max_df * 0.25, 2.0), max_df)
        else:
            df = min(max(float(edge_array[0]) * 0.25, 2.0), max_df)
        filtorder = int(np.ceil((3.3 / (df / float(srate))) / 2.0) * 2)
    else:
        filtorder = _validate_even_order(filtorder)
        df = 3.3 / filtorder * float(srate)
        min_order = int(np.ceil(3.3 / ((max_df * 2.0) / float(srate)) / 2.0) * 2)
        if filtorder < min_order:
            raise ValueError(f"Filter order too low. Minimum required filter order is {min_order}.")

    if revfilt:
        offsets = np.asarray([-df]) if edge_array.size == 1 else np.asarray([df, -df])
    else:
        offsets = np.asarray([df]) if edge_array.size == 1 else np.asarray([-df, df])
    cutoff = edge_array + offsets / 2.0
    if np.any(cutoff <= 0) or np.any(cutoff >= nyquist):
        raise ValueError("Cutoff frequency out of range")

    window = signal_windows.hamming(filtorder + 1, sym=True)
    filter_type = ""
    if revfilt:
        filter_type = "high" if cutoff.size == 1 else "stop"
    b, _a = firws(filtorder, cutoff / nyquist, filter_type, window)
    if minphase:
        b = minimum_phase(b, method="homomorphic", half=False)
    metadata = {
        "locutoff": locutoff,
        "hicutoff": hicutoff,
        "filtorder": filtorder,
        "revfilt": revfilt,
        "minphase": bool(minphase),
        "cutoff": cutoff.tolist(),
        "df": float(df),
    }
    return np.asarray(b, dtype=float), metadata


def design_firws(
    srate: float,
    *,
    fcutoff: Any,
    forder: int,
    ftype: str | None = None,
    wtype: str | None = None,
    warg: float | None = None,
    minphase: bool = False,
) -> np.ndarray:
    """Design an EEGLAB firfilt ``pop_firws`` windowed-sinc FIR."""
    cutoff = _as_float_vector(fcutoff)
    order = _validate_even_order(forder)
    ftype = _normalized_filter_type(ftype, cutoff)
    window = _window(wtype or "hamming", order + 1, warg)
    fir_type = {"highpass": "high", "bandstop": "stop"}.get(ftype, "")
    b, _a = firws(order, np.sort(cutoff / (float(srate) / 2.0)), fir_type, window)
    if minphase:
        b = minimum_phase(b, method="homomorphic", half=False)
    return np.asarray(b, dtype=float)


def design_firpm(
    srate: float,
    *,
    fcutoff: Any,
    ftrans: float,
    ftype: str,
    forder: int,
    wtpass: float | None = None,
    wtstop: float | None = None,
) -> np.ndarray:
    """Design an EEGLAB firfilt ``pop_firpm`` equiripple FIR."""
    cutoff = _as_float_vector(fcutoff)
    order = _validate_even_order(forder)
    ftype = _normalized_filter_type(ftype, cutoff)
    transition = float(ftrans)
    if transition <= 0:
        raise ValueError("Transition band width must be positive")
    edges = np.sort(np.concatenate([cutoff - transition / 2.0, cutoff + transition / 2.0]) / (float(srate) / 2.0))
    if np.any(edges < 0):
        raise ValueError("Cutoff frequencies - transition band width / 2 must not be < DC")
    if np.any(edges > 1):
        raise ValueError("Cutoff frequencies + transition band width / 2 must not be > Nyquist")
    bands = [0.0, *edges.tolist(), 1.0]
    desired_by_type = {
        "bandpass": [0, 1, 0],
        "bandstop": [1, 0, 1],
        "highpass": [0, 1],
        "lowpass": [1, 0],
    }
    desired = desired_by_type[ftype]
    weight = None
    if wtpass is not None and wtstop is not None:
        weight_values = [float(wtstop), float(wtpass)]
        weight = [weight_values[int(value)] for value in desired]
    return remez(order + 1, bands, desired, weight=weight, fs=2.0)


def design_firma(*, forder: int) -> np.ndarray:
    """Design a moving-average FIR filter like EEGLAB ``pop_firma``."""
    order = _validate_even_order(forder)
    return np.ones(order + 1, dtype=float) / float(order + 1)


def apply_fir_filter(
    EEG: dict[str, Any],
    coefficients: Any,
    *,
    channels: Any = None,
    chantype: Any = None,
    causal: bool = False,
    usefftfilt: bool = False,
) -> dict[str, Any]:
    """Apply FIR coefficients to an EEG dataset using EEGLAB boundary semantics."""
    b = np.asarray(coefficients, dtype=float).ravel()
    if b.size % 2 != 1:
        raise ValueError("Filter order is not even.")
    output = deepcopy(EEG)
    data = np.asarray(output["data"])
    if data.ndim not in {2, 3}:
        raise ValueError("FIR filtering supports continuous or epoched EEG data")
    nbchan = int(output.get("nbchan", data.shape[0]))
    pnts = int(output.get("pnts", data.shape[1]))
    trials = int(output.get("trials", data.shape[2] if data.ndim == 3 else 1))
    channel_indices, _history = resolve_channel_indices(output, channels, chantype)

    flattened = data.transpose(0, 2, 1).reshape(nbchan, pnts * trials) if data.ndim == 3 else data.copy()
    filtered = flattened.astype(np.float64, copy=True)
    bounds = _epoched_bounds(pnts, trials) if trials > 1 else _continuous_bounds(output, pnts)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        window = np.ix_(channel_indices, np.arange(start, stop))
        filtered[window] = _filter_segment(b, filtered[window], causal=causal, usefftfilt=usefftfilt)

    output["data"] = filtered.reshape(nbchan, trials, pnts).transpose(0, 2, 1) if data.ndim == 3 else filtered
    output["icaact"] = np.array([])
    output["saved"] = "no"
    return output


def resolve_channel_indices(
    EEG: dict[str, Any], channels: Any = None, chantype: Any = None
) -> tuple[list[int], list[int] | None]:
    """Resolve EEGLAB-facing channel labels/types/indices to Python indices."""
    nbchan = int(EEG.get("nbchan", np.asarray(EEG.get("data")).shape[0]))
    if chantype is not None and _has_values(chantype):
        requested = {str(value).strip().lower() for value in _as_list(chantype)}
        chanlocs = _chanlocs(EEG)
        indices = [
            index for index, chan in enumerate(chanlocs) if str(chan.get("type", "")).strip().lower() in requested
        ]
        if not indices:
            raise ValueError("No channels matched the selected channel type(s)")
        return indices, [index + 1 for index in indices]
    if channels is None or not _has_values(channels):
        return list(range(nbchan)), None
    tokens = _as_list(channels)
    numeric = []
    all_numeric = True
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError):
            all_numeric = False
            break
        if not value.is_integer():
            all_numeric = False
            break
        numeric.append(int(value))
    if all_numeric:
        indices = [value - 1 for value in numeric]
        if any(index < 0 or index >= nbchan for index in indices):
            raise ValueError("Channel indices must be 1-based and within EEG.nbchan")
        return list(dict.fromkeys(indices)), list(dict.fromkeys(numeric))
    labels = {str(chan.get("labels", "")).lower(): index for index, chan in enumerate(_chanlocs(EEG))}
    indices = []
    for token in tokens:
        key = str(token).strip().lower()
        if key not in labels:
            raise ValueError(f"Channel '{token}' not found")
        indices.append(labels[key])
    indices = list(dict.fromkeys(indices))
    return indices, [index + 1 for index in indices]


def _filter_segment(b: np.ndarray, segment: np.ndarray, *, causal: bool, usefftfilt: bool) -> np.ndarray:
    group_delay = (b.size - 1) // 2
    if group_delay == 0:
        return segment.copy()
    if segment.shape[1] == 0:
        return segment.copy()
    start_width = 2 * group_delay if causal else group_delay
    start_pad = np.repeat(segment[:, :1], start_width, axis=1)
    end_pad = (
        np.empty((segment.shape[0], 0), dtype=segment.dtype)
        if causal
        else np.repeat(segment[:, -1:], group_delay, axis=1)
    )
    padded = np.concatenate([start_pad, segment, end_pad], axis=1)
    if usefftfilt:
        filtered = fftconvolve(padded, b.reshape(1, -1), mode="full", axes=1)[:, : padded.shape[1]]
    else:
        filtered = lfilter(b, [1.0], padded, axis=1)
    return filtered[:, 2 * group_delay :]


def _continuous_bounds(EEG: dict[str, Any], pnts: int) -> np.ndarray:
    starts = [0]
    for event in events_to_records(EEG.get("event")):
        event_type = event.get("type")
        is_boundary = str(event_type).startswith("boundary") if isinstance(event_type, str) else event_type == -99
        if not is_boundary:
            continue
        try:
            latency = float(event.get("latency"))
        except (TypeError, ValueError):
            continue
        start = int(round_mat(latency + 0.5)) - 1
        if 0 <= start <= pnts:
            starts.append(start)
    starts.append(pnts)
    return np.asarray(sorted(set(starts)), dtype=int)


def _epoched_bounds(pnts: int, trials: int) -> np.ndarray:
    return np.arange(0, pnts * trials + 1, pnts, dtype=int)


def _window(wtype: str, length: int, warg: float | None) -> np.ndarray:
    key = str(wtype).lower()
    if key == "rectangular":
        return np.ones(length, dtype=float)
    if key == "hann":
        return signal_windows.hann(length, sym=True)
    if key == "hamming":
        return signal_windows.hamming(length, sym=True)
    if key == "blackman":
        return signal_windows.blackman(length, sym=True)
    if key == "kaiser":
        if warg is None:
            raise ValueError("Kaiser window beta is required")
        return signal_windows.kaiser(length, float(warg), sym=True)
    raise ValueError("Unknown window type.")


def _normalized_filter_type(ftype: str | None, cutoff: np.ndarray) -> str:
    if ftype is None or str(ftype).strip() == "":
        return "lowpass" if cutoff.size == 1 else "bandpass"
    value = str(ftype).lower()
    if value not in FILTER_TYPES:
        raise ValueError("Unknown filter type.")
    if value in {"bandpass", "bandstop"} and cutoff.size != 2:
        raise ValueError("Not enough input arguments.")
    if value in {"highpass", "lowpass"} and cutoff.size != 1:
        raise ValueError("Too many input arguments.")
    return value


def _validate_even_order(value: Any) -> int:
    order = int(value)
    if order < 2 or order % 2 != 0:
        raise ValueError("Filter order must be a real, even, positive integer.")
    return order


def _none_if_zero(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    number = float(value)
    return None if number == 0 else number


def _as_float_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return np.asarray([], dtype=float)
        values = [float(token) for token in text.replace(",", " ").split()]
    else:
        values = np.asarray(value).ravel().tolist()
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Not enough input arguments.")
    return array


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [token for token in value.replace(",", " ").split() if token]
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip().strip("[]"))
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return True


def _chanlocs(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    chanlocs = EEG.get("chanlocs", [])
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    return [chan if isinstance(chan, dict) else {} for chan in list(chanlocs or [])]
