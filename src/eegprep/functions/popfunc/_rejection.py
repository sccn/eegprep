"""Shared EEGLAB-style rejection helpers."""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
from scipy import signal, stats

from eegprep.functions.popfunc._plot_utils import component_activations as plot_component_activations
from eegprep.functions.popfunc._pop_utils import parse_numeric_sequence


def copy_eeg(EEG: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy suitable for pop-function mutation."""
    return copy.deepcopy(EEG)


def one_based_indices(value: Any, *, limit: int, default_all: bool = False) -> list[int]:
    """Normalize EEGLAB-facing 1-based indices."""
    values = parse_numeric_sequence(value, dtype=int)
    if not values and default_all:
        values = list(range(1, limit + 1))
    if any(item < 1 or item > limit for item in values):
        raise ValueError("Index out of range")
    return values


def component_rejection_flags(
    EEG: dict[str, Any],
    component_total: int,
    *,
    create: bool = False,
) -> np.ndarray:
    """Return ``EEG.reject.gcompreject`` as a component-length boolean vector."""
    count = int(component_total)
    if count < 0:
        raise ValueError("component_total must be non-negative")
    reject = EEG.get("reject")
    raw = [] if not isinstance(reject, dict) else reject.get("gcompreject", [])
    flags = np.asarray(raw, dtype=bool).ravel()
    if flags.size != count:
        flags = np.zeros(count, dtype=bool)
    else:
        flags = np.array(flags, dtype=bool, copy=True)
    if create:
        stored_reject = reject if isinstance(reject, dict) else {}
        stored_reject["gcompreject"] = flags.astype(int)
        EEG["reject"] = stored_reject
    return flags


def set_component_rejection_flag(
    EEG: dict[str, Any],
    component_index: int,
    rejected: bool,
    component_total: int,
) -> np.ndarray:
    """Set one EEGLAB-facing component rejection flag and return all flags."""
    index = int(component_index)
    count = int(component_total)
    if index < 1 or index > count:
        raise ValueError("component index is outside available ICA components")
    flags = component_rejection_flags(EEG, count, create=False).astype(int)
    flags[index - 1] = int(bool(rejected))
    reject = EEG.get("reject")
    if not isinstance(reject, dict):
        reject = {}
    reject["gcompreject"] = flags
    EEG["reject"] = reject
    return flags


def expand_values(values: Any, count: int, default: float = 0.0) -> np.ndarray:
    """Expand scalar/short EEGLAB threshold vectors to ``count`` entries."""
    parsed = parse_numeric_sequence(values, dtype=float)
    if not parsed:
        parsed = [default]
    if len(parsed) < count:
        parsed.extend([parsed[-1]] * (count - len(parsed)))
    return np.asarray(parsed[:count], dtype=float)


def bool_marks(value: Any, trials: int) -> np.ndarray:
    """Normalize trial flags or 1-based trial indices to a boolean vector."""
    values = np.asarray(value if value is not None else [], dtype=float).ravel()
    marks = np.zeros(trials, dtype=bool)
    if values.size == 0:
        return marks
    if values.size == trials and np.all(np.isin(values, [0, 1, False, True])):
        return values.astype(bool)
    indices = values.astype(int)
    if np.any(indices < 1) or np.any(indices > trials):
        raise ValueError("Trial index out of range")
    marks[indices - 1] = True
    return marks


def data_3d(EEG: dict[str, Any]) -> np.ndarray:
    """Return channel data as channels x points x trials."""
    data = np.asarray(EEG.get("data"))
    if data.ndim == 2:
        return data[:, :, np.newaxis]
    if data.ndim == 3:
        return data
    raise ValueError("EEG data must be 2-D or 3-D")


def component_activations(EEG: dict[str, Any]) -> np.ndarray:
    """Return ICA activations as components x points x trials.

    Rejection always recomputes from the weight and sphere matrices (ignoring a
    stored ``EEG['icaact']``) so scoring matches EEGLAB's rejection path.
    """
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    sphere = np.asarray(EEG.get("icasphere", []), dtype=float)
    if weights.size == 0 or sphere.size == 0:
        raise ValueError("ICA decomposition is required")
    return plot_component_activations(EEG, use_stored=False)


def rejection_data(EEG: dict[str, Any], icacomp: int | bool) -> tuple[np.ndarray, int]:
    """Return data to reject on and row count for data/component modes."""
    if int(bool(icacomp)):
        data = data_3d(EEG)
        return data, int(data.shape[0])
    data = component_activations(EEG)
    return data, int(data.shape[0])


def eegthresh_marks(
    data: np.ndarray,
    elecrange: list[int],
    negthresh: Any,
    posthresh: Any,
    timerange: tuple[float, float],
    starttime: Any,
    endtime: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return EEGLAB-style epoch/channel marks for amplitude thresholding."""
    selected = [item - 1 for item in elecrange]
    count = len(selected)
    neg = expand_values(negthresh, count)
    pos = expand_values(posthresh, count)
    starts = expand_values(starttime, count, timerange[0])
    ends = expand_values(endtime, count, timerange[1])
    trials = data.shape[2]
    row_marks = np.zeros((data.shape[0], trials), dtype=bool)
    reject = np.zeros(trials, dtype=bool)
    for out_index, row_index in enumerate(selected):
        first, last = _time_window(data.shape[1], timerange, starts[out_index], ends[out_index])
        segment = data[row_index, first:last, :]
        marks = (np.min(segment, axis=0) < neg[out_index]) | (np.max(segment, axis=0) > pos[out_index])
        row_marks[row_index, :] = marks
        reject |= marks
    return reject, row_marks


def jointprob_marks(
    data: np.ndarray,
    elecrange: list[int],
    locthresh: Any,
    globthresh: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local/global joint-probability marks."""
    selected = [item - 1 for item in elecrange]
    local_scores, local_reject = jointprob(data[selected], locthresh, normalize=1)
    row_marks = np.zeros((data.shape[0], data.shape[2]), dtype=bool)
    row_marks[selected, :] = local_reject
    # Match EEGLAB pop_jointprob.m: trials become rows, then jointprob()
    # normalizes that 2-D score vector across trial rows.
    global_data = data[selected].transpose(2, 0, 1).reshape(data.shape[2], -1)
    global_scores, global_reject = jointprob(global_data, globthresh, normalize=1)
    reject = local_reject.any(axis=0) | global_reject.ravel()
    return reject, row_marks, local_scores, global_scores.ravel()


def kurtosis_marks(
    data: np.ndarray,
    elecrange: list[int],
    locthresh: Any,
    globthresh: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return local/global kurtosis marks."""
    selected = [item - 1 for item in elecrange]
    local_scores, local_reject = rejkurt(data[selected], locthresh, normalize=1)
    row_marks = np.zeros((data.shape[0], data.shape[2]), dtype=bool)
    row_marks[selected, :] = local_reject
    # Match EEGLAB pop_rejkurt.m global mode: trials become rows before
    # rejkurt() normalizes that 2-D score vector across trial rows.
    global_data = data[selected].transpose(2, 0, 1).reshape(data.shape[2], -1)
    global_scores, global_reject = rejkurt(global_data, globthresh, normalize=1)
    reject = local_reject.any(axis=0) | global_reject.ravel()
    return reject, row_marks, local_scores, global_scores.ravel()


def trend_marks(
    data: np.ndarray, elecrange: list[int], winsize: int, maxslope: float, min_r: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear-trend rejection marks."""
    selected = [item - 1 for item in elecrange]
    winsize = int(winsize)
    if winsize < 2 or winsize > data.shape[1]:
        raise ValueError("winsize must be between 2 and EEG.pnts")
    row_marks = np.zeros((data.shape[0], data.shape[2]), dtype=bool)
    x = np.linspace(1 / winsize, 1, winsize)
    x_centered = x - np.mean(x)
    x_denom = float(np.sum(x_centered**2))
    tolerance = 1000 * winsize * 1.1921e-7
    starts = range(0, data.shape[1] - winsize + 1, winsize)
    for row_index in selected:
        windows = np.stack([data[row_index, start : start + winsize, :].T for start in starts], axis=1)
        means = np.mean(windows, axis=2, keepdims=True)
        slopes = np.tensordot(windows - means, x_centered, axes=([2], [0])) / x_denom
        intercepts = means[:, :, 0] - slopes * float(np.mean(x))
        fitted = slopes[:, :, np.newaxis] * x + intercepts[:, :, np.newaxis]
        sst = np.maximum(np.sum((windows - means) ** 2, axis=2), tolerance)
        sse = np.sum((windows - fitted) ** 2, axis=2)
        row_marks[row_index, :] = ((np.abs(slopes) >= maxslope) & (1 - sse / sst > min_r)).any(axis=1)
    return row_marks.any(axis=0), row_marks


def spectrum_marks(
    data: np.ndarray,
    elecrange: list[int],
    srate: float,
    threshold: Any,
    freqlimits: Any,
    method: str = "multitaper",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spectral threshold rejection marks."""
    selected = [item - 1 for item in elecrange]
    thresholds = np.asarray(_matrix_arg(threshold, columns=2, default=[-30.0, 30.0]), dtype=float)
    freqs_limits = np.asarray(_matrix_arg(freqlimits, columns=2, default=[15.0, 30.0]), dtype=float)
    if thresholds.shape[0] < len(selected):
        thresholds = np.vstack([thresholds, np.tile(thresholds[-1], (len(selected) - thresholds.shape[0], 1))])
    if freqs_limits.shape[0] < len(selected):
        freqs_limits = np.vstack([freqs_limits, np.tile(freqs_limits[-1], (len(selected) - freqs_limits.shape[0], 1))])
    method = str(method).lower()
    if method == "fft":
        power, freqs = _fft_power(data, srate)
    elif method == "multitaper":
        power, freqs = _multitaper_power(data, srate)
    else:
        raise ValueError("method must be 'fft' or 'multitaper'")
    power = power - power.mean(axis=2, keepdims=True)
    row_marks = np.zeros((data.shape[0], data.shape[2]), dtype=bool)
    for out_index, row_index in enumerate(selected):
        low, high = freqs_limits[out_index]
        freq_mask = (freqs >= low) & (freqs <= high)
        if not freq_mask.any():
            continue
        segment = power[row_index, freq_mask, :]
        row_marks[row_index, :] = (segment.min(axis=0) < thresholds[out_index, 0]) | (
            segment.max(axis=0) > thresholds[out_index, 1]
        )
    return row_marks.any(axis=0), row_marks, power


def _fft_power(data: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray]:
    demeaned = data - data.mean(axis=1, keepdims=True)
    spectra = np.fft.rfft(demeaned, axis=1)[:, 1:, :]
    freqs = np.fft.rfftfreq(data.shape[1], d=1 / float(srate))[1:]
    power = 10 * np.log10(np.maximum(np.abs(spectra) ** 2, np.finfo(float).tiny))
    return power, freqs


def _multitaper_power(data: np.ndarray, srate: float) -> tuple[np.ndarray, np.ndarray]:
    points = data.shape[1]
    nw = min(4.0, max(0.5, (points - 1) / 2 - np.finfo(float).eps))
    kmax = max(1, min(int(2 * nw) - 1, points))
    tapers = signal.windows.dpss(points, nw, Kmax=kmax, sym=False)
    if tapers.ndim == 1:
        tapers = tapers[np.newaxis, :]
    demeaned = data - data.mean(axis=1, keepdims=True)
    tapered = demeaned[:, np.newaxis, :, :] * tapers[np.newaxis, :, :, np.newaxis]
    spectra = np.fft.rfft(tapered, axis=2)[:, :, 1:, :]
    freqs = np.fft.rfftfreq(points, d=1 / float(srate))[1:]
    power = np.mean(np.abs(spectra) ** 2, axis=1)
    return 10 * np.log10(np.maximum(power, np.finfo(float).tiny)), freqs


def jointprob(
    signal: np.ndarray, threshold: Any = 0, oldjp: Any = None, normalize: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Port of EEGLAB ``jointprob`` for deterministic rejection scoring."""
    if oldjp is not None and np.asarray(oldjp).size:
        scores = np.asarray(oldjp, dtype=float)
    else:
        signal_ndim = np.asarray(signal).ndim
        arr = _as_rows_points_trials(signal)
        scores = np.zeros((arr.shape[0], arr.shape[2]), dtype=float)
        for row in range(arr.shape[0]):
            probabilities = _realproba(arr[row].reshape(-1, order="F"))
            for trial in range(arr.shape[2]):
                data_prob = probabilities[trial * arr.shape[1] : (trial + 1) * arr.shape[1]]
                scores[row, trial] = -np.sum(np.log(np.maximum(data_prob, np.finfo(float).tiny)))
        scores = _normalize_scores(scores, int(normalize), signal_ndim=signal_ndim)
    reject = _threshold_scores(scores, threshold)
    return scores, reject


def rejkurt(
    signal: np.ndarray, threshold: Any = 0, oldkurtosis: Any = None, normalize: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Port of EEGLAB ``rejkurt`` for deterministic rejection scoring."""
    if oldkurtosis is not None and np.asarray(oldkurtosis).size:
        scores = np.asarray(oldkurtosis, dtype=float)
    else:
        signal_ndim = np.asarray(signal).ndim
        arr = _as_rows_points_trials(signal)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = stats.kurtosis(arr, axis=1, fisher=False, bias=True)
        scores = np.asarray(scores, dtype=float)
        if scores.ndim == 1:
            scores = scores[:, np.newaxis]
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = _normalize_scores(scores, int(normalize), signal_ndim=signal_ndim)
    reject = _threshold_scores(scores, threshold)
    return scores, reject


def update_reject_fields(
    EEG: dict[str, Any],
    *,
    icacomp: int | bool,
    kind: str,
    reject: np.ndarray,
    reject_e: np.ndarray,
) -> None:
    """Store rejection marks in EEGLAB-compatible field names."""
    reject_struct = EEG.setdefault("reject", {})
    prefix = "" if int(bool(icacomp)) else "ica"
    reject_struct[f"{prefix}{kind}"] = np.asarray(reject, dtype=bool)
    reject_struct[f"{prefix}{kind}E"] = np.asarray(reject_e, dtype=bool)


def reject_field_names(icacomp: int | bool, kind: str) -> tuple[str, str]:
    """Return EEGLAB reject field names for a mode/kind."""
    prefix = "" if int(bool(icacomp)) else "ica"
    return f"{prefix}{kind}", f"{prefix}{kind}E"


def history_vector(value: Any) -> str:
    """Return compact MATLAB-style numeric vector text."""
    values = parse_numeric_sequence(value, dtype=float)
    if not values:
        return "[]"
    formatted = [str(int(item)) if float(item).is_integer() else f"{float(item):g}" for item in values]
    return "[" + " ".join(formatted) + "]"


def _time_window(pnts: int, timerange: tuple[float, float], start: float, end: float) -> tuple[int, int]:
    low = max(start, timerange[0])
    high = min(end, timerange[1])
    if high < low:
        raise ValueError("End time must be greater than or equal to start time")
    span = timerange[1] - timerange[0]
    if span == 0:
        return 0, pnts
    first = max(0, int(np.floor((low - timerange[0]) / span * (pnts - 1))))
    last = min(pnts, int(np.floor((high - timerange[0]) / span * (pnts - 1))) + 1)
    return first, max(first + 1, last)


def _as_rows_points_trials(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, arr.size, 1)
    elif arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim != 3:
        raise ValueError("Signal must be 1-D, 2-D, or 3-D")
    return arr


def _realproba(data: np.ndarray, bins: int = 1000) -> np.ndarray:
    data = np.asarray(data, dtype=float).ravel()
    if data.size == 0:
        return data
    minimum = float(np.nanmin(data))
    maximum = float(np.nanmax(data))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum == minimum:
        return np.ones(data.shape, dtype=float)
    indices = np.floor((data - minimum) / (maximum - minimum) * (bins - 1)).astype(int)
    indices = np.clip(indices, 0, bins - 1)
    counts = np.bincount(indices, minlength=bins).astype(float)
    return counts[indices] / data.size


def _normalize_scores(scores: np.ndarray, mode: int, *, signal_ndim: int) -> np.ndarray:
    if not mode:
        return scores
    baseline = np.asarray(scores, dtype=float)
    if mode == 2:
        sorted_scores = np.sort(baseline, axis=1)
        trim = np.maximum(np.rint(sorted_scores.shape[1] * 0.1).astype(int), 1)
        if sorted_scores.shape[1] > 2 * trim:
            baseline = sorted_scores[:, trim:-trim]
    if signal_ndim == 2:
        mean = np.mean(baseline)
        std = _nonzero_std(baseline, axis=None)
    else:
        mean = baseline.mean(axis=1, keepdims=True)
        std = _nonzero_std(baseline, axis=1)
    return (scores - mean) / std


def _nonzero_std(values: np.ndarray, *, axis: int | None) -> np.ndarray | float:
    count = values.size if axis is None else values.shape[axis]
    std = np.std(values, axis=axis, ddof=1 if count > 1 else 0, keepdims=axis is not None)
    std = np.asarray(std, dtype=float)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    if axis is None:
        return float(std)
    return std


def _threshold_scores(scores: np.ndarray, threshold: Any) -> np.ndarray:
    values = parse_numeric_sequence(threshold, dtype=float)
    if not values or values[0] == 0:
        return np.zeros(scores.shape, dtype=bool)
    if len(values) > 1:
        return (scores < values[0]) | (scores > values[-1])
    return np.abs(scores) > values[0]


def _matrix_arg(value: Any, *, columns: int, default: list[float]) -> list[list[float]]:
    if value is None:
        values = default
    elif isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == columns:
            return arr.tolist()
        values = arr.ravel().tolist()
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple, np.ndarray)):
        return [parse_numeric_sequence(row, dtype=float)[:columns] for row in value]
    else:
        values = parse_numeric_sequence(value, dtype=float)
    if not values:
        values = default
    rows = []
    for index in range(0, len(values), columns):
        row = list(values[index : index + columns])
        if len(row) < columns:
            row.extend(default[len(row) : columns])
        rows.append(row)
    return rows
