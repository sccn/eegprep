"""EEGLAB-style EEG resampling pop function."""

from copy import deepcopy
import math
from math import ceil, floor, gcd

import numpy as np
import sympy as sp
from scipy import signal
from scipy.signal import resample, resample_poly
from scipy.signal.windows import kaiser

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.plugins.firfilt import firws, firwsord


def pop_resample(
    EEG,
    freq=None,
    engine=None,
    *,
    gui=None,
    renderer=None,
    return_com=False,
    fc=None,
    df=None,
):
    """Resample EEG data to a new sampling rate.

    Parameters
    ----------
    EEG : dict
        EEGLAB EEG structure.
    freq : float
        New sampling rate in Hz.
    engine : str or None
        Engine to use for implementation. Options are:
        - None: Use the default Python implementation
        - 'poly': Use scipy's resample_poly function
        - 'matlab': Use MATLAB engine
        - 'octave': Use Octave engine

    Returns
    -------
    EEG : dict
        EEGLAB EEG structure with resampled data.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = freq is None
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        freq = result["freq"]
    if freq is None:
        raise ValueError("freq argument is required when gui=False")
    freq = float(freq)
    if freq <= 0:
        raise ValueError("New sampling rate must be positive")
    fc = 0.9 if fc is None else fc
    df = 0.2 if df is None else df

    if isinstance(EEG, list):
        output = [
            pop_resample(item, freq, engine=engine, gui=False, fc=fc, df=df)
            for item in EEG
        ]
        command = _history_command(freq)
        return (output, command) if return_com else output

    # Check if using MATLAB or Octave implementation
    if engine in ['matlab', 'octave']:
        eeglab = get_eeglab(runtime='MAT' if engine == 'matlab' else 'OCT')
        EEG_new = eeglab.pop_resample(EEG, freq)
        command = _history_command(freq)
        return (EEG_new, command) if return_com else EEG_new

    if engine not in {None, "poly", "scipy"}:
        raise ValueError("Unsupported engine: {engine}. Should be None, 'poly', 'scipy', 'matlab', or 'octave'".format(engine=engine))
    EEG_new = resample_eeg(EEG, freq, method="poly" if engine is None else engine, fc=fc, df=df)
    command = _history_command(freq)
    return (EEG_new, command) if return_com else EEG_new


def pop_resample_dialog_spec(srate) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_resample``."""
    return DialogSpec(
        title="Resample current dataset -- pop_resample()",
        function_name="pop_resample",
        eeglab_source="functions/popfunc/pop_resample.m",
        geometry=((1,), (1,)),
        size=(300, 199),
        help_text="pophelp('pop_resample')",
        controls=(
            ControlSpec("text", "New sampling rate"),
            ControlSpec(
                "edit",
                tag="freq",
                value=f"{float(srate):g}",
                callback=CallbackSpec("validate_numeric_range", params={"columns": 1, "lower": 0, "upper": np.inf}),
            ),
        ),
    )


def _run_gui(EEG, renderer=None):
    spec = pop_resample_dialog_spec(EEG.get("srate", 1))
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    text = str(result.get("freq", "")).strip()
    if not text:
        return None
    return {"freq": float(text)}


def _history_command(freq):
    return f"EEG = pop_resample( EEG, {_format_number(freq)});"


def _format_number(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def resample_eeg(EEG, freq, method='poly', fc=0.9, df=0.2):
    """Port of EEGLAB's pop_resample behavior.

    Parameters
    ----------
    EEG : dict
        EEGLAB EEG structure.
    freq : float
        New sampling rate in Hz.
    method : str
        Resampling method. Options are 'poly' or 'octave'.
    fc : float
        Anti-aliasing filter cutoff frequency.
    df : float
        Transition width of the filter.

    Returns
    -------
    EEG : dict
        EEGLAB EEG structure with resampled data.
    """
    if not 0 <= fc <= 1:
        raise ValueError("Anti-aliasing filter cutoff frequency out of range.")
    if method not in {"poly", "scipy", "octave"}:
        raise ValueError(f"Unsupported method: {method}. Should be 'poly', 'scipy', or 'octave'")

    p, q = _resample_ratio(freq, EEG["srate"])
    ratio = p / q
    data = np.asarray(EEG["data"])
    if data.ndim not in {2, 3}:
        raise ValueError("pop_resample supports continuous or epoched EEG data")
    old_pnts = int(EEG.get("pnts", data.shape[1]))
    data_3d = data[:, :, np.newaxis] if data.ndim == 2 else data
    bounds = _segment_bounds(EEG, old_pnts) if data_3d.shape[2] == 1 else np.asarray([1, old_pnts + 1], dtype=int)
    segments = []
    indices = [1]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        segment = data_3d[:, start - 1:stop - 1, :]
        resampled = _resample_segment(segment, p, q, method=method, fc=fc, df=df)
        segments.append(resampled)
        indices.append(indices[-1] + resampled.shape[1])
    resampled_data = np.concatenate(segments, axis=1) if segments else data_3d[:, :0, :]

    output = deepcopy(EEG)
    output["data"] = resampled_data[:, :, 0] if data.ndim == 2 else resampled_data
    output["pnts"] = int(resampled_data.shape[1])
    output["trials"] = int(resampled_data.shape[2])
    output["srate"] = float(freq)
    output["xmin"] = float(output.get("xmin", EEG.get("xmin", 0.0)) or 0.0)
    output["xmax"] = output["xmin"] + ((output["pnts"] - 1) / output["srate"] if output["pnts"] else 0.0)
    output["times"] = np.linspace(output["xmin"] * 1000, output["xmax"] * 1000, output["pnts"]) if output["pnts"] else np.array([])
    _resample_event_latencies(output, old_pnts, ratio, np.asarray(bounds), indices)
    output["icaact"] = np.array([])
    if output.get("setname"):
        output["setname"] = f"{output['setname']} resampled"
    output["saved"] = "no"
    return output


def _resample_ratio(freq, srate):
    rational_approx = sp.nsimplify(float(freq) / float(srate), tolerance=1e-12)
    p, q = rational_approx.as_numer_denom()
    return int(p), int(q)


def _segment_bounds(EEG, old_pnts):
    bounds = [1]
    for event in events_to_records(EEG.get("event")):
        if not _is_boundary_event(event):
            continue
        try:
            latency = float(event.get("latency"))
        except (TypeError, ValueError):
            continue
        if latency <= 0 or latency > old_pnts:
            continue
        if not latency.is_integer():
            latency = round(latency + 0.5)
        bounds.append(int(latency))
    bounds.append(old_pnts + 1)
    return np.asarray(sorted(set(bounds)), dtype=int)


def _is_boundary_event(event):
    event_type = event.get("type") if isinstance(event, dict) else None
    if isinstance(event_type, str):
        return event_type.lower().startswith("boundary")
    return bool(EEG_OPTIONS.get("option_boundary99")) and event_type == -99


def _resample_segment(segment, p, q, *, method, fc, df):
    if segment.shape[1] < 2:
        return segment.astype(np.float32, copy=True)
    if method == "scipy":
        return resample(segment.astype(np.float64), int(np.ceil(segment.shape[1] * p / q)), axis=1).astype(np.float32)
    if method == "octave":
        flattened = segment.transpose(1, 0, 2).reshape(segment.shape[1], -1)
        resampled, _h = resample_raw(flattened.astype(np.float64), p, q)
        return resampled.reshape(resampled.shape[0], segment.shape[0], segment.shape[2]).transpose(1, 0, 2).astype(np.float32)
    return _resample_poly_segment(segment, p, q, fc=fc, df=df)


def _resample_poly_segment(segment, p, q, *, fc, df):
    nyq = 1 / np.maximum(p, q)
    cutoff = fc * nyq
    transition = df * nyq
    m, _ = firwsord("kaiser", 2, transition, 0.002)
    wnd = kaiser(m + 1, beta=5)
    b, _ = firws(m, cutoff, w=wnd)
    n_pad = int(np.ceil((m / 2) / q) * q)
    pad_width = [(0, 0), (n_pad, n_pad), *[(0, 0) for _ in range(segment.ndim - 2)]]
    padded = np.pad(segment, pad_width, mode="edge").astype(np.float64)
    resampled = resample_poly(padded, p, q, axis=1, window=b).astype(np.float32)
    n_pad_after = n_pad * p // q
    if n_pad_after == 0:
        return resampled
    return resampled[:, n_pad_after:-n_pad_after, :]


def _resample_event_latencies(output, old_pnts, ratio, bounds, indices):
    events = events_to_records(output.get("event"))
    urevents = events_to_records(output.get("urevent"))
    if output["trials"] > 1:
        _resample_epoched_events(events, old_pnts, output["pnts"], ratio)
        output["urevent"] = []
    else:
        _resample_continuous_events(events, bounds, indices, ratio)
        _resample_continuous_events(urevents, bounds, indices, ratio)
        output["urevent"] = urevents
    output["event"] = events


def _resample_epoched_events(events, old_pnts, new_pnts, ratio):
    for event in events:
        if "latency" not in event:
            continue
        epoch = int(event.get("epoch", 1) or 1)
        event["latency"] = (float(event["latency"]) - (epoch - 1) * old_pnts - 1) * ratio + (epoch - 1) * new_pnts + 1
        _scale_duration(event, ratio)


def _resample_continuous_events(events, bounds, indices, ratio):
    for event in events:
        if "latency" not in event:
            continue
        latency = float(event["latency"])
        if _is_boundary_event(event) and abs(latency % 1 - 0.5) < 1e-12:
            segment_index = _segment_index(bounds, latency + 0.5)
            event["latency"] = indices[segment_index] - 0.5
        else:
            segment_index = _segment_index(bounds, latency)
            event["latency"] = (latency - bounds[segment_index]) * ratio + indices[segment_index]
        _scale_duration(event, ratio)


def _segment_index(bounds, latency):
    index = int(np.searchsorted(bounds, latency, side="right") - 1)
    return max(0, min(index, len(bounds) - 2))


def _scale_duration(event, ratio):
    if "duration" not in event or event["duration"] in (None, ""):
        return
    event["duration"] = float(event["duration"]) * ratio


def upfirdn_raw(x, h, p, q):
    """Upfirdn implementation for resampling.

    Parameters
    ----------
    x : array_like
        Input signal.
    h : array_like
        Filter coefficients.
    p : int
        Upsampling factor.
    q : int
        Downsampling factor.

    Returns
    -------
    y : ndarray
        Filtered and resampled signal.
    """
    # Ensure x is a numpy array and h is 1D.
    x = np.array(x, copy=True)
    h = np.array(h).flatten()

    # If x is a row vector, convert it to a column vector.
    is_row_vector = False
    if x.ndim == 2 and x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T
        is_row_vector = True

    rx, cx = x.shape
    Lh = h.size
    Ly = math.ceil(((rx - 1) * p + Lh) / q)
    y = np.zeros((Ly, cx))

    for c in range(cx):
        for m in range(Ly):
            n = (m * q) // p
            lm = (m * q) % p
            # k goes from max(0, n - rx + 1) to n (inclusive)
            for k in range(max(0, n - rx + 1), n + 1):
                if k * p + lm < Lh:
                    y[m, c] += h[k * p + lm] * x[n - k, c]

    if is_row_vector:
        y = y.T

    return y

def resample_raw(x, p, q, h=None):
    """Change the sample rate of x by a factor of p/q.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    p : int
        The upsampling factor.
    q : int
        The downsampling factor.
    h : array_like, optional
        The filter coefficients. If not provided, a Kaiser-windowed sinc filter is used.

    Returns
    -------
    y : ndarray
        The resampled array. If input is a vector, output will be a vector.
    h : ndarray
        The filter coefficients used.
    """
    # Input validation
    if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)):
        raise ValueError("p and q must be positive integers")
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive integers")

    # Convert x to numpy array and handle row vectors
    x = np.asarray(x)
    input_shape = x.shape
    is_1d = x.ndim == 1

    # Reshape input to 2D array with shape (samples, channels)
    if is_1d:
        x = x.reshape(-1, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x = x.T

    # Simplify decimation and interpolation factors
    great_common_divisor = gcd(p, q)
    if great_common_divisor > 1:
        p = p // great_common_divisor
        q = q // great_common_divisor

    # Filter design if required
    if h is None:
        # Properties of the antialiasing filter
        log10_rejection = -3.0
        stopband_cutoff_f = 1.0 / (2.0 * max(p, q))
        roll_off_width = stopband_cutoff_f / 10.0

        # Determine filter length
        rejection_dB = -20.0 * log10_rejection
        L = ceil((rejection_dB - 8.0) / (28.714 * roll_off_width))

        # Ideal sinc filter
        t = np.arange(-L, L + 1)
        ideal_filter = 2 * p * stopband_cutoff_f * np.sinc(2 * stopband_cutoff_f * t)

        # Determine parameter of Kaiser window
        if 21 <= rejection_dB <= 50:
            beta = 0.5842 * (rejection_dB - 21.0)**0.4 + 0.07886 * (rejection_dB - 21.0)
        elif rejection_dB > 50:
            beta = 0.1102 * (rejection_dB - 8.7)
        else:
            beta = 0.0

        # Apply Kaiser window to ideal filter
        h = ideal_filter * signal.windows.kaiser(2 * L + 1, beta)

    if not np.isrealobj(h):
        raise ValueError("The filter h should be a real vector")

    h = np.asarray(h)
    if h.ndim != 1:
        raise ValueError("The filter h should be a vector")

    Lx = x.shape[0]
    Lh = len(h)
    L = (Lh - 1) / 2.0
    Ly = ceil(Lx * p / q)

    # Pre and postpad filter response
    nz_pre = floor(q - np.mod(L, q))
    h_padded = np.pad(h, (nz_pre, 0), 'constant')

    offset = floor((L + nz_pre) / q)
    nz_post = 0
    while ceil(((Lx - 1) * p + nz_pre + Lh + nz_post) / q) - offset < Ly:
        nz_post += 1
    h_padded = np.pad(h_padded, (0, nz_post), 'constant')

    # Filtering - fixed upfirdn usage
    y = upfirdn_raw(x, h_padded, p, q)
    y = y[offset:offset + Ly]

    # Restore original dimensionality
    if is_1d:
        y = y.flatten()
    else:
        y = y.reshape(-1, x.shape[1])

    return y, h
