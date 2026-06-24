"""EEGLAB-style EEG resampling pop function."""

from copy import deepcopy
import logging
from math import ceil

import numpy as np
import sympy as sp
from scipy.signal import resample, resample_poly
from scipy.signal.windows import kaiser

from eegprep.functions.adminfunc.eeglabcompat import get_eeglab
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.miscfunc.event_utils import is_boundary_event as _shared_is_boundary_event
from eegprep.functions.miscfunc.parity import resample_raw
from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.plugins.firfilt.firws import firws
from eegprep.plugins.firfilt.firwsord import firwsord


logger = logging.getLogger(__name__)


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
        output = []
        for index, item in enumerate(EEG, start=1):
            logger.info("Processing group dataset %s of %s.", index, len(EEG))
            output.append(pop_resample(item, freq, engine=engine, gui=False, fc=fc, df=df))
        command = _history_command(freq)
        return (output, command) if return_com else output

    # Check if using MATLAB or Octave implementation
    if engine in ['matlab', 'octave']:
        eeglab = get_eeglab(runtime='MAT' if engine == 'matlab' else 'OCT')
        EEG_new = eeglab.pop_resample(EEG, freq)
        command = _history_command(freq)
        return (EEG_new, command) if return_com else EEG_new

    if engine not in {None, "poly", "scipy"}:
        raise ValueError(
            "Unsupported engine: {engine}. Should be None, 'poly', 'scipy', 'matlab', or 'octave'".format(engine=engine)
        )
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

    logger.info("resampling data %g Hz", float(freq))
    p, q = _resample_ratio(freq, EEG["srate"])
    ratio = p / q
    data = np.asarray(EEG["data"])
    if data.ndim not in {2, 3}:
        raise ValueError("pop_resample supports continuous or epoched EEG data")
    old_pnts = int(EEG.get("pnts", data.shape[1]))
    data_3d = data[:, :, np.newaxis] if data.ndim == 2 else data
    bounds = _segment_bounds(EEG, old_pnts) if data_3d.shape[2] == 1 else np.asarray([1, old_pnts + 1], dtype=int)
    if len(bounds) > 2:
        logger.info("Data break detected; resampling continuous segments separately.")
    segments = []
    indices = [1]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        segment = data_3d[:, start - 1 : stop - 1, :]
        logger.info("resampling channel data segment %s of %s", len(segments) + 1, len(bounds) - 1)
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
    output["times"] = (
        np.linspace(output["xmin"] * 1000, output["xmax"] * 1000, output["pnts"]) if output["pnts"] else np.array([])
    )
    logger.info("resampling event latencies...")
    _resample_event_latencies(output, old_pnts, ratio, np.asarray(bounds), indices, EEG)
    output["icaact"] = np.array([])
    if output.get("setname"):
        output["setname"] = f"{output['setname']} resampled"
    output["saved"] = "no"
    logger.info("resampling finished")
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
            latency = ceil(latency)
        bounds.append(int(latency))
    bounds.append(old_pnts + 1)
    return np.asarray(sorted(set(bounds)), dtype=int)


def _is_boundary_event(event):
    return isinstance(event, dict) and _shared_is_boundary_event(event)


def _resample_segment(segment, p, q, *, method, fc, df):
    if segment.shape[1] < 2:
        return segment.astype(np.float32, copy=True)
    if method == "scipy":
        return resample(segment.astype(np.float64), int(np.ceil(segment.shape[1] * p / q)), axis=1).astype(np.float32)
    if method == "octave":
        flattened = segment.transpose(1, 0, 2).reshape(segment.shape[1], -1)
        resampled, _h = resample_raw(flattened.astype(np.float64), p, q)
        return (
            resampled.reshape(resampled.shape[0], segment.shape[0], segment.shape[2])
            .transpose(1, 0, 2)
            .astype(np.float32)
        )
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


def _resample_event_latencies(output, old_pnts, ratio, bounds, indices, original):
    events = events_to_records(output.get("event"))
    urevents = events_to_records(output.get("urevent"))
    if output["trials"] > 1:
        _resample_epoched_events(events, old_pnts, output["pnts"], ratio)
        output["urevent"] = _restore_event_container(original.get("urevent"), [])
    else:
        _resample_continuous_events(events, bounds, indices, ratio)
        _resample_continuous_events(urevents, bounds, indices, ratio)
        output["urevent"] = _restore_event_container(original.get("urevent"), urevents)
    output["event"] = _restore_event_container(original.get("event"), events)


def _restore_event_container(original_events, events):
    if isinstance(original_events, np.ndarray):
        return np.asarray(events, dtype=object)
    if isinstance(original_events, dict):
        return events[0] if events else {}
    return events


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
