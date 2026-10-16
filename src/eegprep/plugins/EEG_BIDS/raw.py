"""Raw EEG file readers used by EEG-BIDS import workflows."""

from __future__ import annotations

from collections.abc import Callable
import copy
import logging
import os
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import ToolError
from eegprep.functions.popfunc.pop_loadbv import pop_loadbv

logger = logging.getLogger(__name__)


def load_raw_eeg_file(
    filename: str,
    *,
    dtype: np.dtype,
    numeric_null: Any,
    warning: Callable[[str], None],
    verbose: bool = True,
) -> tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]:
    """Load a supported BIDS raw EEG file into an EEG dictionary."""
    _path, ext = os.path.splitext(filename)
    ext = ext.lower()
    basename = os.path.basename(filename)
    report: dict[str, Any] = {}

    if ext == ".set":
        from eegprep.functions.popfunc.pop_loadset import pop_loadset

        eeg = pop_loadset(filename)
        eeg["data"] = eeg["data"].astype(dtype)
        report["ImporterUsed"] = "pop_loadset"
        srate = eeg["srate"]
        times_sec = eeg["times"] / 1000.0
        return eeg, srate, times_sec, report

    if ext == ".vhdr":
        eeg, srate, times_sec, raw_report = _load_brainvision_file(filename, dtype=dtype)
        report.update(raw_report)
        return eeg, srate, times_sec, report

    if ext in [".edf", ".bdf"]:
        eeg, srate, times_sec, raw_report = _load_neo_raw_file(
            filename,
            ext=ext,
            basename=basename,
            dtype=dtype,
            numeric_null=numeric_null,
            warning=warning,
            verbose=verbose,
        )
        report.update(raw_report)
        return eeg, srate, times_sec, report

    if ext in [".fdt", ".vmrk", ".eeg"]:
        raise ValueError(
            f"pop_load_frombids should be called with the main data file, but was called on a sidecar file: {filename}."
        )
    raise ValueError(f"Unsupported file format: {ext}. Supported formats are .set, .edf, .bdf, .vhdr.")


def _load_brainvision_file(
    filename: str,
    *,
    dtype: np.dtype,
) -> tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]:
    eeg = pop_loadbv(filename)
    eeg["data"] = np.asarray(eeg["data"], dtype=dtype)
    srate = float(np.asarray(eeg["srate"]).reshape(-1)[0])
    times_sec = np.asarray(eeg["times"], dtype=float) / 1000.0
    # BrainVision stores the reference on each channel. The dataset reference
    # from pop_loadbv is EEGLAB's default, so a BIDS EEGReference can still apply.
    if eeg.get("ref") == "common":
        eeg["ref"] = "unknown"
    eeg["chaninfo"].setdefault("nosedir", "+X")
    return eeg, srate, times_sec, {"ImporterUsed": "pop_loadbv"}


def _load_neo_raw_file(
    filename: str,
    *,
    ext: str,
    basename: str,
    dtype: np.dtype,
    numeric_null: Any,
    warning: Callable[[str], None],
    verbose: bool,
) -> tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]:
    from neo import NeoReadWriteError
    from neo.rawio.edfrawio import EDFRawIO

    if ext not in {".edf", ".bdf"}:
        raise ValueError(f"Unexpected file format: {ext}. Please add support for this format if needed.")

    importer_used = "neo.rawio.edfrawio.EDFRawIO"
    io = EDFRawIO(filename)
    try:
        io.parse_header()
    except NeoReadWriteError as exc:
        classname = io.__class__.__name__
        raise ToolError(
            f"Encountered error with NEO {classname} importer on {filename!r}: {exc}. Skipping file."
        ) from exc

    if (n_streams := io.signal_streams_count()) > 1:
        warning(f"The raw data file {filename} appears to contain more than one stream; using only the first stream.")
    elif not n_streams:
        raise ValueError(f"The raw data file {filename} does not contain any data.")
    if (n_blocks := io.block_count()) > 1:
        warning(
            f"The raw data file {filename} appears to contain "
            f"more than one recording; this is not meaningful "
            f"in a BIDS context; using only the first block."
        )
    elif not n_blocks:
        raise ValueError(f"The raw data file {filename} does not contain any data.")
    if (n_segments := io.segment_count(0)) > 1:
        raise NotImplementedError(
            f"The raw data file {filename} appears to contain "
            f"more than one segment; This importer currently "
            f"only supports continuous EEG data."
        )
    elif not n_segments:
        raise ValueError(f"The raw data file {filename} does not contain any data.")

    n_channels = io.signal_channels_count(0)
    n_samples = io.get_signal_size(0, 0, 0)
    channel_indexes = list(range(n_channels))
    report = {
        "ImporterUsed": importer_used,
        "NumStreams": n_streams,
        "NumBlocks": n_blocks,
        "NumSegments": n_segments,
    }

    if verbose:
        logger.info("  retrieving EEG data from file...")
    data_t = io.get_analogsignal_chunk(
        block_index=0,
        seg_index=0,
        channel_indexes=channel_indexes,
        i_start=None,
        i_stop=None,
    )
    old_scale = np.std(data_t, axis=0)
    data_t = io.rescale_signal_raw_to_float(data_t, dtype=dtype, channel_indexes=channel_indexes)
    new_scale = np.std(data_t, axis=0)
    scale_ratios = new_scale / old_scale
    unique_ratios = np.unique(scale_ratios)
    if len(unique_ratios) == 1:
        report["ScaleApplied"] = unique_ratios.item()
    else:
        report["ScalesApplied"] = scale_ratios.tolist()

    srate = io.get_signal_sampling_rate(0)
    t0 = io.get_signal_t_start(block_index=0, seg_index=0, stream_index=0)
    report["RawStartTime"] = t0
    time_offset = getattr(io, "_global_time", 0.0)
    report["StartTimeOffset"] = time_offset
    t0 += time_offset
    report["CombinedStartTime"] = t0
    times_sec = t0 + np.arange(0, n_samples, dtype=float) / srate

    channels = io.header["signal_channels"]
    try:
        units = channels["units"].tolist()
    except KeyError:
        units = ["uV"] * n_channels
    unique_units = np.unique(units)
    if len(unique_units) == 1 and unique_units[0] not in ("uV", "microvolts"):
        warning(
            f"Your channel unit does not appear to be in microvolts (uV) "
            f"but is documented instead as {unique_units[0]}. EEG scale might be incorrect. "
        )

    labels = channels["name"].tolist()
    chanlocs = np.asarray([_empty_chanloc(label, numeric_null) for label in labels])
    events = _read_neo_events(io, ext, times_sec, numeric_null, verbose=verbose)

    eeg = {
        "setname": "",
        "filename": basename,
        "filepath": os.path.dirname(filename),
        "subject": "",
        "group": "",
        "condition": "",
        "session": numeric_null,
        "comments": "",
        "nbchan": n_channels,
        "trials": 1,
        "pnts": n_samples,
        "srate": srate,
        "xmin": times_sec[0],
        "xmax": times_sec[-1],
        "times": times_sec * 1000,
        "data": data_t.T,
        "icaact": numeric_null,
        "icawinv": numeric_null,
        "icasphere": numeric_null,
        "icaweights": numeric_null,
        "icachansind": numeric_null,
        "chanlocs": chanlocs,
        "urchanlocs": numeric_null,
        "chaninfo": {
            "plotrad": numeric_null,
            "shrink": numeric_null,
            "nosedir": "+X",
            "nodatchans": numeric_null,
            "icachansind": numeric_null,
        },
        "ref": "unknown",
        "event": events,
        "urevent": copy.deepcopy(events),
        "eventdescription": [],
        "epoch": numeric_null,
        "epochdescription": [],
        "reject": {},
        "stats": {},
        "specdata": numeric_null,
        "specicaact": numeric_null,
        "splinefile": "",
        "icasplinefile": "",
        "dipfit": numeric_null,
        "history": "",
        "saved": "justloaded",
        "etc": {},
        "run": numeric_null,
    }
    return eeg, srate, times_sec, report


def _empty_chanloc(label: str, numeric_null: Any) -> dict[str, Any]:
    return {
        "labels": label,
        "sph_radius": numeric_null,
        "sph_theta": numeric_null,
        "sph_phi": numeric_null,
        "theta": numeric_null,
        "radius": numeric_null,
        "X": numeric_null,
        "Y": numeric_null,
        "Z": numeric_null,
        "type": "EEG",
        "ref": numeric_null,
    }


def _read_neo_events(io: Any, ext: str, times_sec: np.ndarray, numeric_null: Any, *, verbose: bool) -> Any:
    if (event_channels := io.event_channels_count()) <= 0:
        return numeric_null

    if verbose:
        logger.info("  reading in event data from EEG file...")
    all_times = []
    all_durations = []
    all_channels = []
    all_data = []
    for event_channel_index in range(event_channels):
        event_times, event_durations, event_labels = io.get_event_timestamps(
            block_index=0,
            seg_index=0,
            event_channel_index=event_channel_index,
            t_start=None,
            t_stop=None,
        )
        all_times.extend(io.rescale_event_timestamp(event_times))
        if event_durations is not None:
            all_durations.extend(event_durations)
        else:
            all_durations.extend([1] * len(event_times))
        all_channels.extend(np.repeat(io.header["event_channels"][event_channel_index]["name"], len(event_times)))
        all_data.extend(event_labels)

    if ext not in {".edf", ".bdf"}:
        raise ValueError(f"Unsupported file format for event extraction: {ext}. Supported formats are .edf, .bdf.")
    event_types = [str(value) for value in all_data]
    event_codes = [str(channel) for channel in all_channels]

    # 1-based sample indices (EEGLAB convention), same rounding as the events.tsv path
    event_latencies = np.searchsorted(times_sec, all_times) + 1
    event_durations = np.array(all_durations, dtype=float)
    urevents = np.arange(len(all_times))
    return np.array(
        [
            {
                "duration": duration,
                "latency": latency,
                "type": event_type or ("boundary" if code == "New Segment" else ""),
                "code": code,
                "urevent": urevent,
            }
            for duration, latency, event_type, code, urevent in zip(
                event_durations,
                event_latencies,
                event_types,
                event_codes,
                urevents,
            )
        ]
    )
