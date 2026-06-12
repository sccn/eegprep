"""Raw EEG file readers used by EEG-BIDS import workflows."""

from __future__ import annotations

from collections.abc import Callable
import copy
import logging
import os
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import ToolError

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

    if ext in [".edf", ".bdf", ".vhdr"]:
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

    if ext == ".vhdr":
        from neo.rawio.brainvisionrawio import BrainVisionRawIO as NeoIO

        importer_used = "neo.rawio.brainvisionrawio.BrainVisionRawIO"
    elif ext in [".edf", ".bdf"]:
        from neo.rawio.edfrawio import EDFRawIO as NeoIO

        importer_used = "neo.rawio.edfrawio.EDFRawIO"
    else:
        raise ValueError(f"Unexpected file format: {ext}. Please add support for this format if needed.")

    io = NeoIO(filename)
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
    _apply_neo_channel_coordinates(io, ext, filename, chanlocs, n_channels, warning=warning, verbose=verbose)
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


def _apply_neo_channel_coordinates(
    io: Any,
    ext: str,
    filename: str,
    chanlocs: np.ndarray,
    n_channels: int,
    *,
    warning: Callable[[str], None],
    verbose: bool,
) -> None:
    if ext == ".vhdr":
        if verbose:
            logger.info("  parsing VHDR-specific channel locations...")
        try:
            annots = io.raw_annotations["blocks"][0]["segments"][0]["signals"][0]["__array_annotations__"]
            sph_radius = annots["coordinates_0"]
            theta = annots["coordinates_1"]
            phi = annots["coordinates_2"]
            valid = (sph_radius != 0) | (theta != 0) | (phi != 0)
            sph_theta = phi - 90 * np.sign(theta)
            sph_phi = -np.abs(theta) + 90
        except KeyError:
            warning(f"Channel coordinates not found in {filename}. Using default values for channel locations.")
            valid = np.zeros(n_channels, dtype=bool)
    elif ext in [".edf", ".bdf"]:
        valid = np.zeros(n_channels, dtype=bool)
    else:
        raise ValueError(
            f"Unsupported file format for channel coordinates extraction: {ext}. "
            f"Supported formats are .edf, .bdf, .vhdr."
        )

    if not np.any(valid):
        return

    if verbose:
        logger.info("  applying channel locations from EEG file...")
    for loc, val, sph_r, sph_p, sph_t in zip(chanlocs, valid, sph_radius, sph_phi, sph_theta):
        if not val:
            continue
        loc["sph_radius"] = sph_r
        loc["sph_theta"] = sph_t
        loc["sph_phi"] = sph_p
        az = sph_p
        horiz = sph_t
        loc["theta"] = -horiz
        loc["radius"] = 0.5 - az / 180
        az = np.deg2rad(sph_t)
        elev = np.deg2rad(sph_p)
        loc["Z"] = sph_r * np.sin(elev)
        loc["X"] = sph_r * np.cos(elev) * np.cos(az)
        loc["Y"] = sph_r * np.cos(elev) * np.sin(az)


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

    if ext == ".vhdr":
        event_types = all_data
        event_codes = all_channels
    elif ext in [".edf", ".bdf"]:
        event_types = [str(value) for value in all_data]
        event_codes = [str(channel) for channel in all_channels]
    else:
        raise ValueError(
            f"Unsupported file format for event extraction: {ext}. Supported formats are .edf, .bdf, .vhdr."
        )

    event_latencies = np.searchsorted(times_sec, all_times)
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
