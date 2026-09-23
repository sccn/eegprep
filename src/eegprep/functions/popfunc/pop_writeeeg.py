"""Export EEG data to EDF, BDF, or GDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyedflib
from pyedflib import highlevel

from eegprep.functions.popfunc._file_io import channel_labels, events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.sigprocfunc.writegdf import writegdf


def pop_writeeeg(EEG: dict[str, Any], filename: str | Path, *args: Any, **kwargs: Any) -> str:
    """Write continuous EEG data to EDF, BDF, or GDF."""
    path = Path(filename)
    options = parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    output_type = str(options.pop("type", path.suffix.lstrip("."))).lower()
    if options:
        raise ValueError(f"Unsupported pop_writeeeg option(s): {', '.join(sorted(options))}")
    if path.suffix.lower() not in {".edf", ".bdf", ".gdf"}:
        raise ValueError("pop_writeeeg output must end in .edf, .bdf, or .gdf")
    if output_type != path.suffix.lstrip(".").lower():
        raise ValueError("TYPE must match the output filename extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_type == "gdf":
        writegdf(
            path,
            EEG["data"],
            EEG["srate"],
            labels=channel_labels(EEG),
            events=events_to_records(EEG.get("event")),
            subject=str(EEG.get("subject", "")),
            recording=str(EEG.get("setname", "")),
        )
    else:
        _write_edf_family(EEG, path, output_type)
    pieces = [format_history_value(path)]
    if "type" in {str(key).lower() for key in kwargs} or args:
        pieces.extend(["'TYPE'", format_history_value(output_type.upper())])
    return f"LASTCOM = pop_writeeeg(EEG, {', '.join(pieces)});"


def _write_edf_family(EEG: dict[str, Any], path: Path, output_type: str) -> None:
    data = np.ascontiguousarray(EEG["data"])
    if data.ndim != 2:
        raise ValueError("pop_writeeeg requires continuous 2-D channel-by-sample data")
    if not np.isfinite(data).all():
        raise ValueError("pop_writeeeg requires finite EEG data")
    file_type = pyedflib.FILETYPE_BDFPLUS if output_type == "bdf" else pyedflib.FILETYPE_EDFPLUS
    digital_min, digital_max = (-8_388_608, 8_388_607) if output_type == "bdf" else (-32_768, 32_767)
    headers = [
        _signal_header(label, signal, float(EEG["srate"]), digital_min, digital_max)
        for label, signal in zip(channel_labels(EEG), data)
    ]
    annotations = [
        [
            (float(event.get("latency", 1)) - 1) / float(EEG["srate"]),
            float(event.get("duration", 0) or 0) / float(EEG["srate"]),
            str(event.get("type", "event")),
        ]
        for event in events_to_records(EEG.get("event"))
    ]
    if not highlevel.write_edf(str(path), data, headers, header={"annotations": annotations}, file_type=file_type):
        raise OSError(f"Could not write {output_type.upper()} file: {path}")


def _signal_header(label: str, signal: np.ndarray, srate: float, digital_min: int, digital_max: int) -> dict[str, Any]:
    physical_min = float(np.floor(np.min(signal)))
    physical_max = float(np.ceil(np.max(signal)))
    if physical_min == physical_max:
        physical_min -= 1
        physical_max += 1
    return highlevel.make_signal_header(
        label,
        dimension="uV",
        sample_frequency=srate,
        physical_min=physical_min,
        physical_max=physical_max,
        digital_min=digital_min,
        digital_max=digital_max,
    )
