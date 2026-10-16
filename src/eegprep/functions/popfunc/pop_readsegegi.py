"""Import a numbered series of EGI Simple Binary RAW files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_readegi import _egi_to_eeg
from eegprep.functions.sigprocfunc.readegi import readegi


_NUMBERED_RAW = re.compile(r"^(?P<prefix>.*?)(?P<number>\d{3})(?P<suffix>\.[^.]+)$")
_COMPATIBILITY_FIELDS = ("version", "samp_rate", "nchan", "gain", "bits", "range", "eventtypes", "eventcode")


def pop_readsegegi(
    filename: str | Path,
    *,
    fileloc: str | Path | None = "auto",
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import the contiguous ``001``, ``002``, ... EGI RAW file series.

    The selected filename may be any member of the series; loading always
    begins at ``001``. Files must be compatible continuous EGI recordings.
    Missing next-number files end the series, while corrupt or incompatible
    files raise an error.
    """
    selected = Path(filename)
    match = _NUMBERED_RAW.match(selected.name)
    if match is None:
        raise ValueError("filename must end in a three-digit sequence number and extension")
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    first = selected.with_name(f"{prefix}001{suffix}")
    if not first.is_file():
        raise FileNotFoundError(f"First EGI series file not found: {first}")

    paths = []
    headers = []
    signal_blocks = []
    event_blocks = []
    index = 1
    while True:
        path = selected.with_name(f"{prefix}{index:03d}{suffix}")
        if not path.is_file():
            break
        header, signals, events, _categories = readegi(path)
        if header["segmented"]:
            raise ValueError("pop_readsegegi accepts continuous EGI series; use pop_readegi for epoched files")
        if headers:
            _check_compatible(headers[0], header, path)
        paths.append(path)
        headers.append(header)
        signal_blocks.append(signals)
        event_blocks.append(events)
        index += 1

    later_files = []
    for candidate in selected.parent.iterdir():
        candidate_match = _NUMBERED_RAW.match(candidate.name)
        if candidate_match is None:
            continue
        if candidate_match.group("prefix") != prefix or candidate_match.group("suffix") != suffix:
            continue
        if int(candidate_match.group("number")) > index:
            later_files.append(candidate)
    if later_files:
        missing = selected.with_name(f"{prefix}{index:03d}{suffix}")
        raise ValueError(f"EGI series is missing {missing.name} before {min(later_files).name}")

    header = deepcopy(headers[0])
    trial_data = np.concatenate(signal_blocks, axis=1)
    event_data = np.concatenate(event_blocks, axis=1)
    header["samples"] = trial_data.shape[1]
    eeg = _egi_to_eeg(
        header,
        trial_data,
        event_data,
        np.array([], dtype=np.int64),
        filename=first,
        fileloc=fileloc,
        comments=f"Original files: {first} to {paths[-1]}",
    )
    command = f"EEG = pop_readsegegi({format_history_value(selected)});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _check_compatible(reference: dict[str, Any], candidate: dict[str, Any], path: Path) -> None:
    mismatches = [field for field in _COMPATIBILITY_FIELDS if candidate[field] != reference[field]]
    if mismatches:
        fields = ", ".join(mismatches)
        raise ValueError(f"Incompatible EGI series header in {path}: {fields}")


__all__ = ["pop_readsegegi"]
