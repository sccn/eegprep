"""Import EEG data from a BIDS folder or BIDS EEG file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_load_frombids import pop_load_frombids
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files


def pop_importbids(
    path: str | Path,
    *,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any] | list[dict[str, Any]], str]:
    """Load one or more EEG files from a BIDS dataset."""
    source = Path(path)
    files = bids_list_eeg_files(str(source)) if source.is_dir() else [str(source)]
    if not files:
        raise ValueError(f"No supported EEG files found in BIDS path: {source}")
    eegs = [pop_load_frombids(filename, **kwargs) for filename in files]
    result: dict[str, Any] | list[dict[str, Any]] = eegs[0] if len(eegs) == 1 else eegs
    command = f"EEG = pop_importbids({format_history_value(source)});"
    if isinstance(result, list):
        for eeg in result:
            eeg["history"] = command
    else:
        result["history"] = command
    return (result, command) if return_com else result
