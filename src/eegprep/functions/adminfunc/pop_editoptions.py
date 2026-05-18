"""Edit EEGPrep options used by EEGLAB-style workflows."""

from __future__ import annotations

from typing import Any

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS


def pop_editoptions(options: dict[str, Any] | None = None, **updates: Any) -> str:
    """Update EEGPrep's EEGLAB-compatible options dictionary."""
    target = EEG_OPTIONS if options is None else options
    for key, value in updates.items():
        if key not in target:
            raise KeyError(f"Unknown EEGPrep option: {key}")
        target[key] = int(value) if isinstance(target[key], int) and isinstance(value, bool) else value
    return "LASTCOM = pop_editoptions();"
