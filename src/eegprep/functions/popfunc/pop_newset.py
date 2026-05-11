"""Minimal EEGLAB-style dataset switching/storing helper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.adminfunc.eeg_store import eeg_store


def pop_newset(
    ALLEEG: list[dict[str, Any]] | None,
    EEG: dict[str, Any] | list[dict[str, Any]],
    CURRENTSET: int | list[int] | tuple[int, ...] | None,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]], int | list[int], str]:
    """Store or retrieve datasets with the subset needed by the EEGPrep GUI."""
    retrieve = kwargs.pop("retrieve", None)
    setname = kwargs.pop("setname", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(f"Unsupported pop_newset option(s): {unknown}")
    if args:
        options = dict(zip(args[0::2], args[1::2]))
        retrieve = options.get("retrieve", retrieve)
        setname = options.get("setname", setname)

    if retrieve is not None and retrieve != []:
        current, alleeg, current_set = eeg_retrieve(ALLEEG, retrieve)
        return alleeg, current, current_set, f"[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', {retrieve});"

    if isinstance(EEG, dict) and setname is not None:
        EEG = dict(EEG)
        EEG["setname"] = setname
    alleeg, current, current_set = eeg_store(ALLEEG, EEG, CURRENTSET if CURRENTSET else None)
    return alleeg, current, current_set, "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET);"
