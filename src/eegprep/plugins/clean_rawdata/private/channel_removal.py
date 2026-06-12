"""Shared clean_rawdata channel-removal helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

CHANNEL_DEPENDENT_FIELDS = ("icawinv", "icasphere", "icaweights", "icaact", "stats", "specdata", "specicaact")


def remove_channels_without_pop_select(EEG: dict[str, Any], removed_channels: np.ndarray) -> dict[str, Any]:
    """Remove channels when ``pop_select`` is unavailable."""
    removed = np.asarray(removed_channels, dtype=bool).ravel()
    data = np.asarray(EEG["data"], dtype=np.float32)
    chanlocs = EEG.get("chanlocs", [])
    if len(chanlocs) == data.shape[0]:
        if isinstance(chanlocs, np.ndarray):
            EEG["chanlocs"] = chanlocs[~removed]
        else:
            EEG["chanlocs"] = np.asarray([chanloc for index, chanloc in enumerate(chanlocs) if not removed[index]])
    EEG["data"] = data[~removed, ...]
    EEG["nbchan"] = EEG["data"].shape[0]
    for field in CHANNEL_DEPENDENT_FIELDS:
        if field in EEG:
            EEG[field] = np.array([])
    return EEG


def update_clean_channel_mask(EEG: dict[str, Any], removed_channels: np.ndarray) -> None:
    """Update ``EEG.etc.clean_channel_mask`` after current-channel removal."""
    removed = np.asarray(removed_channels, dtype=bool).ravel()
    etc = EEG.setdefault("etc", {})
    mask = etc.get("clean_channel_mask")
    if mask is not None:
        existing = np.asarray(mask, dtype=bool).ravel()
        if int(np.sum(existing)) == removed.size:
            updated = np.array(existing, dtype=bool, copy=True)
            updated[updated] = ~removed
            etc["clean_channel_mask"] = updated
            return
    etc["clean_channel_mask"] = ~removed
