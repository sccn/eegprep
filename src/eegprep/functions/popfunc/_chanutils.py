"""Shared channel-location helpers for EEGLAB-style pop functions."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def chanlocs_as_list(chanlocs: Any) -> list[dict[str, Any]]:
    """Return channel-location structures as a Python list."""
    if chanlocs is None:
        return []
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    if isinstance(chanlocs, dict):
        return [chanlocs]
    return list(chanlocs)


def is_number_like(value: Any) -> bool:
    """Return true when a value can be converted to a finite or NaN float."""
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def normalise_reflocs(refloc: Any) -> list[dict[str, Any]]:
    """Normalise old-reference channel locations to dictionaries."""
    if isinstance(refloc, dict):
        return [copy.deepcopy(refloc)]
    if isinstance(refloc, np.ndarray):
        refloc = refloc.tolist()
    if (
        isinstance(refloc, (list, tuple))
        and len(refloc) == 3
        and isinstance(refloc[0], str)
        and is_number_like(refloc[1])
        and is_number_like(refloc[2])
    ):
        return [{"labels": refloc[0], "theta": refloc[1], "radius": refloc[2]}]

    locs = []
    for loc in list(refloc):
        if not isinstance(loc, dict):
            raise TypeError("refloc entries must be dictionaries or [label, theta, radius]")
        locs.append(copy.deepcopy(loc))
    return locs
