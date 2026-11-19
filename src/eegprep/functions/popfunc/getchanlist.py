"""Select channels by their channel-location type."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import cell_items, is_cell
from eegprep.functions.miscfunc.matlab_outputs import as_row, empty_matrix
from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def getchanlist(chanlocs: Any, channel_type: str | list[str] | tuple[str, ...] | None = None) -> np.ndarray:
    """Return 1-based channel indices matching one or more types.

    Matching is case-insensitive. With no type, or when locations have no
    type field, every channel is returned. A requested type with no match is
    omitted. EEGLAB also prints a warning for that type. The result is a row,
    or ``[]`` when nothing matches.
    """
    locations = chanlocs_as_list(chanlocs)
    if channel_type is None or not locations or "type" not in locations[0]:
        return as_row(np.arange(1, len(locations) + 1, dtype=float)) if locations else empty_matrix()
    requested_lower = {value.lower() for value in _requested_types(channel_type)}
    matches = [index + 1 for index, loc in enumerate(locations) if str(loc.get("type", "")).lower() in requested_lower]
    return as_row(np.asarray(matches, dtype=float)) if matches else empty_matrix()


def _requested_types(channel_type: Any) -> list[str]:
    if is_cell(channel_type):
        return [str(item) for item in cell_items(channel_type)]
    if isinstance(channel_type, str):
        return [channel_type]
    return [str(item) for item in np.asarray(channel_type, dtype=object).reshape(-1)]


__all__ = ["getchanlist"]
