"""Select channels by their channel-location type."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def getchanlist(chanlocs: Any, channel_type: str | list[str] | tuple[str, ...] | None = None) -> list[int]:
    """Return 1-based channel indices matching one or more types.

    Matching is case-insensitive and the result is sorted. With no type, or
    when locations have no type field, every channel is returned. A requested
    type with no match is omitted. EEGLAB also prints a warning for that type.
    """
    locations = chanlocs_as_list(chanlocs)
    if channel_type is None or not locations or "type" not in locations[0]:
        return list(range(1, len(locations) + 1))
    requested = [channel_type] if isinstance(channel_type, str) else list(channel_type)
    requested_lower = {str(value).lower() for value in requested}
    return [index + 1 for index, loc in enumerate(locations) if str(loc.get("type", "")).lower() in requested_lower]


__all__ = ["getchanlist"]
