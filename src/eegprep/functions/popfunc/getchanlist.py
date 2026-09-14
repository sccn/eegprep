"""Select channels by their channel-location type."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def getchanlist(chanlocs: Any, channel_type: str | list[str] | tuple[str, ...] | None = None) -> list[int]:
    """Return 0-based channel indices matching one or more types.

    Matching is case-insensitive. With no type, every channel is returned.
    Missing requested types do not prevent other requested types from matching.
    """
    locations = chanlocs_as_list(chanlocs)
    if channel_type is None or not locations or "type" not in locations[0]:
        return list(range(len(locations)))
    requested = [channel_type] if isinstance(channel_type, str) else list(channel_type)
    requested_lower = {str(value).lower() for value in requested}
    return [index for index, loc in enumerate(locations) if str(loc.get("type", "")).lower() in requested_lower]


__all__ = ["getchanlist"]
