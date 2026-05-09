"""EEGLAB-style channel selection dialog."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from eegprep.functions.guifunc.listdlg2 import listdlg2


def pop_chansel(
    chans: Any,
    *,
    withindex: str | Sequence[int] = "off",
    select: Any = None,
    field: str = "labels",
    handle: Any | None = None,
    selectionmode: str = "multiple",
    parent: Any | None = None,
) -> tuple[list[int], str, list[str]]:
    """Open an EEGLAB ``pop_chansel``-style channel selector."""
    channel_values = _channel_values(chans, field)
    if not channel_values:
        return [], "", []

    chan_indices, withindex_value = _channel_indices(withindex, len(channel_values))
    selected_indices = _selection_to_indices(select, channel_values)
    display_values = _display_values(channel_values, chan_indices, withindex_value)
    prompt = "Select channel below" if selectionmode == "single" else "(use shift|Ctrl to\nselect several)"
    chanlist, ok, _strval = listdlg2(
        promptstring=prompt,
        liststring=display_values,
        initialvalue=selected_indices,
        selectionmode=selectionmode,
        parent=parent,
    )
    if not ok:
        return [], "", []

    allchanstr = [channel_values[index - 1] for index in chanlist]
    chanliststr = _selected_string(allchanstr, withindex_value)
    if handle is not None and hasattr(handle, "setText"):
        handle.setText(chanliststr)
    return chanlist, chanliststr, allchanstr


def pop_chansel_display_values(
    chans: Any,
    *,
    withindex: str | Sequence[int] = "off",
    field: str = "labels",
) -> list[str]:
    """Return the display labels that EEGLAB ``pop_chansel`` shows."""
    channel_values = _channel_values(chans, field)
    chan_indices, withindex_value = _channel_indices(withindex, len(channel_values))
    return _display_values(channel_values, chan_indices, withindex_value)


def pop_chansel_selected_string(
    chans: Any,
    select: Any,
    *,
    field: str = "labels",
    withindex: str = "off",
) -> str:
    """Return EEGLAB's selected channel string without opening the dialog."""
    channel_values = _channel_values(chans, field)
    selected = _selection_to_indices(select, channel_values)
    return _selected_string([channel_values[index - 1] for index in selected], withindex)


def _channel_values(chans: Any, field: str) -> list[str]:
    if chans is None:
        return []
    if isinstance(chans, (int, float)):
        chans = [chans]
    if isinstance(chans, dict) and "chanlocs" in chans:
        chans = chans["chanlocs"]
    if isinstance(chans, dict):
        chans = [chans]
    values = []
    for chan in list(chans):
        if isinstance(chan, dict):
            values.append(str(chan.get(field, "")))
        else:
            values.append(str(chan))
    if field.lower() == "type":
        deduped = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        values = deduped
    return values


def _channel_indices(withindex: str | Sequence[int], nchan: int) -> tuple[list[int], str]:
    if isinstance(withindex, str):
        return list(range(1, nchan + 1)), withindex.lower()
    return [int(value) for value in withindex], "on"


def _display_values(values: list[str], chan_indices: list[int], withindex: str) -> list[str]:
    if withindex == "on":
        return [f"{chan_indices[index]}  -  {value}" for index, value in enumerate(values)]
    return values


def _selection_to_indices(select: Any, values: list[str]) -> list[int]:
    if select is None or select == "":
        return []
    if isinstance(select, str):
        tokens = _parse_text(select)
    elif isinstance(select, (int, float)):
        tokens = [select]
    else:
        tokens = list(select)

    lower_values = [value.lower() for value in values]
    selected = []
    for token in tokens:
        if isinstance(token, str) and not _is_int_text(token):
            value = token.strip().lower()
            if value not in lower_values:
                raise ValueError(f"Cannot find '{token}'")
            selected.append(lower_values.index(value) + 1)
        else:
            index = int(token)
            if 1 <= index <= len(values):
                selected.append(index)
    return selected


def _parse_text(text: str) -> list[str]:
    tokens = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", text)
    return [next(part for part in token if part) for token in tokens]


def _selected_string(values: list[str], withindex: str) -> str:
    if not values:
        return ""
    space_present = any(" " in value or "\t" in value for value in values)
    if space_present:
        return " ".join(f"'{value}'" for value in values)
    return " ".join(values)


def _is_int_text(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True
