"""Channel-selection compatibility wrapper for EEGLAB ``pop_topochansel``."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_chansel import pop_chansel


def pop_topochansel(
    chanlocs: Any,
    select: Any = None,
    *args: Any,
    labels: str = "off",
    cellstrout: str = "off",
    gui: bool | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Select channels by label/index using EEGPrep's channel selector."""
    del args, kwargs, labels
    if gui is None:
        gui = select is None
    if gui:
        chanlist, strchannames, cellchannames = pop_chansel(chanlocs, select=select, withindex="on")
    else:
        channel_values = _channel_labels(chanlocs)
        chanlist = _resolve_selection(select, channel_values)
        cellchannames = [channel_values[index - 1] for index in chanlist]
        strchannames = " ".join(cellchannames)
    first_output: Any = cellchannames if str(cellstrout).lower() == "on" else chanlist
    command = (
        "pop_topochansel("
        f"{format_history_value(_channel_labels(chanlocs), cell_for_sequence='all_strings')}, "
        f"{format_history_value(select, cell_for_sequence=None)});"
    )
    return (first_output, command) if return_com else (first_output, cellchannames, strchannames)


def _channel_labels(chanlocs: Any) -> list[str]:
    if isinstance(chanlocs, dict) and "chanlocs" in chanlocs:
        chanlocs = chanlocs["chanlocs"]
    return [str(chan.get("labels", chan)) if isinstance(chan, dict) else str(chan) for chan in (chanlocs or [])]


def _resolve_selection(select: Any, labels: list[str]) -> list[int]:
    if select is None or select == "":
        return []
    if isinstance(select, str):
        tokens = select.split()
    elif isinstance(select, (int, float)):
        tokens = [select]
    else:
        tokens = list(select)
    lowered = [label.lower() for label in labels]
    selected = []
    for token in tokens:
        if isinstance(token, (int, float)) or str(token).isdigit():
            index = int(token)
        else:
            try:
                index = lowered.index(str(token).lower()) + 1
            except ValueError as exc:
                raise ValueError(f"Unknown channel label {token!r}") from exc
        if index < 1 or index > len(labels):
            raise ValueError("Selected channel index out of range")
        selected.append(index)
    return selected


__all__ = ["pop_topochansel"]
