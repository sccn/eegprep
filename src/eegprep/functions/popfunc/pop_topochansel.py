"""Channel-selection compatibility wrapper for EEGLAB ``pop_topochansel``."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_chansel import pop_chansel, pop_chansel_resolve


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
    channel_values, resolved = pop_chansel_resolve(chanlocs, select)
    if gui:
        chanlist, strchannames, cellchannames = pop_chansel(chanlocs, select=select, withindex="on")
    else:
        chanlist = resolved
        cellchannames = [channel_values[index - 1] for index in chanlist]
        strchannames = " ".join(cellchannames)
    first_output: Any = cellchannames if str(cellstrout).lower() == "on" else chanlist
    command = (
        "pop_topochansel("
        f"{format_history_value(channel_values, cell_for_sequence='all_strings')}, "
        f"{format_history_value(select, cell_for_sequence=None)});"
    )
    if return_com:
        return first_output, cellchannames, strchannames, command
    return first_output, cellchannames, strchannames


__all__ = ["pop_topochansel"]
