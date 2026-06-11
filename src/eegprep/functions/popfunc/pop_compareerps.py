"""Legacy EEGLAB ERP comparison wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_comperp import pop_comperp


def pop_compareerps(
    ALLEEG: list[dict[str, Any]] | None = None,
    setlist: Any | None = None,
    chansubset: Any | None = None,
    plottitle: str = "",
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Compare ERP averages for selected datasets using ``pop_comperp``."""
    if not ALLEEG:
        return (None, "") if return_com else None
    if gui is None:
        gui = setlist is None
    if gui:
        result, _modern_command = pop_comperp(ALLEEG, 1, gui=True, renderer=renderer, return_com=True)
        command = "pop_compareerps(ALLEEG);"
        return (result, command) if return_com else result
    if setlist is None:
        raise ValueError("setlist is required when gui=False")
    channels = [] if chansubset is None else chansubset
    result, _modern_command = pop_comperp(
        ALLEEG,
        1,
        setlist,
        [],
        chans=channels,
        title=plottitle or "Compare dataset ERPs",
        addavg="off",
        addall="on",
        return_com=True,
    )
    command = _history_command(setlist, channels, plottitle)
    return (result, command) if return_com else result


def _history_command(setlist: Any, chansubset: Any, plottitle: str) -> str:
    return (
        "figure; pop_compareerps( ALLEEG, "
        f"{format_history_value(setlist, cell_for_sequence=None)}, "
        f"{format_history_value(chansubset, cell_for_sequence=None)}, "
        f"{format_history_value(plottitle)});"
    )


__all__ = ["pop_compareerps"]
