"""Legacy EEGLAB average-reference wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.pop_reref import pop_reref


def pop_averef(
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    confirm: int | bool = 0,
    *,
    return_com: bool = False,
):
    """Convert data to average reference through ``pop_reref(EEG, [])``."""
    if EEG is None:
        return (None, "") if return_com else None
    out, _command = pop_reref(EEG, [], gui=False, return_com=True)
    command = f"EEG = pop_averef( EEG, {int(bool(confirm))});"
    return (out, command) if return_com else out


__all__ = ["pop_averef"]
