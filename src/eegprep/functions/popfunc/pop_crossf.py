"""Legacy EEGLAB ``pop_crossf`` wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.pop_newcrossf import pop_newcrossf


def pop_crossf(
    EEG: dict[str, Any] | None = None,
    typeproc: int = 1,
    num1: Any = None,
    num2: Any = None,
    tlimits: Any = None,
    cycles: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Run legacy ``pop_crossf`` through EEGPrep's ``pop_newcrossf`` implementation."""
    result, command = pop_newcrossf(
        EEG,
        typeproc,
        num1,
        num2,
        tlimits,
        cycles,
        *args,
        gui=gui,
        renderer=renderer,
        return_com=True,
        **kwargs,
    )
    if command:
        command = command.replace("pop_newcrossf", "pop_crossf", 1)
    return (result, command) if return_com else result


__all__ = ["pop_crossf"]
