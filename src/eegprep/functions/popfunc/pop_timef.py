"""Legacy EEGLAB ``pop_timef`` wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.pop_newtimef import pop_newtimef


def pop_timef(
    EEG: dict[str, Any] | None = None,
    typeproc: int = 1,
    num: Any = None,
    tlimits: Any = None,
    cycles: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Run legacy ``pop_timef`` through EEGPrep's ``pop_newtimef`` implementation."""
    result, command = pop_newtimef(
        EEG,
        typeproc,
        num,
        tlimits,
        cycles,
        *args,
        gui=gui,
        renderer=renderer,
        return_com=True,
        **kwargs,
    )
    if command:
        command = command.replace("pop_newtimef", "pop_timef", 1)
    return (result, command) if return_com else result


__all__ = ["pop_timef"]
