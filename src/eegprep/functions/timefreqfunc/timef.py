"""Legacy EEGLAB ``timef`` entry point backed by ``newtimef``."""

from __future__ import annotations

from typing import Any

from eegprep.functions.timefreqfunc.newtimef import TimeFrequencyResult, newtimef


def timef(
    data: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    **kwargs: Any,
) -> TimeFrequencyResult:
    """Compute legacy ``timef`` outputs using EEGPrep's ``newtimef`` core."""
    return newtimef(data, frames, tlimits, srate, cycles, **kwargs)


__all__ = ["timef"]
