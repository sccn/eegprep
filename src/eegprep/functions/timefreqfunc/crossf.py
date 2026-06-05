"""Legacy EEGLAB ``crossf`` entry point backed by ``newcrossf``."""

from __future__ import annotations

from typing import Any

from eegprep.functions.timefreqfunc.newcrossf import CrossFrequencyResult, newcrossf


def crossf(
    x: Any,
    y: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    **kwargs: Any,
) -> CrossFrequencyResult:
    """Compute legacy ``crossf`` outputs using EEGPrep's ``newcrossf`` core."""
    return newcrossf(x, y, frames, tlimits, srate, cycles, **kwargs)


__all__ = ["crossf"]
