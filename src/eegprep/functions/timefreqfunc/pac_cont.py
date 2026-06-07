"""Continuous EEGLAB-style phase-amplitude coupling."""

from __future__ import annotations

from typing import Any

from eegprep.functions.timefreqfunc._pac_support import PacContResult, compute_pac_cont


def pac_cont(X: Any, Y: Any, srate: float, **kwargs: Any) -> PacContResult:
    """Compute sliding-window phase-amplitude coupling from continuous data."""
    return compute_pac_cont(X, Y, srate, **kwargs)


__all__ = ["PacContResult", "pac_cont"]
