"""EEGLAB-style phase-amplitude coupling."""

from __future__ import annotations

from typing import Any

from eegprep.functions.timefreqfunc._pac_support import PacResult, compute_pac


def pac(X: Any, Y: Any, srate: float, **kwargs: Any) -> PacResult:
    """Compute phase-amplitude coupling from epoched signals.

    The output mirrors EEGLAB's compute-only helper with named fields for the
    PAC matrix, output times, amplitude frequencies, phase frequencies, and
    single-trial spectral decompositions.
    """
    return compute_pac(X, Y, srate, **kwargs)


__all__ = ["PacResult", "pac"]
