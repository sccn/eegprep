"""Compatibility entry point for EEGLAB's low-memory infomax variant."""

from __future__ import annotations

from typing import Any

from eegprep.functions.sigprocfunc.runica import runica


def runicalowmem(data: Any, *args: Any, **kwargs: Any) -> tuple:
    """Run infomax ICA through EEGPrep's single maintained ICA engine.

    Positional EEGLAB name/value pairs are forwarded to :func:`runica`.
    EEGPrep's ``runica`` already trains in sample blocks. This compatibility
    name deliberately delegates to that implementation so algorithm fixes and
    numerical qualification cannot diverge between two copied engines.
    """
    return runica(data, *args, **kwargs)


__all__ = ["runicalowmem"]
