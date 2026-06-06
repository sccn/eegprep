"""Explicit limitation for EEGLAB LIMO result helpers."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._limo_limitations import raise_limo_limitation


def std_limoresults(*args: Any, **kwargs: Any) -> None:
    """Report that standalone EEGPrep does not compute external LIMO results."""
    raise_limo_limitation("std_limoresults", *args, **kwargs)


__all__ = ["std_limoresults"]
