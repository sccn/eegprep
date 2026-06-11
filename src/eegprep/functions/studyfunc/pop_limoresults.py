"""Explicit limitation for EEGLAB LIMO result browsing."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._limo_limitations import raise_limo_limitation


def pop_limoresults(*args: Any, **kwargs: Any) -> None:
    """Report that standalone EEGPrep does not browse external LIMO results."""
    raise_limo_limitation("pop_limoresults", *args, **kwargs)


__all__ = ["pop_limoresults"]
