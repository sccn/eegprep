"""Explicit limitation for EEGLAB LIMO preparation workflows."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._limo_limitations import raise_limo_limitation


def pop_limo(*args: Any, **kwargs: Any) -> None:
    """Report that standalone EEGPrep does not run external LIMO workflows."""
    raise_limo_limitation("pop_limo", *args, **kwargs)


__all__ = ["pop_limo"]
