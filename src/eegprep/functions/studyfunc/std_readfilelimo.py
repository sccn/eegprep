"""Explicit limitation for EEGLAB LIMO file readers."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._limo_limitations import raise_limo_limitation


def std_readfilelimo(*args: Any, **kwargs: Any) -> None:
    """Report that standalone EEGPrep does not parse external LIMO files."""
    raise_limo_limitation("std_readfilelimo", *args, **kwargs)


__all__ = ["std_readfilelimo"]
