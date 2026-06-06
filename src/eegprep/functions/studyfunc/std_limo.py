"""Explicit limitation for EEGLAB LIMO model helpers."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._limo_limitations import raise_limo_limitation


def std_limo(*args: Any, **kwargs: Any) -> None:
    """Report that standalone EEGPrep does not run external LIMO models."""
    raise_limo_limitation("std_limo", *args, **kwargs)


__all__ = ["std_limo"]
