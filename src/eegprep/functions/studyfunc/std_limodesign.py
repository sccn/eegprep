"""Explicit limitation for EEGLAB LIMO design helpers."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._limo_limitations import raise_limo_limitation


def std_limodesign(*args: Any, **kwargs: Any) -> None:
    """Report that standalone EEGPrep does not build external LIMO designs."""
    raise_limo_limitation("std_limodesign", *args, **kwargs)


__all__ = ["std_limodesign"]
