"""Continuous phase-amplitude coupling limitation entry point."""

from __future__ import annotations

from typing import Any

from eegprep.functions.timefreqfunc._pac_support import raise_pac_not_implemented


def pac_cont(*_args: Any, **_kwargs: Any) -> None:
    """Raise a clear limitation for EEGLAB's unported ``pac_cont`` helper."""
    raise_pac_not_implemented()


__all__ = ["pac_cont"]
