"""STUDY PAC computation limitation entry point."""

from __future__ import annotations

from typing import Any

from eegprep.functions.timefreqfunc._pac_support import raise_pac_not_implemented


def std_pac(*_args: Any, **_kwargs: Any) -> None:
    """Raise a clear limitation for EEGLAB's unported STUDY PAC computation."""
    raise_pac_not_implemented()


__all__ = ["std_pac"]
