"""Typed access to stable EEGLAB ICA and display defaults."""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class ICADefaults:
    """Stable constants historically introduced by EEGLAB's ``icadefs`` script."""

    ICABINARY: str
    DEFAULT_SRATE: float = 256.0175
    DEFAULT_TIMLIM: tuple[int, int] = (-1000, 2000)
    DEFAULT_EPOCH: int = 10
    YDIR: int = 1
    HZDIR: str = "up"
    BACKCOLOR: tuple[float, float, float] = (0.93, 0.96, 1.0)
    BACKEEGLABCOLOR: tuple[float, float, float] = (0.66, 0.76, 1.0)


def icadefs() -> ICADefaults:
    """Return platform-aware ICA defaults without mutating process globals."""
    system = platform.system()
    binary = "binica.exe" if system == "Windows" else "ica_osx" if system == "Darwin" else "ica_linux"
    return ICADefaults(ICABINARY=binary)


__all__ = ["ICADefaults", "icadefs"]
