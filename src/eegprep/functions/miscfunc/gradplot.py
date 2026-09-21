"""Plot spatial gradients of EEG scalp maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.gradmap import gradmap


def gradplot(maps: Any, locations: Any, draw: bool | int = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute scalp-map gradients and optionally draw their vector fields."""
    return gradmap(maps, locations, draw=draw)


__all__ = ["gradplot"]
