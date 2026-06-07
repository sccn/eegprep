"""Kaiser-window beta helper for the bundled firfilt plugin."""

from __future__ import annotations

import numpy as np


def kaiserbeta(dev: float) -> float:
    """Estimate the Kaiser window beta from maximum passband deviation."""
    dev = float(dev)
    if dev <= 0:
        raise ValueError("Passband deviation must be positive.")
    devdb = -20.0 * np.log10(dev)
    if devdb > 50:
        return float(0.1102 * (devdb - 8.7))
    if devdb >= 21:
        return float(0.5842 * (devdb - 21) ** 0.4 + 0.07886 * (devdb - 21))
    return 0.0
