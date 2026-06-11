"""Power-unit labels for EEGLAB-style ``newtimef`` outputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def newtimefpowerunit(params: dict[str, Any] | None = None) -> str:
    """Return the display unit implied by ``newtimef`` baseline settings."""
    values = {} if params is None else dict(params)
    baseline = np.asarray(values.get("baseline", 0), dtype=float).reshape(-1)
    scale = str(values.get("scale", "log")).lower()
    basenorm = str(values.get("basenorm", "off")).lower()
    no_baseline = bool(baseline.size and np.isnan(baseline[0]))
    if scale == "log":
        if basenorm == "on":
            return "10*log(std.)"
        if no_baseline:
            return "10*log10(uV^{2}/Hz)"
        return "dB"
    if basenorm == "on":
        return "std."
    if no_baseline:
        return "uV^{2}/Hz"
    return "% of baseline"


__all__ = ["newtimefpowerunit"]
