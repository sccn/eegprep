"""Pure signal transform example."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def pop_demo_center(EEG: dict[str, Any] | None, *, return_com: bool = False):
    """Subtract each channel's mean without changing EEG dimensions."""
    if EEG is None:
        return (None, "") if return_com else None
    output = deepcopy(EEG)
    data = np.asarray(output["data"], dtype=float)
    axis = 1 if data.ndim == 2 else (1, 2)
    output["data"] = data - np.mean(data, axis=axis, keepdims=True)
    output.setdefault("etc", {})["eegprep_ext_signal_transform"] = {"centered": True}
    command = "EEG = pop_demo_center(EEG);"
    return (output, command) if return_com else output
