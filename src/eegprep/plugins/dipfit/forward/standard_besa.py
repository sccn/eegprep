"""Standard BESA four-shell volume used by DIPFIT ``model='standardBESA'``.

Values are ``vol.r`` and ``vol.c`` in ``standard_BESA.mat`` (DIPFIT plugin).
FieldTrip's ``ft_datatype_headmodel`` renames ``c`` to ``cond`` and casts the
radii to double. Units are millimeters.
"""

from __future__ import annotations

import numpy as np


STANDARD_BESA_RADII_MM = np.array([71.0, 72.0, 79.0, 85.0])
STANDARD_BESA_CONDUCTIVITY = np.array([0.33, 1.0, 0.0042, 0.33])
STANDARD_BESA_ORIGIN_MM = np.zeros(3)


def standard_besa_headmodel() -> dict[str, np.ndarray | str]:
    """Return the concentric-sphere head model stored in ``standard_BESA.mat``."""
    return {
        "r": STANDARD_BESA_RADII_MM.copy(),
        "cond": STANDARD_BESA_CONDUCTIVITY.copy(),
        "o": STANDARD_BESA_ORIGIN_MM.copy(),
        "type": "concentricspheres",
        "unit": "mm",
    }
