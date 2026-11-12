"""Concentric-sphere leadfield hook for ``ft_compute_leadfield``.

FieldTrip dispatches ``singlesphere`` and ``concentricspheres`` to
``eeg_leadfield1`` and ``eeg_leadfield4``. That spherical implementation is
separate from the BEM module so it can be added without changing the dipoli
forward model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ft_prepare_vol_sens_spheres(
    headmodel: dict[str, Any], sens: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Raise until concentric-sphere sensor preparation is available."""
    del sens
    kind = str(headmodel.get("type", "concentricspheres"))
    raise NotImplementedError(
        f"ft_prepare_vol_sens has no {kind} implementation yet. "
        "Plug the four-shell preparation into ft_prepare_vol_sens_spheres."
    )


def eeg_leadfield_spheres(dippos: np.ndarray, elecpos: np.ndarray, headmodel: dict[str, Any]) -> np.ndarray:
    """Raise until the concentric-sphere leadfield is available."""
    del dippos, elecpos
    kind = str(headmodel.get("type", "concentricspheres"))
    raise NotImplementedError(
        f"ft_compute_leadfield has no {kind} implementation yet. "
        "Plug eeg_leadfield1 / eeg_leadfield4 into eeg_leadfield_spheres."
    )
