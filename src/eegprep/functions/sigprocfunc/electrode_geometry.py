"""Scale-invariant checks for degenerate electrode positions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# Second-moment ratio σ3/σ1 of a uniform hemispherical shell. With the shell
# measured from its centroid, Var(z) / Var(x) = 1/4, so the singular-value
# ratio is 1/2. σ2/σ1 is 1 by rotational symmetry.
HEMISPHERE_THICKNESS_RATIO = 0.5

# An axis is absent when its RMS is under a quarter of that hemispherical
# thickness. Packaged 10-20, 10-10, 128-channel, and 256-channel caps stay
# several times thicker, including after channels are removed. A flat cloud
# with sub-millimetre noise falls below the cutoff.
DEGENERATE_THICKNESS_RATIO = HEMISPHERE_THICKNESS_RATIO / 4.0


def validate_electrode_positions(positions: Any, *, operation: str) -> None:
    """Raise on identical or collinear positions and warn on planar ones.

    ``positions`` has shape ``(channels, 2)`` or ``(channels, 3)``. The
    singular values σ1 ≥ σ2 ≥ σ3 of the centered coordinate matrix are the
    RMS extents of the cloud along its principal axes, up to one common
    factor. The ratios σk/σ1 are unchanged by translation and by uniform
    scaling.

    The cloud has no extent when σ1 is at most ``max(channels, axes) * eps
    * ||P||_F``. That catches exact duplicates and floating-point dust.
    Otherwise an axis whose ratio is below
    :data:`DEGENERATE_THICKNESS_RATIO` is treated as missing. Identical
    positions and collinear positions raise ``ValueError``. Planar or
    nearly planar positions, including a ring, emit ``UserWarning`` and
    return. Two-dimensional positions only take the identical and collinear
    tests. EEGLAB accepts all of these layouts.

    Args:
        positions: Cartesian electrode coordinates, one row per channel.
        operation: Name used in the error or warning.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"{operation}: electrode positions must have shape (channels, 2 or 3)")
    n_channels, n_axes = points.shape
    if n_channels == 0:
        raise ValueError(f"{operation}: electrode positions are empty")
    if not np.isfinite(points).all():
        raise ValueError(f"{operation}: electrode positions must be finite")

    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    extent = float(singular[0]) if singular.size else 0.0
    scale = float(np.linalg.norm(points))
    roundoff = float(np.finfo(float).eps) * max(n_channels, n_axes) * scale
    if extent <= roundoff:
        raise ValueError(f"{operation}: electrode positions are identical and have no spatial extent")

    relative = np.zeros(n_axes, dtype=float)
    relative[: singular.size] = singular / extent
    if relative[1] < DEGENERATE_THICKNESS_RATIO:
        raise ValueError(
            f"{operation}: electrode positions are collinear "
            f"(second principal axis is {relative[1]:.3g} of the first; "
            f"values below {DEGENERATE_THICKNESS_RATIO} are degenerate)"
        )
    if n_axes == 3 and relative[2] < DEGENERATE_THICKNESS_RATIO:
        warnings.warn(
            f"{operation}: electrode positions are planar or nearly planar "
            f"(third principal axis is {relative[2]:.3g} of the first; "
            f"values below {DEGENERATE_THICKNESS_RATIO} are degenerate). "
            "EEGLAB accepts this layout and the computation will continue.",
            UserWarning,
            stacklevel=2,
        )


__all__ = [
    "DEGENERATE_THICKNESS_RATIO",
    "HEMISPHERE_THICKNESS_RATIO",
    "validate_electrode_positions",
]
