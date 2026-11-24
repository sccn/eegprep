"""Desikan-Killiany labels for fitted dipoles, ported from ``eeg_compatlas.m``."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from eegprep.plugins.dipfit._coordinates import traditionaldipfit


_MNI_TRANSFORM = (
    0.0,
    -26.604623,
    -46.0,
    0.12346256,
    0.0,
    -1.5707963,
    1000.0,
    1000.0,
    1000.0,
)
_SPHERICAL_TRANSFORM = (
    -5.658258,
    1.039259,
    -42.80596,
    -0.00981033,
    0.03362692,
    0.004391199,
    860.8199,
    926.6112,
    858.162,
)


def eeg_compatlas(EEG: dict[str, Any], components: Any = None) -> dict[str, Any]:
    """Store ``model.areadk`` for each single dipole, matching ``eeg_compatlas.m``."""
    dipfit = EEG.get("dipfit")
    if not isinstance(dipfit, dict) or not dipfit.get("model"):
        raise ValueError("eeg_compatlas: dipole localization is required before looking up a brain area.")
    vertices, colors, labels, tree = _atlas()
    inverse = np.linalg.pinv(_transform(dipfit))
    models = list(dipfit["model"])
    selected = _component_set(components, len(models))
    for index, model in enumerate(models, start=1):
        if index not in selected or not isinstance(model, dict):
            continue
        positions = np.asarray(model.get("posxyz", []), dtype=float).reshape(-1, 3)
        if positions.size < 3 or (positions.shape[0] > 1 and np.any(positions[1] != 0)):
            continue
        atlas_point = inverse @ np.array([positions[0, 0], positions[0, 1], positions[0, 2], 1.0])
        _distance, nearest = tree.query(atlas_point[:3])
        area = int(colors[int(nearest)])
        model["areadk"] = str(labels[area - 1]) if area > 0 else "no area"
    dipfit["model"] = models
    return EEG


def _coord_transform_values(stored: Any) -> np.ndarray:
    if stored is None or _empty_transform(stored):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    try:
        values = np.asarray(stored, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "eeg_compatlas: coord_transform must be 9 numbers. "
            "Replace the malformed value with 9 numbers, or leave it empty for the identity transform."
        ) from exc
    if values.size == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    if values.size not in {6, 9} or not np.all(np.isfinite(values)):
        raise ValueError(
            "eeg_compatlas: coord_transform must be 9 numbers. "
            "Replace the malformed value with 9 numbers, or leave it empty for the identity transform."
        )
    return values


def _empty_transform(stored: Any) -> bool:
    if isinstance(stored, np.ndarray):
        return stored.size == 0
    return isinstance(stored, (list, tuple)) and len(stored) == 0


def _component_set(components: Any, count: int) -> set[int]:
    if components is None:
        return set(range(1, count + 1))
    values = np.asarray(components, dtype=int).ravel()
    return {int(value) for value in values}


def _transform(dipfit: dict[str, Any]) -> np.ndarray:
    if "coord_transform" in dipfit:
        return traditionaldipfit(_coord_transform_values(dipfit.get("coord_transform")))
    coordformat = str(dipfit.get("coordformat", "")).lower()
    if coordformat == "mni":
        return traditionaldipfit(_MNI_TRANSFORM)
    if coordformat == "spherical":
        return traditionaldipfit(_SPHERICAL_TRANSFORM)
    raise ValueError(
        "eeg_compatlas: the coordinate format is not MNI or spherical. "
        "Set dipfit.coordformat, or pass a 9-value coord_transform."
    )


@lru_cache(maxsize=1)
def _atlas():
    path = files("eegprep").joinpath("resources/dipfit/standard_BEM/dk_atlas.npz")
    with np.load(path) as data:
        vertices = np.asarray(data["vertices"], dtype=float)
        colors = np.asarray(data["color_table"])
        labels = np.asarray(data["labels"])
    return vertices, colors, labels, KDTree(vertices)
