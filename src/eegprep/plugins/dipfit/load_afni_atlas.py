"""AFNI atlas loading boundary for DIPFIT leadfield workflows."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.plugins.dipfit._coordinates import traditionaldipfit
from eegprep.plugins.dipfit._utils import DIPFITUnavailableError


def load_afni_atlas(
    sourcemodel: str | Path,
    headmodel: Any | None = None,
    sourcemodel2mni: Any | None = None,
    downsample: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and downsample an AFNI/NIfTI atlas into source points.

    The FieldTrip MATLAB function reads AFNI HEAD/BRIK atlases and clips
    points to the head model. EEGPrep keeps the same data contract when
    ``nibabel`` can read the file; head-model clipping is limited to simple
    spherical models unless a richer Python backend is added.
    """
    if find_spec("nibabel") is None:
        raise DIPFITUnavailableError("load_afni_atlas requires nibabel to read AFNI or NIfTI atlas files")
    import nibabel as nib

    path = Path(sourcemodel)
    if not path.exists():
        raise FileNotFoundError(f"atlas file not found: {path}")
    if int(downsample) < 1:
        raise ValueError("downsample must be a positive integer")
    image: Any = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=float)
    if data.ndim == 4:
        data = data[..., 0]
    labelsstr = _atlas_labels(image)
    reduced = _block_mode(data.astype(int), int(downsample))
    indices = np.argwhere(reduced != 0)
    labels = reduced[tuple(indices.T)] if indices.size else np.asarray([], dtype=int)
    voxel_centers = indices * int(downsample) + (int(downsample) - 1) / 2.0
    homogeneous = np.column_stack([voxel_centers, np.ones(indices.shape[0])])
    xyz = homogeneous @ np.asarray(image.affine, dtype=float).T
    if sourcemodel2mni is not None and np.asarray(sourcemodel2mni, dtype=float).size:
        xyz = xyz @ traditionaldipfit(sourcemodel2mni).T
    xyz = xyz[:, :3]
    inside = _inside_headmodel(xyz, headmodel)
    return reduced, xyz[inside], labels[inside], labelsstr


def _block_mode(data: np.ndarray, factor: int) -> np.ndarray:
    shape = tuple(int(np.ceil(size / factor)) for size in data.shape[:3])
    out = np.zeros(shape, dtype=int)
    for index in np.ndindex(shape):
        slices = tuple(
            slice(axis * factor, min((axis + 1) * factor, data.shape[dim])) for dim, axis in enumerate(index)
        )
        values, counts = np.unique(data[slices], return_counts=True)
        out[index] = int(values[np.argmax(counts)]) if values.size else 0
    return out


def _inside_headmodel(points: np.ndarray, headmodel: Any) -> np.ndarray:
    if headmodel is None or points.size == 0:
        return np.ones(points.shape[0], dtype=bool)
    model = headmodel.get("vol", headmodel) if isinstance(headmodel, dict) else headmodel
    if isinstance(model, dict) and "r" in model:
        radius = float(np.max(np.asarray(model["r"], dtype=float)))
        origin = np.asarray(model.get("o", [0.0, 0.0, 0.0]), dtype=float).ravel()[:3]
        return np.linalg.norm(points - origin, axis=1) <= radius
    return np.ones(points.shape[0], dtype=bool)


def _atlas_labels(image: Any) -> list[str]:
    header = getattr(image, "header", None)
    if header is None:
        return []
    labels = []
    for key in ("BRICK_LABS", "brick_labs"):
        try:
            value = header[key]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        labels.extend(part for part in str(value).split("~") if part)
    return labels


__all__ = ["load_afni_atlas"]
