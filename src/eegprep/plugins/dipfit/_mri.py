"""Packaged MNI MRI dictionaries for DIPFIT slice displays."""

from functools import lru_cache
from importlib.resources import files

import numpy as np


@lru_cache(maxsize=1)
def load_standard_mri_volume() -> dict:
    """Load the packaged Colin ``standard_mri`` anatomy used for MNI models."""
    return _load_mri_volume("resources/dipfit/standard_BEM/standard_mri_mni.npz")


@lru_cache(maxsize=1)
def load_besa_mri_volume() -> dict:
    """Load the packaged ``avg152t1`` anatomy used for spherical BESA models."""
    return _load_mri_volume("resources/dipfit/standard_BESA/avg152t1.npz")


def load_dipfit_mri_volume(coordformat: str) -> tuple[dict, str]:
    """Return the EEGLAB template volume and its figure label for ``coordformat``."""
    if str(coordformat).lower() == "spherical":
        return load_besa_mri_volume(), "avg152t1"
    return load_standard_mri_volume(), "standard_mri"


def _load_mri_volume(resource_path: str) -> dict:
    resource = files("eegprep") / resource_path
    with resource.open("rb") as stream, np.load(stream) as data:
        return {key: data[key] for key in ("anatomy", "transform", "xgrid", "ygrid", "zgrid")}


_SLICE_VOXEL_INDEX_OFFSET = np.array([-3.0, 3.0, -3.0])


def dipfit_mri_slice_indices(volume: dict, positions) -> tuple[int, int, int]:
    """Return EEGLAB's slice offsets as bounded, zero-based voxel indices."""
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    voxels = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(volume["transform"]).T
    indices = (
        np.ceil(np.nanmean(voxels[:, :3], axis=0) + _SLICE_VOXEL_INDEX_OFFSET * _voxel_index_steps(volume)).astype(int)
        - 1
    )
    return tuple(np.clip(indices, 0, np.asarray(volume["anatomy"].shape) - 1).tolist())


def dipfit_mri_slices(volume: dict, positions) -> tuple[dict, ...]:
    """Return sagittal, coronal, and axial images with their MNI extents."""
    indices = dipfit_mri_slice_indices(volume, positions)
    coordinates = [_world_axis(volume, axis, key) for axis, key in enumerate(("xgrid", "ygrid", "zgrid"))]
    slices = []
    for axis, (x_axis, y_axis) in enumerate(((1, 2), (0, 2), (0, 1))):
        x, y = coordinates[x_axis], coordinates[y_axis]
        slices.append(
            {
                "image": np.take(volume["anatomy"], indices[axis], axis=axis).astype(float).T,
                "extent": (float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
                "x_axis": x_axis,
                "y_axis": y_axis,
                "index": indices[axis],
            }
        )
    return tuple(slices)


def _voxel_index_steps(volume: dict) -> np.ndarray:
    steps = []
    for key in ("xgrid", "ygrid", "zgrid"):
        grid = np.asarray(volume[key], dtype=float)
        steps.append(float(grid[1] - grid[0]) if grid.size > 1 else 1.0)
    return np.asarray(steps, dtype=float)


def _world_axis(volume: dict, axis: int, key: str) -> np.ndarray:
    grid = np.asarray(volume[key], dtype=float)
    return float(volume["transform"][axis, axis]) * grid + float(volume["transform"][axis, 3])
