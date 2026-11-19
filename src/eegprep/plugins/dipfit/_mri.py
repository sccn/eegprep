"""Packaged MNI MRI dictionaries for DIPFIT slice displays."""

from functools import lru_cache
from importlib.resources import files

import numpy as np


@lru_cache(maxsize=1)
def load_standard_mri_volume() -> dict:
    """Load the packaged MNI anatomy, voxel transform, and coordinate grids."""
    resource = files("eegprep") / "resources/dipfit/standard_BEM/standard_mri_mni.npz"
    with resource.open("rb") as stream, np.load(stream) as data:
        return {key: data[key] for key in ("anatomy", "transform", "xgrid", "ygrid", "zgrid")}


def dipfit_mri_slice_indices(volume: dict, positions) -> tuple[int, int, int]:
    """Return EEGLAB's slice offsets as bounded, zero-based voxel indices."""
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    voxels = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(volume["transform"]).T
    indices = np.ceil(np.nanmean(voxels[:, :3], axis=0) + [-3.0, 3.0, -3.0]).astype(int) - 1
    return tuple(np.clip(indices, 0, np.asarray(volume["anatomy"].shape) - 1).tolist())


def dipfit_mri_slices(volume: dict, positions) -> tuple[dict, ...]:
    """Return sagittal, coronal, and axial images with their MNI extents."""
    indices = dipfit_mri_slice_indices(volume, positions)
    coordinates = [volume[key] + volume["transform"][axis, 3] for axis, key in enumerate(("xgrid", "ygrid", "zgrid"))]
    slices = []
    for axis, (x_axis, y_axis) in enumerate(((1, 2), (0, 2), (0, 1))):
        x, y = coordinates[x_axis], coordinates[y_axis]
        slices.append(
            {
                "image": np.rot90(np.take(volume["anatomy"], indices[axis], axis=axis).astype(float)),
                "extent": (float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
                "x_axis": x_axis,
                "y_axis": y_axis,
                "index": indices[axis],
            }
        )
    return tuple(slices)
