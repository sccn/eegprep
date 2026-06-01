"""Packaged DIPFIT MRI volume helpers for EEGPrep-native visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DipfitMriVolume:
    """Standard MNI MRI volume used for DIPFIT viewprops-style slice panels."""

    anatomy: np.ndarray
    transform: np.ndarray
    xgrid: np.ndarray
    ygrid: np.ndarray
    zgrid: np.ndarray

    @property
    def inverse_transform(self) -> np.ndarray:
        """Return the MNI-to-voxel homogeneous transform."""
        return np.linalg.inv(self.transform)

    @property
    def axis_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return voxel grid coordinates transformed to MNI millimeters."""
        return (
            self.xgrid + float(self.transform[0, 3]),
            self.ygrid + float(self.transform[1, 3]),
            self.zgrid + float(self.transform[2, 3]),
        )


@dataclass(frozen=True)
class DipfitMriSlice:
    """One 2-D MRI slice and its displayed MNI coordinate extent."""

    image: np.ndarray
    extent: tuple[float, float, float, float]
    x_axis: int
    y_axis: int
    index: int


def load_standard_mri_volume() -> DipfitMriVolume:
    """Load EEGPrep's packaged standard MNI MRI volume."""
    return _cached_standard_mri_volume()


def dipfit_mri_slice_indices(volume: DipfitMriVolume, positions: np.ndarray) -> tuple[int, int, int]:
    """Return EEGLAB-like 0-based slice indices for localized dipole positions."""
    mri_coordinates = _mni_to_voxel(volume, positions)
    center = np.nanmean(mri_coordinates, axis=0)
    offsets = np.asarray([-3.0, 3.0, -3.0])
    return tuple(_clip_voxel_indices(volume, np.ceil(center + offsets)).tolist())


def dipfit_mri_slices(volume: DipfitMriVolume, positions: np.ndarray) -> tuple[DipfitMriSlice, ...]:
    """Return sagittal, coronal, and axial slices through the dipole neighborhood."""
    sagittal_index, coronal_index, axial_index = dipfit_mri_slice_indices(volume, positions)
    x_coords, y_coords, z_coords = volume.axis_coordinates
    anatomy = np.asarray(volume.anatomy)
    return (
        DipfitMriSlice(
            np.rot90(np.asarray(anatomy[sagittal_index, :, :], dtype=float)),
            (float(y_coords[0]), float(y_coords[-1]), float(z_coords[0]), float(z_coords[-1])),
            1,
            2,
            sagittal_index,
        ),
        DipfitMriSlice(
            np.rot90(np.asarray(anatomy[:, coronal_index, :], dtype=float)),
            (float(x_coords[0]), float(x_coords[-1]), float(z_coords[0]), float(z_coords[-1])),
            0,
            2,
            coronal_index,
        ),
        DipfitMriSlice(
            np.rot90(np.asarray(anatomy[:, :, axial_index], dtype=float)),
            (float(x_coords[0]), float(x_coords[-1]), float(y_coords[0]), float(y_coords[-1])),
            0,
            1,
            axial_index,
        ),
    )


@lru_cache(maxsize=1)
def _cached_standard_mri_volume() -> DipfitMriVolume:
    resource = (
        resources.files("eegprep")
        .joinpath("resources")
        .joinpath("dipfit")
        .joinpath("standard_BEM")
        .joinpath("standard_mri_mni.npz")
    )
    with resources.as_file(resource) as path:
        with np.load(path) as data:
            return DipfitMriVolume(
                anatomy=np.asarray(data["anatomy"], dtype=np.uint8),
                transform=np.asarray(data["transform"], dtype=float),
                xgrid=np.asarray(data["xgrid"], dtype=float),
                ygrid=np.asarray(data["ygrid"], dtype=float),
                zgrid=np.asarray(data["zgrid"], dtype=float),
            )


def _mni_to_voxel(volume: DipfitMriVolume, positions: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1, -1)
    homogeneous = np.column_stack([coordinates[:, :3], np.ones(coordinates.shape[0])])
    voxels = homogeneous @ volume.inverse_transform.T
    return voxels[:, :3]


def _clip_voxel_indices(volume: DipfitMriVolume, coordinates: Any) -> np.ndarray:
    indices = np.asarray(coordinates, dtype=float).astype(int) - 1
    shape = np.asarray(volume.anatomy.shape[:3], dtype=int)
    return np.clip(indices, 0, shape - 1)


__all__ = [
    "DipfitMriSlice",
    "DipfitMriVolume",
    "dipfit_mri_slice_indices",
    "dipfit_mri_slices",
    "load_standard_mri_volume",
]
