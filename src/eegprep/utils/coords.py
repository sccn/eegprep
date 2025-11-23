"""Coordinate system utilities."""

from typing import Dict, Any, Sequence
import numpy as np

__all__ = ['coords_to_mm', 'coords_any_to_RAS', 'coords_RAS_to_ALS', 'coords_ALS_to_angular',
           'clear_chanloc', 'chanloc_has_coords', 'chanlocs_to_coords']


def coords_to_mm(coords: np.ndarray, unit: str) -> np.ndarray:
    """Convert the given coordinates array from the specified unit to millimeters."""
    if unit in ('mm', 'millimeters'):
        pass
    elif unit in ('cm', 'centimeters'):
        coords *= 10.0
    elif unit in ('m', 'meters'):
        coords *= 1000.0
    else:
        raise ValueError(f"Unsupported coordinate unit: {unit}. "
                         f"Supported units are 'mm', 'cm', 'm'.")
    return coords


def coords_RAS_to_ALS(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates from RAS (Right-Anterior-Superior) to ALS (Anterior- Left-
    Superior) convention.
    """
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]  # Ensure 2D array for consistent processing
    if coords.shape[1] != 3:
        raise ValueError("Coordinates must have three dimensions (x, y, z).")
    coords = coords[:, [1, 0, 2]]  # Swap x and y coordinates
    coords[:, 1] = -coords[:, 1]  # Invert the y coordinate
    return coords


def coords_any_to_RAS(coords: np.ndarray, x: str, y: str, z: str) -> np.ndarray:
    """Convert the given coordinates (Nx3 array) to the RAS (Right-Anterior-Superior) system.

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of coordinates to convert.
    x : str
        Orientation of the X axis relative to the head in coords, e.g., 'front'.
    y : str
        Orientation of the Y axis relative to the head in coords, e.g., 'left'.
    z : str
        Orientation of the Z axis relative to the head in coords, e.g., 'up'.

    Returns
    -------
    coords : np.ndarray
        The transformed coordinates.
    """
    coords = np.array(coords, copy=False, dtype=float)
    if x == 'front' and y == 'left' and z == 'up':
        # +X nose direction
        # rotate 90 degrees clockwise (looking down on head)
        coords = np.dot(coords, np.array([[0, 1, 0],
                                          [-1, 0, 0],
                                          [0, 0, 1]]))
    elif x == 'back' and y == 'right' and z == 'up':
        # -X nose direction
        coords = np.dot(coords, np.array([[0, -1, 0],
                                          [1, 0, 0],
                                          [0, 0, 1]]))
    elif not (x == 'right' and y == 'front' and z == 'up'):
        # not +Y nose direction
        raise RuntimeError("Unsupported input coordinate system (%s,%s,%s)" % (x, y, z))
    return coords


def coords_ALS_to_angular(coords: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates (sph_theta, sph_phi, sph_radius) and 2d polar coordinates (polar_theta, polar_radius).

    Parameters
    ----------
    coords : np.ndarray
        Nx3 array of Cartesian coordinates (x, y, z).

    Returns
    -------
    sph_theta : np.ndarray
        Nx1 arrays of spherical coordinates.
    sph_phi : np.ndarray
        Nx1 arrays of spherical coordinates.
    sph_radius : np.ndarray
        Nx1 arrays of spherical coordinates.
    polar_theta : np.ndarray
        2d polar coordinates.
    polar_radius : np.ndarray
        2d polar coordinates.
    """
    x,y,z = coords.T
    hypotxy = np.hypot(x, y)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, hypotxy)
    sph_radius = np.hypot(hypotxy, z)
    sph_theta = theta / np.pi * 180
    sph_phi = phi / np.pi * 180

    polar_theta = -sph_theta
    polar_radius = 0.5 - sph_phi / 180

    return sph_theta, sph_phi, sph_radius, polar_theta, polar_radius


def clear_chanloc(ch: Dict[str, Any], numeric_null: Any) -> None:
    """Clear a channel-location record for a single channel in-place."""
    ch['sph_radius'] = numeric_null
    ch['sph_theta'] = numeric_null
    ch['sph_phi'] = numeric_null
    ch['theta'] = numeric_null
    ch['radius'] = numeric_null
    ch['X'] = numeric_null
    ch['Y'] = numeric_null
    ch['Z'] = numeric_null


def chanloc_has_coords(ch: Dict[str, Any]) -> bool:
    """Check if a given channel location record has valid (Cartesian) coordinates."""
    if ch.get('X') is None or ch.get('Y') is None or ch.get('Z') is None:
        return False
    elif isinstance(ch['X'], np.ndarray) and not len(ch['X']):
        return False
    elif isinstance(ch['Y'], np.ndarray) and not len(ch['Y']):
        return False
    elif isinstance(ch['Z'], np.ndarray) and not len(ch['Z']):
        return False
    elif np.isnan(ch['X']) or np.isnan(ch['Y']) or np.isnan(ch['Z']):
        return False
    return True


def chanlocs_to_coords(chanlocs: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Convert an EEGLAB chanlocs data structure to a Nx3 coordinates array."""
    coords = np.array([[cl['X'], cl['Y'], cl['Z']]
                       if chanloc_has_coords(cl)
                       else [np.nan, np.nan, np.nan]
                       for cl in chanlocs])
    return coords
