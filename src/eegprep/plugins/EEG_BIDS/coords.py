"""Coordinate system utilities."""

from typing import Dict, Any, Optional, Sequence
import numpy as np

__all__ = [
    'coords_to_mm',
    'coords_any_to_RAS',
    'coords_RAS_to_ALS',
    'coords_ALS_to_angular',
    'clear_chanloc',
    'chanloc_has_coords',
    'chanlocs_to_coords',
    'BIDS_ALS_SYSTEMS',
    'BIDS_RAS_SYSTEMS',
    'bids_coordsystem_orientation',
]

# The BIDS coordinate-systems appendix, as in eegprep_lean.locations: frames that are
# ALS already (EEGLAB's own), and frames that are RAS. The appendix gives no orientation
# to Pixels, ACPC, ScanRAS, Other, individual, study or scanner, so they are in neither.
BIDS_ALS_SYSTEMS = frozenset({'EEGLAB', 'EEGLAB-HJ', 'CTF', '4DBti', 'KitYokogawa'})
BIDS_RAS_SYSTEMS = frozenset(
    {
        'CapTrak',
        'NeuromagElektaMEGIN',
        'ElektaNeuromag',
        'ChietiItab',
        'ICBM452AirSpace',
        'ICBM452Warp5Space',
        'IXI549Space',
        'fsaverage',
        'fsaverageSym',
        'fsLR',
        'MNIColin27',
        'MNI152Lin',
        'MNI152NLin6Sym',
        # The appendix's table prints MNI152NLin6ASym; the schema's value is that up to BIDS
        # 1.10.1 and MNI152NLin6Asym from 1.11.0.
        'MNI152NLin6ASym',
        'MNI152NLin6Asym',
        'MNI305',
        'NIHPD',
        'OASIS30AntsOASISAnts',
        'OASIS30Atropos',
        'Talairach',
        'UNCInfant',
        'fsaverage3',
        'fsaverage4',
        'fsaverage5',
        'fsaverage6',
        'fsaveragesym',
        *(f'MNI152NLin2009{version}{symmetry}' for version in 'abc' for symmetry in ('Sym', 'Asym')),
        *(f'UNCInfant{age}V{volume}' for age in '012' for volume in ('21', '22', '23')),
    }
)

# Names are looked up without regard to case or surrounding spaces; no two frames above differ
# only in case.
_ORIENTATION_BY_SYSTEM = {
    **{system.lower(): 'RAS' for system in BIDS_RAS_SYSTEMS},
    **{system.lower(): 'ALS' for system in BIDS_ALS_SYSTEMS},
}


def bids_coordsystem_orientation(coordsystem: Dict[str, Any]) -> Optional[str]:
    """Return ``'ALS'`` or ``'RAS'`` for the EEG frame a BIDS ``coordsystem.json`` names.

    An ``EEGCoordinateSystemDescription`` of ``EEGLAB``, or a system with ``EEGLAB`` in its
    name, is ALS. Any other system is looked up in the BIDS coordinate-systems appendix.
    Returns ``None`` for ``Other``, a name the appendix does not list, or no name at all.
    """
    if coordsystem.get('EEGCoordinateSystemDescription') == 'EEGLAB':
        return 'ALS'
    system = coordsystem.get('EEGCoordinateSystem')
    if not isinstance(system, str):
        return None
    if 'EEGLAB' in system.upper():
        return 'ALS'
    return _ORIENTATION_BY_SYSTEM.get(system.strip().lower())


def coords_to_mm(coords: np.ndarray, unit: str) -> np.ndarray:
    """Convert the given coordinates array from the specified unit to millimeters."""
    if unit in ('mm', 'millimeters'):
        return np.array(coords, copy=True)
    elif unit in ('cm', 'centimeters'):
        return coords * 10.0
    elif unit in ('m', 'meters'):
        return coords * 1000.0
    else:
        raise ValueError(f"Unsupported coordinate unit: {unit}. Supported units are 'mm', 'cm', 'm'.")


def coords_RAS_to_ALS(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates from RAS (Right-Anterior-Superior) to ALS (Anterior-Left-Superior) convention."""
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
        coords = np.dot(coords, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    elif x == 'back' and y == 'right' and z == 'up':
        # -X nose direction
        coords = np.dot(coords, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
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
    x, y, z = coords.T
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
    coords = np.array(
        [[cl['X'], cl['Y'], cl['Z']] if chanloc_has_coords(cl) else [np.nan, np.nan, np.nan] for cl in chanlocs]
    )
    return coords
