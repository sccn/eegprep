"""FieldTrip dipoli/bemcp EEG forward model.

``eeg_infinite_dipole``, ``eeg_leadfieldb``, and the BEM branch of
``ft_prepare_vol_sens`` follow FieldTrip 20250523. The packaged head model is
the DIPFIT ``standard_vol.mat`` dipoli solution (system matrix already
inverted). Building a new bemcp or dipoli system matrix is not implemented.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from eegprep.plugins.dipfit.fieldtrip.geometry import (
    find_innermost_boundary,
    find_outermost_boundary,
    project_elec,
    transfer_elec,
)
from eegprep.plugins.dipfit.fieldtrip.spheres import ft_prepare_vol_sens_spheres


BEM_TYPES = frozenset({"bem", "dipoli", "asa", "bemcp"})
_STANDARD_VOL_NAME = "standard_vol.mat"


def eeg_infinite_dipole(dippos: np.ndarray, elc: np.ndarray, vol: Any) -> np.ndarray:
    """Return the infinite-medium EEG leadfield at ``elc`` for dipoles ``dippos``.

    ``vol`` may be a conductivity scalar or a headmodel dict with ``cond``.
    The result has one row per electrode and three columns per dipole, in
    x, y, z order.
    """
    conductivity = _conductivity(vol)
    positions = _dipole_positions(dippos)
    electrodes = np.asarray(elc, dtype=float)
    if electrodes.ndim == 1:
        electrodes = electrodes.reshape(1, 3)
    relative = electrodes[None, :, :] - positions[:, None, :]
    scale = (4.0 * np.pi * conductivity) * np.sum(relative**2, axis=2) ** 1.5
    leadfield = relative / scale[:, :, None]
    return np.transpose(leadfield, (1, 0, 2)).reshape(electrodes.shape[0], 3 * positions.shape[0])


def eeg_leadfieldb(dippos: np.ndarray, elc: np.ndarray, vol: dict[str, Any]) -> np.ndarray:
    """Return the BEM EEG leadfield for one or more dipoles."""
    boundaries = _boundaries(vol)
    conductivity = np.asarray(vol["cond"], dtype=float).ravel()
    if len(boundaries) != conductivity.size:
        raise ValueError("the number of compartments in the volume is inconsistent")
    system = vol.get("mat")
    if system is None:
        raise ValueError("there is no system matrix present")
    kind = str(vol.get("type", "bem")).lower()
    electrodes = np.asarray(elc, dtype=float)
    if electrodes.size == 0:
        electrodes = np.zeros((0, 3))
    elif electrodes.ndim == 1:
        electrodes = electrodes.reshape(1, 3)
    _check_system_rows(vol, system, electrodes.shape[0], kind)
    if "source" not in vol:
        raise ValueError("the BEM volume has no source compartment")
    if kind == "dipoli":
        vertices = np.vstack([boundary["pos"] for boundary in boundaries])
        leadfield = eeg_infinite_dipole(dippos, vertices, 1.0)
    elif kind == "asa":
        vertices = np.vstack([boundary["pos"] for boundary in boundaries])
        source = int(vol["source"])
        leadfield = eeg_infinite_dipole(dippos, vertices, float(conductivity[source]))
    elif kind == "bemcp":
        extended = np.concatenate([conductivity, [0.0]])
        pieces = []
        for index, boundary in enumerate(boundaries):
            compartment = (extended[index] + extended[index + 1]) / 2.0
            pieces.append(eeg_infinite_dipole(dippos, boundary["pos"], compartment))
        leadfield = np.vstack(pieces)
    else:
        raise ValueError(f"unsupported type of volume conductor ({kind})")
    return system @ leadfield


def ft_prepare_vol_sens(headmodel: dict[str, Any], sens: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project EEG electrodes onto the skin and fold that transfer into ``mat``.

    This is the ``bem`` / ``dipoli`` / ``asa`` / ``bemcp`` branch of FieldTrip
    ``ft_prepare_vol_sens``. Other head-model types are left untouched so a
    spherical implementation can extend the same entry point.
    """
    kind = str(headmodel.get("type", "")).lower()
    if kind in {"singlesphere", "concentricspheres"}:
        return ft_prepare_vol_sens_spheres(headmodel, sens)
    if kind not in BEM_TYPES and "mat" not in headmodel:
        return headmodel, sens
    if kind not in BEM_TYPES:
        raise ValueError(f"ft_prepare_vol_sens BEM branch does not accept headmodel type {kind!r}")
    prepared = {
        "type": kind,
        "cond": np.asarray(headmodel["cond"], dtype=float).ravel().copy(),
        "bnd": [
            {"pos": np.asarray(item["pos"], dtype=float).copy(), "tri": np.asarray(item["tri"], dtype=int).copy()}
            for item in _boundaries(headmodel)
        ],
        "mat": _as_dense(headmodel["mat"]).copy(),
        "unit": str(headmodel.get("unit", "mm")),
    }
    if "skin_surface" in headmodel:
        prepared["skin_surface"] = int(headmodel["skin_surface"])
    else:
        prepared["skin_surface"] = find_outermost_boundary(prepared["bnd"])
    if "source" in headmodel:
        prepared["source"] = int(headmodel["source"])
    else:
        prepared["source"] = find_innermost_boundary(prepared["bnd"])
    electrodes = np.asarray(sens["elecpos"], dtype=float)
    if electrodes.ndim == 1:
        electrodes = electrodes.reshape(1, 3)
    system = prepared["mat"]
    skin = prepared["bnd"][prepared["skin_surface"]]
    if system.shape[0] != system.shape[1] and system.shape[0] == electrodes.shape[0]:
        combined = system
    else:
        weights, projected = project_elec(electrodes, skin["pos"], skin["tri"])
        skin_transfer = transfer_elec(skin["pos"], skin["tri"], weights)
        if system.shape[0] == skin["pos"].shape[0]:
            interpolation = skin_transfer
        else:
            blocks = []
            for index, boundary in enumerate(prepared["bnd"]):
                if index == prepared["skin_surface"]:
                    blocks.append(skin_transfer)
                else:
                    blocks.append(np.zeros((electrodes.shape[0], boundary["pos"].shape[0])))
            interpolation = np.hstack(blocks)
        combined = interpolation @ system
        electrodes = projected
    average = np.mean(combined, axis=0, keepdims=True)
    prepared["mat"] = combined - average
    prepared_sens = {
        "elecpos": np.asarray(electrodes, dtype=float),
        "label": list(sens.get("label", [])),
        "unit": str(sens.get("unit", prepared["unit"])),
        "tra": np.eye(electrodes.shape[0]),
    }
    return prepared, prepared_sens


def load_bem_headmodel(hdmfile: str | None = None) -> dict[str, Any]:
    """Load a dipoli/bemcp volume, defaulting to the packaged MNI model."""
    if hdmfile in (None, "", _STANDARD_VOL_NAME) or _is_packaged_standard_vol(str(hdmfile or "")):
        return _copy_headmodel(_cached_standard_bem())
    path = Path(str(hdmfile))
    if not path.is_file():
        raise FileNotFoundError(
            f"DIPFIT head model {hdmfile!r} was not found. The bundled MNI model is "
            f"{_STANDARD_VOL_NAME}; creating a BEM system matrix from an MRI is not implemented."
        )
    if path.suffix == ".npz":
        return _headmodel_from_npz(path)
    return _headmodel_from_mat(path)


def standard_bem_resource() -> Path:
    """Return the packaged MNI BEM volume path."""
    resource = (
        resources.files("eegprep")
        .joinpath("resources")
        .joinpath("dipfit")
        .joinpath("standard_BEM")
        .joinpath("standard_vol.npz")
    )
    with resources.as_file(resource) as path:
        return Path(path)


def is_bem_hdmfile(hdmfile: Any) -> bool:
    """Return whether a DIPFIT head-model file names the MNI boundary-element template."""
    return _is_packaged_standard_vol(str(hdmfile or ""))


def _cached_standard_bem() -> dict[str, Any]:
    return _cached_standard_bem_from(str(standard_bem_resource()))


@lru_cache(maxsize=1)
def _cached_standard_bem_from(path: str) -> dict[str, Any]:
    return _headmodel_from_npz(Path(path))


def _headmodel_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        boundaries = []
        index = 0
        while f"bnd{index}_pos" in data:
            boundaries.append(
                {
                    "pos": np.asarray(data[f"bnd{index}_pos"], dtype=float),
                    "tri": np.asarray(data[f"bnd{index}_tri"], dtype=int) - 1,
                }
            )
            index += 1
        system = sparse.csr_matrix(
            (data["mat_data"], data["mat_indices"], data["mat_indptr"]),
            shape=tuple(int(value) for value in data["mat_shape"]),
        )
        return {
            "type": str(data["type"]),
            "cond": np.asarray(data["cond"], dtype=float).copy(),
            "bnd": boundaries,
            "mat": system,
            "unit": "mm",
        }


def _headmodel_from_mat(path: Path) -> dict[str, Any]:
    from scipy.io import loadmat

    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    volume = loaded.get("vol", loaded.get("headmodel"))
    if volume is None:
        raise ValueError(f"{path} does not contain a FieldTrip vol/headmodel structure")
    boundaries = []
    raw_boundaries = volume.bnd if isinstance(volume.bnd, np.ndarray) else [volume.bnd]
    for boundary in np.atleast_1d(raw_boundaries):
        pos = getattr(boundary, "pos", None)
        if pos is None:
            pos = boundary.pnt
        boundaries.append({"pos": np.asarray(pos, dtype=float), "tri": np.asarray(boundary.tri, dtype=int) - 1})
    kind = str(getattr(volume, "type", "dipoli"))
    if kind.lower() not in BEM_TYPES:
        raise ValueError(
            f"{path.name} is a {kind!r} head model. EEGPrep applies dipoli/bemcp system matrices; "
            "creating one from an MRI is not implemented."
        )
    return {
        "type": kind.lower(),
        "cond": np.asarray(volume.cond, dtype=float).ravel(),
        "bnd": boundaries,
        "mat": volume.mat,
        "unit": str(getattr(volume, "unit", "mm")),
    }


def _copy_headmodel(headmodel: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": headmodel["type"],
        "cond": np.asarray(headmodel["cond"], dtype=float).copy(),
        "bnd": [
            {"pos": np.asarray(item["pos"], dtype=float).copy(), "tri": np.asarray(item["tri"], dtype=int).copy()}
            for item in headmodel["bnd"]
        ],
        "mat": headmodel["mat"].copy(),
        "unit": headmodel.get("unit", "mm"),
    }


def _boundaries(vol: dict[str, Any]) -> list[dict[str, Any]]:
    boundaries = vol.get("bnd")
    if not boundaries:
        raise ValueError("there are no compartment boundaries present")
    return list(boundaries)


def _check_system_rows(vol: dict[str, Any], system: Any, n_electrodes: int, kind: str) -> None:
    rows = int(system.shape[0])
    skin_index = int(vol["skin_surface"]) if "skin_surface" in vol else find_outermost_boundary(_boundaries(vol))
    n_skin = int(vol["bnd"][skin_index]["pos"].shape[0])
    n_vertices = int(sum(boundary["pos"].shape[0] for boundary in vol["bnd"]))
    if rows in {n_electrodes, n_skin, n_vertices}:
        return
    if kind == "openmeeg":
        return
    raise ValueError("unexpected size of system matrix")


def _conductivity(vol: Any) -> float:
    if isinstance(vol, dict):
        values = np.asarray(vol["cond"], dtype=float).ravel()
        return float(values[0])
    return float(np.asarray(vol, dtype=float).ravel()[0])


def _dipole_positions(dippos: np.ndarray) -> np.ndarray:
    positions = np.asarray(dippos, dtype=float)
    if positions.ndim == 1:
        if positions.size % 3 != 0:
            raise ValueError("incorrect specification of dipole locations")
        return positions.reshape(-1, 3)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("incorrect specification of dipole locations")
    return positions


def _as_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _is_packaged_standard_vol(hdmfile: str) -> bool:
    normalized = hdmfile.replace("\\", "/").lower()
    return normalized.endswith("standard_vol.mat") or normalized.endswith("standard_vol.npz")
