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
from scipy.io import loadmat

from eegprep.plugins.dipfit.fieldtrip.geometry import (
    find_innermost_boundary,
    find_outermost_boundary,
    project_elec,
    transfer_elec,
)
from eegprep.plugins.dipfit.fieldtrip.spheres import ft_prepare_vol_sens_spheres


BEM_TYPES = frozenset({"bem", "dipoli", "asa", "bemcp", "openmeeg"})
_IMPLEMENTED_BEM_TYPES = frozenset({"bem", "dipoli", "asa", "bemcp"})
_PACKAGED_VOL_NAMES = frozenset({"standard_vol.mat", "standard_vol.npz"})
_BEM_FILE_REQUIREMENT = (
    "A boundary-element file must contain a FieldTrip vol or headmodel with bnd, mat, and cond, "
    "and type dipoli, bemcp, or openmeeg."
)


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
    if kind == "openmeeg":
        raise ValueError("OpenMEEG head models are not implemented")
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
    ``ft_prepare_vol_sens``. ``singlesphere`` and ``concentricspheres`` go to
    the sphere hook. OpenMEEG and any other type raise.
    """
    kind = str(headmodel.get("type", "")).lower()
    if kind in {"singlesphere", "concentricspheres"}:
        return ft_prepare_vol_sens_spheres(headmodel, sens)
    if kind == "openmeeg":
        raise ValueError("OpenMEEG head models are not implemented")
    if kind not in _IMPLEMENTED_BEM_TYPES:
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
    """Load a dipoli, bemcp, or OpenMEEG volume.

    A filename that is not on disk and names ``standard_vol.mat`` loads the
    packaged MNI model. Any other existing ``.mat`` or ``.npz`` file is read
    from its contents. A missing or unrecognizable head-model file raises.
    """
    if hdmfile in (None, ""):
        return _copy_headmodel(_cached_standard_bem())
    path = Path(str(hdmfile))
    if path.is_file():
        if path.suffix.lower() == ".npz":
            return _validate_bem_volume(_headmodel_from_npz(path), path)
        if path.suffix.lower() == ".mat":
            return _validate_bem_volume(_headmodel_from_mat(path), path)
        raise ValueError(f"{path.name} is not a FieldTrip .mat or .npz head model. {_BEM_FILE_REQUIREMENT}")
    if _is_packaged_standard_vol(str(hdmfile)):
        return _copy_headmodel(_cached_standard_bem())
    raise FileNotFoundError(
        f"DIPFIT head model {hdmfile!r} was not found. The bundled MNI model is "
        "standard_BEM/standard_vol.mat; creating a BEM system matrix from an MRI is not implemented."
    )


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


def _cached_standard_bem() -> dict[str, Any]:
    return _cached_standard_bem_from(str(standard_bem_resource()))


@lru_cache(maxsize=1)
def _cached_standard_bem_from(path: str) -> dict[str, Any]:
    return _headmodel_from_npz(Path(path))


def _headmodel_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        if "mat_data" not in data or "cond" not in data or "type" not in data or "bnd0_pos" not in data:
            raise ValueError(f"{path.name} is not a boundary-element head model. {_BEM_FILE_REQUIREMENT}")
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
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    volume = loaded.get("vol", loaded.get("headmodel"))
    if volume is None or not hasattr(volume, "bnd") or not hasattr(volume, "mat") or not hasattr(volume, "cond"):
        raise ValueError(f"{path.name} is not a boundary-element head model. {_BEM_FILE_REQUIREMENT}")
    raw_boundaries = volume.bnd
    boundaries_in = list(raw_boundaries) if isinstance(raw_boundaries, np.ndarray) else [raw_boundaries]
    boundaries = []
    for boundary in boundaries_in:
        pos = getattr(boundary, "pos", None)
        if pos is None:
            pos = getattr(boundary, "pnt", None)
        triangles = getattr(boundary, "tri", None)
        if pos is None or triangles is None:
            raise ValueError(f"{path.name} is not a boundary-element head model. {_BEM_FILE_REQUIREMENT}")
        boundaries.append({"pos": np.asarray(pos, dtype=float), "tri": np.asarray(triangles, dtype=int) - 1})
    kind = str(getattr(volume, "type", "") or "").lower()
    return {
        "type": kind,
        "cond": np.asarray(volume.cond, dtype=float).ravel(),
        "bnd": boundaries,
        "mat": volume.mat,
        "unit": str(getattr(volume, "unit", "mm")),
    }


def _validate_bem_volume(headmodel: dict[str, Any], path: Path) -> dict[str, Any]:
    kind = str(headmodel.get("type", "")).lower()
    has_volume = headmodel.get("bnd") and headmodel.get("mat") is not None and headmodel.get("cond") is not None
    if kind not in BEM_TYPES or not has_volume:
        label = kind or "missing"
        raise ValueError(
            f"{path.name} is a {label!r} head model. EEGPrep applies dipoli and bemcp system matrices. "
            "OpenMEEG files are recognized but not implemented. "
            "This file is not one of those volumes."
        )
    return headmodel


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
    normalized = hdmfile.replace("\\", "/").lower().rsplit("/", 1)[-1]
    return normalized in _PACKAGED_VOL_NAMES
