"""Leadfield matrix for explicit DIPFIT source points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.plugins.dipfit._utils import (
    DIPFITUnavailableError,
    copy_eeg,
    dipfit_history,
    ensure_dipfit_settings,
)
from eegprep.plugins.dipfit.dipfit_gridsearch import fitting_config
from eegprep.plugins.dipfit.eeglab2fieldtrip import eeglab2fieldtrip
from eegprep.plugins.dipfit.ft_dipolefitting import (
    compute_dipole_leadfield,
    inside_headmodel,
    prepare_headmodel_sens,
)


def loreta_key_source_points() -> np.ndarray:
    """Return the 20 mm LORETA-KEY grid kept inside a 75 mm radius.

    FieldTrip's LORETA-KEY atlas file is not bundled. The coordinates run from
    -80 mm to 80 mm.
    """
    coords = np.arange(-80.0, 80.0 + 20.0, 20.0)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[np.linalg.norm(grid, axis=1) <= 75.0]


def source_model_from_points(EEG: dict[str, Any], points: Any) -> dict[str, Any]:
    """Build a source model whose leadfield comes from ``ft_dipolefitting``."""
    array = np.asarray(points, dtype=float)
    if array.size == 0 or array.size % 3 or not np.all(np.isfinite(array)):
        raise ValueError(
            "pop_leadfield: source points must be finite and shaped N x 3. Pass one row per source position."
        )
    source_points = array.reshape(-1, 3)
    comp = eeglab2fieldtrip(EEG)
    cfg = fitting_config(EEG)
    vol, sens = prepare_headmodel_sens(cfg, comp["elec"], EEG.get("chanlocs", []))
    stacked = compute_dipole_leadfield(source_points, sens, vol)
    inside = inside_headmodel(source_points, vol)
    return {
        "pos": source_points.tolist(),
        "inside": [bool(value) for value in np.asarray(inside).reshape(-1)],
        "leadfield": [stacked[:, 3 * index : 3 * index + 3].tolist() for index in range(source_points.shape[0])],
        "label": list(sens["label"]),
        "unit": "mm",
        "coordformat": ensure_dipfit_settings(EEG).get("coordformat", ""),
    }


def pop_leadfield(EEG=None, *args, gui=None, renderer=None, return_com=False, **kwargs):
    """Store an average-referenced leadfield for explicit source points."""
    if EEG is None:
        return (None, "") if return_com else None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options
    if gui:
        result = inputgui(pop_leadfield_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options.update(_leadfield_options_from_result(result))
    if not options.get("sourcemodel"):
        raise ValueError(
            "pop_leadfield: sourcemodel is empty. Pass source points, a position file, or sourcemodel='loreta-key'."
        )
    points = _source_points_from_option(options.get("sourcemodel"))
    out = copy_eeg(EEG)
    ensure_dipfit_settings(out)
    out["dipfit"]["sourcemodel"] = source_model_from_points(out, points)
    out["dipfit"]["sourcemodel"]["file"] = options.get("sourcemodel")
    if options.get("sourcemodel2mni") not in (None, ""):
        out["dipfit"]["sourcemodel"]["coordtransform"] = numeric_vector(options["sourcemodel2mni"]).tolist()
    command = dipfit_history(
        "pop_leadfield",
        sourcemodel=options.get("sourcemodel"),
        sourcemodel2mni=options.get("sourcemodel2mni", ""),
        downsample=options.get("downsample", ""),
    )
    return (out, command) if return_com else out


def pop_leadfield_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like leadfield source-model dialog."""
    return DialogSpec(
        title="Compute ROI activity",
        function_name="pop_leadfield",
        eeglab_source="plugins/dipfit/pop_leadfield.m",
        help_text="pophelp('pop_leadfield')",
        size=(650, 230),
        geometry=((1,), (1,), (0.1, 0.8, 0.8, 0.2), (0.1, 0.8, 0.8, 0.2), (0.1, 0.8, 0.2, 0.8)),
        controls=(
            ControlSpec("text", "Choose source model for Leadfield matrix", font_weight="bold"),
            ControlSpec(
                "popupmenu",
                "Surface source model: Colin27 (with Desikan-Kilianny atlas)|Surface source model: Use Brainstorm ICBM152 (with Desikan-Kilianny atlas)|Volumetric source model: LORETA-KEY|Volumetric source model: AFNI with TTatlas+tlrc atlas (Fieldtrip)|Custom source model",
                tag="source_choice",
                value=3,
                callback=CallbackSpec(
                    "choice_text",
                    params={
                        "source": "source_choice",
                        "target": "sourcemodel",
                        "browse": "source_browse",
                        "texts": ["", "", "loreta-key", "", ""],
                    },
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "File name"),
            ControlSpec("edit", tag="sourcemodel", value="loreta-key"),
            ControlSpec(
                "pushbutton",
                "...",
                tag="source_browse",
                enabled=False,
                callback=CallbackSpec("select_file", params={"button": "source_browse", "target": "sourcemodel"}),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Transformation to MNI (if any)"),
            ControlSpec("edit", tag="sourcemodel2mni", value=""),
            ControlSpec("pushbutton", "...", tag="transform_browse", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Downsampling (AFNI only)"),
            ControlSpec("edit", tag="downsample", value="4", enabled=False),
            ControlSpec("spacer"),
        ),
        known_differences=(
            "The LORETA-KEY choice builds a 20 mm in-head grid. The leadfield uses the selected head model.",
            "Colin27, ICBM152, and AFNI atlas files are not bundled. MRI source-model construction and eLORETA are not implemented.",
        ),
    )


def _leadfield_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    choice = int(result.get("source_choice", 3) or 3)
    sourcemodel = str(result.get("sourcemodel") or "").strip()
    if not sourcemodel and choice == 3:
        sourcemodel = "loreta-key"
    if not sourcemodel:
        raise ValueError(
            "pop_leadfield: that source model file is not bundled. "
            "Leave the LORETA-KEY choice selected, or pass source points in sourcemodel."
        )
    return {
        "sourcemodel": sourcemodel,
        "sourcemodel2mni": result.get("sourcemodel2mni", ""),
        "downsample": result.get("downsample", ""),
    }


def _source_points_from_option(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "pos" not in value:
            raise ValueError(
                "pop_leadfield: sourcemodel dictionaries must contain 'pos'. Pass an N x 3 array under that key."
            )
        return np.asarray(value["pos"], dtype=float)
    if isinstance(value, str) and value.strip().lower() in {"loreta-key", "loreta_key"}:
        return loreta_key_source_points()
    if not isinstance(value, str):
        array = np.asarray(value, dtype=float)
        if array.size:
            return array
        raise ValueError(
            "pop_leadfield: sourcemodel has no coordinates. "
            "Pass source points, a position file, or sourcemodel='loreta-key'."
        )
    path = value.strip()
    if not path:
        raise ValueError(
            "pop_leadfield: sourcemodel is empty. Pass source points, a position file, or sourcemodel='loreta-key'."
        )
    lower = path.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(path), dtype=float)
    if lower.endswith(".npz"):
        with np.load(path) as data:
            key = "pos" if "pos" in data else next(iter(data.files))
            return np.asarray(data[key], dtype=float)
    if lower.endswith(".mat"):
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        points = _find_mat_points(data)
        if points is None:
            raise ValueError(
                f"pop_leadfield: {path} has no source positions in pos, vertices, or Vertices. "
                "Save an N x 3 array under one of those names."
            )
        return points
    if lower.endswith((".head", ".brik", ".nii", ".nii.gz")):
        raise DIPFITUnavailableError(
            "pop_leadfield: AFNI and NIfTI atlases are not clipped into a leadfield. "
            "Call load_afni_atlas to inspect the atlas points, or pass an N x 3 source array."
        )
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"pop_leadfield: source model file {path} was not found. "
            "Pass an existing .npy, .npz, or .mat position file, or sourcemodel='loreta-key'."
        )
    raise DIPFITUnavailableError(
        f"pop_leadfield: {path} is not a supported source-position file. "
        "Use sourcemodel='loreta-key', an N x 3 array, a dictionary with 'pos', or a .npy, .npz, or .mat file."
    )


def _find_mat_points(data: dict[str, Any]) -> np.ndarray | None:
    for key in ("pos", "vertices", "Vertices"):
        if key in data:
            points = np.asarray(data[key], dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
    for value in data.values():
        if hasattr(value, "pos"):
            points = np.asarray(value.pos, dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
        if hasattr(value, "vertices"):
            points = np.asarray(value.vertices, dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
    return None
