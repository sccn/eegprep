"""Plot existing DIPFIT component dipoles with a standalone matplotlib view."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.plugins.dipfit._mri import dipfit_mri_slices, load_standard_mri_volume
from eegprep.plugins.dipfit._utils import (
    DIPFITUnavailableError,
    dipfit_history,
    localized_components,
    normalize_model_list,
    one_based_indices,
)


def pop_dipplot(
    EEG: dict[str, Any] | None = None,
    comps: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: bool = True,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot localized DIPFIT component positions."""
    if EEG is None:
        return ([], "") if return_com else []
    models = normalize_model_list(EEG)
    if not models:
        raise ValueError("No dipole information in dataset")
    if gui is None:
        gui = comps is None and not kwargs
    options = dict(kwargs)
    if gui:
        gui_result = _run_gui(EEG, renderer=renderer)
        if gui_result is None:
            return ([], "") if return_com else []
        comps = gui_result.pop("comps")
        options.update(gui_result)
    default_components = localized_components(EEG)
    components = one_based_indices(comps, limit=len(models), default_all=False)
    if not components:
        components = default_components
    if not components:
        raise ValueError("No localized dipoles are available to plot")
    components = _filter_components_by_rv(models, components, options.get("rvrange"))
    if not components:
        raise ValueError("No localized dipoles are inside the requested RV range")
    _validate_components_have_positions(models, components)
    figures = [_plot_dipoles(EEG, models, components, options)] if plot else []
    command = dipfit_history("pop_dipplot", components, assignment=False, **_history_options(options))
    return (figures, command) if return_com else figures


def pop_dipplot_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dipole plotting dialog spec."""
    raw_dipfit = EEG.get("dipfit")
    dipfit: dict[str, Any] = raw_dipfit if isinstance(raw_dipfit, dict) else {}
    mrifile = dipfit.get("mrifile", "")
    return DialogSpec(
        title="Plot dipoles - pop_dipplot",
        function_name="pop_dipplot",
        eeglab_source="plugins/dipfit/pop_dipplot.m",
        help_text="pophelp('pop_dipplot')",
        size=(660, 410),
        geometry=(
            (2, 1),
            (2, 1),
            (0.8, 0.3, 1.5),
            (2.05, 0.26, 0.75),
            (2.05, 0.26, 0.75),
            (2.05, 0.26, 0.75),
            (2.05, 0.26, 0.75),
            (2.05, 0.26, 0.75),
            (2.05, 0.26, 0.75),
            (2.05, 0.26, 0.75),
            (2, 1),
        ),
        controls=(
            ControlSpec("text", "Components indices ([]=all avaliable)"),
            ControlSpec("edit", tag="comps", value=""),
            ControlSpec("text", "Plot dipoles within RV (%) range ([min max])"),
            ControlSpec("edit", tag="rvrange", value=""),
            ControlSpec("text", "Background image"),
            ControlSpec("pushbutton", "...", tag="mri_button", enabled=False),
            ControlSpec("edit", tag="mri", value=str(mrifile)),
            ControlSpec("text", "Plot summary mode"),
            ControlSpec("checkbox", tag="summary", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Plot edges"),
            ControlSpec("checkbox", tag="drawedges", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Plot closest MRI slide"),
            ControlSpec("checkbox", tag="cornermri", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Plot dipole's 2-D projections"),
            ControlSpec("checkbox", tag="projimg", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Plot projection lines"),
            ControlSpec("checkbox", tag="projlines", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Make all dipoles point out"),
            ControlSpec("checkbox", tag="pointout", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Normalized dipole length"),
            ControlSpec("checkbox", tag="normlen", value=True),
            ControlSpec("spacer"),
            ControlSpec("text", "Additionnal dipplot() options"),
            ControlSpec("edit", tag="extra_options", value=""),
        ),
        known_differences=(
            "EEGPrep plots a standalone 3-D dipole-position view rather than FieldTrip MRI slice overlays.",
        ),
        row_spacing=4,
    )


def _run_gui(EEG: dict[str, Any], renderer: Any | None) -> dict[str, Any] | None:
    result = inputgui(pop_dipplot_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options = {
        "comps": result.get("comps", ""),
        "rvrange": _vector_or_empty(result.get("rvrange", "")),
        "mri": result.get("mri", ""),
        "summary": "on" if result.get("summary", False) else "off",
        "drawedges": "on" if result.get("drawedges", False) else "off",
        "cornermri": "on" if result.get("cornermri", False) else "off",
        "projimg": "on" if result.get("projimg", False) else "off",
        "projlines": "on" if result.get("projlines", False) else "off",
        "pointout": "on" if result.get("pointout", False) else "off",
        "normlen": "on" if result.get("normlen", False) else "off",
        "extra_options": result.get("extra_options", ""),
    }
    return options


def _validate_components_have_positions(models: list[dict[str, Any]], components: list[int]) -> None:
    missing = [component for component in components if not np.asarray(models[component - 1].get("posxyz", [])).size]
    if missing:
        raise ValueError(f"Localization not found for selected components: {missing}")


def _plot_dipoles(EEG: dict[str, Any], models: list[dict[str, Any]], components: list[int], options: dict[str, Any]):
    use_mri = (
        _option_on(options.get("summary")) or _option_on(options.get("projimg")) or _option_on(options.get("cornermri"))
    )
    if use_mri:
        fig = plt.figure(figsize=(11, 8))
        grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])
        axis = fig.add_subplot(grid[0, :], projection="3d")
        slice_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    else:
        fig = plt.figure(figsize=(7, 6))
        axis = fig.add_subplot(111, projection="3d")
        slice_axes = []
    all_positions = []
    for component in components:
        model = models[component - 1]
        positions = np.asarray(model.get("posxyz", []), dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.shape[1] < 3:
            continue
        moments = _model_moments(model, positions.shape[0], options)
        all_positions.extend(positions[:, :3].tolist())
        rv = model.get("rv", np.nan)
        label = f"IC {component}"
        if _is_scalar(rv) and np.isfinite(float(rv)):
            label += f" (RV {float(rv) * 100:.1f}%)"
        axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=55, label=label)
        axis.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            moments[:, 0],
            moments[:, 1],
            moments[:, 2],
            length=18.0,
            normalize=_option_on(options.get("normlen")),
            linewidth=1.4,
        )
        for row in positions[:, :3]:
            axis.text(row[0], row[1], row[2], str(component), fontsize=9)
            if _option_on(options.get("projlines")):
                axis.plot([row[0], row[0]], [row[1], row[1]], [0, row[2]], linestyle="--", color="0.4", linewidth=0.8)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    coordformat = (EEG.get("dipfit") or {}).get("coordformat", "")
    axis.set_title(f"DIPFIT dipoles ({coordformat or 'unknown coordinates'})")
    _set_equal_3d_axes(axis, np.asarray(all_positions, dtype=float) if all_positions else np.zeros((1, 3)))
    if components:
        axis.legend(loc="best")
    if slice_axes:
        _plot_mri_slices(slice_axes, np.asarray(all_positions, dtype=float), options)
    fig.tight_layout()
    return fig


def _history_options(options: dict[str, Any]) -> dict[str, Any]:
    values = {}
    for key in ("rvrange", "mri", "summary", "drawedges", "cornermri", "projimg", "projlines", "pointout", "normlen"):
        value = options.get(key)
        if value not in (None, "", [], "off"):
            values[key] = value
    return values


def _vector_or_empty(value: Any) -> list[float]:
    if value is None or str(value).strip() == "":
        return []
    return numeric_vector(value, dtype=float).tolist()


def _is_scalar(value: Any) -> bool:
    try:
        np.asarray(value, dtype=float).reshape(())
    except ValueError:
        return False
    return True


def _filter_components_by_rv(models: list[dict[str, Any]], components: list[int], rvrange: Any) -> list[int]:
    bounds = _vector_or_empty(rvrange)
    if not bounds:
        return components
    if len(bounds) == 1:
        low, high = 0.0, bounds[0]
    elif len(bounds) == 2:
        low, high = min(bounds), max(bounds)
    else:
        raise ValueError("rvrange must contain [max] or [min max] in percent")
    selected = []
    for component in components:
        rv = models[component - 1].get("rv", np.nan)
        if _is_scalar(rv) and low <= float(rv) * 100.0 <= high:
            selected.append(component)
    return selected


def _model_moments(model: dict[str, Any], count: int, options: dict[str, Any]) -> np.ndarray:
    moments = np.asarray(model.get("momxyz", []), dtype=float)
    if moments.size == 0:
        moments = np.zeros((count, 3))
    if moments.ndim == 1:
        moments = moments.reshape(1, -1)
    if moments.shape[0] < count:
        moments = np.vstack([moments, np.zeros((count - moments.shape[0], moments.shape[1]))])
    moments = moments[:, :3]
    if _option_on(options.get("pointout")):
        positions = np.asarray(model.get("posxyz", []), dtype=float).reshape(count, -1)[:, :3]
        inward = np.sum(positions * moments, axis=1) < 0
        moments[inward] *= -1
    return moments


def _plot_mri_slices(axes: list[Any], positions: np.ndarray, options: dict[str, Any]) -> None:
    _validate_mri_option(options.get("mri", ""))
    volume = load_standard_mri_volume()
    for axis, mri_slice, title in zip(
        axes,
        dipfit_mri_slices(volume, positions),
        ("Sagittal", "Coronal", "Axial"),
        strict=True,
    ):
        axis.imshow(mri_slice.image, cmap="gray", origin="lower", extent=mri_slice.extent)
        axis.scatter(
            positions[:, mri_slice.x_axis],
            positions[:, mri_slice.y_axis],
            s=35,
            c="tab:red",
            edgecolors="white",
            linewidths=0.5,
        )
        if _option_on(options.get("projlines")):
            for row in positions:
                axis.axvline(row[mri_slice.x_axis], color="tab:red", linestyle=":", linewidth=0.6, alpha=0.5)
                axis.axhline(row[mri_slice.y_axis], color="tab:red", linestyle=":", linewidth=0.6, alpha=0.5)
        if _option_on(options.get("drawedges")):
            for spine in axis.spines.values():
                spine.set_color("white")
                spine.set_linewidth(1.2)
        axis.set_title(title)
        axis.set_xlabel(("X", "Y", "Z")[mri_slice.x_axis])
        axis.set_ylabel(("X", "Y", "Z")[mri_slice.y_axis])


def _validate_mri_option(value: Any) -> None:
    text = str(value or "").strip()
    if not text or "standard_mri" in text or "standard_BEM" in text:
        return
    raise DIPFITUnavailableError(
        "pop_dipplot can render EEGPrep's packaged standard MNI MRI slices; custom MRI plotting assets are not bundled"
    )


def _set_equal_3d_axes(axis: Any, positions: np.ndarray) -> None:
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    if finite.size == 0:
        return
    center = np.mean(finite, axis=0)
    radius = max(float(np.max(np.linalg.norm(finite - center, axis=1))), 50.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _option_on(value: Any) -> bool:
    return is_on(value)


__all__ = ["pop_dipplot", "pop_dipplot_dialog_spec"]
