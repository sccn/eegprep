"""Plot existing DIPFIT component dipoles with a standalone matplotlib view."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import is_on
from eegprep.functions.popfunc.plot_utils import numeric_vector, show_figures
from eegprep.plugins.dipfit._coordinates import sph2spm
from eegprep.plugins.dipfit._mri import dipfit_mri_slices, load_dipfit_mri_volume
from eegprep.plugins.dipfit._utils import (
    DIPFITUnavailableError,
    dipfit_history,
    dipfit_record,
    localized_components,
    matlab_plot_options,
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
    options.update(matlab_plot_options(options.pop("extra_options", "")))
    figures = [_plot_dipoles(EEG, models, components, options)] if plot else []
    show_figures(figures, plot=plot)
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
            ControlSpec(
                "pushbutton",
                "...",
                tag="mri_button",
                callback=CallbackSpec(
                    "select_file",
                    params={
                        "button": "mri_button",
                        "target": "mri",
                        "caption": "Background MRI",
                        "filter": "MRI files (*.mat *.nii *.nii.gz);;All files (*)",
                    },
                ),
            ),
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
            "Spherical plots use the packaged avg152t1 volume. MNI plots use standard_mri. A custom MRI file is refused.",
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
    coordformat = str(dipfit_record(EEG).get("coordformat", ""))
    use_mri = coordformat.lower() in {"mni", "spherical"}
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
    located: list[np.ndarray] = []
    groups: list[list[Any]] = []
    plotted: list[int] = []
    volume, volume_name = load_dipfit_mri_volume(coordformat) if use_mri else ({}, "")
    for component in components:
        model = models[component - 1]
        positions = np.asarray(model.get("posxyz", []), dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.size == 0 or positions.shape[1] < 3:
            continue
        positions = positions[:, :3]
        chosen = _selected_row_indices(model, positions.shape[0])
        if not chosen:
            continue
        moments = _model_moments(model, positions.shape[0], options)[:, :3]
        positions = positions[chosen]
        moments = moments[chosen]
        positions, moments = _display_dipole(positions, moments, coordformat)
        all_positions.extend(positions.tolist())
        rv = model.get("rv", np.nan)
        label = f"IC {component}"
        if _is_scalar(rv) and np.isfinite(float(rv)):
            label += f" (RV {float(rv) * 100:.1f}%)"
        located.append(positions)
        artists: list[Any] = [
            axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=55, label=label),
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
            ),
        ]
        for row in positions:
            artists.append(axis.text(row[0], row[1], row[2], str(component), fontsize=9))
            if _option_on(options.get("projlines")):
                artists.extend(
                    axis.plot(
                        [row[0], row[0]], [row[1], row[1]], [0, row[2]], linestyle="--", color="0.4", linewidth=0.8
                    )
                )
        groups.append(artists)
        plotted.append(component)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    title = options.get("title")
    axis.set_title(str(title) if title else f"DIPFIT dipoles ({coordformat or 'unknown coordinates'})")
    _set_equal_3d_axes(axis, np.asarray(all_positions, dtype=float) if all_positions else np.zeros((1, 3)))
    if plotted:
        axis.legend(loc="best")
    if slice_axes:
        setattr(fig, "eegprep_mri", volume_name)
    fig.tight_layout()
    _add_dipole_navigation(fig, groups, models, plotted, slice_axes, located, options, volume)
    return fig


def _display_dipole(positions, moments, coordformat):
    if str(coordformat).lower() != "spherical":
        return positions, moments
    matrix = sph2spm()
    ones = np.ones((positions.shape[0], 1))
    transformed = np.hstack((positions, ones)) @ matrix.T
    moved = np.hstack((positions + moments, ones)) @ matrix.T
    return transformed[:, :3], moved[:, :3] - transformed[:, :3]


def _selected_row_indices(model: dict[str, Any], count: int) -> list[int]:
    if "select" not in model or model.get("select") is None:
        return list(range(count))
    raw = np.atleast_1d(np.asarray(model.get("select"), dtype=float)).ravel()
    chosen = []
    for value in raw:
        index = int(value) - 1
        if 0 <= index < count:
            chosen.append(index)
    return chosen


def _add_dipole_navigation(fig, groups, models, components, slice_axes, located, options, volume) -> None:
    from matplotlib.widgets import Button

    if not groups:
        return
    state = {"index": 0, "single": False, "slice_artists": []}
    area = fig.text(0.02, 0.62, "", fontsize=8)
    fig.subplots_adjust(left=0.18)

    def apply() -> None:
        for index, artists in enumerate(groups):
            visible = not state["single"] or index == state["index"]
            for artist in artists:
                artist.set_visible(visible)
        model = models[components[state["index"]] - 1]
        label = str(model.get("areadk") or "").strip()
        area.set_text(f"Comp: {components[state['index']]} {label}".strip())
        if slice_axes:
            for artist in state["slice_artists"]:
                artist.remove()
            state["slice_artists"] = _plot_mri_slices(slice_axes, located[state["index"]], options, volume)
        fig.canvas.draw_idle()

    def plot_one(_event=None) -> None:
        state["single"] = not state["single"]
        plot_button.label.set_text("Plot all" if state["single"] else "Plot one")
        apply()

    def previous(_event=None) -> None:
        state["index"] = max(0, state["index"] - 1)
        apply()

    def following(_event=None) -> None:
        state["index"] = min(len(groups) - 1, state["index"] + 1)
        apply()

    plot_button = Button(fig.add_axes((0.01, 0.90, 0.14, 0.05)), "Plot one")
    next_button = Button(fig.add_axes((0.01, 0.84, 0.14, 0.05)), "Next")
    prev_button = Button(fig.add_axes((0.01, 0.78, 0.14, 0.05)), "Prev")
    plot_button.on_clicked(plot_one)
    next_button.on_clicked(following)
    prev_button.on_clicked(previous)
    setattr(
        fig,
        "eegprep_dipplot",
        {
            "plot_one": plot_one,
            "next": following,
            "prev": previous,
            "state": state,
            "area": area,
            "buttons": (plot_button, next_button, prev_button),
        },
    )
    apply()


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


def _plot_mri_slices(axes: list[Any], positions: np.ndarray, options: dict[str, Any], volume: dict) -> list[Any]:
    _validate_mri_option(options.get("mri", ""))
    artists: list[Any] = []
    for axis, mri_slice, title in zip(
        axes,
        dipfit_mri_slices(volume, positions),
        ("Sagittal", "Coronal", "Axial"),
        strict=True,
    ):
        artists.append(axis.imshow(mri_slice["image"], cmap="gray", origin="lower", extent=mri_slice["extent"]))
        artists.append(
            axis.scatter(
                positions[:, mri_slice["x_axis"]],
                positions[:, mri_slice["y_axis"]],
                s=35,
                c="tab:red",
                edgecolors="white",
                linewidths=0.5,
            )
        )
        if _option_on(options.get("projlines")):
            for row in positions:
                artists.append(
                    axis.axvline(row[mri_slice["x_axis"]], color="tab:red", linestyle=":", linewidth=0.6, alpha=0.5)
                )
                artists.append(
                    axis.axhline(row[mri_slice["y_axis"]], color="tab:red", linestyle=":", linewidth=0.6, alpha=0.5)
                )
        if _option_on(options.get("drawedges")):
            for spine in axis.spines.values():
                spine.set_color("white")
                spine.set_linewidth(1.2)
        axis.set_title(title)
        axis.set_xlabel(("X", "Y", "Z")[mri_slice["x_axis"]])
        axis.set_ylabel(("X", "Y", "Z")[mri_slice["y_axis"]])
    return artists


_PACKAGED_MRI_MARKERS = ("standard_mri", "standard_bem", "standard_besa")


def _validate_mri_option(value: Any) -> None:
    text = str(value or "").replace("\\", "/").strip().lower()
    if not text or any(marker in text for marker in _PACKAGED_MRI_MARKERS):
        return
    raise DIPFITUnavailableError(
        "pop_dipplot: custom MRI files are not plotted. "
        "Omit mri, or pass a packaged template MRI (standard_BEM or standard_BESA), "
        "so the standard_mri slices are used."
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
