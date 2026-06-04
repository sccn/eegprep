"""ICLabel extended channel/component property dashboard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import (
    channel_labels,
    component_activations,
    component_channel_indices,
    component_map_data,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
    parse_plot_options_text,
)
from eegprep.functions.popfunc._property_browser import property_activity_browser
from eegprep.functions.popfunc._rejection import (
    component_rejection_flags,
    one_based_indices,
    set_component_rejection_flag,
)
from eegprep.functions.sigprocfunc.spectopo import compute_spectra
from eegprep.functions.sigprocfunc.topoplot import topoplot
from eegprep.plugins.dipfit._mri import dipfit_mri_slices, load_standard_mri_volume
from eegprep.plugins.dipfit._utils import normalize_model_list


DEFAULT_ICLABEL_CLASSES = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other")
_DASHBOARD_SIZE = (12.0, 7.0)
_SCROLL_SECONDS = 5.0
_EVENT_COLORS = (
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)
_DIPFIT_COLORS = ("#00cc00", "#d336d3", "#e0c21a")
_REJECT_COLOR = "#ff9999"
_ACCEPT_COLOR = "#bfffbf"
_CONTROL_COLOR = "#e6e6e6"
_DISABLED_CONTROL_COLOR = "#d0d0d0"


@dataclass(frozen=True)
class ClassifierData:
    """Normalized component-classifier output from ``EEG.etc.ic_classification``."""

    name: str
    classes: tuple[str, ...]
    probabilities: np.ndarray


@dataclass(frozen=True)
class DipfitData:
    """Normalized localized DIPFIT model for one ICA component."""

    positions: np.ndarray
    moments: np.ndarray | None
    rv_percent: float | None
    dmr: float | None
    coordformat: str


@dataclass(frozen=True)
class ExtendedPropertyData:
    """Data assembled for one extended channel/component property dashboard."""

    typecomp: int
    index: int
    label: str
    figure_title: str
    topography_title: str
    topography_values: Any
    topography_chanlocs: list[dict[str, Any]]
    activity: np.ndarray
    times_ms: np.ndarray
    activity_title: str
    image_data: np.ndarray
    image_extent: tuple[float, float, float, float]
    image_title: str
    spectrum_freqs: np.ndarray
    spectrum_power: np.ndarray
    spectrum_title: str
    classifier: ClassifierData | None
    class_probabilities: np.ndarray | None
    pvaf: float | None
    dipfit: DipfitData | None
    rejected: bool | None


@dataclass(frozen=True)
class ActivityTraceData:
    """Inline scroll-plot trace using EEGLAB-compatible event coordinates."""

    x_values: np.ndarray
    y_values: np.ndarray
    times_ms: np.ndarray
    pnts: int
    epoched: bool


def pop_prop_extended(
    EEG: dict[str, Any] | None = None,
    typecomp: int | bool = 1,
    chanorcomp: Any = None,
    winhandle: Any = None,
    spec_opt: Any = None,
    erp_opt: Any = None,
    scroll_event: int | bool = 1,
    classifier_name: str = "",
    fig: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: bool = True,
    show_activity: bool = False,
    reject_callback: Any | None = None,
    return_com: bool = False,
):
    """Render the EEGLAB viewprops-style extended property dashboard.

    Args:
        EEG: EEGLAB-style EEG dictionary.
        typecomp: ``1`` for channels, ``0`` for ICA components.
        chanorcomp: EEGLAB-facing 1-based channel/component index or indices.
        winhandle: Accepted for EEGLAB signature compatibility.
        spec_opt: EEGLAB-style ``spectopo`` option text or parsed options.
        erp_opt: Accepted for EEGLAB signature compatibility; image options are
            currently interpreted by EEGPrep's native image summary.
        scroll_event: Include events in the attached activity browser.
        classifier_name: Component classifier field in ``EEG.etc.ic_classification``.
        fig: Optional Matplotlib figure to reuse.
        gui: Force or suppress the EEGLAB-like input dialog.
        renderer: Optional dialog renderer for tests.
        plot: Build the Matplotlib dashboard when true.
        show_activity: Open the browser-backed activity window in addition to
            attaching its model/window to the dashboard figure.
        reject_callback: Optional callable invoked after OK commits component
            rejection states. It receives ``(EEG, states)`` where ``states`` maps
            EEGLAB-facing component indices to booleans.
        return_com: Return ``(figure, command)`` when true.
    """
    if EEG is None:
        return (None, "") if return_com else None
    typecomp = int(bool(typecomp))
    if gui is None:
        gui = chanorcomp is None
    if gui:
        result = inputgui(pop_prop_extended_dialog_spec(EEG, typecomp), renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        chanorcomp = result.get("chanorcomp", "1")
        spec_opt = result.get("spec_opt", "")
        erp_opt = result.get("erp_opt", "")
        scroll_event = int(bool(result.get("scroll_event", True)))
        if not typecomp:
            classifier_name = classifier_name_from_gui(EEG, result.get("classifier_name", classifier_name))
    if chanorcomp is None:
        chanorcomp = 1
    indices = selected_property_indices(EEG, typecomp, chanorcomp, default_all=False)
    command = _history_command(typecomp, indices, winhandle, spec_opt, erp_opt, scroll_event, classifier_name)
    figure = None
    if plot:
        figure = _build_navigable_dashboard(
            EEG,
            typecomp,
            indices,
            winhandle,
            spec_opt,
            erp_opt,
            scroll_event,
            classifier_name,
            fig=fig,
            show_activity=show_activity,
            reject_callback=reject_callback,
        )
    return (figure, command) if return_com else figure


def pop_prop_extended_dialog_spec(EEG: dict[str, Any], typecomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like ``pop_prop_extended`` prompt."""
    is_channel = int(bool(typecomp))
    limit = int(EEG.get("nbchan", 0) or 0) if is_channel else component_count(EEG)
    label = "Channel index(ices) to plot:" if is_channel else "Component index(ices) to plot:"
    controls = [
        ControlSpec("text", label),
        ControlSpec("edit", tag="chanorcomp", value="1" if limit else ""),
        ControlSpec("text", "Spectral options (see spectopo() help):"),
        ControlSpec("edit", tag="spec_opt", value=f"'freqrange', [2 {min(80, float(EEG.get('srate', 1)) / 2):g}]"),
        ControlSpec("text", "Erpimage options (see erpimage() help):"),
        ControlSpec("edit", tag="erp_opt", value=""),
        ControlSpec(
            "checkbox",
            f"Draw events over scrolling {'channel' if is_channel else 'component'} activity",
            tag="scroll_event",
            value=True,
        ),
    ]
    classifiers = classifier_names(EEG) if not is_channel else []
    if classifiers:
        controls.append(
            ControlSpec(
                "popupmenu",
                "|".join(classifiers),
                tag="classifier_name",
                value=classifier_default_index(classifiers),
            )
        )
    return DialogSpec(
        title=f"{'Channel' if is_channel else 'Component'} properties - pop_prop_extended()",
        function_name="pop_prop_extended",
        eeglab_source="plugins/ICLabel/viewprops/pop_prop_extended.m",
        help_text="pophelp('pop_prop_extended')",
        size=(600, 318 if classifiers else 288),
        geometry=((1.3, 1), (1.3, 1), (1.3, 1), (1,), *((1,) if classifiers else ())),
        controls=tuple(controls),
        known_differences=(
            "EEGPrep uses one navigable dashboard for multiple selected components instead of opening one "
            "separate figure per component.",
        ),
    )


def classifier_names(EEG: dict[str, Any]) -> list[str]:
    """Return classifier field names available under ``EEG.etc.ic_classification``."""
    etc = EEG.get("etc") or {}
    if not isinstance(etc, dict):
        return []
    classifications = etc.get("ic_classification") or {}
    if not isinstance(classifications, dict):
        return []
    return [str(name) for name in classifications if str(name)]


def classifier_default_index(classifiers: list[str]) -> int:
    """Return EEGLAB popup index for the default component classifier."""
    for index, name in enumerate(classifiers, start=1):
        if name.lower() == "iclabel":
            return index
    return 1


def classifier_name_from_gui(EEG: dict[str, Any], value: Any) -> str:
    """Resolve a GUI popup value to a classifier field name."""
    classifiers = classifier_names(EEG)
    if not classifiers:
        return ""
    if isinstance(value, str):
        for classifier in classifiers:
            if classifier.lower() == value.lower():
                return classifier
        return classifiers[classifier_default_index(classifiers) - 1]
    try:
        index = int(value) - 1
    except (TypeError, ValueError):
        index = classifier_default_index(classifiers) - 1
    if 0 <= index < len(classifiers):
        return classifiers[index]
    return classifiers[classifier_default_index(classifiers) - 1]


def resolve_classifier_data(
    EEG: dict[str, Any],
    classifier_name: str = "",
    *,
    component_total: int | None = None,
    require: bool = False,
) -> ClassifierData | None:
    """Return normalized classifier data or ``None`` when no classifier is available."""
    classifiers = classifier_names(EEG)
    if not classifiers:
        if require:
            raise ValueError("No component classifier data found in EEG.etc.ic_classification")
        return None
    resolved_name = _resolve_classifier_name(classifiers, classifier_name)
    record = (EEG.get("etc") or {})["ic_classification"][resolved_name]
    if not isinstance(record, dict):
        raise ValueError(f"Classifier {resolved_name!r} must be stored as a dictionary")
    probabilities = np.asarray(record.get("classifications", []), dtype=float)
    if probabilities.ndim != 2 or probabilities.size == 0:
        raise ValueError(f"Classifier {resolved_name!r} is missing a 2-D classifications matrix")
    if component_total is not None and probabilities.shape[0] != int(component_total):
        raise ValueError(
            f"Classifier {resolved_name!r} has {probabilities.shape[0]} rows for {component_total} ICA components"
        )
    classes = _classifier_classes(record, resolved_name, probabilities.shape[1])
    return ClassifierData(resolved_name, classes, probabilities)


def resolve_dipfit_data(EEG: dict[str, Any], component_index: int) -> DipfitData | None:
    """Return normalized DIPFIT model data for a 1-based component index."""
    models = normalize_model_list(EEG)
    index = int(component_index)
    if index < 1:
        raise ValueError("component index must be 1-based")
    if index > len(models):
        return None
    model = models[index - 1]
    raw_positions = np.asarray(model.get("posxyz", []), dtype=float)
    if raw_positions.size == 0:
        return None
    positions = _dipfit_matrix(raw_positions, "posxyz", index)
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"DIPFIT model for component {index} contains non-finite posxyz values")
    moments = _dipfit_moments(model.get("momxyz", []), positions.shape[0], index)
    rv = _finite_float(model.get("rv"))
    coordformat = ""
    dipfit = EEG.get("dipfit")
    if isinstance(dipfit, dict):
        coordformat = str(dipfit.get("coordformat") or "")
    return DipfitData(
        positions=positions,
        moments=moments,
        rv_percent=None if rv is None else rv * 100.0,
        dmr=_dipole_moment_ratio(moments),
        coordformat=coordformat,
    )


def selected_property_indices(
    EEG: dict[str, Any],
    typecomp: int | bool,
    values: Any,
    *,
    default_all: bool = True,
) -> list[int]:
    """Normalize EEGLAB-facing channel/component selections to 1-based indices."""
    limit = int(EEG.get("nbchan", 0) or 0) if int(bool(typecomp)) else component_count(EEG)
    return one_based_indices(values, limit=limit, default_all=default_all)


def component_count(EEG: dict[str, Any]) -> int:
    """Return the number of available ICA components."""
    icaact = EEG.get("icaact")
    if icaact is not None and np.asarray(icaact).size:
        values = np.asarray(icaact)
        if values.ndim >= 2:
            return int(values.shape[0])
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0


def has_component_classifier(EEG: dict[str, Any], classifier_name: str = "") -> bool:
    """Return whether usable classifier data are available for the current ICA."""
    try:
        return (
            resolve_classifier_data(EEG, classifier_name, component_total=component_count(EEG), require=False)
            is not None
        )
    except ValueError as exc:
        if "missing a 2-D classifications matrix" in str(exc):
            return False
        raise


def component_rejection_status(
    EEG: dict[str, Any],
    component_index: int,
    *,
    component_total: int | None = None,
) -> bool:
    """Return the current ``EEG.reject.gcompreject`` status for one component."""
    total = component_count(EEG) if component_total is None else int(component_total)
    index = int(component_index)
    if index < 1 or index > total:
        raise ValueError("component index is outside available ICA components")
    return bool(component_rejection_flags(EEG, total, create=False)[index - 1])


def build_extended_property_data(
    EEG: dict[str, Any],
    typecomp: int | bool,
    index: int,
    *,
    spec_opt: Any = None,
    erp_opt: Any = None,
    classifier_name: str = "",
) -> ExtendedPropertyData:
    """Assemble the dashboard data for one EEGLAB-facing channel/component index."""
    del erp_opt
    typecomp = int(bool(typecomp))
    data = eeg_epoch_data(EEG)
    times_ms = eeg_times_ms(EEG)
    if typecomp:
        return _channel_dashboard_data(EEG, data, times_ms, int(index), spec_opt)
    return _component_dashboard_data(EEG, data, times_ms, int(index), spec_opt, classifier_name)


def _build_navigable_dashboard(
    EEG: dict[str, Any],
    typecomp: int,
    indices: list[int],
    winhandle: Any,
    spec_opt: Any,
    erp_opt: Any,
    scroll_event: int | bool,
    classifier_name: str,
    *,
    fig: Any,
    show_activity: bool,
    reject_callback: Any | None,
) -> Any:
    figure = fig if fig is not None else plt.figure(figsize=_DASHBOARD_SIZE)
    state = {
        "EEG": EEG,
        "typecomp": int(typecomp),
        "indices": tuple(indices),
        "position": 0,
        "spec_opt": spec_opt,
        "erp_opt": erp_opt,
        "scroll_event": int(bool(scroll_event)),
        "classifier_name": classifier_name,
        "show_activity": bool(show_activity),
        "winhandle": winhandle,
        "reject_callback": reject_callback,
        "rejection_pending": _initial_rejection_state(EEG, int(typecomp), indices),
    }
    figure.eegprep_dashboard_state = state
    _render_dashboard(figure)
    return figure


def _render_dashboard(figure: Any) -> None:
    state = figure.eegprep_dashboard_state
    indices = state["indices"]
    position = int(state["position"])
    index = int(indices[position])
    dashboard = build_extended_property_data(
        state["EEG"],
        state["typecomp"],
        index,
        spec_opt=state["spec_opt"],
        erp_opt=state["erp_opt"],
        classifier_name=state["classifier_name"],
    )
    figure.clf()
    figure.set_size_inches(*_DASHBOARD_SIZE, forward=True)
    figure.patch.set_facecolor((0.93, 0.96, 1.0))
    _set_window_title(figure, dashboard.figure_title)
    has_rejection_controls = _has_rejection_controls(state)
    bottom = (
        0.17
        if has_rejection_controls and len(indices) > 1
        else 0.125
        if has_rejection_controls
        else 0.12
        if len(indices) > 1
        else 0.075
    )
    if dashboard.dipfit is None:
        grid = figure.add_gridspec(
            2,
            4,
            left=0.055,
            right=0.97,
            top=0.88,
            bottom=bottom,
            wspace=0.65,
            hspace=0.55,
            width_ratios=(1.2, 0.95, 1.25, 1.25),
            height_ratios=(0.9, 1.2),
        )
        topo_ax = figure.add_subplot(grid[0, 0])
        if dashboard.classifier is None:
            class_ax = None
            activity_ax = figure.add_subplot(grid[0, 1:])
        else:
            class_ax = figure.add_subplot(grid[0, 1])
            activity_ax = figure.add_subplot(grid[0, 2:])
        image_ax = figure.add_subplot(grid[1, :2])
        dipfit_axes = []
        spectrum_ax = figure.add_subplot(grid[1, 2:])
    else:
        grid = figure.add_gridspec(
            2,
            5,
            left=0.055,
            right=0.97,
            top=0.88,
            bottom=bottom,
            wspace=0.58,
            hspace=0.55,
            width_ratios=(1.15, 0.9, 0.85, 1.25, 1.25),
            height_ratios=(0.9, 1.2),
        )
        topo_ax = figure.add_subplot(grid[0, 0])
        if dashboard.classifier is None:
            class_ax = None
            activity_ax = figure.add_subplot(grid[0, 1:])
        else:
            class_ax = figure.add_subplot(grid[0, 1])
            activity_ax = figure.add_subplot(grid[0, 2:])
        image_ax = figure.add_subplot(grid[1, :2])
        dipfit_grid = grid[1, 2].subgridspec(3, 1, hspace=0.04)
        dipfit_axes = [figure.add_subplot(dipfit_grid[row, 0]) for row in range(3)]
        spectrum_ax = figure.add_subplot(grid[1, 3:])

    _plot_topography(topo_ax, dashboard)
    if class_ax is not None:
        _plot_classifier(class_ax, dashboard)
    events = state["EEG"].get("event", []) if bool(state["scroll_event"]) else []
    _plot_activity(activity_ax, dashboard, events)
    _plot_activity_image(image_ax, dashboard)
    if dipfit_axes:
        _plot_dipfit(dipfit_axes, dashboard.dipfit)
    _plot_spectrum(spectrum_ax, dashboard)
    figure.suptitle(dashboard.figure_title, fontsize=14, fontweight="bold")
    figure.eegprep_dashboard_data = dashboard
    figure.eegprep_activity_view = property_activity_browser(
        state["EEG"],
        dashboard.typecomp,
        dashboard.index,
        scroll_event=state["scroll_event"],
        show=state["show_activity"],
    )
    if len(indices) > 1:
        _add_navigation_controls(
            figure,
            bottom=0.075 if has_rejection_controls else 0.025,
            count_y=0.1 if has_rejection_controls else 0.092,
        )
    else:
        figure.eegprep_dashboard_navigation = {}
        figure.eegprep_dashboard_navigation_buttons = ()
    if has_rejection_controls:
        _add_rejection_controls(figure, dashboard)
    else:
        figure.eegprep_dashboard_rejection = {}
        figure.eegprep_dashboard_rejection_buttons = {}
        figure.eegprep_dashboard_rejection_button_list = ()
    buttons = []
    buttons.extend(getattr(figure, "eegprep_dashboard_navigation_buttons", ()))
    buttons.extend(getattr(figure, "eegprep_dashboard_rejection_button_list", ()))
    figure.eegprep_dashboard_buttons = tuple(buttons)
    figure.canvas.draw_idle()


def _plot_topography(axis: Any, dashboard: ExtendedPropertyData) -> None:
    if dashboard.typecomp:
        topoplot(
            dashboard.topography_values,
            dashboard.topography_chanlocs,
            axes=axis,
            style="blank",
            electrodes="off",
        )
    else:
        topoplot(
            dashboard.topography_values,
            dashboard.topography_chanlocs,
            axes=axis,
            electrodes="on",
            colorbar=False,
        )
    axis.set_title(dashboard.topography_title, fontsize=12, fontweight="normal")
    if dashboard.pvaf is not None:
        axis.text(
            0.5,
            -0.13,
            f"{{% scalp data var. accounted for}}: {dashboard.pvaf:.1f}%",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )


def _plot_classifier(axis: Any, dashboard: ExtendedPropertyData) -> None:
    assert dashboard.classifier is not None
    assert dashboard.class_probabilities is not None
    labels = list(reversed(dashboard.classifier.classes))
    probabilities = np.asarray(dashboard.class_probabilities, dtype=float)[::-1]
    y_values = np.arange(len(labels))
    axis.barh(y_values, probabilities, color="#4c78a8")
    axis.set_yticks(y_values, labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.grid(axis="x", alpha=0.3)
    axis.set_xlabel("Probability")
    axis.set_title(dashboard.classifier.name, fontsize=12, fontweight="normal")
    for y_value, probability in zip(y_values, probabilities):
        axis.text(0.5, y_value, f"{probability * 100:.1f}%", ha="center", va="center", fontsize=8)


def _plot_activity(axis: Any, dashboard: ExtendedPropertyData, events: Any) -> None:
    trace = _activity_trace(dashboard)
    axis.plot(trace.x_values, trace.y_values, color="black", linewidth=0.85)
    axis.axhline(0.0, color="0.75", linewidth=0.6)
    _plot_epoch_markers(axis, trace)
    _plot_event_markers(axis, trace, events)
    axis.set_title(dashboard.activity_title, fontsize=12, fontweight="normal")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("uV")
    axis.grid(True, alpha=0.2)
    _format_scrollplot_axis(axis, trace)


def _plot_activity_image(axis: Any, dashboard: ExtendedPropertyData) -> None:
    image = np.asarray(dashboard.image_data, dtype=float)
    handle = axis.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=dashboard.image_extent,
        cmap="RdBu_r",
    )
    axis.set_title(dashboard.image_title, fontsize=12, fontweight="normal")
    axis.set_xlabel("Time (ms)" if dashboard.activity.shape[2] > 1 else "Data")
    axis.set_ylabel("Epoch" if dashboard.activity.shape[2] > 1 else "Data")
    plt.colorbar(handle, ax=axis, fraction=0.046, pad=0.035)


def _plot_spectrum(axis: Any, dashboard: ExtendedPropertyData) -> None:
    axis.plot(dashboard.spectrum_freqs, dashboard.spectrum_power, color="black", linewidth=1.0)
    axis.set_title(dashboard.spectrum_title, fontsize=12, fontweight="normal")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power 10*log10(uV^2/Hz)")
    axis.grid(True, alpha=0.25)
    finite = np.isfinite(dashboard.spectrum_power)
    if np.any(finite):
        finite_values = dashboard.spectrum_power[finite]
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
        if high == low:
            padding = 1.0 if low == 0.0 else abs(low) * 0.05
            low -= padding
            high += padding
        axis.set_ylim(low, high)


def _plot_dipfit(axes: list[Any], dipfit: DipfitData | None) -> None:
    assert dipfit is not None
    volume = load_standard_mri_volume()
    for axis, mri_slice in zip(axes, dipfit_mri_slices(volume, dipfit.positions)):
        axis.set_facecolor("black")
        axis.imshow(mri_slice.image, cmap="gray", origin="lower", extent=mri_slice.extent, interpolation="nearest")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        _plot_dipfit_points(axis, dipfit, mri_slice.x_axis, mri_slice.y_axis)
    axes[0].set_title("Dipole Position", fontsize=12, fontweight="normal", pad=7)
    _plot_dipfit_values(axes[-1], dipfit)


def _plot_dipfit_points(axis: Any, dipfit: DipfitData, x_index: int, y_index: int) -> None:
    for row, position in enumerate(dipfit.positions):
        color = _DIPFIT_COLORS[row % len(_DIPFIT_COLORS)]
        x_value = float(position[x_index])
        y_value = float(position[y_index])
        axis.plot(
            x_value,
            y_value,
            marker="o",
            markersize=5.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
        )
        if dipfit.moments is not None and row < dipfit.moments.shape[0]:
            _plot_dipfit_moment(axis, x_value, y_value, dipfit.moments[row], x_index, y_index, color)


def _plot_dipfit_moment(
    axis: Any,
    x_value: float,
    y_value: float,
    moment: np.ndarray,
    x_index: int,
    y_index: int,
    color: str,
) -> None:
    dx = float(moment[x_index])
    dy = float(moment[y_index])
    norm = float(np.hypot(dx, dy))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    axis.arrow(
        x_value,
        y_value,
        dx / norm * 18.0,
        dy / norm * 18.0,
        color=color,
        width=0.8,
        head_width=5.0,
        length_includes_head=True,
        alpha=0.9,
    )


def _plot_dipfit_values(axis: Any, dipfit: DipfitData) -> None:
    lines = []
    if dipfit.rv_percent is not None:
        lines.append(f"RV: {dipfit.rv_percent:.1f}%")
    if dipfit.dmr is not None:
        lines.append(f"DMR: {dipfit.dmr:.1f}")
    if lines:
        axis.text(
            0.5,
            -0.02,
            "\n".join(lines),
            transform=axis.transAxes,
            color="black",
            fontsize=8,
            ha="center",
            va="top",
        )


def _add_navigation_controls(figure: Any, *, bottom: float, count_y: float) -> None:
    previous_axis = figure.add_axes((0.37, bottom, 0.105, 0.05))
    next_axis = figure.add_axes((0.525, bottom, 0.105, 0.05))
    previous_button = Button(previous_axis, "Previous")
    next_button = Button(next_axis, "Next")

    def previous(_event: Any = None) -> None:
        _navigate_dashboard(figure, -1)

    def next_(_event: Any = None) -> None:
        _navigate_dashboard(figure, 1)

    previous_button.on_clicked(previous)
    next_button.on_clicked(next_)
    figure.eegprep_dashboard_navigation_buttons = (previous_button, next_button)
    figure.eegprep_dashboard_navigation = {"previous": previous, "next": next_}
    state = figure.eegprep_dashboard_state
    figure.text(
        0.5,
        count_y,
        f"{int(state['position']) + 1} / {len(state['indices'])}",
        ha="center",
        va="center",
        fontsize=9,
    )


def _add_rejection_controls(figure: Any, dashboard: ExtendedPropertyData) -> None:
    state = figure.eegprep_dashboard_state
    index = int(dashboard.index)
    rejected = _pending_rejection_status(state, index)
    cancel_button = Button(figure.add_axes((0.2, 0.015, 0.1, 0.045)), "Cancel", color=_CONTROL_COLOR)
    values_button = Button(figure.add_axes((0.325, 0.015, 0.1, 0.045)), "Values", color=_CONTROL_COLOR)
    status_button = Button(
        figure.add_axes((0.45, 0.015, 0.1, 0.045)),
        _rejection_label(rejected),
        color=_rejection_color(rejected),
        hovercolor=_rejection_color(rejected),
    )
    help_button = Button(figure.add_axes((0.575, 0.015, 0.1, 0.045)), "HELP", color=_CONTROL_COLOR)
    ok_button = Button(figure.add_axes((0.7, 0.015, 0.1, 0.045)), "OK", color=_CONTROL_COLOR)

    if not _component_values_available(state["EEG"]):
        values_button.set_active(False)
        values_button.ax.set_facecolor(_DISABLED_CONTROL_COLOR)

    def cancel(_event: Any = None) -> None:
        plt.close(figure)

    def values(_event: Any = None) -> None:
        if _component_values_available(state["EEG"]):
            _show_component_values(state["EEG"], index, _pending_rejection_status(state, index))

    def toggle(_event: Any = None) -> None:
        state["rejection_pending"][index] = not _pending_rejection_status(state, index)
        _style_rejection_status_button(status_button, state["rejection_pending"][index])
        figure.canvas.draw_idle()

    def help_(_event: Any = None) -> None:
        pophelp("pop_prop_extended")

    def ok(_event: Any = None) -> None:
        _commit_rejection_state(figure)
        plt.close(figure)

    cancel_button.on_clicked(cancel)
    values_button.on_clicked(values)
    status_button.on_clicked(toggle)
    help_button.on_clicked(help_)
    ok_button.on_clicked(ok)
    figure.eegprep_dashboard_rejection_button_list = (
        cancel_button,
        values_button,
        status_button,
        help_button,
        ok_button,
    )
    figure.eegprep_dashboard_rejection_buttons = {
        "cancel": cancel_button,
        "values": values_button,
        "status": status_button,
        "help": help_button,
        "ok": ok_button,
    }
    figure.eegprep_dashboard_rejection = {
        "cancel": cancel,
        "values": values,
        "toggle": toggle,
        "help": help_,
        "ok": ok,
        "pending": state["rejection_pending"],
    }


def _navigate_dashboard(figure: Any, step: int) -> None:
    state = figure.eegprep_dashboard_state
    state["position"] = (int(state["position"]) + int(step)) % len(state["indices"])
    _render_dashboard(figure)


def _initial_rejection_state(EEG: dict[str, Any], typecomp: int, indices: list[int]) -> dict[int, bool]:
    if int(typecomp):
        return {}
    total = component_count(EEG)
    flags = component_rejection_flags(EEG, total, create=False)
    return {int(index): bool(flags[int(index) - 1]) for index in indices}


def _has_rejection_controls(state: dict[str, Any]) -> bool:
    return int(state["typecomp"]) == 0 and bool(state["indices"])


def _pending_rejection_status(state: dict[str, Any], component_index: int) -> bool:
    pending = state["rejection_pending"]
    index = int(component_index)
    if index not in pending:
        pending[index] = component_rejection_status(state["EEG"], index)
    return bool(pending[index])


def _rejection_label(rejected: bool) -> str:
    return "REJECT" if rejected else "ACCEPT"


def _rejection_color(rejected: bool) -> str:
    return _REJECT_COLOR if rejected else _ACCEPT_COLOR


def _style_rejection_status_button(button: Button, rejected: bool) -> None:
    button.label.set_text(_rejection_label(rejected))
    button.ax.set_facecolor(_rejection_color(rejected))
    button.color = _rejection_color(rejected)
    button.hovercolor = _rejection_color(rejected)


def _commit_rejection_state(figure: Any) -> None:
    state = figure.eegprep_dashboard_state
    if not _has_rejection_controls(state):
        return
    EEG = state["EEG"]
    total = component_count(EEG)
    committed: dict[int, bool] = {}
    for index in sorted(state["rejection_pending"]):
        rejected = bool(state["rejection_pending"][index])
        set_component_rejection_flag(EEG, index, rejected, total)
        _style_rejection_winhandle(state["winhandle"], index, rejected)
        committed[int(index)] = rejected
    callback = state.get("reject_callback")
    if callback is not None:
        callback(EEG, dict(committed))


def _style_rejection_winhandle(winhandle: Any, component_index: int, rejected: bool) -> None:
    if _is_empty_winhandle(winhandle):
        return
    handle = winhandle
    if isinstance(winhandle, Mapping):
        handle = winhandle.get(int(component_index))
    if isinstance(handle, Button):
        handle.ax.set_facecolor(_rejection_color(rejected))


def _is_empty_winhandle(winhandle: Any) -> bool:
    if winhandle is None:
        return True
    if isinstance(winhandle, (int, float, np.integer, np.floating)):
        return bool(winhandle == 0) or bool(np.isnan(float(winhandle)))
    return False


def _component_values_available(EEG: dict[str, Any]) -> bool:
    stats = EEG.get("stats")
    if not isinstance(stats, dict):
        return False
    return np.asarray(stats.get("compenta", [])).size > 0


def _show_component_values(EEG: dict[str, Any], component_index: int, rejected: bool) -> None:
    values = _component_value_lines(EEG, component_index, rejected)
    figure = plt.figure(figsize=(3.4, 3.4))
    manager = getattr(figure.canvas, "manager", None)
    if manager is not None:
        manager.set_window_title("Statistics of the component")
    axis = figure.add_subplot(1, 1, 1)
    axis.axis("off")
    axis.text(0.04, 0.96, "\n".join(values), va="top", ha="left", fontsize=9, family="monospace")
    close_button = Button(figure.add_axes((0.375, 0.03, 0.25, 0.08)), "Close", color=_CONTROL_COLOR)
    close_button.on_clicked(lambda _event=None: plt.close(figure))
    setattr(figure, "eegprep_component_values_button", close_button)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.canvas.draw_idle()


def _component_value_lines(EEG: dict[str, Any], component_index: int, rejected: bool) -> list[str]:
    raw_stats = EEG.get("stats")
    stats = raw_stats if isinstance(raw_stats, dict) else {}
    raw_reject = EEG.get("reject")
    reject = raw_reject if isinstance(raw_reject, dict) else {}
    index = int(component_index)
    return [
        "(",
        f"Entropy of component activity       {_indexed_stat(stats.get('compenta'), index):>8}",
        f"> Rejection threshold                {_scalar_stat(reject.get('threshentropy')):>8}",
        "",
        " AND                                ----",
        "",
        f"Kurtosis of component activity      {_indexed_stat(stats.get('compkurta'), index):>8}",
        f"> Rejection threshold                {_scalar_stat(reject.get('threshkurtact')):>8}",
        "",
        ") OR                               ----",
        "",
        f"Kurtosis distribution                {_indexed_stat(stats.get('compkurtdist'), index):>8}",
        f"> Rejection threshold                {_scalar_stat(reject.get('threshkurtdist')):>8}",
        "",
        f"Current thresholds suggest to {_rejection_label(rejected)} the component",
        "",
        "After manually accepting/rejecting the component, recalibrate",
        "thresholds before applying automatic rejection to other datasets.",
    ]


def _indexed_stat(values: Any, component_index: int) -> str:
    vector = np.asarray(values, dtype=float).ravel()
    index = int(component_index) - 1
    if index < 0 or index >= vector.size or not np.isfinite(vector[index]):
        return "----"
    return f"{float(vector[index]):2.2f}"


def _scalar_stat(value: Any) -> str:
    vector = np.asarray(value, dtype=float).ravel()
    if vector.size == 0 or not np.isfinite(vector[0]):
        return "----"
    return f"{float(vector[0]):2.2f}"


def _channel_dashboard_data(
    EEG: dict[str, Any],
    data: np.ndarray,
    times_ms: np.ndarray,
    index: int,
    spec_opt: Any,
) -> ExtendedPropertyData:
    labels = channel_labels(EEG)
    if index < 1 or index > data.shape[0]:
        raise ValueError("channel index is outside available channels")
    label = labels[index - 1] if index - 1 < len(labels) else str(index)
    activity = np.array(data[index - 1 : index], dtype=float, copy=True)
    spectrum_freqs, spectrum_power = _spectrum(activity, EEG, spec_opt)
    image_data, image_extent, image_title = _activity_image(activity, times_ms, f"Epoched Channel {label} Activity")
    return ExtendedPropertyData(
        typecomp=1,
        index=index,
        label=label,
        figure_title=f"Channel {label} - pop_prop_extended()",
        topography_title=f"Channel {label}",
        topography_values=index,
        topography_chanlocs=list(EEG.get("chanlocs", []) or []),
        activity=activity,
        times_ms=times_ms,
        activity_title="Channel Time Series",
        image_data=image_data,
        image_extent=image_extent,
        image_title=image_title,
        spectrum_freqs=spectrum_freqs,
        spectrum_power=spectrum_power,
        spectrum_title="Channel Activity Power Spectrum",
        classifier=None,
        class_probabilities=None,
        pvaf=None,
        dipfit=None,
        rejected=None,
    )


def _component_dashboard_data(
    EEG: dict[str, Any],
    data: np.ndarray,
    times_ms: np.ndarray,
    index: int,
    spec_opt: Any,
    classifier_name: str,
) -> ExtendedPropertyData:
    activity_all = component_activations(EEG)
    maps, map_chanlocs = component_map_data(EEG)
    if index < 1 or index > activity_all.shape[0]:
        raise ValueError("component index is outside available ICA components")
    activity = np.array(activity_all[index - 1 : index], dtype=float, copy=True)
    classifier = resolve_classifier_data(EEG, classifier_name, component_total=activity_all.shape[0], require=False)
    probabilities = (
        None if classifier is None else np.array(classifier.probabilities[index - 1], dtype=float, copy=True)
    )
    spectrum_freqs, spectrum_power = _spectrum(activity, EEG, spec_opt)
    image_data, image_extent, image_title = _activity_image(activity, times_ms, f"Epoched IC{index} Activity")
    return ExtendedPropertyData(
        typecomp=0,
        index=index,
        label=f"IC{index}",
        figure_title=f"IC{index} - pop_prop_extended()",
        topography_title=f"IC{index}",
        topography_values=maps[:, index - 1],
        topography_chanlocs=map_chanlocs,
        activity=activity,
        times_ms=times_ms,
        activity_title=f"Scrolling IC{index} Activity",
        image_data=image_data,
        image_extent=image_extent,
        image_title=image_title,
        spectrum_freqs=spectrum_freqs,
        spectrum_power=spectrum_power,
        spectrum_title=f"IC{index} Activity Power Spectrum",
        classifier=classifier,
        class_probabilities=probabilities,
        pvaf=_component_pvaf(EEG, data, maps, activity, index),
        dipfit=resolve_dipfit_data(EEG, index),
        rejected=component_rejection_status(EEG, index, component_total=activity_all.shape[0]),
    )


def _spectrum(activity: np.ndarray, EEG: dict[str, Any], spec_opt: Any) -> tuple[np.ndarray, np.ndarray]:
    options = parse_plot_options_text(spec_opt)
    flat = np.asarray(activity, dtype=float).reshape(1, -1)
    spectra, freqs, _std = compute_spectra(
        flat,
        int(EEG.get("pnts", flat.shape[1]) or flat.shape[1]),
        float(EEG.get("srate", 1.0) or 1.0),
        winsize=_first_int(options.get("winsize")),
        overlap=_first_int(options.get("overlap")) or 0,
        nfft=_first_int(options.get("nfft")),
    )
    return freqs, spectra[0]


def _activity_image(
    activity: np.ndarray,
    times_ms: np.ndarray,
    epoched_title: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    trace = np.asarray(activity[0], dtype=float)
    trace = trace - float(np.nanmean(trace))
    if trace.ndim == 1:
        trace = trace[:, np.newaxis]
    if trace.shape[1] > 1:
        image = trace.T
        extent = (float(times_ms[0]), float(times_ms[-1]), 1.0, float(trace.shape[1]))
        return image, extent, epoched_title
    flat = trace[:, 0]
    line_count = min(200, max(1, int(np.floor(np.sqrt(flat.size)))))
    frame_count = max(1, flat.size // line_count)
    image = flat[: line_count * frame_count].reshape(line_count, frame_count)
    extent = (0.0, float(frame_count - 1), 1.0, float(line_count))
    return image, extent, "Continuous Data"


def _activity_trace(dashboard: ExtendedPropertyData) -> ActivityTraceData:
    trace = np.asarray(dashboard.activity[0], dtype=float)
    if trace.ndim == 1:
        trace = trace[:, np.newaxis]
    pnts = int(dashboard.times_ms.size)
    srate = _srate_from_times(dashboard.times_ms)
    window_samples = max(1, int(round(_SCROLL_SECONDS * srate)))
    if trace.shape[1] == 1:
        sample_count = min(pnts, window_samples)
        return ActivityTraceData(
            x_values=np.arange(1, sample_count + 1, dtype=float),
            y_values=trace[:sample_count, 0],
            times_ms=dashboard.times_ms,
            pnts=pnts,
            epoched=False,
        )
    flat = trace.T.reshape(-1)
    sample_count = min(flat.size, window_samples)
    return ActivityTraceData(
        x_values=np.arange(1, sample_count + 1, dtype=float),
        y_values=flat[:sample_count],
        times_ms=dashboard.times_ms,
        pnts=pnts,
        epoched=True,
    )


def _plot_event_markers(
    axis: Any,
    trace: ActivityTraceData,
    events: Any,
) -> None:
    event_items = _event_items(events)
    if not event_items:
        return
    first = float(np.nanmin(trace.x_values))
    last = float(np.nanmax(trace.x_values))
    colors = _event_color_map(event_items)
    for event in event_items:
        if "latency" not in event:
            continue
        try:
            latency = float(_event_scalar(event["latency"]))
        except (TypeError, ValueError):
            continue
        x_value = latency
        if first <= x_value <= last:
            label = str(_event_scalar(event.get("type", "")))
            axis.axvline(x_value, color=colors[label], linestyle="--", linewidth=0.75)
            axis.text(
                x_value,
                0.98,
                label,
                color=colors[label],
                transform=axis.get_xaxis_transform(),
                rotation=45,
                fontsize=8,
            )


def _plot_epoch_markers(axis: Any, trace: ActivityTraceData) -> None:
    if not trace.epoched:
        return
    first = float(np.nanmin(trace.x_values))
    last = float(np.nanmax(trace.x_values))
    start = int(np.ceil(first / trace.pnts) * trace.pnts)
    for boundary in range(start, int(np.floor(last / trace.pnts) * trace.pnts) + 1, trace.pnts):
        if boundary < first or boundary > last:
            continue
        axis.axvline(float(boundary), color="red", linestyle="-", linewidth=0.75)
        axis.text(
            float(boundary),
            0.98,
            f"epoch {boundary // trace.pnts}",
            color="red",
            transform=axis.get_xaxis_transform(),
            rotation=45,
            fontsize=8,
        )


def _format_scrollplot_axis(axis: Any, trace: ActivityTraceData) -> None:
    first = float(trace.x_values[0])
    last = float(trace.x_values[-1])
    axis.set_xlim(first, last)
    tick_count = min(6, trace.x_values.size)
    ticks = np.unique(np.linspace(first, last, tick_count).round().astype(int))
    labels = []
    for tick in ticks:
        sample = (int(tick) - 1) % trace.pnts if trace.epoched else min(max(int(tick) - 1, 0), trace.pnts - 1)
        labels.append(f"{trace.times_ms[sample]:g}")
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)


def _event_items(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, dict):
        if "latency" not in events:
            return []
        latencies = np.asarray(events["latency"], dtype=object).ravel()
        types = np.asarray(events.get("type", [""] * latencies.size), dtype=object).ravel()
        epochs = np.asarray(events.get("epoch", [None] * latencies.size), dtype=object).ravel()
        event_items = []
        for index, latency in enumerate(latencies):
            event = {
                "type": _event_scalar(types[min(index, types.size - 1)]),
                "latency": _event_scalar(latency),
            }
            epoch = _event_scalar(epochs[min(index, epochs.size - 1)])
            if epoch is not None:
                event["epoch"] = epoch
            event_items.append(event)
        return event_items
    if isinstance(events, np.ndarray):
        events = events.tolist()
    try:
        event_values = list(events)
    except TypeError:
        return []
    return [event for event in event_values if isinstance(event, dict)]


def _event_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        item = array.ravel()[0]
        return item.item() if hasattr(item, "item") else item
    return value


def _event_color_map(event_items: list[dict[str, Any]]) -> dict[str, str]:
    labels: list[str] = []
    for event in event_items:
        label = str(_event_scalar(event.get("type", "")))
        if label not in labels:
            labels.append(label)
    return {label: _EVENT_COLORS[index % len(_EVENT_COLORS)] for index, label in enumerate(labels)}


def _dipfit_matrix(values: np.ndarray, field_name: str, component_index: int) -> np.ndarray:
    matrix = values
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError(
            f"DIPFIT model for component {component_index} must contain {field_name} rows with 3 coordinates"
        )
    return np.array(matrix[:, :3], dtype=float, copy=True)


def _dipfit_moments(values: Any, position_count: int, component_index: int) -> np.ndarray | None:
    raw_moments = np.asarray(values, dtype=float)
    if raw_moments.size == 0:
        return None
    moments = _dipfit_matrix(raw_moments, "momxyz", component_index)
    if moments.shape[0] != position_count:
        raise ValueError(f"DIPFIT model for component {component_index} must have matching posxyz and momxyz rows")
    if not np.all(np.isfinite(moments)):
        raise ValueError(f"DIPFIT model for component {component_index} contains non-finite momxyz values")
    return moments


def _dipole_moment_ratio(moments: np.ndarray | None) -> float | None:
    if moments is None or moments.shape[0] != 2:
        return None
    norms = np.linalg.norm(moments, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return None
    ratio = float(norms[0] / norms[1])
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _component_pvaf(
    EEG: dict[str, Any],
    data: np.ndarray,
    maps: np.ndarray,
    activity: np.ndarray,
    index: int,
) -> float | None:
    if maps.shape[0] == 0:
        return None
    icachansind = component_channel_indices(EEG, data.shape[0])
    if maps.shape[0] != icachansind.size:
        return None
    flat_data = data[icachansind, :, :].reshape(icachansind.size, -1)
    component_trace = activity.reshape(1, -1)
    projection = maps[:, index - 1 : index] @ component_trace
    datavar = float(np.nanmean(np.nanvar(flat_data, axis=1)))
    if not np.isfinite(datavar) or datavar <= 0:
        return None
    projvar = float(np.nanmean(np.nanvar(flat_data - projection, axis=1)))
    if not np.isfinite(projvar):
        return None
    return 100.0 * (1.0 - projvar / datavar)


def _resolve_classifier_name(classifiers: list[str], classifier_name: str) -> str:
    if classifier_name:
        for name in classifiers:
            if name.lower() == str(classifier_name).lower():
                return name
        raise ValueError(f"Classifier {classifier_name!r} was not found in EEG.etc.ic_classification")
    return classifiers[classifier_default_index(classifiers) - 1]


def _classifier_classes(record: dict[str, Any], classifier_name: str, class_count: int) -> tuple[str, ...]:
    raw_classes = record.get("classes", [])
    classes = [str(item) for item in np.asarray(raw_classes, dtype=object).ravel().tolist() if str(item)]
    if not classes:
        if classifier_name.lower() == "iclabel" and class_count == len(DEFAULT_ICLABEL_CLASSES):
            return DEFAULT_ICLABEL_CLASSES
        return tuple(f"Class {index}" for index in range(1, class_count + 1))
    if len(classes) != class_count:
        raise ValueError(
            f"Classifier {classifier_name!r} has {class_count} probability columns but {len(classes)} class names"
        )
    return tuple(classes)


def _first_int(value: Any) -> int | None:
    vector = numeric_vector(value)
    if vector.size == 0:
        return None
    return int(vector[0])


def _srate_from_times(times_ms: np.ndarray) -> float:
    if times_ms.size < 2:
        return 1.0
    interval_ms = float(np.nanmedian(np.diff(times_ms)))
    if not np.isfinite(interval_ms) or interval_ms <= 0:
        return 1.0
    return 1000.0 / interval_ms


def _set_window_title(figure: Any, title: str) -> None:
    figure.set_label(title)
    manager = getattr(getattr(figure, "canvas", None), "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _history_command(
    typecomp: int,
    indices: list[int],
    winhandle: Any,
    spec_opt: Any,
    erp_opt: Any,
    scroll_event: int | bool,
    classifier_name: str,
) -> str:
    return history_command(
        "pop_prop_extended",
        int(bool(typecomp)),
        indices if len(indices) != 1 else indices[0],
        winhandle,
        [] if spec_opt is None else spec_opt,
        [] if erp_opt is None else erp_opt,
        int(bool(scroll_event)),
        classifier_name,
    )


__all__ = [
    "ClassifierData",
    "DipfitData",
    "ExtendedPropertyData",
    "build_extended_property_data",
    "classifier_default_index",
    "classifier_name_from_gui",
    "classifier_names",
    "component_count",
    "component_rejection_status",
    "has_component_classifier",
    "pop_prop_extended",
    "pop_prop_extended_dialog_spec",
    "resolve_classifier_data",
    "resolve_dipfit_data",
    "selected_property_indices",
]
