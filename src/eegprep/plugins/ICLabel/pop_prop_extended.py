"""ICLabel extended channel/component property dashboard."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import history_command
from eegprep.plugins.ICLabel._prop_browser import build_navigable_dashboard
from eegprep.plugins.ICLabel._prop_numerics import (
    DEFAULT_ICLABEL_CLASSES,
    ClassifierData,
    DipfitData,
    ExtendedPropertyData,
    build_extended_property_data,
    classifier_default_index,
    classifier_name_from_gui,
    classifier_names,
    component_count,
    component_rejection_status,
    has_component_classifier,
    resolve_classifier_data,
    resolve_dipfit_data,
    selected_property_indices,
)


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
        figure = build_navigable_dashboard(
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
    "DEFAULT_ICLABEL_CLASSES",
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
