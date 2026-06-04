"""View channel or component property thumbnails."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._property_browser import property_activity_browser
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import one_based_indices
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot
from eegprep.plugins.ICLabel.pop_prop_extended import (
    classifier_default_index,
    classifier_name_from_gui,
    classifier_names,
    component_count,
    has_component_classifier,
    pop_prop_extended,
)


PLOTS_PER_FIGURE = 35


def pop_viewprops(
    EEG: dict[str, Any],
    typecomp: int | bool = 1,
    chanorcomp: Any = None,
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
    """Render channel/component property overview figures and activity views."""
    if EEG is None:
        return ([], "") if return_com else []
    if gui is None:
        gui = chanorcomp is None
    if gui:
        result = inputgui(pop_viewprops_dialog_spec(EEG, typecomp), renderer=renderer)
        if result is None:
            return ([], "") if return_com else []
        chanorcomp = result.get("chanorcomp", "")
        spec_opt = result.get("spec_opt", "")
        erp_opt = result.get("erp_opt", "")
        scroll_event = int(bool(result.get("scroll_event", True)))
        if not int(bool(typecomp)):
            classifier_name = classifier_name_from_gui(EEG, result.get("classifier_name", classifier_name))
    limit = int(EEG.get("nbchan", 0) or 0) if int(bool(typecomp)) else component_count(EEG)
    indices = one_based_indices(chanorcomp, limit=limit, default_all=True)
    figures = []
    if plot:
        figures = _plot_props(
            EEG,
            int(bool(typecomp)),
            indices,
            spec_opt,
            erp_opt,
            classifier_name,
            scroll_event,
            show_activity,
            reject_callback,
        )
    command = _history_command(typecomp, indices, spec_opt, erp_opt, scroll_event, classifier_name)
    return (figures, command) if return_com else figures


def pop_viewprops_dialog_spec(EEG: dict[str, Any], typecomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like ``pop_viewprops`` prompt."""
    is_channel = int(bool(typecomp))
    limit = int(EEG.get("nbchan", 0) or 0) if is_channel else component_count(EEG)
    label = "Channel indices to plot:" if is_channel else "Component indices to plot:"
    title = "View many chan or comp. properties -- pop_viewprops"
    geometry = [(1.3, 1), (1.3, 1), (1.3, 1), (1,)]
    controls = [
        ControlSpec("text", label),
        ControlSpec("edit", tag="chanorcomp", value=f"1:{limit}"),
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
        geometry.append((1,))
        controls.append(
            ControlSpec(
                "popupmenu",
                "|".join(classifiers),
                tag="classifier_name",
                value=classifier_default_index(classifiers),
            )
        )
    return DialogSpec(
        title=title,
        function_name="pop_viewprops",
        eeglab_source="plugins/ICLabel/viewprops/pop_viewprops.m",
        size=(600, 318 if classifiers else 288),
        geometry=tuple(geometry),
        controls=tuple(controls),
    )


def _plot_props(
    EEG: dict[str, Any],
    typecomp: int,
    indices: list[int],
    spec_opt: Any,
    erp_opt: Any,
    classifier_name: str,
    scroll_event: int | bool,
    show_activity: bool,
    reject_callback: Any | None,
) -> list[Any]:
    if not indices:
        return []
    visible_indices = indices[:PLOTS_PER_FIGURE]
    if not typecomp and has_component_classifier(EEG, classifier_name):
        figure = pop_prop_extended(
            EEG,
            0,
            visible_indices,
            None,
            spec_opt,
            erp_opt,
            scroll_event,
            classifier_name,
            gui=False,
            show_activity=show_activity,
            reject_callback=reject_callback,
        )
        return [figure] if figure is not None else []
    activity_views = [
        property_activity_browser(EEG, typecomp, index, scroll_event=scroll_event, show=show_activity)
        for index in visible_indices
    ]
    if not typecomp:
        figures = pop_topoplot(
            EEG, 0, visible_indices, "View components properties - pop_viewprops()", [], 0, gui=False
        )
        _attach_activity_views(figures, activity_views)
        return figures
    fig_obj, axes = plt.subplots(1, min(len(indices), PLOTS_PER_FIGURE), squeeze=False)
    labels = _channel_labels(EEG)
    for axis, index in zip(axes.ravel(), visible_indices):
        axis.axis("off")
        label = labels[index - 1] if index - 1 < len(labels) else str(index)
        axis.text(0.5, 0.5, label, ha="center", va="center", fontsize=10)
        axis.set_title(str(index))
    fig_obj.suptitle("View channels properties - pop_viewprops()")
    fig_obj.tight_layout()
    _attach_activity_views([fig_obj], activity_views)
    return [fig_obj]


def _attach_activity_views(figures: list[Any], activity_views: list[Any]) -> None:
    for figure in figures:
        figure.eegprep_activity_views = activity_views


def _history_command(
    typecomp: int | bool,
    indices: list[int],
    spec_opt: Any,
    erp_opt: Any,
    scroll_event: int | bool,
    classifier_name: str,
) -> str:
    args = [
        int(bool(typecomp)),
        indices,
        [] if spec_opt is None else spec_opt,
        [] if erp_opt is None else erp_opt,
        int(bool(scroll_event)),
        classifier_name,
    ]
    return "pop_viewprops(EEG, " + ", ".join(format_history_value(arg, cell_for_sequence=None) for arg in args) + ");"


def _channel_labels(EEG: dict[str, Any]) -> list[str]:
    labels = []
    chanlocs = EEG.get("chanlocs", [])
    if chanlocs is None:
        chanlocs = []
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    for index, chanloc in enumerate(chanlocs):
        labels.append(str(chanloc.get("labels", index + 1)) if isinstance(chanloc, dict) else str(index + 1))
    return labels or [str(index + 1) for index in range(int(EEG.get("nbchan", 0) or 0))]
