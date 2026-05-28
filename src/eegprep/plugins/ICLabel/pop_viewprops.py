"""View channel or component property thumbnails."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import one_based_indices
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot


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
    return_com: bool = False,
):
    """Render a non-scrolling channel/component property overview."""
    if EEG is None:
        return ([], "") if return_com else []
    if gui is None:
        gui = chanorcomp is None
    if gui:
        result = inputgui(pop_viewprops_dialog_spec(EEG, typecomp), renderer=renderer)
        if result is None:
            return ([], "") if return_com else []
        chanorcomp = result.get("chanorcomp", "")
        scroll_event = int(bool(result.get("scroll_event", True)))
    limit = int(EEG.get("nbchan", 0) or 0) if int(bool(typecomp)) else _component_count(EEG)
    indices = one_based_indices(chanorcomp, limit=limit, default_all=True)
    figures = []
    if plot:
        figures = _plot_props(EEG, int(bool(typecomp)), indices, classifier_name)
    command = _history_command(typecomp, indices, spec_opt, erp_opt, scroll_event, classifier_name)
    return (figures, command) if return_com else figures


def pop_viewprops_dialog_spec(EEG: dict[str, Any], typecomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like ``pop_viewprops`` prompt."""
    is_channel = int(bool(typecomp))
    limit = int(EEG.get("nbchan", 0) or 0) if is_channel else _component_count(EEG)
    label = "Channel indices to plot:" if is_channel else "Component indices to plot:"
    title = "View many chan or comp. properties -- pop_viewprops"
    return DialogSpec(
        title=title,
        function_name="pop_viewprops",
        eeglab_source="plugins/ICLabel/viewprops/pop_viewprops.m",
        size=(600, 288),
        geometry=((1.3, 1), (1.3, 1), (1.3, 1), (1.3, 0.3)),
        controls=(
            ControlSpec("text", label),
            ControlSpec("edit", tag="chanorcomp", value=f"1:{limit}"),
            ControlSpec("text", "Spectral options (see spectopo() help):"),
            ControlSpec("edit", tag="spec_opt", value=f"'freqrange', [2 {min(80, float(EEG.get('srate', 1)) / 2):g}]"),
            ControlSpec("text", "Erpimage options (see erpimage() help):"),
            ControlSpec("edit", tag="erp_opt", value=""),
            ControlSpec("text", f"Draw events over scrolling {'channel' if is_channel else 'component'} activity"),
            ControlSpec("checkbox", tag="scroll_event", value=True, enabled=False),
        ),
        known_differences=("EEGPrep omits EEGBrowser/eegplot scrolling activity from the property overview.",),
    )


def _plot_props(EEG: dict[str, Any], typecomp: int, indices: list[int], classifier_name: str) -> list[Any]:
    if not indices:
        return []
    if not typecomp:
        return pop_topoplot(
            EEG, 0, indices[:PLOTS_PER_FIGURE], "View components properties - pop_viewprops()", [], 0, gui=False
        )
    fig_obj, axes = plt.subplots(1, min(len(indices), PLOTS_PER_FIGURE), squeeze=False)
    labels = _channel_labels(EEG)
    for axis, index in zip(axes.ravel(), indices[:PLOTS_PER_FIGURE]):
        axis.axis("off")
        label = labels[index - 1] if index - 1 < len(labels) else str(index)
        axis.text(0.5, 0.5, label, ha="center", va="center", fontsize=10)
        axis.set_title(str(index))
    fig_obj.suptitle("View channels properties - pop_viewprops()")
    fig_obj.tight_layout()
    return [fig_obj]


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


def _component_count(EEG: dict[str, Any]) -> int:
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0


def _channel_labels(EEG: dict[str, Any]) -> list[str]:
    labels = []
    for index, chanloc in enumerate(EEG.get("chanlocs", []) or []):
        labels.append(str(chanloc.get("labels", index + 1)) if isinstance(chanloc, dict) else str(index + 1))
    return labels or [str(index + 1) for index in range(int(EEG.get("nbchan", 0) or 0))]
