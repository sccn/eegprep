"""Edit and plot STUDY component clusters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.studyfunc._cluster_utils import checked_study_and_datasets, cluster_command, cluster_summary
from eegprep.functions.studyfunc.std_clustplot import std_clustplot
from eegprep.functions.studyfunc.std_mergeclust import std_mergeclust
from eegprep.functions.studyfunc.std_movecomp import std_movecomp
from eegprep.functions.studyfunc.std_moveoutlier import std_moveoutlier
from eegprep.functions.studyfunc.std_rejectoutliers import std_rejectoutliers
from eegprep.functions.studyfunc.std_renameclust import std_renameclust


ACTIONS = ("plot", "rename", "merge", "moveoutlier", "movecomp", "rejectoutliers")


def pop_clustedit(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    clusters: Any = None,
    *,
    action: str = "plot",
    cluster: int | None = None,
    name: str = "",
    to_cluster: int | None = None,
    comps: Any = None,
    threshold: float = 3.0,
    gui: bool = False,
    renderer: Any | None = None,
    return_com: bool = False,
) -> Any:
    """Run cluster edit and plotting actions on ``STUDY.cluster``."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    if gui:
        result = inputgui(pop_clustedit_dialog_spec(study), renderer=renderer)
        if result is None:
            return (study, "", None) if return_com else study
        action = _action_from_gui(result.get("action"))
        cluster = _optional_int(result.get("cluster"))
        to_cluster = _optional_int(result.get("to_cluster"))
        comps = numeric_vector(result.get("comps", []), dtype=int).tolist()
        clusters = numeric_vector(result.get("clusters", []), dtype=int).tolist()
        name = str(result.get("name") or name)
        threshold = float(numeric_vector(result.get("threshold", threshold), dtype=float)[0])

    figure = None
    command_kwargs = {"action": action}
    if action == "plot":
        study, _plot_command, figure = std_clustplot(study, datasets, clusters=clusters, return_com=True)
        command_kwargs["clusters"] = clusters
    elif action == "rename":
        if cluster is None:
            raise ValueError("rename requires a cluster index")
        study = std_renameclust(study, datasets, cluster, name)
        command_kwargs.update({"cluster": cluster, "name": name})
    elif action == "merge":
        merge_indices = clusters if clusters else []
        study = std_mergeclust(study, datasets, merge_indices, name or "Cls")
        command_kwargs.update({"clusters": merge_indices, "name": name or "Cls"})
    elif action == "moveoutlier":
        if cluster is None:
            raise ValueError("moveoutlier requires a cluster index")
        study = std_moveoutlier(study, datasets, cluster, comps or [])
        command_kwargs.update({"cluster": cluster, "comps": comps or []})
    elif action == "movecomp":
        if cluster is None or to_cluster is None:
            raise ValueError("movecomp requires source and target clusters")
        study = std_movecomp(study, datasets, cluster, to_cluster, comps or [])
        command_kwargs.update({"cluster": cluster, "to_cluster": to_cluster, "comps": comps or []})
    elif action == "rejectoutliers":
        study = std_rejectoutliers(study, datasets, clusters or "all", threshold)
        command_kwargs.update({"clusters": clusters or "all", "threshold": threshold})
    else:
        raise ValueError(f"Unknown pop_clustedit action: {action}")

    command = cluster_command("pop_clustedit", ("STUDY",), "STUDY", "ALLEEG", **command_kwargs)
    return (study, command, figure) if return_com else study


def pop_clustedit_dialog_spec(STUDY: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like cluster editing dialog spec."""
    return DialogSpec(
        title="Edit/plot component clusters -- pop_clustedit()",
        controls=(
            ControlSpec("text", "Select cluster to plot or edit", font_weight="bold"),
            ControlSpec("text", "Clusters: " + "; ".join(cluster_summary(STUDY))),
            ControlSpec("text", "Action:"),
            ControlSpec(
                "popupmenu",
                "Plot Cluster properties|Rename selected cluster|Merge clusters|Remove selected outlier component(s)|Reassign selected component(s)|Reject outlier components",
                tag="action",
                value=1,
            ),
            ControlSpec("text", "Cluster:"),
            ControlSpec("edit", tag="cluster", value="2"),
            ControlSpec("text", "Clusters to plot/merge/reject:"),
            ControlSpec("edit", tag="clusters", value=""),
            ControlSpec("text", "Component positions:"),
            ControlSpec("edit", tag="comps", value=""),
            ControlSpec("text", "Target cluster:"),
            ControlSpec("edit", tag="to_cluster", value=""),
            ControlSpec("text", "New cluster name:"),
            ControlSpec("edit", tag="name", value=""),
            ControlSpec("text", "Outlier threshold:"),
            ControlSpec("edit", tag="threshold", value="3"),
        ),
        geometry=((1,), (1,), (1, 1.2), (1, 0.7), (1, 0.7), (1, 0.7), (1, 0.7), (1, 0.7)),
        function_name="pop_clustedit",
        eeglab_source="functions/studyfunc/pop_clustedit.m",
        help_text="pop_clustedit",
        size=(720, 420),
    )


def _action_from_gui(value: Any) -> str:
    try:
        index = int(value) - 1
    except (TypeError, ValueError):
        index = 0
    return ACTIONS[index] if 0 <= index < len(ACTIONS) else "plot"


def _optional_int(value: Any) -> int | None:
    values = numeric_vector(value, dtype=int)
    return int(values[0]) if values.size else None


__all__ = ["pop_clustedit", "pop_clustedit_dialog_spec"]
