"""Edit and plot STUDY component clusters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_command,
    cluster_list,
    sets_array,
)
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
        cluster = _optional_int(result.get("cluster")) or _cluster_from_list_selection(study, result.get("clus_list"))
        to_cluster = _optional_int(result.get("to_cluster"))
        comps = numeric_vector(result.get("comps", []), dtype=int).tolist() or _components_from_list_selection(
            result.get("clust_comp")
        )
        clusters = numeric_vector(result.get("clusters", []), dtype=int).tolist() or _clusters_from_list_selection(
            study, result.get("clus_list")
        )
        name = str(result.get("name") or name)
        threshold = _optional_float(result.get("threshold"), threshold)

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
    design_names = _design_names(STUDY)
    current_design = int(STUDY.get("currentdesign") or 1)
    show_options = _cluster_list_options(STUDY)
    component_options = _component_list_options(STUDY)
    controls = [
        ControlSpec("text", "Select design:", font_weight="bold"),
        ControlSpec("popupmenu", "|".join(design_names), tag="design", value=current_design, font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("text", "Select cluster to plot", font_weight="bold"),
        ControlSpec("spacer"),
        ControlSpec("text", "Select component to plot        ", font_weight="bold"),
        ControlSpec("listbox", "|".join(show_options), tag="clus_list", value=1),
        _clustedit_button("STATS", "plot", "stats"),
        ControlSpec("listbox", "|".join(component_options), tag="clust_comp", value=[1]),
        _clustedit_button("Plot scalp maps", "plot", "plot_clus_maps", enabled=False),
        ControlSpec("spacer"),
        _clustedit_button("Plot scalp map(s)", "plot", "plot_comp_maps", enabled=False),
        _clustedit_button("Plot dipoles", "plot", "plot_clus_dip", enabled=False),
        _clustedit_button("Params", "plot", "dip_params", enabled=False),
        _clustedit_button("Plot dipole(s)", "plot", "plot_comp_dip", enabled=False),
        _clustedit_button("Plot ERPs", "plot", "plot_clus_erp", enabled=False),
        _clustedit_button("Params", "plot", "erp_params", enabled=False),
        _clustedit_button("Plot ERP(s)", "plot", "plot_comp_erp", enabled=False),
        _clustedit_button("Plot spectra", "plot", "plot_clus_spectra", enabled=False),
        _clustedit_button("Params", "plot", "spec_params", enabled=False),
        _clustedit_button("Plot spectra", "plot", "plot_comp_spectra", enabled=False),
        _clustedit_button("Plot ERPimage", "plot", "plot_clus_erpimage", enabled=False),
        _clustedit_button("Params", "plot", "erpimage_params", enabled=False),
        _clustedit_button("Plot ERPimage(s)", "plot", "plot_comp_erpimage", enabled=False),
        _clustedit_button("Plot ERSPs", "plot", "plot_clus_ersp", enabled=False),
        _clustedit_button("Params", "plot", "ersp_params", enabled=False),
        _clustedit_button("Plot ERSP(s)", "plot", "plot_comp_ersp", enabled=False),
        _clustedit_button("Plot ITCs", "plot", "plot_clus_itc", enabled=False),
        ControlSpec("spacer"),
        _clustedit_button("Plot ITC(s)", "plot", "plot_comp_itc", enabled=False),
        ControlSpec("spacer", tag="action"),
        _clustedit_button("Create new cluster", "plot", "create_cluster", enabled=False),
        ControlSpec("spacer"),
        _clustedit_button("Reassign selected component(s)", "movecomp", "move_comp", enabled=False),
        _clustedit_button("Rename selected cluster", "rename", "rename_cluster", enabled=False),
        ControlSpec("spacer"),
        _clustedit_button("Remove selected outlier comps.", "moveoutlier", "move_outlier"),
    ]
    return DialogSpec(
        title="View and edit current component clusters -- pop_clustedit()",
        controls=tuple(controls),
        geometry=(
            (0.8, 3),
            (1,),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1, 0.35, 1),
            (1,),
            (1, 0.35, 1),
            (1, 0.35, 1),
        ),
        function_name="pop_clustedit",
        eeglab_source="functions/studyfunc/pop_clustedit.m",
        help_text="pop_clustedit",
        size=(770, 672),
        content_margins=(48, 24, 48, 14),
        row_spacing=6,
        geomvert=(1, 0.5, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        button_size=(58, 20),
        extra_stylesheet="""
            QDialog#pop_clustedit QListWidget {
                min-height: 126px;
                max-height: 126px;
            }
            QDialog#pop_clustedit QPushButton {
                min-width: 170px;
                max-width: 360px;
                padding: 0 8px;
            }
            QDialog#pop_clustedit QPushButton#stats,
            QDialog#pop_clustedit QPushButton#dip_params,
            QDialog#pop_clustedit QPushButton#erp_params,
            QDialog#pop_clustedit QPushButton#spec_params,
            QDialog#pop_clustedit QPushButton#erpimage_params,
            QDialog#pop_clustedit QPushButton#ersp_params {
                min-width: 82px;
                max-width: 82px;
            }
            QDialog#pop_clustedit QPushButton#stats {
                min-height: 94px;
                max-height: 126px;
            }
        """,
    )


def _action_from_gui(value: Any) -> str:
    if isinstance(value, str) and value in ACTIONS:
        return value
    try:
        index = int(value) - 1
    except (TypeError, ValueError):
        index = 0
    return ACTIONS[index] if 0 <= index < len(ACTIONS) else "plot"


def _optional_int(value: Any) -> int | None:
    values = numeric_vector(value, dtype=int)
    return int(values[0]) if values.size else None


def _optional_float(value: Any, default: float) -> float:
    values = numeric_vector(value, dtype=float)
    return float(values[0]) if values.size else float(default)


def _clustedit_button(label: str, action: str, tag: str, *, enabled: bool = True) -> ControlSpec:
    return ControlSpec(
        "pushbutton",
        label,
        tag=tag,
        enabled=enabled,
        callback=CallbackSpec("set_value", {"source": tag, "target": "action", "value": action}),
    )


def _design_names(study: dict[str, Any]) -> list[str]:
    designs = study.get("design") or []
    if isinstance(designs, dict):
        designs = [designs]
    names = [str(design.get("name") or f"STUDY.design {index}") for index, design in enumerate(designs, start=1)]
    return names or ["STUDY.design 1"]


def _cluster_list_options(study: dict[str, Any]) -> list[str]:
    options = ["All cluster centroids"]
    for cluster in cluster_list(study):
        name = str(cluster.get("name") or "Cluster")
        count = len(cluster.get("comps") or [])
        indent = "   " if cluster.get("parent") else ""
        options.append(f"{indent}{name} ({count} ICs)")
    return options


def _component_list_options(study: dict[str, Any]) -> list[str]:
    options = []
    clusters = cluster_list(study)
    datasetinfo = study.get("datasetinfo") or []
    for cluster in clusters[1:] or clusters:
        name = str(cluster.get("name") or "Cluster")
        sets = sets_array(cluster.get("sets"))
        comps = list(cluster.get("comps") or [])
        for row, comp in enumerate(comps, start=1):
            study_set = int(sets[0, row - 1]) if sets.size and row - 1 < sets.shape[1] else 1
            subject = _subject_label(datasetinfo, study_set)
            options.append(f"'{name}' comp. {row} ({subject} IC{int(comp)})")
    return options or ["No components"]


def _subject_label(datasetinfo: Any, study_set: int) -> str:
    if (
        isinstance(datasetinfo, list)
        and 1 <= study_set <= len(datasetinfo)
        and isinstance(datasetinfo[study_set - 1], dict)
    ):
        return str(datasetinfo[study_set - 1].get("subject") or f"S{study_set:02d}")
    return f"S{study_set:02d}"


def _clusters_from_list_selection(study: dict[str, Any], selection: Any) -> list[int]:
    indices = numeric_vector(selection, dtype=int)
    if indices.size == 0 or bool((indices == 1).any()):
        return []
    clusters = list(range(1, len(cluster_list(study)) + 1))
    return [clusters[index - 2] for index in indices.tolist() if 2 <= index <= len(clusters) + 1]


def _cluster_from_list_selection(study: dict[str, Any], selection: Any) -> int | None:
    clusters = _clusters_from_list_selection(study, selection)
    return clusters[0] if clusters else None


def _components_from_list_selection(selection: Any) -> list[int]:
    return numeric_vector(selection, dtype=int).tolist()


__all__ = ["pop_clustedit", "pop_clustedit_dialog_spec"]
