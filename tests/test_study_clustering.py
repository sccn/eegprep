from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.studyfunc.pop_clust import pop_clust, pop_clust_dialog_spec
from eegprep.functions.studyfunc.pop_clustedit import pop_clustedit, pop_clustedit_dialog_spec
from eegprep.functions.studyfunc.pop_preclust import pop_preclust, pop_preclust_dialog_spec
from eegprep.functions.studyfunc.pop_precomp import pop_precomp
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.optimal_kmeans import optimal_kmeans
from eegprep.functions.studyfunc.robust_kmeans import robust_kmeans
from eegprep.functions.studyfunc.std_apcluster import std_apcluster
from eegprep.functions.studyfunc.std_centroid import std_centroid
from eegprep.functions.studyfunc.std_createclust import std_createclust
from eegprep.functions.studyfunc.std_findoutlierclust import std_findoutlierclust
from eegprep.functions.studyfunc.std_mergeclust import std_mergeclust
from eegprep.functions.studyfunc.std_movecomp import std_movecomp
from eegprep.functions.studyfunc.std_moveoutlier import std_moveoutlier
from eegprep.functions.studyfunc.std_preclust import normalize_preclust_specs, std_preclust
from eegprep.functions.studyfunc.std_rejectoutliers import std_rejectoutliers


class _Renderer:
    def __init__(self, result):
        self.result = result

    def run(self, spec, initial_values=None):
        self.spec = spec
        return self.result


def _ica_eeg(setname: str, offset: float = 0.0) -> dict:
    data = np.arange(40, dtype=float).reshape(4, 10) + offset
    return {
        "setname": setname,
        "subject": setname.upper(),
        "data": data,
        "nbchan": 4,
        "pnts": 10,
        "trials": 1,
        "srate": 100.0,
        "xmin": 0.0,
        "xmax": 0.09,
        "chanlocs": [{"labels": label} for label in ("Fz", "Cz", "Pz", "Oz")],
        "icaweights": np.eye(3, 4),
        "icasphere": np.eye(4),
        "icawinv": np.array(
            [
                [1.0 + offset, 0.1, 0.2],
                [0.2, 1.2 + offset, 0.3],
                [0.1, 0.4, 1.4 + offset],
                [0.0, 0.2, 0.5],
            ]
        ),
        "icachansind": [0, 1, 2, 3],
        "event": [],
        "urevent": [],
        "epoch": [],
    }


def _study_with_ica() -> tuple[dict, list[dict]]:
    return pop_study(None, [_ica_eeg("s1"), _ica_eeg("s2", 4.0)], name="Cluster study")


def _preclustered_study() -> tuple[dict, list[dict]]:
    study, alleeg = _study_with_ica()
    return pop_preclust(
        study,
        alleeg,
        preproc=[{"measure": "scalp", "npca": 2, "norm": 1, "weight": 1}],
    )


def test_std_preclust_builds_parent_component_matrix_from_scalp_maps():
    study, alleeg = _study_with_ica()

    study, alleeg, command = std_preclust(
        study,
        alleeg,
        1,
        [{"measure": "scalp", "npca": 2, "norm": 1, "weight": 2}],
        return_com=True,
    )

    preclust = study["etc"]["preclust"]
    assert np.asarray(preclust["preclustdata"]).shape == (6, 2)
    assert preclust["clustlevel"] == 1
    assert len(study["cluster"][0]["comps"]) == 6
    assert command.startswith("STUDY, ALLEEG = std_preclust(")


def test_std_preclust_reads_phase_5b_component_measure_cache():
    study, alleeg = _study_with_ica()
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on")

    study, _alleeg = std_preclust(
        study,
        alleeg,
        1,
        [{"measure": "erp", "npca": 2, "timewindow": [0, 90]}],
    )

    assert np.asarray(study["etc"]["preclust"]["preclustdata"]).shape == (6, 2)
    assert study["etc"]["eegprep"]["preclust_contract"]["component_measure_root"] == "STUDY.cluster[0]"
    assert study["cluster"][0]["sets"] == [[1, 1, 1, 2, 2, 2]]


def test_preclust_missing_component_measure_and_missing_ica_errors():
    study, alleeg = _study_with_ica()
    with pytest.raises(ValueError, match="component measure 'erp' is missing"):
        std_preclust(study, alleeg, 1, [{"measure": "erp"}])

    no_ica = dict(_ica_eeg("bad"))
    no_ica.pop("icaweights")
    no_ica.pop("icawinv")
    study, alleeg = pop_study(None, [no_ica], name="No ICA")
    with pytest.raises(ValueError, match="no ICA components"):
        std_preclust(study, alleeg)


def test_pop_clust_creates_deterministic_child_clusters():
    study, alleeg = _preclustered_study()

    clustered, command = pop_clust(study, alleeg, clus_num=2, random_state=11, return_com=True)

    assert command.startswith("STUDY = pop_clust(")
    assert len(clustered["cluster"]) == 3
    assert sum(len(cluster["comps"]) for cluster in clustered["cluster"][1:]) == 6
    assert clustered["cluster"][0]["child"] == [cluster["name"] for cluster in clustered["cluster"][1:]]


def test_deterministic_clustering_helpers_return_stable_shapes():
    data = np.asarray([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [20.0, 20.0]])

    optimal_labels, optimal_centers, optimal_sumd, optimal_distances = optimal_kmeans(data[:4], [2, 3], random_state=7)
    robust_labels, robust_centers, robust_sumd, robust_distances, outliers = robust_kmeans(
        data,
        2,
        STD=1.1,
        MAXiter=3,
        random_state=7,
    )
    ap_labels, ap_centers, ap_sumd = std_apcluster(data[:4], maxits=50, convits=5, dampfact=0.7)
    centroids = std_centroid(data[:4], optimal_labels)
    outlier_rows = std_findoutlierclust(data, [1, 1, 1, 1, 1], threshold=1.0)

    # The two tight pairs must each share a label and the pairs must differ,
    # so a regression that swaps or misindexes cluster assignments is caught.
    assert set(optimal_labels.tolist()) == {1, 2}
    assert optimal_labels[0] == optimal_labels[1]
    assert optimal_labels[2] == optimal_labels[3]
    assert optimal_labels[0] != optimal_labels[2]
    assert optimal_centers.shape == (2, 2)
    assert optimal_sumd.shape == (2,)
    assert optimal_distances.shape == (4, 2)

    # The far row [20, 20] (row index 4) must land alone in its own cluster,
    # distinct from the four tight rows.
    assert robust_labels.shape == (5,)
    assert robust_labels[4] not in set(robust_labels[:4].tolist())
    assert int(np.sum(robust_labels == robust_labels[4])) == 1
    assert robust_centers.shape[1] == 2
    assert robust_sumd.ndim == 1
    assert robust_distances.shape[0] == 5
    assert outliers.size >= 1
    assert ap_labels.shape == (4,)
    assert ap_centers.shape[1] == 2
    assert ap_sumd.ndim == 1
    assert centroids.shape == (2, 2)
    # std_findoutlierclust must flag the far row (1-based index 5) as the outlier.
    assert outlier_rows.tolist() == [5]


def test_kmeans_kernel_is_shared_by_clustering_callers():
    # optimal_kmeans and robust_kmeans must reuse the one canonical k-means
    # kernel so the numerics cannot drift between copies. Single-cluster runs
    # let us compare labels/centers directly against the kernel output.
    from eegprep.functions.studyfunc._cluster_kmeans import kmeans_labels, squared_distances

    data = np.asarray([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    kernel_labels, kernel_centers = kmeans_labels(data, 2, random_state=7)

    optimal_labels, optimal_centers, _sumd, optimal_distances = optimal_kmeans(data, 2, random_state=7)
    np.testing.assert_array_equal(optimal_labels, kernel_labels)
    np.testing.assert_allclose(optimal_centers, kernel_centers)
    np.testing.assert_allclose(optimal_distances, np.sqrt(squared_distances(data, kernel_centers)))

    robust_labels, robust_centers, _rsumd, _rdist, _outliers = robust_kmeans(
        data, 2, STD=float("inf"), MAXiter=1, random_state=7
    )
    np.testing.assert_array_equal(robust_labels, kernel_labels)
    np.testing.assert_allclose(robust_centers, kernel_centers)


def test_pop_clust_rejects_invalid_cluster_counts_and_outlier_thresholds():
    study, alleeg = _preclustered_study()

    with pytest.raises(ValueError, match="at least 2"):
        pop_clust(study, alleeg, clus_num=1)

    with pytest.raises(ValueError, match="greater than 0"):
        pop_clust(study, alleeg, clus_num=2, outliers=-1)


def test_preclust_spec_command_precedence_is_explicit():
    command_spec = normalize_preclust_specs([{"measure": "erp", "command": "spec"}])[0]
    measure_spec = normalize_preclust_specs([{"measure": "erp", "command": ""}])[0]

    assert command_spec["measure"] == "spec"
    assert measure_spec["measure"] == "erp"


def test_pop_clust_outlier_threshold_uses_mean_distance_guard():
    study, alleeg = _preclustered_study()
    study["etc"]["preclust"]["preclustdata"] = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]

    clustered = pop_clust(study, alleeg, clus_num=2, outliers=2, random_state=11)

    assert not any(str(cluster["name"]).startswith("outlier") for cluster in clustered["cluster"])


def test_std_createclust_numbers_outliers_separately_from_clusters():
    study, alleeg = _preclustered_study()

    clustered = std_createclust(study, alleeg, clusterind=[0, 1, 1, 2, 2, 2])

    assert [cluster["name"] for cluster in clustered["cluster"][1:]] == ["outlier 1", "Cls 1", "Cls 2"]


def test_cluster_edit_rename_merge_moveoutlier_and_plot():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)

    renamed, command, _figure = pop_clustedit(
        study,
        alleeg,
        action="rename",
        cluster=2,
        name="Alpha",
        return_com=True,
    )
    assert renamed["cluster"][1]["name"].startswith("Alpha")
    assert "action='rename'" in command

    merged = std_mergeclust(renamed, alleeg, [2, 3], "Merged")
    assert merged["cluster"][-1]["name"].startswith("Merged")
    assert set(merged["cluster"][-1]["parent"]) == {renamed["cluster"][1]["name"], renamed["cluster"][2]["name"]}

    moved = std_moveoutlier(renamed, alleeg, 2, [1])
    assert moved["cluster"][1]["comps"] != renamed["cluster"][1]["comps"]
    assert moved["cluster"][-1]["name"].startswith("Outliers")
    assert np.asarray(moved["cluster"][-1]["preclust"]["preclustdata"]).shape[0] == 1

    plotted, _command, figure = pop_clustedit(renamed, alleeg, action="plot", clusters=[2, 3], return_com=True)
    assert plotted["cluster"][1]["name"].startswith("Alpha")
    assert figure is not None
    plt.close(figure)


def test_cluster_gui_all_selection_expands_to_all_clusters():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)
    renderer = _Renderer({"action": "plot", "clus_list": [1, 3]})

    _plotted, command, figure = pop_clustedit(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert "clusters=[1, 2, 3]" in command
    plt.close(figure)


def test_moveoutlier_reuses_outlier_cluster_after_source_rename():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)
    source = 2 if len(study["cluster"][1]["comps"]) > 1 else 3

    moved = std_moveoutlier(study, alleeg, source, [1])
    renamed, _command, _figure = pop_clustedit(
        moved,
        alleeg,
        action="rename",
        cluster=source,
        name="Alpha",
        return_com=True,
    )
    moved_again = std_moveoutlier(renamed, alleeg, source, [1])

    outliers = [cluster for cluster in moved_again["cluster"] if str(cluster["name"]).startswith("Outliers")]
    assert len(outliers) == 1
    assert "Alpha" in outliers[0]["name"]


def test_movecomp_does_not_rewrite_destination_parent():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)
    study["cluster"].append(
        {
            "name": "Manual 99",
            "sets": [],
            "comps": [],
            "parent": [],
            "child": [],
            "preclust": {"preclustparams": [], "preclustdata": []},
        }
    )

    moved = std_movecomp(study, alleeg, 2, 4, [1])

    assert moved["cluster"][3]["parent"] == []


def test_std_rejectoutliers_uses_euclidean_distance_threshold():
    study, alleeg = _study_with_ica()
    study["cluster"] = [
        {
            "name": "ParentCluster",
            "sets": [[1, 1, 1]],
            "comps": [1, 2, 3],
            "parent": [],
            "child": ["Cls 1"],
            "preclust": {"preclustparams": [], "preclustdata": []},
        },
        {
            "name": "Cls 1",
            "sets": [[1, 1, 1]],
            "comps": [1, 2, 3],
            "parent": ["ParentCluster"],
            "child": [],
            "preclust": {
                "preclustparams": [{"measure": "fixture"}],
                "preclustdata": [[0.0], [1.0], [4.0]],
            },
        },
    ]

    rejected = std_rejectoutliers(study, alleeg, clusters=[2], th=2.7)

    assert rejected["cluster"][1]["comps"] == [1, 2]
    assert rejected["cluster"][-1]["comps"] == [3]


def test_pop_preclust_clust_and_clustedit_history_replays():
    study, alleeg = _study_with_ica()
    study, alleeg, preclust_command = pop_preclust(
        study,
        alleeg,
        preproc=[{"measure": "scalp", "npca": 2}],
        return_com=True,
    )
    namespace = {
        "STUDY": study,
        "ALLEEG": alleeg,
        "pop_preclust": pop_preclust,
        "pop_clust": pop_clust,
        "pop_clustedit": pop_clustedit,
    }

    exec(preclust_command, namespace)
    assert namespace["STUDY"]["etc"]["preclust"]["clustlevel"] == 1

    clustered, _command = pop_clust(namespace["STUDY"], namespace["ALLEEG"], clus_num=2, return_com=True)
    namespace["STUDY"] = clustered
    _renamed, rename_command, _figure = pop_clustedit(
        clustered,
        namespace["ALLEEG"],
        action="rename",
        cluster=2,
        name="Replay",
        return_com=True,
    )
    exec(rename_command, namespace)
    assert namespace["STUDY"]["cluster"][1]["name"].startswith("Replay")


def test_console_bare_pop_clust_updates_study_workspace():
    study, alleeg = _preclustered_study()
    session = EEGPrepSession()
    session.ALLEEG = alleeg
    session.STUDY = study
    session.CURRENTSTUDY = 1
    workspace = EEGPrepConsoleWorkspace(session, exports={"pop_clust": pop_clust})

    result = workspace.namespace["pop_clust"](workspace.namespace["STUDY"], workspace.namespace["ALLEEG"], clus_num=2)

    assert session.STUDY is workspace.namespace["STUDY"]
    assert len(session.STUDY["cluster"]) == 3
    assert result.command.startswith("STUDY = pop_clust(")
    assert session.ALLCOM[-1].startswith("STUDY = pop_clust(")


def test_console_bare_preclust_clustedit_and_plot_update_study_workspace():
    study, alleeg = _study_with_ica()
    session = EEGPrepSession()
    session.ALLEEG = alleeg
    session.STUDY = study
    session.CURRENTSTUDY = 1
    workspace = EEGPrepConsoleWorkspace(
        session,
        exports={"pop_preclust": pop_preclust, "pop_clust": pop_clust, "pop_clustedit": pop_clustedit},
    )

    preclust_result = workspace.namespace["pop_preclust"](
        workspace.namespace["STUDY"],
        workspace.namespace["ALLEEG"],
        preproc=[{"measure": "scalp", "npca": 2}],
    )
    cluster_result = workspace.namespace["pop_clust"](
        workspace.namespace["STUDY"],
        workspace.namespace["ALLEEG"],
        clus_num=2,
        random_state=11,
    )
    edit_result = workspace.namespace["pop_clustedit"](
        workspace.namespace["STUDY"],
        workspace.namespace["ALLEEG"],
        action="rename",
        cluster=2,
        name="Console",
    )
    plot_result = workspace.namespace["pop_clustedit"](
        workspace.namespace["STUDY"],
        workspace.namespace["ALLEEG"],
        action="plot",
        clusters=[2, 3],
    )

    assert preclust_result.command.startswith("STUDY, ALLEEG = pop_preclust(")
    assert cluster_result.command.startswith("STUDY = pop_clust(")
    assert edit_result.command.startswith("STUDY = pop_clustedit(")
    assert plot_result.command.startswith("STUDY = pop_clustedit(")
    assert session.CURRENTSTUDY == 1
    assert session.STUDY["cluster"][1]["name"].startswith("Console")
    assert session.ALLCOM[-1] == plot_result.command
    plt.close("all")


def test_cluster_gui_specs_and_cancel_paths_are_stable():
    study, alleeg = _preclustered_study()
    preclust_spec = pop_preclust_dialog_spec(study)
    clust_spec = pop_clust_dialog_spec(study)
    edit_spec = pop_clustedit_dialog_spec(study)

    assert controls_by_tag(preclust_spec)["scalp_on"].value == 0
    assert (
        controls_by_tag(preclust_spec)["scalp_choice"].string
        == "Use channel values|Use Laplacian values|Use Gradient values"
    )
    assert "double_dip_help" in controls_by_tag(preclust_spec)
    assert controls_by_tag(clust_spec)["clus_num"].value
    assert controls_by_tag(edit_spec)["clus_list"].string.startswith("All cluster centroids|ParentCluster")
    assert controls_by_tag(edit_spec)["plot_clus_maps"].string == "Plot scalp maps"
    assert controls_by_tag(edit_spec)["move_outlier"].string == "Remove selected outlier comps."
    assert controls_by_tag(edit_spec)["create_cluster"].enabled is False
    assert controls_by_tag(edit_spec)["move_comp"].enabled is False
    assert controls_by_tag(edit_spec)["rename_cluster"].enabled is False

    assert pop_preclust(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)[2] == ""
    assert pop_clust(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)[1] == ""
    assert pop_clustedit(study, alleeg, gui=True, renderer=_Renderer(None), return_com=True)[1] == ""


def test_cluster_gui_submit_paths_accept_blank_optional_numeric_fields():
    study, alleeg = _study_with_ica()
    preclust_renderer = _Renderer(
        {"cluster_ind": "", "scalp_on": 1, "scalp_PCA": "2", "scalp_weight": "", "scalp_absolute": 1}
    )
    study, alleeg, preclust_command = pop_preclust(
        study,
        alleeg,
        gui=True,
        renderer=preclust_renderer,
        return_com=True,
    )
    clust_renderer = _Renderer({"algorithm": 1, "clus_num": "", "outliers_on": 1, "outliers": ""})

    clustered, clust_command = pop_clust(study, alleeg, gui=True, renderer=clust_renderer, return_com=True)

    assert preclust_command.startswith("STUDY, ALLEEG = pop_preclust(")
    assert clust_command.startswith("STUDY = pop_clust(")
    assert len(clustered["cluster"]) >= 3


def test_clustedit_gui_empty_threshold_falls_back_for_non_reject_actions():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)
    renderer = _Renderer(
        {
            "action": 1,
            "cluster": "",
            "clusters": "2 3",
            "comps": "",
            "to_cluster": "",
            "name": "",
            "threshold": "",
        }
    )

    _study, command, figure = pop_clustedit(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert "action='plot'" in command
    assert figure is not None
    plt.close(figure)


def test_clustedit_gui_maps_component_list_rows_to_cluster_local_positions():
    study, alleeg = _preclustered_study()
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)
    source_cluster = 3
    global_component_row = len(study["cluster"][1]["comps"]) + 1
    source_before = list(study["cluster"][source_cluster - 1]["comps"])
    renderer = _Renderer({"action": "moveoutlier", "clus_list": 1, "clust_comp": [global_component_row]})

    moved, command, _figure = pop_clustedit(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert "action='moveoutlier'" in command
    assert f"cluster={source_cluster}" in command
    assert moved["cluster"][source_cluster - 1]["comps"] == source_before[1:]
    assert moved["cluster"][-1]["comps"] == [source_before[0]]
