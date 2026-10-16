from __future__ import annotations

import ast

import numpy as np
import pytest
from matplotlib import pyplot as plt

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.sigprocfunc.kmeanscluster import kmeanscluster
from eegprep.functions.studyfunc.pop_clust import pop_clust
from eegprep.functions.studyfunc.pop_clustedit import pop_clustedit
from eegprep.functions.studyfunc.pop_preclust import pop_preclust
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
from eegprep.functions.studyfunc.std_rejectoutliers import std_rejectoutliers
from tests.eeglab_tests.backend import call_matlab


_INITIAL_CENTROID_OBSERVATIONS = np.array(
    [
        [0.0, 10.0],
        [0.0, 0.0],
        [0.2, 10.1],
        [-0.1, 9.7],
        [0.3, 0.2],
        [-0.2, 0.4],
    ]
)
_INITIAL_CENTROID_LABELS = np.array([0, 1, 0, 0, 1, 1])


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


def test_pop_clust_outlier_threshold_uses_mean_distance_guard():
    study, alleeg = _preclustered_study()
    study["etc"]["preclust"]["preclustdata"] = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]

    clustered = pop_clust(study, alleeg, clus_num=2, outliers=2, random_state=11)

    assert not any(str(cluster["name"]).startswith("outlier") for cluster in clustered["cluster"])


def test_pop_clust_finite_outliers_uses_robust_kmeans_provenance_and_labels():
    study, alleeg = _preclustered_study()
    data = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [5.0, 5.0], [5.1, 5.0], [20.0, 20.0]])
    study["etc"]["preclust"]["preclustdata"] = data.tolist()
    expected_labels, _centers, _sumd, _distances, expected_outliers = robust_kmeans(
        data,
        2,
        STD=2.5,
        MAXiter=5,
        method="kmeans",
        random_state=7,
    )

    clustered, command = pop_clust(study, alleeg, clus_num=2, outliers=2.5, random_state=7, return_com=True)

    outlier_clusters = [
        cluster for cluster in clustered["cluster"] if str(cluster.get("name") or "").startswith("outlier")
    ]
    assert len(outlier_clusters) == 1
    np.testing.assert_allclose(outlier_clusters[0]["preclust"]["preclustdata"], data[expected_outliers - 1])
    assert all(cluster.get("algorithm") == ["robust_kmeans", 2] for cluster in clustered["cluster"][1:])
    assert "outliers=2.5" in command
    ast.parse(command)

    rows_by_cluster = []
    for cluster in clustered["cluster"][1:]:
        rows_by_cluster.extend(np.asarray(cluster["preclust"]["preclustdata"], dtype=float).tolist())
    np.testing.assert_allclose(sorted(rows_by_cluster), sorted(data[expected_labels >= 0].tolist()))


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


def test_kmeanscluster_labels_follow_initial_centroid_rows():
    labels, centers, previous, _observations = kmeanscluster(_INITIAL_CENTROID_OBSERVATIONS, 2)
    order = np.lexsort(tuple(centers[:, column] for column in reversed(range(centers.shape[1]))))

    np.testing.assert_array_equal(labels, _INITIAL_CENTROID_LABELS)
    np.testing.assert_array_equal(previous, labels)
    assert centers[0, 1] > centers[1, 1]
    assert not np.array_equal(order, np.arange(order.size))


def test_pop_clust_sample_study_keeps_initial_centroid_numbering():
    eeg = pop_loadset("sample_data/eeglab_data_epochs_ica.set")
    study, alleeg = pop_study(None, [eeg], name="Sample study")
    study, alleeg = pop_preclust(
        study,
        alleeg,
        preproc=[{"measure": "scalp", "npca": 3, "norm": 1, "weight": 1, "abso": 1}],
    )

    clustered = pop_clust(study, alleeg, algorithm="kmeans", clus_num=4, random_state=0)
    repeated = pop_clust(study, alleeg, algorithm="kmeans", clus_num=4, random_state=0)

    expected = {
        "Cls 1": [4, 5, 6, 8, 9, 10, 18, 19, 26, 28, 30, 32],
        "Cls 2": [1, 2, 14, 16, 21, 22, 27, 29],
        "Cls 3": [7, 12, 13, 15, 17, 24, 25],
        "Cls 4": [3, 11, 20, 23, 31],
    }
    actual = {cluster["name"]: cluster["comps"] for cluster in clustered["cluster"][1:]}
    assert actual == expected
    assert {cluster["name"]: cluster["comps"] for cluster in repeated["cluster"][1:]} == expected


def test_matlab_kmeans_numbering_matches_fixed_initial_centroids(eeglab_matlab_engine):
    found = eeglab_matlab_engine.which("kmeans")
    found_text = found if isinstance(found, str) else ""
    if "toolbox/stats" not in found_text.replace("\\", "/").lower():
        pytest.skip(f"Statistics Toolbox kmeans is not on the path ({found_text})")

    data = _INITIAL_CENTROID_OBSERVATIONS
    labels, centers, _previous, _observations = kmeanscluster(data, 2)
    np.testing.assert_array_equal(labels, _INITIAL_CENTROID_LABELS)
    matlab_labels, matlab_centers, _matlab_previous, _matlab_data = call_matlab(
        eeglab_matlab_engine,
        "kmeanscluster",
        data,
        2.0,
        nargout=4,
    )
    stats_labels, stats_centers = call_matlab(
        eeglab_matlab_engine,
        "kmeans",
        data,
        2.0,
        "Start",
        data[:2].copy(),
        "EmptyAction",
        "error",
        "OnlinePhase",
        "off",
        "MaxIter",
        300.0,
        "Display",
        "off",
        nargout=2,
    )

    np.testing.assert_array_equal(np.asarray(matlab_labels).ravel(), labels + 1)
    np.testing.assert_allclose(np.asarray(matlab_centers, dtype=float), centers, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(stats_labels).ravel(), labels + 1)
    np.testing.assert_allclose(np.asarray(stats_centers, dtype=float), centers, rtol=0, atol=1e-12)
