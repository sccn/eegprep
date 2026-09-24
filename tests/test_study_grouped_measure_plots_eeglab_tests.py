"""Ports of the maintained grouped STUDY measure-plot wrapper methods.

Upstream suite: sccn/eeglab_tests@ff605546f3f70868916fb8d49c007472b3257b50
EEGLAB tree: sccn/eeglab@8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba
"""

from __future__ import annotations

from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

from eegprep import (
    pop_statparams,
    pop_study,
    std_erpplot,
    std_erspplot,
    std_itcplot,
    std_precomp,
    std_specplot,
    std_stat,
    std_topoplot,
)
from tests.eeglab_tests import eeglab_test


STUDYFUNC_ROOT = "unittesting_studyfunc"


def _reference(wrapper: str, test: str):
    source = f"{STUDYFUNC_ROOT}/{wrapper}/studyfunc_{wrapper}_wrapperTest.m"
    return eeglab_test(source, test)


def _factorial_study(*, n_channels: int = 6, n_components: int = 2) -> tuple[dict, list[dict]]:
    datasets = []
    srate = 64.0
    pnts = 64
    trials = 4
    seconds = np.arange(pnts, dtype=float) / srate
    for group_index, group in enumerate(("control", "patient")):
        for subject_index in range(2):
            subject = f"{group[0].upper()}{subject_index + 1:02d}"
            subject_shift = 0.025 * (subject_index + 1)
            for condition_index, condition in enumerate(("standard", "target")):
                condition_shift = condition_index * (0.35 + 0.04 * subject_index)
                amplitude = 1.0 + 0.2 * group_index + 0.1 * condition_index + subject_shift
                data = np.empty((n_channels, pnts, trials), dtype=float)
                activations = np.empty((n_components, pnts, trials), dtype=float)
                for trial in range(trials):
                    phase = trial * np.pi / 12
                    for channel in range(n_channels):
                        data[channel, :, trial] = (
                            amplitude * np.sin(2 * np.pi * (6 + channel) * seconds + phase)
                            + condition_shift
                            + 0.15 * group_index
                        )
                    for component in range(n_components):
                        activations[component, :, trial] = (amplitude + 0.1 * component) * np.sin(
                            2 * np.pi * (8 + 2 * component) * seconds + phase
                        ) + condition_shift
                mixing = np.zeros((n_channels, n_components), dtype=float)
                mixing[:n_components] = np.eye(n_components)
                if group_index == 1 and subject_index == 1 and condition_index == 1:
                    mixing[:, 0] *= -1
                weights = np.zeros((n_components, n_channels), dtype=float)
                weights[:, :n_components] = np.eye(n_components)
                chanlocs = []
                for channel in range(n_channels):
                    angle = 2 * np.pi * channel / n_channels
                    chanlocs.append(
                        {
                            "labels": f"Ch{channel + 1}",
                            "theta": float(np.degrees(angle)),
                            "radius": 0.4,
                            "X": float(np.cos(angle)),
                            "Y": float(np.sin(angle)),
                            "Z": 0.0,
                        }
                    )
                datasets.append(
                    {
                        "setname": f"{subject}_{condition}",
                        "subject": subject,
                        "condition": condition,
                        "group": group,
                        "session": 1,
                        "run": 1,
                        "data": data,
                        "nbchan": n_channels,
                        "pnts": pnts,
                        "trials": trials,
                        "srate": srate,
                        "xmin": 0.0,
                        "xmax": float(seconds[-1]),
                        "times": seconds * 1000.0,
                        "chanlocs": chanlocs,
                        "icaact": activations,
                        "icawinv": mixing,
                        "icaweights": weights,
                        "icasphere": np.eye(n_channels),
                        "icachansind": list(range(n_channels)),
                        "event": [],
                        "urevent": [],
                        "epoch": [{} for _trial in range(trials)],
                        "etc": {},
                    }
                )
    return pop_study(None, datasets, name="Deterministic 2 x 2 study")


def _component_clusters(study: dict) -> dict:
    study = deepcopy(study)
    parent = study["cluster"][0]
    dataset_ids = list(range(1, len(study["datasetinfo"]) + 1))
    study["cluster"] = [
        parent,
        {"name": "Cluster 1", "sets": [dataset_ids], "comps": [1] * len(dataset_ids), "child": []},
        {"name": "Cluster 2", "sets": [dataset_ids], "comps": [2] * len(dataset_ids), "child": []},
    ]
    parent["child"] = ["Cluster 1", "Cluster 2"]
    return study


def test_std_stat_fdr_preserves_undefined_samples_and_graded_thresholds():
    condition_a = np.asarray([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    condition_b = np.asarray([[1.0, 1.0, 1.0], [2.0, 4.0, 8.0]])

    pcond, _pgroup, _pinter = std_stat(
        [condition_a, condition_b],
        condstats="on",
        paired=["on", "off"],
        method="param",
        mcorrect="fdr",
    )
    assert np.isnan(pcond[0][0])
    assert np.isfinite(pcond[0][1])

    pvalue = float(pcond[0][1])
    masks, _group_masks, _interaction_masks = std_stat(
        [condition_a, condition_b],
        condstats="on",
        paired=["on", "off"],
        method="param",
        mcorrect="fdr",
        threshold=[pvalue / 2, min(pvalue * 2, 1.0)],
    )
    np.testing.assert_array_equal(masks[0], [0.0, 1.0])


@_reference("std_erpplot", "test_test_std_erpplot")
def test_std_erpplot_groups_design_cells_and_returns_statistics_and_masks():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, [1], erp="on", recompute="on")
    study = pop_statparams(study, condstats="on", groupstats="on", threshold=np.nan, method="param")

    result = std_erpplot(study, alleeg, channels=[1], return_stats=True, plotstderr="on")
    _study, cells, times, pgroup, pcond, pinter, figure = result

    assert [[cell.shape for cell in row] for row in cells] == [[(64, 2), (64, 2)], [(64, 2), (64, 2)]]
    expected = ttest_rel(cells[0][0], cells[1][0], axis=-1).pvalue
    np.testing.assert_allclose(pcond[0], expected)
    assert len(pgroup) == 2
    assert len(pinter) == 3
    assert len(figure.axes) == 4
    assert figure.eegprep_plot_metadata["statistics"].condmask == []
    plt.close(figure)

    mask_result = std_erpplot(study, alleeg, channels=[1], threshold=0.05, return_stats=True)
    _study, _cells, _times, _pgroup, condition_masks, _pinter, mask_figure = mask_result
    assert set(np.unique(condition_masks[0])) <= {0.0, 1.0}
    np.testing.assert_array_equal(condition_masks, mask_figure.eegprep_plot_metadata["statistics"].condmask)
    plt.close(mask_figure)

    _study, _cells, _times, together = std_erpplot(
        study,
        alleeg,
        channels=[1],
        condstats="off",
        groupstats="off",
        plotconditions="together",
        plotgroups="together",
        plotsubjects="on",
    )
    assert len(together.axes) == 1
    assert len(together.axes[0].lines) == 12
    plt.close(together)


def test_std_erspplot_supports_clusters_subject_panels_and_channel_topographies():
    study, alleeg = _factorial_study()
    tf_params = {"cycles": 0, "nfreqs": 5, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(study, alleeg, "channels", ersp="on", savetrials="on", erspparams=tf_params)
    study, alleeg = std_precomp(study, alleeg, "components", ersp="on", savetrials="on", erspparams=tf_params)
    study = _component_clusters(study)

    _study, cluster_cells, times, freqs, cluster_figure = std_erspplot(study, alleeg, clusters=[2, 3])
    assert cluster_cells[0][0].shape == (freqs.size, times.size, 2)
    assert len(cluster_figure.axes) >= 2
    plt.close(cluster_figure)

    _study, subject_cells, _times, _freqs, subject_figure = std_erspplot(
        study, alleeg, channels=[1], subject="C01", plotsubjects="on"
    )
    assert sum(cell.shape[-1] for row in subject_cells for cell in row) == 2
    assert len(subject_figure.axes) >= 4
    plt.close(subject_figure)

    _study, topo_cells, _times, _freqs, topo_figure = std_erspplot(
        study, alleeg, channels="channels", topofreq=8, topotime=400, caxis=[-3, 3]
    )
    assert topo_cells[0][0].shape[-2] == 6
    assert len(topo_figure.axes) == 4
    assert all(axis.images or not axis.get_visible() for axis in topo_figure.axes)
    plt.close(topo_figure)


@_reference("std_erspplot", "test_test_std_erspplot2_2")
def test_std_erspplot_channel_saved_trials_reproduce_the_cached_ersp():
    study, alleeg = _factorial_study()
    params = {"cycles": 0, "nfreqs": 5, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(study, alleeg, [1], ersp="on", savetrials="on", recompute="on", erspparams=params)
    cache = study["changrp"][0]

    for dataset_index, trials in enumerate(cache["erspdatatrials"]):
        reconstructed = 10 * np.log10(np.mean(np.asarray(trials), axis=-1))
        np.testing.assert_allclose(reconstructed, np.asarray(cache["erspdata"])[dataset_index], atol=1e-12)
    _study, cells, times, freqs, figure = std_erspplot(study, alleeg, channels=[1])
    assert cells[0][0].shape == (freqs.size, times.size, 2)
    assert cache["measureinfo"]["trial_cache"]["erspdatatrials"] == "linear baseline-corrected power"
    plt.close(figure)


@_reference("std_erspplot", "test_test_std_erspplot3_2")
def test_std_erspplot_component_saved_trials_reproduce_the_cached_ersp():
    study, alleeg = _factorial_study()
    for info in study["datasetinfo"]:
        info["comps"] = [2]
    params = {"cycles": 0, "nfreqs": 5, "timesout": 5, "baseline": np.nan}
    study, alleeg = std_precomp(
        study, alleeg, "components", ersp="on", savetrials="on", recompute="on", erspparams=params
    )
    cache = study["cluster"][0]

    for dataset_index, component_trials in enumerate(cache["erspdatatrials"]):
        reconstructed = 10 * np.log10(np.mean(np.asarray(component_trials[0]), axis=-1))
        np.testing.assert_allclose(reconstructed, np.asarray(cache["erspdata"])[dataset_index, 0], atol=1e-12)
    _study, cells, times, freqs, figure = std_erspplot(study, alleeg, clusters=1, components=[2])
    assert cells[0][0].shape == (freqs.size, times.size, 2)
    plt.close(figure)


def test_std_itcplot_supports_centroids_component_panels_channels_and_subjects():
    study, alleeg = _factorial_study()
    tf_params = {"cycles": 0, "nfreqs": 4, "timesout": 4, "baseline": np.nan}
    study, alleeg = std_precomp(study, alleeg, [1], itc="on", savetrials="on", erspparams=tf_params)
    channel_cache = study["changrp"][0]
    for dataset_index, phases in enumerate(channel_cache["itcdatatrials"]):
        reconstructed = np.abs(np.mean(np.exp(1j * np.asarray(phases)), axis=-1))
        np.testing.assert_allclose(reconstructed, np.asarray(channel_cache["itcdata"])[dataset_index])
    study, alleeg = std_precomp(study, alleeg, "components", itc="on", erspparams=tf_params)
    study = _component_clusters(study)

    _study, cells, times, freqs, centroid = std_itcplot(study, alleeg, clusters=2, mode="centroid")
    assert cells[0][0].shape == (freqs.size, times.size, 2)
    assert np.nanmin(cells[0][0]) >= 0
    plt.close(centroid)

    _study, _cells, _times, _freqs, components = std_itcplot(study, alleeg, clusters=2, mode="comps")
    assert len(components.axes) >= 8
    plt.close(components)

    _study, channel_cells, _times, _freqs, channel_figure = std_itcplot(
        study, alleeg, channels=[1], subject="P01", plotsubjects="on"
    )
    assert sum(cell.shape[-1] for row in channel_cells for cell in row) == 2
    plt.close(channel_figure)


def test_std_specplot_supports_clusters_fdr_subject_traces_and_channel_topography():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, "channels", spec="on", recompute="on")
    study, alleeg = std_precomp(study, alleeg, "components", spec="on", recompute="on")
    study = _component_clusters(study)

    result = std_specplot(
        study,
        alleeg,
        clusters=2,
        condstats="on",
        plotconditions="together",
        threshold=0.05,
        mcorrect="fdr",
        return_stats=True,
    )
    _study, cells, frequencies, _pgroup, pcond, _pinter, figure = result
    assert cells[0][0].shape == (frequencies.size, 2)
    assert len(pcond) == 2
    assert figure.eegprep_plot_metadata["statistics"].mcorrect == "fdr"
    plt.close(figure)

    _study, _cells, _frequencies, subject_figure = std_specplot(
        study, alleeg, channels=[1], subject="C01", plotsubjects="on", plotconditions="together"
    )
    assert sum(len(axis.lines) for axis in subject_figure.axes) >= 4
    plt.close(subject_figure)

    _study, topo_cells, _frequencies, topo_figure = std_specplot(study, alleeg, channels="channels", topofreq=8)
    assert topo_cells[0][0].shape[-2] == 6
    assert len(topo_figure.axes) == 4
    plt.close(topo_figure)


@_reference("std_specplot", "test_test_std_specplot2")
def test_std_specplot_group_and_condition_layout_controls_preserve_design_cells():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, [1], spec="on", recompute="on")

    for plotconditions, plotgroups, expected_axes in (
        ("apart", "apart", 4),
        ("together", "apart", 2),
        ("apart", "together", 2),
        ("together", "together", 1),
    ):
        _study, cells, frequencies, figure = std_specplot(
            study,
            alleeg,
            channels=[1],
            plotconditions=plotconditions,
            plotgroups=plotgroups,
            plotsubjects="on",
        )
        assert [[cell.shape for cell in row] for row in cells] == [
            [(frequencies.size, 2), (frequencies.size, 2)],
            [(frequencies.size, 2), (frequencies.size, 2)],
        ]
        assert len(figure.axes) == expected_axes
        plt.close(figure)


@_reference("std_topoplot", "test_test_std_topoplot")
def test_reference_std_topoplot(eeglab_backend, eeglab_sample_study):
    study, alleeg = eeglab_sample_study
    for options in (
        {"clusters": "all", "mode": "centroid"},
        {"clusters": 3.0, "mode": "centroid"},
        {"clusters": 3.0, "mode": "comps"},
        {"clusters": 3.0, "comps": 4.0},
    ):
        eeglab_backend("std_topoplot", study, alleeg, **options, nargout=0)
        eeglab_backend("close", nargout=0)


def test_std_topoplot_draws_all_centroids_component_maps_and_selected_members():
    study, alleeg = _factorial_study()
    study, alleeg = std_precomp(study, alleeg, "components", erp="on", scalp="on", recompute="on")
    study = _component_clusters(study)

    study, all_figure = std_topoplot(study, alleeg, clusters="all", mode="centroid")
    assert len(all_figure.axes) == 2
    assert all(cluster.get("topo") for cluster in study["cluster"][1:])
    plt.close(all_figure)

    study, component_figure = std_topoplot(study, alleeg, clusters=2, mode="comps")
    assert len(component_figure.axes) == len(study["cluster"][1]["comps"]) + 1
    assert set(study["cluster"][1]["topopol"]) <= {-1, 1}
    plt.close(component_figure)

    _study, selected_figure = std_topoplot(study, alleeg, clusters=2, components=[4], mode="comps")
    assert len(selected_figure.axes) == 2
    assert selected_figure.axes[1].get_title().endswith("/IC1")
    plt.close(selected_figure)
