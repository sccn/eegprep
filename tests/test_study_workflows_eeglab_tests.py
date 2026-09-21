"""Generated-fixture ports of current EEGLAB STUDY workflow tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pytest

from eegprep.functions.popfunc.plot_utils import component_activations
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.studyfunc.pop_clust import pop_clust
from eegprep.functions.studyfunc.pop_corrmap import pop_corrmap
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.corrmap import corrmap
from eegprep.functions.studyfunc.std_editset import std_editset
from eegprep.functions.studyfunc.std_erpplot import std_erpplot
from eegprep.functions.studyfunc.std_erspplot import std_erspplot
from eegprep.functions.studyfunc.std_makedesign import std_makedesign
from eegprep.functions.studyfunc.std_preclust import std_preclust
from eegprep.functions.studyfunc.std_precomp import std_precomp
from eegprep.functions.studyfunc.std_selectdesign import std_selectdesign
from eegprep.functions.studyfunc.std_specplot import std_specplot
from tests.eeglab_tests import eeglab_test


STUDYFUNC_ROOT = "unittesting_studyfunc"


def _reference(wrapper: str, test: str):
    source = f"{STUDYFUNC_ROOT}/{wrapper}/studyfunc_{wrapper}_wrapperTest.m"
    return eeglab_test(source, test)


def _deterministic_eeg(
    setname: str,
    subject: str,
    condition: str,
    *,
    offset: float = 0.0,
    n_channels: int = 4,
    n_components: int = 3,
) -> dict:
    srate = 64.0
    pnts = 128
    trials = 4
    seconds = np.arange(pnts, dtype=float) / srate
    data = np.empty((n_channels, pnts, trials), dtype=float)
    activations = np.empty((n_components, pnts, trials), dtype=float)
    for trial in range(trials):
        phase = trial * np.pi / 8
        for channel in range(n_channels):
            frequency = 6.0 + 2.0 * channel
            data[channel, :, trial] = np.sin(2 * np.pi * frequency * seconds + phase) + offset
        for component in range(n_components):
            frequency = 8.0 + 2.0 * component
            activations[component, :, trial] = np.sin(2 * np.pi * frequency * seconds + phase) + offset / 2
    mixing = np.zeros((n_channels, n_components), dtype=float)
    mixing[:n_components, :] = np.eye(n_components)
    weights = np.zeros((n_components, n_channels), dtype=float)
    weights[:, :n_components] = np.eye(n_components)
    chanlocs = []
    for channel in range(n_channels):
        angle = 2 * np.pi * channel / n_channels
        chanlocs.append(
            {
                "labels": f"Ch{channel + 1}",
                "theta": float(np.degrees(angle)),
                "radius": 0.35,
                "X": float(np.cos(angle)),
                "Y": float(np.sin(angle)),
                "Z": 0.0,
            }
        )
    dipoles = [
        {
            "posxyz": [float(component + 1), float((-1) ** component), float(component) / 2],
            "momxyz": [1.0, 0.0, 0.5],
            "rv": 0.05,
        }
        for component in range(n_components)
    ]
    return {
        "setname": setname,
        "subject": subject,
        "condition": condition,
        "group": "control",
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
        "event": [
            {"type": "stim", "latency": trial * pnts + 1, "epoch": trial + 1, "urevent": trial + 1}
            for trial in range(trials)
        ],
        "urevent": [{"type": "stim", "latency": trial * pnts + 1, "epoch": trial + 1} for trial in range(trials)],
        "epoch": [{"event": [trial], "eventtype": ["stim"]} for trial in range(trials)],
        "dipfit": {"model": dipoles},
        "etc": {},
    }


def _study_pair(*, n_channels: int = 4, n_components: int = 3) -> tuple[dict, list[dict]]:
    datasets = [
        _deterministic_eeg("s01_target", "S01", "target", n_channels=n_channels, n_components=n_components),
        _deterministic_eeg(
            "s02_standard",
            "S02",
            "standard",
            offset=0.2,
            n_channels=n_channels,
            n_components=n_components,
        ),
    ]
    return pop_study(None, datasets, name="Generated N400 study")


def _corrmap_study(*, scales: tuple[float, float] = (4.0, 0.2)) -> tuple[dict, list[dict]]:
    template = np.array([-2.0, -1.0, -0.2, 0.5, 1.2, 2.0])
    alternating = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    biphasic = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
    unrelated = np.array([-0.5, 0.5, 1.0, -1.0, 0.5, -0.5])
    near_template = template + np.array([0.1, -0.05, 0.03, -0.04, 0.02, -0.08])
    inverse_maps = (
        np.column_stack([template, alternating, biphasic]),
        np.column_stack([alternating, -scales[0] * template, unrelated]),
        np.column_stack([biphasic, unrelated, scales[1] * near_template]),
        np.column_stack([alternating, biphasic, unrelated]),
    )
    datasets = []
    for dataset_index, maps in enumerate(inverse_maps, start=1):
        eeg = _deterministic_eeg(
            f"corrmap_{dataset_index}",
            f"S{dataset_index:02d}",
            "target",
            n_channels=6,
            n_components=3,
        )
        eeg["icawinv"] = maps
        eeg["icaweights"] = np.linalg.pinv(maps)
        eeg["icasphere"] = np.eye(6)
        datasets.append(eeg)
    return pop_study(None, datasets, name="Generated CORRMAP study")


@_reference("pop_corrmap", "test_test_pop_corrmap")
def test_pop_corrmap_matches_polarity_builds_cluster_and_is_scale_invariant():
    study, alleeg = _corrmap_study()

    result, matched_study, matched_datasets, command = pop_corrmap(
        study,
        alleeg,
        1,
        1,
        "chanlocs",
        "",
        "th",
        "auto",
        "ics",
        1,
        "title",
        "Cluster test2",
        "clname",
        "test2",
        "badcomps",
        "yes",
        "resetclusters",
        "off",
        return_com=True,
    )

    second_pairs = dict(zip(result["output"]["sets"][1], result["output"]["ics"][1], strict=True))
    second_polarities = dict(zip(result["output"]["sets"][1], result["output"]["polarity"][1], strict=True))
    # Direct output from CORRMAP 6d1b06e on these maps is first-pass sets
    # [2, 3], components [2, 3], then second-pass sets/components [1, 2, 3].
    assert second_pairs == {1: 1, 2: 2, 3: 3}
    assert second_polarities == {1: 1, 2: -1, 3: 1}
    np.testing.assert_array_equal(result["output"]["sets"][0], [2, 3])
    np.testing.assert_array_equal(result["output"]["ics"][0], [2, 3])
    np.testing.assert_allclose(result["corr"]["abs_values"][0][:3], [1.0, 0.9993692940674627, 0.5046949386828399])
    np.testing.assert_array_equal(result["clust"]["sets"]["absent"][0], [1, 4])
    np.testing.assert_array_equal(result["clust"]["sets"]["absent"][1], [4])
    assert result["clust"]["best_th"] == 0.95
    assert result["clust"]["similarity"] > 0.999

    child = matched_study["cluster"][1]
    assert child["name"] == "test2 1"
    assert child["algorithm"][0] == "correlation (CORRMAP)"
    assert dict(zip(child["sets"][0], child["comps"], strict=True)) == second_pairs
    assert matched_study["cluster"][0]["child"] == ["test2 1"]
    assert [eeg.get("badcomps", []) for eeg in matched_datasets] == [[1], [2], [3], []]
    assert all("badcomps" not in eeg for eeg in alleeg)
    assert command.startswith("CORRMAP, STUDY, ALLEEG = pop_corrmap(")

    scaled_study, scaled_alleeg = _corrmap_study(scales=(400.0, 0.002))
    scaled, _study, _datasets = pop_corrmap(scaled_study, scaled_alleeg, 1, 1, th="auto", ics=1)
    np.testing.assert_allclose(result["output"]["average_plot"], scaled["output"]["average_plot"], atol=1e-12)


def test_corrmap_aligns_labelled_montages_and_rejects_unsupported_inputs():
    study, alleeg = _corrmap_study()
    expected, _study, _datasets = corrmap(study, alleeg, 1, 1, th=0.95, ics=1)
    permutation = np.array([5, 3, 1, 4, 2, 0])
    alleeg[1]["icawinv"] = np.asarray(alleeg[1]["icawinv"])[permutation]
    alleeg[1]["icaweights"] = np.linalg.pinv(alleeg[1]["icawinv"])
    alleeg[1]["chanlocs"] = [alleeg[1]["chanlocs"][index] for index in permutation]
    reordered_study, alleeg = pop_study(None, alleeg, name="Reordered CORRMAP study")

    actual, _study, _datasets = corrmap(reordered_study, alleeg, 1, 1, th=0.95, ics=1)

    np.testing.assert_allclose(actual["corr"]["abs_values"], expected["corr"]["abs_values"], atol=1e-12)
    np.testing.assert_allclose(actual["output"]["average_plot"], expected["output"]["average_plot"], atol=1e-12)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        corrmap(study, alleeg, 1, 1, th=1.0)
    with pytest.raises(ValueError, match="1, 2, or 3"):
        corrmap(study, alleeg, 1, 1, ics=4)
    with pytest.raises(ValueError, match="template component"):
        corrmap(study, alleeg, 1, 4, th=0.8, ics=1)
    with pytest.raises(NotImplementedError, match="summary plotting"):
        corrmap(study, alleeg, 1, 1, th=0.8, ics=1, plot="on")


@_reference("std_editset", "test_test_std_editset")
def test_std_editset_loads_generated_sets_assigns_metadata_and_removes_a_dataset(tmp_path: Path):
    first = _deterministic_eeg("ignore", "", "")
    second = _deterministic_eeg("probe", "", "", offset=0.2)
    first_path = tmp_path / "Ignore.set"
    second_path = tmp_path / "Probe.set"
    pop_saveset(first, first_path)
    pop_saveset(second, second_path)

    study, alleeg = std_editset(
        None,
        None,
        "commands",
        [
            ["index", 1, "load", first_path, "subject", "S01", "condition", "ignore"],
            ["index", 2, "load", second_path, "subject", "S01", "condition", "probe"],
        ],
        "updatedat",
        "off",
    )
    study = std_makedesign(study, alleeg, 1, variable1="condition", values1=["ignore", "probe"])
    study, alleeg = std_editset(study, alleeg, commands=[["remove", 2]], updatedat="off")

    assert len(alleeg) == 1
    assert alleeg[0]["setname"] == "ignore"
    assert [(row["subject"], row["condition"]) for row in study["datasetinfo"]] == [("S01", "ignore")]
    assert study["design"][0]["variable"][0]["value"] == ["ignore", "probe"]


@_reference("std_makedesign", "test_test_std_makedesign")
def test_std_makedesign_preserves_subject_selection_and_combined_factor_levels():
    datasets = []
    for subject in ("S02", "S07", "S08", "S10"):
        datasets.extend(
            [
                _deterministic_eeg(f"{subject}_syn", subject, "synonyms"),
                _deterministic_eeg(f"{subject}_nonsyn", subject, "non-synonyms", offset=0.2),
            ]
        )
    study, alleeg = pop_study(None, datasets, name="Generated design study")
    selected_subjects = ["S02", "S07", "S08", "S10"]

    study = std_makedesign(
        study,
        alleeg,
        1,
        variable1="condition",
        name="STUDY.design 1",
        values1=["non-synonyms", "synonyms"],
        subjselect=selected_subjects,
    )
    study = std_makedesign(
        study,
        alleeg,
        2,
        variable1="condition",
        name="Design 2 test",
        values1=["non-synonyms", ["non-synonyms", "synonyms"]],
        subjselect=selected_subjects,
    )

    assert len(study["design"]) == 2
    assert study["design"][0]["cases"]["value"] == selected_subjects
    assert study["design"][1]["cases"]["value"] == selected_subjects
    assert study["design"][1]["variable"][0]["value"][1] == ["non-synonyms", "synonyms"]


@_reference("std_precomp", "test_test_std_precomp")
def test_std_precomp_computes_channel_and_component_erp_spectrum_ersp_and_itc():
    study, alleeg = _study_pair()
    tf_params = {"cycles": 0, "nfreqs": 8, "timesout": 8}

    study, alleeg = std_precomp(
        study,
        alleeg,
        "components",
        recompute="on",
        erp="on",
        scalp="on",
        spec="on",
        ersp="on",
        itc="on",
        erspparams=tf_params,
    )
    study, alleeg = std_precomp(
        study,
        alleeg,
        "channels",
        recompute="on",
        erp="on",
        spec="on",
        ersp="on",
        itc="on",
        erspparams=tf_params,
    )

    channel = study["changrp"][0]
    component = study["cluster"][0]
    np.testing.assert_allclose(np.asarray(channel["erpdata"])[0], np.mean(alleeg[0]["data"][0], axis=1))
    assert np.asarray(component["erpdata"]).shape == (2, 3, 128)
    assert np.asarray(component["topo"]).shape == (2, 3, 4)
    assert np.asarray(channel["erspdata"]).shape == np.asarray(channel["itcdata"]).shape
    assert np.nanmin(channel["itcdata"]) >= 0.0
    assert np.nanmax(channel["itcdata"]) <= 1.0 + 1e-12
    peak = int(np.argmax(np.asarray(channel["specdata"])[0]))
    assert channel["specfreqs"][peak] == 6.0


@_reference("std_preclust", "test_test_std_preclust")
def test_std_preclust_combines_all_current_measure_families_and_final_pca():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(
        study,
        alleeg,
        "components",
        recompute="on",
        erp="on",
        scalp="on",
        spec="on",
        ersp="on",
        itc="on",
        erspparams={"cycles": 0, "nfreqs": 6, "timesout": 6},
    )

    study, _alleeg = std_preclust(
        study,
        alleeg,
        1,
        ["spec", "npca", 4, "norm", 1, "weight", 1, "freqrange", [3, 25]],
        ["erp", "npca", 4, "norm", 1, "weight", 1, "timewindow", []],
        ["scalp", "npca", 4, "norm", 1, "weight", 1, "abso", 1],
        ["dipoles", "norm", 1, "weight", 10],
        ["ersp", "npca", 4, "freqrange", [], "timewindow", [], "norm", 1, "weight", 1],
        ["itc", "npca", 4, "freqrange", [], "timewindow", [], "norm", 1, "weight", 1],
        ["finaldim", "npca", 4],
    )

    preclust = study["etc"]["preclust"]
    assert np.asarray(preclust["preclustdata"]).shape == (6, 4)
    assert [item["measure"] for item in preclust["preclustparams"]] == [
        "spec",
        "erp",
        "scalp",
        "dipoles",
        "ersp",
        "itc",
        "finaldim",
    ]
    assert np.isfinite(preclust["preclustdata"]).all()


@_reference("pop_clust", "test_test_pop_clust")
def test_pop_clust_runs_current_kmeanscluster_scenario_with_ten_clusters():
    study, alleeg = _study_pair(n_channels=6, n_components=6)
    study, alleeg = std_preclust(study, alleeg, 1, ["scalp", "npca", 4, "norm", 1, "weight", 1])

    study = pop_clust(study, alleeg, algorithm="kmeanscluster", clus_num=10, random_state=11)

    children = study["cluster"][1:]
    assert len(children) == 10
    assert sum(len(cluster["comps"]) for cluster in children) == 12
    assert study["cluster"][0]["child"] == [cluster["name"] for cluster in children]


@_reference("std_selectdesign", "test_test_std_selectdesign")
def test_std_selectdesign_scans_generated_designs_without_corrupting_component_membership():
    study, alleeg = _study_pair()
    study = std_makedesign(study, alleeg, 2, variable1="subject", values1=["S01"], name="S01")
    study = std_makedesign(study, alleeg, 3, variable1="subject", values1=["S02"], name="S02")
    study, alleeg = std_preclust(study, alleeg)
    original_pairs = (deepcopy(study["cluster"][0]["sets"]), deepcopy(study["cluster"][0]["comps"]))

    for design_index in range(1, 4):
        selected = std_selectdesign(study, alleeg, design_index)
        assert selected["currentdesign"] == design_index
        assert (selected["cluster"][0]["sets"], selected["cluster"][0]["comps"]) == original_pairs


@_reference("std_erpplot", "test_test_stderpplot2")
def test_std_erpplot_channel_output_matches_direct_epoch_average():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(study, alleeg, [1], erp="on", recompute="on")

    _study, erpdata, erptimes, figure = std_erpplot(study, alleeg, channels=[1])

    expected = {eeg["condition"]: np.mean(eeg["data"][0], axis=1) for eeg in alleeg}
    for condition, cell in zip(["standard", "target"], erpdata):
        np.testing.assert_allclose(cell[:, 0], expected[condition], atol=1e-12)
    np.testing.assert_allclose(erptimes, alleeg[0]["times"], atol=1e-12)
    assert len(figure.axes[0].lines) == 1
    plt.close(figure)


@_reference("std_erpplot", "test_test_stderpplot3")
def test_std_erpplot_component_output_matches_direct_scaled_activation_average():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(study, alleeg, "components", erp="on", scalp="on", recompute="on")

    _study, erpdata, erptimes, figure = std_erpplot(study, alleeg, clusters=1, components=[2])

    expected = []
    for eeg in alleeg:
        scale = float(np.sqrt(np.mean(np.asarray(eeg["icawinv"])[:, 1] ** 2)))
        expected.append(np.mean(component_activations(eeg)[1], axis=1) * scale)
    np.testing.assert_allclose(erpdata[0][:, 0], expected[1], atol=1e-12)
    np.testing.assert_allclose(erpdata[1][:, 0], expected[0], atol=1e-12)
    np.testing.assert_allclose(erptimes, alleeg[0]["times"], atol=1e-12)
    plt.close(figure)


@_reference("std_specplot", "test_test_stdspecplot3")
def test_std_specplot_channel_output_preserves_known_oscillation_peak():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(study, alleeg, [2], spec="on", recompute="on")

    _study, specdata, frequencies, figure = std_specplot(study, alleeg, channels=[1])

    assert all(frequencies[int(np.argmax(cell[:, 0]))] == 8.0 for cell in specdata)
    assert all(np.isfinite(cell).all() for cell in specdata)
    plt.close(figure)


@_reference("std_specplot", "test_test_stdspecplot4")
def test_std_specplot_component_output_preserves_known_activation_peak():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(study, alleeg, "components", spec="on", recompute="on")

    _study, specdata, frequencies, figure = std_specplot(study, alleeg, clusters=1, components=[3])

    assert all(frequencies[int(np.argmax(cell[:, 0]))] == 12.0 for cell in specdata)
    assert all(np.isfinite(cell).all() for cell in specdata)
    plt.close(figure)


@_reference("std_erspplot", "test_test_std_erspplot2")
def test_std_erspplot_channel_output_matches_precomputed_axes_and_cache():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(
        study,
        alleeg,
        [1],
        ersp="on",
        recompute="on",
        erspparams={"cycles": 0, "nfreqs": 8, "timesout": 8},
    )

    _study, erspdata, times, frequencies, figure = std_erspplot(study, alleeg, channels=[1])

    raw = np.asarray(study["changrp"][0]["erspdata"])
    np.testing.assert_allclose(erspdata[0][..., 0], raw[1])
    np.testing.assert_allclose(erspdata[1][..., 0], raw[0])
    np.testing.assert_allclose(times, study["changrp"][0]["ersptimes"])
    np.testing.assert_allclose(frequencies, study["changrp"][0]["erspfreqs"])
    assert all(np.isfinite(cell).all() for cell in erspdata)
    plt.close(figure)


@_reference("std_erspplot", "test_test_std_erspplot3")
def test_std_erspplot_component_output_selects_the_requested_component():
    study, alleeg = _study_pair()
    study, alleeg = std_precomp(
        study,
        alleeg,
        "components",
        ersp="on",
        recompute="on",
        erspparams={"cycles": 0, "nfreqs": 8, "timesout": 8},
    )

    _study, erspdata, times, frequencies, figure = std_erspplot(study, alleeg, clusters=1, components=[2])

    expected = np.asarray(study["cluster"][0]["erspdata"])[:, 1]
    np.testing.assert_allclose(erspdata[0][..., 0], expected[1])
    np.testing.assert_allclose(erspdata[1][..., 0], expected[0])
    assert all(cell.shape == (frequencies.size, times.size, 1) for cell in erspdata)
    assert all(np.isfinite(cell).all() for cell in erspdata)
    plt.close(figure)
