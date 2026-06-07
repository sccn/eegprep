from __future__ import annotations

import ast
from copy import deepcopy
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eegprep
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot
from eegprep.functions.studyfunc.pop_clust import pop_clust
from eegprep.functions.studyfunc.pop_loadstudy import pop_loadstudy
from eegprep.functions.studyfunc.pop_preclust import pop_preclust
from eegprep.functions.studyfunc.pop_precomp import pop_precomp, pop_precomp_dialog_spec
from eegprep.functions.studyfunc.pop_savestudy import pop_savestudy
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_erpplot import std_erpplot
from eegprep.functions.studyfunc.std_erspplot import std_erspplot
from eegprep.functions.studyfunc.std_dipoleclusters import std_dipoleclusters
from eegprep.functions.studyfunc.std_dipplot import std_dipplot
from eegprep.functions.studyfunc.std_interp import std_interp
from eegprep.functions.studyfunc.std_itcplot import std_itcplot
from eegprep.functions.studyfunc.std_limodesign import std_limodesign
from eegprep.functions.studyfunc.std_precomp import std_precomp
from eegprep.functions.studyfunc.std_checkdatasession import std_checkdatasession
from eegprep.functions.studyfunc.std_checkfiles import std_checkfiles
from eegprep.functions.studyfunc.std_prepare_neighbors import std_prepare_neighbors
from eegprep.functions.studyfunc.std_readdata import (
    std_readdata,
    std_readerp,
    std_readitc,
    std_readpac,
    std_readspec,
    std_readtopo,
)
from eegprep.functions.studyfunc.std_savedat import std_savedat
from eegprep.functions.studyfunc.std_specplot import std_specplot
from eegprep.functions.studyfunc.std_uniformfiles import std_uniformfiles
from eegprep.functions.studyfunc.std_uniformsetinds import std_uniformsetinds
from tests.fixtures import create_test_eeg, create_test_eeg_with_ica


class _Renderer:
    def __init__(self, result):
        self.result = result
        self.spec = None

    def run(self, spec, initial_values=None):
        self.spec = spec
        return self.result


def _study_pair():
    first = create_test_eeg(n_channels=3, n_samples=64, n_trials=4, srate=128)
    first.update({"setname": "one", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "two", "subject": "S02", "condition": "standard"})
    second["data"] = second["data"] * 2.0
    return pop_study(None, [first, second], name="Measure study")


def test_std_precomp_channel_measures_store_eeglab_named_fields():
    study, alleeg = _study_pair()

    study, alleeg, command = std_precomp(study, alleeg, [1, 2], erp="on", spec="on", return_com=True)

    assert len(study["changrp"]) == 2
    assert study["changrp"][0]["measureinfo"]["kind"] == "channels"
    assert study["changrp"][0]["measureinfo"]["computed"] == ["erp", "spec"]
    assert np.asarray(study["changrp"][0]["erpdata"]).shape == (2, alleeg[0]["pnts"])
    assert np.asarray(study["changrp"][0]["specdata"]).shape[0] == 2
    np.testing.assert_allclose(
        np.asarray(study["changrp"][0]["erpdata"])[0],
        np.mean(alleeg[0]["data"][0], axis=1),
    )
    ast.parse(command)

    _study, erpdata, times, _freqs = std_readerp(study, alleeg, channels=[1])
    _study, specdata, freqs, _unused = std_readspec(study, alleeg, channels=[1])
    assert erpdata[0].shape == (2, alleeg[0]["pnts"])
    assert times.size == alleeg[0]["pnts"]
    assert specdata[0].shape[0] == 2
    assert freqs.size == specdata[0].shape[1]

    collapsed = deepcopy(study)
    collapsed["changrp"][0]["erpdata"] = np.asarray(collapsed["changrp"][0]["erpdata"])[0]
    with pytest.raises(ValueError, match="dataset-axis"):
        std_readerp(collapsed, alleeg, channels=[1], subject="S01")


def test_std_precomp_baseline_and_design_contract(caplog):
    study, alleeg = _study_pair()
    alleeg[0]["times"] = np.asarray([-100.0, 0.0, 100.0, 200.0])
    alleeg[0]["xmin"] = -0.1
    alleeg[0]["xmax"] = 0.2
    alleeg[0]["pnts"] = 4
    alleeg[0]["data"] = np.asarray([[[1.0, 3.0], [5.0, 7.0], [20.0, 30.0], [40.0, 50.0]]])
    alleeg[0]["nbchan"] = 1
    alleeg[0]["chanlocs"] = [{"labels": "Cz"}]
    alleeg[1] = deepcopy(alleeg[0])
    alleeg[1]["data"] = alleeg[1]["data"] + 10.0

    with caplog.at_level(logging.WARNING, logger="eegprep.functions.studyfunc.std_precomp"):
        study, _alleeg, _command = std_precomp(
            study,
            alleeg,
            [1],
            erp="on",
            design=2,
            erpparams={"rmbase": [-100, 0]},
            interp="on",
            return_com=True,
        )

    erpdata = np.asarray(study["changrp"][0]["erpdata"])
    assert study["changrp"][0]["measureinfo"]["design"] == 2
    assert study["etc"]["eegprep"]["study_measures"]["design"] == 2
    np.testing.assert_allclose(np.mean(erpdata[:, :2], axis=1), 0.0)
    assert "ignoring EEGLAB-only option(s): interp" in caplog.text


def test_pop_precomp_channel_measures_and_cached_plot_are_replayable():
    study, alleeg = _study_pair()

    study, alleeg, command = pop_precomp(study, alleeg, "channels", erp="on", spec="on", return_com=True)
    original_etc = study["etc"]
    study, plot_command, figure = pop_chanplot(study, alleeg, channels=np.asarray([1]), measure="spec", return_com=True)

    assert command.startswith("STUDY, ALLEEG = pop_precomp(")
    assert study["etc"]["last_chanplot"] == {"measure": "spec", "mode": "channels", "channels": [1]}
    assert "last_chanplot" not in original_etc
    assert "channels=[1]" in plot_command
    ast.parse(command)
    ast.parse(plot_command)
    plt.close(figure)


def test_study_measure_roundtrip_and_std_plot_helpers(tmp_path):
    study, alleeg = _study_pair()
    study, alleeg = pop_precomp(
        study,
        alleeg,
        "channels",
        erp="on",
        spec="on",
        ersp="on",
        itc="on",
        erspparams={"cycles": 0, "nfreqs": 5, "timesout": 5},
    )

    study, erpdata, erptimes, erpfig, erpcom = std_erpplot(
        study, alleeg, channels=["Ch1"], timerange=[0, 250], return_com=True
    )
    study, specdata, specfreqs, specfig = std_specplot(study, alleeg, channels=[1])
    study, erspdata, ersptimes, erspfreqs, erspfig, erspcom = std_erspplot(study, alleeg, channels=[1], return_com=True)
    study, itcdata, itctimes, itcfreqs, itcfig = std_itcplot(study, alleeg, channels=[1])
    plotted, plot_command, plotfig = pop_chanplot(study, alleeg, channels=["Ch1"], measure="erp", return_com=True)

    assert erpdata[0].shape[0] == 2
    assert erptimes[0] >= 0
    assert specdata[0].shape[0] == 2
    assert specfreqs.size == specdata[0].shape[1]
    assert erspdata[0].shape == itcdata[0].shape
    assert ersptimes.size == itctimes.size
    assert erspfreqs.size == itcfreqs.size
    assert erpcom.startswith("STUDY, ERPDATA, ERPTIMES, FIGURE = std_erpplot(")
    namespace = {"STUDY": study, "ALLEEG": alleeg, "std_erpplot": std_erpplot}
    exec(erpcom, namespace)
    assert isinstance(namespace["STUDY"], dict)
    assert namespace["ERPDATA"][0].shape == erpdata[0].shape
    assert erspcom.startswith("STUDY, ERSPDATA, ERSPTIMES, ERSPFREQS, FIGURE = std_erspplot(")
    namespace = {"STUDY": study, "ALLEEG": alleeg, "std_erspplot": std_erspplot}
    exec(erspcom, namespace)
    assert isinstance(namespace["STUDY"], dict)
    assert namespace["ERSPDATA"][0].shape == erspdata[0].shape
    assert plotted["etc"]["last_chanplot"]["channels"] == [1]
    assert "channels=['Ch1']" in plot_command

    saved, _command = pop_savestudy(study, alleeg, filename="measures.study", filepath=tmp_path, return_com=True)
    loaded, loaded_alleeg, _load_command = pop_loadstudy(
        "measures.study", filepath=tmp_path, load_datasets=False, return_com=True
    )

    assert saved["saved"] == "yes"
    assert loaded_alleeg == []
    np.testing.assert_allclose(loaded["changrp"][0]["erpdata"], study["changrp"][0]["erpdata"])
    np.testing.assert_allclose(loaded["changrp"][0]["specdata"], study["changrp"][0]["specdata"])
    for figure in (erpfig, specfig, erspfig, itcfig, plotfig):
        plt.close(figure)


def test_std_precomp_ersp_and_itc_store_frequency_time_axes():
    study, alleeg = _study_pair()

    study, _alleeg, _command = std_precomp(
        study,
        alleeg,
        [1],
        ersp="on",
        itc="on",
        erspparams={"cycles": 0, "nfreqs": 6, "timesout": 6},
        return_com=True,
    )

    group = study["changrp"][0]
    assert np.asarray(group["erspdata"]).shape == np.asarray(group["itcdata"]).shape
    assert np.asarray(group["erspdata"]).shape[0] == 2
    assert len(group["ersptimes"]) <= 6
    assert len(group["erspfreqs"]) <= 6

    _study, erspdata, times, freqs = std_readdata(study, alleeg, datatype="ersp", channels=[1])
    assert erspdata[0].shape == np.asarray(group["erspdata"]).shape
    assert times.size == len(group["ersptimes"])
    assert freqs.size == len(group["erspfreqs"])

    _study, itcdata, sliced_times, sliced_freqs = std_readitc(
        study,
        alleeg,
        channels=[1],
        timerange=[group["itctimes"][0], group["itctimes"][-1]],
        freqrange=[group["itcfreqs"][0], group["itcfreqs"][-1]],
        subject="S01",
    )

    assert itcdata[0].shape[0] == 1
    assert sliced_times.size == len(group["itctimes"])
    assert sliced_freqs.size == len(group["itcfreqs"])


def test_std_precomp_component_measures_and_parent_cluster_plot():
    first = create_test_eeg_with_ica(n_channels=5, n_samples=72, n_trials=3, n_components=3)
    first.update({"setname": "ica1", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "ica2", "subject": "S02", "condition": "standard"})
    study, alleeg = pop_study(None, [first, second], name="Component study")

    study, alleeg, command = pop_precomp(study, alleeg, "components", erp="on", spec="on", scalp="on", return_com=True)
    study, plot_command, figure = pop_chanplot(
        study, alleeg, components=[1], measure="erp", mode="components", return_com=True
    )

    cluster = study["cluster"][0]
    assert cluster["measureinfo"]["kind"] == "components"
    assert np.asarray(cluster["erpdata"]).shape == (2, 3, first["pnts"])
    assert np.asarray(cluster["topo"]).shape[:2] == (2, 3)
    assert "components=[1]" in plot_command
    ast.parse(command)
    ast.parse(plot_command)

    _study, topodata, channel_axis = std_readtopo(study, alleeg, clusters=1, components=[1])
    assert topodata[0].shape == (2, 1, first["nbchan"])
    assert channel_axis.tolist() == [1, 2, 3, 4, 5]
    cluster["pacdata"] = np.ones((2, 3, 4))
    cluster["pactimes"] = [0.0, 10.0, 20.0, 30.0]
    cluster["pacfreqs"] = [4.0, 8.0, 12.0]
    _study, pacdata, pactimes, pacfreqs = std_readpac(study, alleeg, clusters=1)
    assert pacdata[0].shape == (2, 3, 4)
    assert pactimes.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert pacfreqs.tolist() == [4.0, 8.0, 12.0]
    with pytest.raises(ValueError, match="component selection"):
        std_readpac(study, alleeg, clusters=1, components=[1])
    del cluster["pacdata"]
    with pytest.raises(NotImplementedError, match="PAC"):
        std_readpac(study, alleeg, clusters=1)
    plt.close(figure)


def test_component_measure_reads_map_requested_component_ids_to_cached_axis():
    eeg = create_test_eeg_with_ica(n_channels=5, n_samples=72, n_trials=3, n_components=4)
    eeg.update({"setname": "ica1", "subject": "S01", "condition": "target"})
    study, alleeg = pop_study(None, [eeg], name="Component subset study")
    study["datasetinfo"][0]["comps"] = [2, 4]
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on", allcomps="off")
    cluster = study["cluster"][0]
    raw = np.asarray(cluster["erpdata"], dtype=float)

    _study, selected, _times, _freqs = std_readerp(study, alleeg, clusters=1, components=[2])

    assert cluster["measureinfo"]["components"] == [2, 4]
    np.testing.assert_allclose(selected[0], raw[:, [0], :])
    with pytest.raises(ValueError, match="available component IDs: 2, 4"):
        std_readerp(study, alleeg, clusters=1, components=[1])

    study, _command, figure = pop_chanplot(
        study, alleeg, components=[4], measure="erp", mode="components", return_com=True
    )

    assert study["etc"]["last_chanplot"]["components"] == [4]
    assert figure.axes[0].get_legend().get_texts()[0].get_text() == "IC 4"
    plt.close(figure)


def test_component_precompute_preserves_per_dataset_component_pairs():
    first = create_test_eeg_with_ica(n_channels=5, n_samples=72, n_trials=3, n_components=4)
    first.update({"setname": "ica1", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "ica2", "subject": "S02", "condition": "standard"})
    study, alleeg = pop_study(None, [first, second], name="Component pair study")
    study["datasetinfo"][0]["comps"] = [1, 2]
    study["datasetinfo"][1]["comps"] = [3, 4]

    study, alleeg = pop_precomp(study, alleeg, "components", erp="on", allcomps="off")
    cluster = study["cluster"][0]
    study, alleeg = pop_preclust(study, alleeg, preproc=[{"measure": "erp", "npca": 2, "norm": 0}])

    assert cluster["sets"] == [[1, 1, 2, 2]]
    assert cluster["comps"] == [1, 2, 3, 4]
    assert np.asarray(cluster["erpdata"], dtype=float).shape == (2, 4, first["pnts"])
    assert study["etc"]["preclust"]["preclustcomps"] == [
        {"set": 1, "comp": 1},
        {"set": 1, "comp": 2},
        {"set": 2, "comp": 3},
        {"set": 2, "comp": 4},
    ]


def test_child_cluster_measure_reads_slice_parent_component_cache():
    first = create_test_eeg_with_ica(n_channels=5, n_samples=72, n_trials=3, n_components=3)
    first.update({"setname": "ica1", "subject": "S01", "condition": "target"})
    second = deepcopy(first)
    second.update({"setname": "ica2", "subject": "S02", "condition": "standard"})
    study, alleeg = pop_study(None, [first, second], name="Cluster measure study")
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on")
    study, alleeg = pop_preclust(study, alleeg, preproc=[{"measure": "erp", "npca": 2}])
    study = pop_clust(study, alleeg, clus_num=2, random_state=11)
    child = study["cluster"][1]

    _study, erpdata, _times, figure, _command = std_erpplot(study, alleeg, clusters=[2], noplot="on", return_com=True)

    assert erpdata[0].shape == (len(child["comps"]), first["pnts"])
    assert figure is None
    with pytest.raises(ValueError, match="subject filter requires"):
        std_readerp(study, alleeg, clusters=[2], subject="S01")


def test_precomp_missing_ica_and_unknown_channel_paths_fail_clearly():
    study, alleeg = _study_pair()

    with pytest.raises(ValueError, match="ICA"):
        pop_precomp(study, alleeg, "components", erp="on")
    with pytest.raises(ValueError, match="Unknown channel"):
        pop_precomp(study, alleeg, ["missing"], erp="on")

    no_locs = deepcopy(alleeg[0])
    no_locs["chanlocs"] = []
    no_locs_study, no_locs_alleeg = pop_study(None, [no_locs], name="No locs")
    computed, _alleeg, _command = pop_precomp(no_locs_study, no_locs_alleeg, [1], erp="on", return_com=True)
    assert computed["changrp"][0]["name"] == "1"


def test_study_consistency_helpers_and_savedat(tmp_path):
    study, alleeg = _study_pair()
    study, alleeg = pop_precomp(study, alleeg, "channels", erp="on")
    ok, report = std_checkfiles(study, alleeg, return_report=True)
    session_ok, session_report = std_checkdatasession(study, alleeg, return_report=True)
    json_path = std_savedat(tmp_path / "erp.json", {"erpdata": study["changrp"][0]["erpdata"]})
    mat_path = std_savedat(tmp_path / "erp.dat", {"erpdata": study["changrp"][0]["erpdata"]})

    assert ok == 1
    assert report["measure_cache"]["checked"]
    assert std_uniformfiles(study, alleeg) == 1
    assert std_uniformsetinds(study) == 1
    assert session_ok is True
    assert session_report["duplicate_subject_sessions"] == []
    assert json_path.exists()
    assert mat_path.exists()

    mismatched = deepcopy(alleeg)
    mismatched[1]["chanlocs"][0]["labels"] = "Mismatch"
    assert std_uniformfiles(study, mismatched) == 0


def test_std_uniformsetinds_treats_nan_as_matching_missing_dataset() -> None:
    study = {"changrp": [{"sets": [1, np.nan]}, {"sets": [1, np.nan]}]}
    assert std_uniformsetinds(study) == 1

    study["changrp"][1]["sets"] = [1, 2]
    assert std_uniformsetinds(study) == 0


def test_pop_precomp_gui_cancel_and_dialog_spec():
    study, alleeg = _study_pair()
    spec = pop_precomp_dialog_spec(study, alleeg, "components")
    controls = controls_by_tag(spec)

    assert spec.function_name == "pop_precomp"
    assert spec.eeglab_source == "functions/studyfunc/pop_precomp.m"
    assert controls["mode"].value == 2
    assert controls["erp_on"].value is True

    returned_study, returned_alleeg, command = pop_precomp(
        study, alleeg, gui=True, renderer=_Renderer(None), return_com=True
    )
    assert returned_study == study
    assert returned_alleeg == alleeg
    assert command == ""


def test_pop_chanplot_gui_component_mode_uses_cached_measures():
    first = create_test_eeg_with_ica(n_channels=4, n_samples=64, n_trials=3, n_components=2)
    first.update({"subject": "S01", "condition": "target"})
    study, alleeg = pop_study(None, [first], name="Component study")
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on")
    renderer = _Renderer({"mode": 4, "channels": "", "components": "1", "measure": 1})

    study, command, figure = pop_chanplot(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert renderer.spec is not None
    assert study["etc"]["last_chanplot"] == {"measure": "erp", "mode": "components", "components": [1]}
    assert "mode='components'" in command
    plt.close(figure)


def test_pop_chanplot_gui_routes_measure_buttons_to_component_cache():
    first = create_test_eeg_with_ica(n_channels=4, n_samples=64, n_trials=3, n_components=2)
    first.update({"subject": "S01", "condition": "target"})
    study, alleeg = pop_study(None, [first], name="Component study")
    study, alleeg = pop_precomp(study, alleeg, "components", erp="on")
    renderer = _Renderer({"chan_list": [1], "measure_action": "erp"})

    study, command, figure = pop_chanplot(study, alleeg, gui=True, renderer=renderer, return_com=True)

    assert study["etc"]["last_chanplot"] == {"measure": "erp", "mode": "components", "components": [1, 2]}
    assert "mode='components'" in command
    plt.close(figure)


def test_study_measure_menu_actions_are_implemented():
    assert action_kind("pop_precomp:channels") == "implemented"
    assert action_kind("pop_precomp:components") == "implemented"
    assert action_kind("pop_chanplot") == "implemented"


def test_study_measure_gui_dispatch_updates_session_history(monkeypatch):
    study, alleeg = _study_pair()
    session = EEGPrepSession(STUDY=study, ALLEEG=alleeg, CURRENTSTUDY=1)
    dispatcher = MenuActionDispatcher(session)

    def fake_pop_precomp(STUDY, ALLEEG, chanorcomp="channels", *args, **kwargs):
        return pop_precomp(STUDY, ALLEEG, chanorcomp, erp="on", return_com=kwargs.get("return_com", False))

    monkeypatch.setattr("eegprep.functions.studyfunc.pop_precomp.pop_precomp", fake_pop_precomp)

    dispatcher.dispatch("pop_precomp:channels")

    assert session.STUDY["changrp"]
    assert session.LASTCOM.startswith("STUDY, ALLEEG = pop_precomp(")


def test_study_chanplot_gui_dispatch_cancel_leaves_history_empty(monkeypatch):
    study, alleeg = _study_pair()
    session = EEGPrepSession(STUDY=study, ALLEEG=alleeg, CURRENTSTUDY=1)
    dispatcher = MenuActionDispatcher(session)

    def fake_pop_chanplot(STUDY, ALLEEG, *args, **kwargs):
        return STUDY, "", None

    monkeypatch.setattr("eegprep.functions.studyfunc.pop_chanplot.pop_chanplot", fake_pop_chanplot)

    dispatcher.dispatch("pop_chanplot")

    assert session.STUDY is study
    assert session.LASTCOM == ""
    assert session.ALLCOM == []


def test_std_limodesign_builds_categorical_continuous_and_split_exports(tmp_path):
    factors = [
        {"label": "condition", "value": "target", "vartype": "categorical"},
        {"label": "condition", "value": "standard", "vartype": "categorical"},
        {"label": "group", "value": "control", "vartype": "categorical"},
        {"label": "group", "value": "patient", "vartype": "categorical"},
        {"label": "rt", "vartype": "continuous"},
    ]
    trialinfo = [
        {"condition": "target", "group": "control", "rt": 1.0},
        {"condition": "standard", "group": "control", "rt": 2.0},
        {"condition": "target", "group": "patient", "rt": 3.0},
        {"condition": "standard", "group": "patient", "rt": 4.0},
    ]

    catmat, contmat, limodesign, command = std_limodesign(
        factors,
        trialinfo,
        interaction="on",
        splitreg="on",
        filepath=tmp_path,
        return_com=True,
    )

    np.testing.assert_allclose(catmat.ravel(), [1, 3, 2, 4])
    assert contmat.shape == (4, 4)
    assert np.count_nonzero(contmat[0]) == 0
    assert len(limodesign["categorical"][0]) == 4
    assert len(limodesign["continuous"]) == 4
    assert (tmp_path / "categorical_variables.txt").is_file()
    assert (tmp_path / "continuous_variables.txt").is_file()
    assert command.startswith("catMat, contMat, limodesign = std_limodesign(")
    ast.parse(command)


def test_std_prepare_neighbors_returns_limo_adjacency_and_exports_name():
    study, alleeg = _study_pair()

    study, neighbors, limostruct, command = std_prepare_neighbors(study, alleeg, force="on", return_com=True)

    adjacency = np.asarray(limostruct["channeighbstructmat"])
    assert len(neighbors) == alleeg[0]["nbchan"]
    assert adjacency.shape == (alleeg[0]["nbchan"], alleeg[0]["nbchan"])
    assert np.array_equal(adjacency, adjacency.T)
    assert np.all(np.diag(adjacency) == 0)
    assert study["etc"]["statistics"]["fieldtrip"]["channelneighbor"] == neighbors
    assert command.startswith("STUDY, neighbors, limostruct = std_prepare_neighbors(")
    assert eegprep.std_prepare_neighbors is std_prepare_neighbors


def test_std_interp_adds_requested_missing_channels_without_dropping_existing():
    study, alleeg = _study_pair()
    reduced = deepcopy(alleeg[1])
    reduced["data"] = reduced["data"][:2]
    reduced["nbchan"] = 2
    reduced["chanlocs"] = reduced["chanlocs"][:2]
    alleeg[1] = reduced

    study, interpolated, command = std_interp(study, alleeg, ["Ch3"], return_com=True)

    assert interpolated[1]["data"].shape[0] == 3
    assert [loc["labels"] for loc in interpolated[1]["chanlocs"]] == ["Ch1", "Ch2", "Ch3"]
    assert study["etc"]["eegprep"]["std_interp"]["changed_datasets"] == [2]
    assert command.startswith("STUDY, ALLEEG = std_interp(")
    assert eegprep.std_interp is std_interp


def test_source_dependent_study_helpers_report_explicit_boundary():
    with pytest.raises(NotImplementedError, match="FieldTrip/DIPFIT STUDY source workflows"):
        std_dipplot({}, [])
    with pytest.raises(NotImplementedError, match="FieldTrip/DIPFIT STUDY source workflows"):
        std_dipoleclusters({}, [])
