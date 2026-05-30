from __future__ import annotations

import ast
from copy import deepcopy
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
from eegprep.functions.studyfunc.std_itcplot import std_itcplot
from eegprep.functions.studyfunc.std_precomp import std_precomp
from eegprep.functions.studyfunc.std_readdata import std_readdata, std_readerp, std_readspec
from eegprep.functions.studyfunc.std_specplot import std_specplot
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
    study, erspdata, ersptimes, erspfreqs, erspfig = std_erspplot(study, alleeg, channels=[1])
    study, itcdata, itctimes, itcfreqs, itcfig = std_itcplot(study, alleeg, channels=[1])
    plotted, plot_command, plotfig = pop_chanplot(study, alleeg, channels=["Ch1"], measure="erp", return_com=True)

    assert erpdata[0].shape[0] == 2
    assert erptimes[0] >= 0
    assert specdata[0].shape[0] == 2
    assert specfreqs.size == specdata[0].shape[1]
    assert erspdata[0].shape == itcdata[0].shape
    assert ersptimes.size == itctimes.size
    assert erspfreqs.size == itcfreqs.size
    assert erpcom.startswith("STUDY = std_erpplot(")
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
