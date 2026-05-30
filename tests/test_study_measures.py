from __future__ import annotations

import ast
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher, action_kind
from eegprep.functions.guifunc.spec import controls_by_tag
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.studyfunc.pop_chanplot import pop_chanplot
from eegprep.functions.studyfunc.pop_precomp import pop_precomp, pop_precomp_dialog_spec
from eegprep.functions.studyfunc.pop_study import pop_study
from eegprep.functions.studyfunc.std_precomp import std_precomp
from eegprep.functions.studyfunc.std_readdata import std_readdata, std_readerp, std_readspec
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


def test_pop_precomp_channel_measures_and_cached_plot_are_replayable():
    study, alleeg = _study_pair()

    study, alleeg, command = pop_precomp(study, alleeg, "channels", erp="on", spec="on", return_com=True)
    study, plot_command, figure = pop_chanplot(study, alleeg, channels=[1], measure="spec", return_com=True)

    assert command.startswith("STUDY, ALLEEG = pop_precomp(")
    assert study["etc"]["last_chanplot"] == {"measure": "spec", "mode": "channels", "channels": [1]}
    ast.parse(command)
    ast.parse(plot_command)
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
    renderer = _Renderer({"mode": 2, "channels": "", "components": "1", "measure": 1})

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
